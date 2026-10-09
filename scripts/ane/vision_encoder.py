"""Qwen3.5 vision encoder with its 27 transformer blocks on the ANE.

Each block is one fixed-size program over 1024 patches, activations [1, 1152, 1, 1024] fp16.
Images and video frame pairs are packed into those 1024 rows; a block-diagonal mask keeps
each one attending only to itself. Patch embedding, positions, and the final 2x2 merger are
small and run on the CPU, so the GPU is never used.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

if __package__:
    from .layers import unit_rms
    from .mil import Graph, T
else:
    from layers import unit_rms
    from mil import Graph, T

ROWS, HIDDEN, HEADS, HEAD_DIM = 1024, 1152, 16, 72
MASK_NEG = -30000.0
FP16_LIMIT = 60000.0


def _linear(g: Graph, x: T, name: str, weight: np.ndarray, bias: np.ndarray) -> T:
    return g.add(g.conv1x1(x, name, weight), g.const_tensor(name + "_b", bias.reshape(1, -1, 1, 1)))


def _layer_norm(g: Graph, x: T, weight: np.ndarray, bias: np.ndarray) -> T:
    mean = g.mul(g.reduce_sum(x, 1), 1.0 / x.shape[1])
    normed = unit_rms(g, g.sub(x, mean), 1)
    return g.add(g.mul(normed, g.const_tensor("ln_w", weight.reshape(1, -1, 1, 1))),
                 g.const_tensor("ln_b", bias.reshape(1, -1, 1, 1)))


def _heads(g: Graph, x: T, rows: int) -> T:
    """[1, 1152, 1, R] -> [1, 16, R, 72]."""
    return g.transpose(g.reshape(x, (1, HEADS, HEAD_DIM, rows)), (0, 1, 3, 2))


def _rope(g: Graph, x: T, cos: T, sin: T) -> T:
    """rotate_half over the whole 72-dim head."""
    rows, half = x.shape[2], HEAD_DIM // 2
    x1 = g.slice(x, (0, 0, 0, 0), (1, HEADS, rows, half))
    x2 = g.slice(x, (0, 0, 0, half), (1, HEADS, rows, half))
    return g.add(g.mul(x, cos), g.mul(g.concat([g.mul(x2, -1.0), x1], 3), sin))


def _gelu_tanh(g: Graph, x: T) -> T:
    """0.5 x (1 + tanh(u)) == x * sigmoid(2u); the clamps keep x^3 and exp inside fp16."""
    xc = g.maximum(g.minimum(x, 12.0), -12.0)
    u2 = g.mul(g.add(xc, g.mul(g.pow(xc, 3.0), 0.044715)), 2 * 0.7978845608028654)
    return g.mul(x, g.sigmoid(g.maximum(g.minimum(u2, 15.0), -15.0)))


def vision_block(w: dict[str, np.ndarray], rows: int = ROWS) -> Graph:
    """norm1 -> bidirectional attention (masked per image / frame pair) -> norm2 -> GELU MLP, both with residuals."""
    g = Graph()
    g.quantize = None  # vision weights keep fp16; LUT-6 was 5-14x less accurate per block without a speed gain
    hid = g.input("hid_in", (1, HIDDEN, 1, rows))
    cos = g.input("cos_in", (1, 1, rows, HEAD_DIM))
    sin = g.input("sin_in", (1, 1, rows, HEAD_DIM))
    mask = g.input("mask_in", (1, 1, rows, rows))
    xn = _layer_norm(g, hid, w["norm1.weight"], w["norm1.bias"])
    qkv = _linear(g, xn, "w_qkv", w["attn.qkv.weight"], w["attn.qkv.bias"])
    part = lambda i: _heads(g, g.slice(qkv, (0, i * HIDDEN, 0, 0), (1, HIDDEN, 1, rows)), rows)
    q, k, v = _rope(g, part(0), cos, sin), _rope(g, part(1), cos, sin), part(2)
    scores = g.add(g.mul(g.matmul(q, k, transpose_y=True), HEAD_DIM ** -0.5), mask)    # [1, 16, R, R]
    e = g.exp(g.sub(scores, g.reduce_max(scores, 3)))
    boost = 16.0                                                                          # probabilities are small; scale for the matmul
    p = g.mul(e, g.div(T(g.scalar(boost), (), "fp16"), g.reduce_sum(e, 3)))
    context = g.mul(g.matmul(p, v), 1.0 / boost)
    flat = g.reshape(g.transpose(context, (0, 1, 3, 2)), (1, HIDDEN, 1, rows))
    hid = g.add(hid, _linear(g, flat, "w_proj", w["attn.proj.weight"], w["attn.proj.bias"]))
    xn = _layer_norm(g, hid, w["norm2.weight"], w["norm2.bias"])
    act = _gelu_tanh(g, _linear(g, xn, "w_fc1", w["mlp.linear_fc1.weight"], w["mlp.linear_fc1.bias"]))
    hid = g.add(hid, _linear(g, act, "w_fc2", w["mlp.linear_fc2.weight"], w["mlp.linear_fc2.bias"]))
    g.output(g.rename(hid, "hid_out"))
    return g


def pack(cu_seqlens: list[int], rows: int = ROWS) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Group consecutive images / frame pairs into calls of at most `rows` patches, never splitting one."""
    packs, start, segments = [], 0, []
    for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        if end - begin > rows:
            raise ValueError(f"a visual input has {end - begin} patches; the ANE encoder accepts at most {rows}")
        if end - start > rows:
            packs.append((start, begin, segments))
            start, segments = begin, []
        segments.append((begin - start, end - start))
    if segments:
        packs.append((start, cu_seqlens[-1], segments))
    return packs


def block_mask(segments: list[tuple[int, int]], rows: int = ROWS) -> np.ndarray:
    """Additive [1, 1, rows, rows] mask: 0 inside each segment (and among padding rows), very negative elsewhere."""
    mask = np.full((1, 1, rows, rows), MASK_NEG, np.float16)
    for begin, end in segments:
        mask[0, 0, begin:end, begin:end] = 0
    used = segments[-1][1]
    mask[0, 0, used:, used:] = 0  # padding rows attend only to padding; their output is discarded
    return mask


class VisionEncoder:
    """Drop-in for Qwen3_5VisionModel's forward used by clef_ane.vision; compiles 27 ANE programs."""
    dtype = torch.float32

    def __init__(self, config, state: dict[str, torch.Tensor], rows: int = ROWS, verbose: bool = True) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
        # Loading the bridge needs the private ANE classes; keep it out of import time.
        if __package__:
            from .ane import Kernel
        else:
            from ane import Kernel

        if (config.hidden_size, config.num_heads) != (HIDDEN, HEADS):
            raise ValueError("unexpected vision encoder shape")
        self.rows, self.kernels = rows, []
        with torch.device("meta"):
            model = Qwen3_5VisionModel(config)
        model.blocks = torch.nn.ModuleList()
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith("blocks.")}, strict=True, assign=True)
        model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(HEAD_DIM // 2)  # non-persistent buffer, not in the checkpoint
        self.model = model.float().eval().requires_grad_(False)
        if any(t.is_meta for t in (*self.model.parameters(), *self.model.buffers())):
            raise ValueError("vision weights are incomplete")
        try:
            for index in range(config.depth):
                prefix = f"blocks.{index}."
                weights = {k[len(prefix):]: v.float().numpy() for k, v in state.items() if k.startswith(prefix)}
                self.kernels.append(Kernel(vision_block(weights, rows)))
                if verbose and (index + 1) % 9 == 0:
                    print(f"Compiled {index + 1}/{config.depth} vision layers for the ANE", flush=True)
        except BaseException:
            self.close()
            raise

    def blocks(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
        source = hidden.float().numpy()
        if not np.isfinite(source).all() or np.abs(source).max() > FP16_LIMIT:
            raise RuntimeError("The vision input is outside the ANE's fp16 range.")
        cos, sin = cos.float().numpy(), sin.float().numpy()
        output = np.empty_like(source)
        for start, end, segments in pack(cu_seqlens, self.rows):
            count = end - start
            hid = np.zeros((1, HIDDEN, 1, self.rows), np.float16)
            hid[0, :, 0, :count] = source[start:end].T
            tables = []
            for table in (cos, sin):
                padded = np.zeros((1, 1, self.rows, HEAD_DIM), np.float16)
                padded[0, 0, :count] = table[start:end]
                tables.append(padded)
            mask = block_mask(segments, self.rows)
            for kernel in self.kernels:
                hid = kernel(hid, *tables, mask)[0]
            result = hid[0, :, 0, :count].T.astype(np.float32)
            if not np.isfinite(result).all():
                raise RuntimeError("The ANE vision encoder overflowed. Restart with --vision-device cpu.")
            output[start:end] = result
        return torch.from_numpy(output)

    @torch.inference_mode()
    def __call__(self, pixels: torch.Tensor, grid_thw: torch.Tensor, return_dict: bool = True):
        from transformers.vision_utils import (get_vision_bilinear_indices_and_weights, get_vision_cu_seqlens,
                                               get_vision_position_ids)

        model = self.model
        indices, weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=model.num_grid_per_side, spatial_merge_size=model.spatial_merge_size)
        positions = get_vision_position_ids(grid_thw, model.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).tolist()
        hidden = model.patch_embed(pixels.float())
        hidden = hidden + (model.pos_embed(indices) * weights[:, :, None]).sum(0)
        rotary = model.rotary_pos_emb(positions).reshape(len(hidden), -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        hidden = self.blocks(hidden, embedding.cos(), embedding.sin(), cu_seqlens)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=model.merger(hidden))

    def close(self) -> None:
        for kernel in self.kernels:
            kernel.close()
        self.kernels = []
