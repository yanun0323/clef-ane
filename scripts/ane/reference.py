"""PyTorch reference for one Clef-Flash decoder layer, exposing every intermediate tensor.

The ANE implementation is checked against these, stage by stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, torch_chunk_gated_delta_rule

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def layer_file(index: int) -> Path:
    path = DATA / "weights" / f"layer_{index:02d}.safetensors"
    if path.exists():
        return path
    for name in ("layer0", "layer1_2", "layer3"):
        path = DATA / f"{name}.safetensors"
        if path.exists() and any(k.startswith(f"model.language_model.layers.{index}.") for k in _keys(path)):
            return path
    raise FileNotFoundError(f"no weights for layer {index} in {DATA}")


def _keys(path: Path):
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return list(f.keys())


def load_layer(index: int = 0, path: Path | None = None, attention: str = "eager") -> tuple[Qwen3_5DecoderLayer, Qwen3_5TextConfig]:
    path = path or layer_file(index)
    config = Qwen3_5TextConfig(**json.loads((DATA / "config.json").read_text())["text_config"])
    config._attn_implementation = attention
    layer = Qwen3_5DecoderLayer(config, index).float().eval()
    prefix = f"model.language_model.layers.{index}."
    state = {k[len(prefix):]: v.float() for k, v in load_file(path).items() if k.startswith(prefix)}
    layer.load_state_dict(state, strict=True)
    return layer, config


def sample_hidden(tokens: int | None = None) -> torch.Tensor:
    import numpy as np

    x = torch.from_numpy(np.load(DATA / "sample_embed.npy"))
    return x[:tokens] if tokens else x


@torch.no_grad()
def pieces(layer: Qwen3_5DecoderLayer, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run a linear-attention (Gated DeltaNet) layer on x [T, hidden], returning all intermediates."""
    attn = layer.linear_attn
    out: dict[str, torch.Tensor] = {"x": x}
    h = x[None]
    xn = layer.input_layernorm(h)
    out["xn"] = xn[0]

    mixed = attn.in_proj_qkv(xn).transpose(1, 2)                     # [1, 8192, T]
    z = attn.in_proj_z(xn).reshape(1, xn.shape[1], -1, attn.head_v_dim)
    b = attn.in_proj_b(xn)
    a = attn.in_proj_a(xn)
    conv = F.silu(attn.conv1d(mixed)[:, :, : mixed.shape[-1]]).transpose(1, 2)  # [1, T, 8192]
    out.update(mixed=mixed[0].T, conv=conv[0])

    q, k, v = torch.split(conv, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    q = q.reshape(1, -1, attn.num_k_heads, attn.head_k_dim)
    k = k.reshape(1, -1, attn.num_k_heads, attn.head_k_dim)
    v = v.reshape(1, -1, attn.num_v_heads, attn.head_v_dim)
    beta = b.sigmoid()
    g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
    rep = attn.num_v_heads // attn.num_k_heads
    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    out.update(q=q[0], k=k[0], v=v[0], beta=beta[0], g=g[0])           # q,k,v: [T, 32, 128]; beta,g: [T, 32]

    core, state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    out.update(core=core[0], state=state[0])                          # state: [32, 128, 128]

    normed = attn.norm(core.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)).reshape(1, xn.shape[1], -1)
    out["normed"] = normed[0]
    mixer = attn.out_proj(normed)
    h1 = h + mixer
    out["h1"] = h1[0]
    mlp_in = layer.post_attention_layernorm(h1)
    out["mlp_in"] = mlp_in[0]
    y = h1 + layer.mlp(mlp_in)
    out["y"] = y[0]
    return out


if __name__ == "__main__":
    layer, config = load_layer()
    x = sample_hidden()
    p = pieces(layer, x)
    with torch.no_grad():
        direct = layer(x[None], position_embeddings=None)[0]
    print("pieces() vs module forward, max abs diff:", float((p["y"] - direct).abs().max()))
    for name, t in p.items():
        t = t.float()
        print(f"{name:<8} {tuple(t.shape)!s:<18} max|.| {t.abs().max():10.3f}  mean|.| {t.abs().mean():9.4f}")


@torch.no_grad()
def layer_forward(layer: Qwen3_5DecoderLayer, config: Qwen3_5TextConfig, x: torch.Tensor) -> torch.Tensor:
    """Any decoder layer on x [T, hidden] (text positions 0..T-1), via the transformers module itself."""
    if layer.layer_type == "linear_attention":
        return pieces(layer, x)["y"]
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    rotary = Qwen3_5TextRotaryEmbedding(config).to(x.device)
    positions = torch.arange(x.shape[0], device=x.device)[None]
    cos, sin = rotary(x[None], positions)
    mask = torch.full((x.shape[0], x.shape[0]), float("-inf"), device=x.device).triu(1)[None, None].to(x.dtype)
    return layer(x[None], position_embeddings=(cos, sin), attention_mask=mask, position_ids=positions)[0]
