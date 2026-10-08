"""Check the ANE delta-rule chunk against PyTorch on real layer-0 activations, carrying state across chunks."""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta  # noqa: E402
import reference  # noqa: E402
from ane import Kernel  # noqa: E402

CHUNK, HEADS, DIM = 64, 32, 128
layer, config = reference.load_layer()
p = reference.pieces(layer, reference.sample_hidden())
chunks = p["q"].shape[0] // CHUNK
print(f"{chunks} chunks of {CHUNK} tokens")

kernel = Kernel(delta.build(CHUNK, HEADS, DIM))
print(f"compiled in {kernel.compile_seconds:.2f}s")


def heads_first(t):  # [C, H, D] -> [1, H, C, D]
    return t.permute(1, 0, 2)[None].numpy()


def rms_rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))


def fp16_floor(q, k, v, beta, g, state):
    """Reference run on inputs rounded to fp16: the error any fp16 pipeline has before doing any arithmetic."""
    r = lambda t: t.half().float()
    out, st = torch_chunk_gated_delta_rule(r(q)[None], r(k)[None], r(v)[None], g=r(g)[None], beta=r(beta)[None], initial_state=state,
                                           output_final_state=True, use_qk_l2norm_in_kernel=True)
    return out[0].permute(1, 0, 2)[None].numpy()


ane_state = np.zeros((1, HEADS, DIM, DIM), np.float16)
ref_state = None
for i in range(chunks):
    s = slice(i * CHUNK, (i + 1) * CHUNK)
    q, k, v = (p[n][s] for n in ("q", "k", "v"))
    beta, g = p["beta"][s].T[None, :, :, None].numpy(), p["g"][s].T[None, :, :, None].numpy()
    out, ane_state = kernel(heads_first(q), heads_first(k), heads_first(v), beta, g, ane_state)

    ref_out, ref_state = torch_chunk_gated_delta_rule(
        q[None], k[None], v[None], g=p["g"][s][None], beta=p["beta"][s][None], initial_state=ref_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    ref_out = ref_out[0].permute(1, 0, 2)[None].numpy()                  # [1, H, C, D]
    floor = rms_rel(fp16_floor(q, k, v, p["beta"][s], p["g"][s], None if i == 0 else prev_state), ref_out)
    prev_state = ref_state.clone()
    print(f"chunk {i}: out rms rel err {rms_rel(out, ref_out):.5f} (fp16-input floor {floor:.5f})   "
          f"state rms rel err {rms_rel(ane_state, ref_state.numpy()):.5f}   max |out| err {np.abs(out.astype(np.float32) - ref_out).max():.4f}")
