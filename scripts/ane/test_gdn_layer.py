"""Run Gated DeltaNet mixer of layer 0 on the ANE over the whole sample, comparing against the PyTorch reference."""
import sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
layer, _ = reference.load_layer()
x = reference.sample_hidden()
p = reference.pieces(layer, x)
w = {k[len("model.language_model.layers.0."):]: v.float().numpy() for k, v in load_file(reference.DATA / "layer0.safetensors").items()}

t0 = time.perf_counter()
kernel = Kernel(layers.gdn_mixer(w, rows=ROWS))
print(f"compiled gdn mixer (rows={ROWS}) in {kernel.compile_seconds:.2f}s")

n = (x.shape[0] // ROWS) * ROWS
conv_state = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16)
rec_state = np.zeros((1, layers.VALUE_HEADS, layers.HEAD_DIM, layers.HEAD_DIM), np.float16)
got = []
for start in range(0, n, ROWS):
    chunk = x[start:start + ROWS].T[None, :, None, :].numpy()            # [1, 4096, 1, R]
    hid, conv_state, rec_state = kernel(chunk, conv_state, rec_state)
    got.append(hid[0, :, 0, :].T)
got = np.concatenate(got).astype(np.float64)
want = p["h1"][:n].double().numpy()
mixer_got, mixer_want = got - x[:n].double().numpy(), want - x[:n].double().numpy()
rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print(f"h1 (mixer + residual) rms rel err: {rms(got, want):.5f}   max abs err {np.abs(got - want).max():.4f}  (max |h1| {np.abs(want).max():.2f})")
print(f"mixer output alone      rms rel err: {rms(mixer_got, mixer_want):.5f}")
print(f"final rec state rms rel err: {rms(rec_state.astype(np.float64)[0], p['state'].double().numpy()) if n == x.shape[0] else float('nan'):.5f}  (only comparable when tokens divide evenly)")
ms = kernel.time(chunk, conv_state, rec_state, iters=10)
print(f"{ms:.2f} ms per call of {ROWS} rows -> {ROWS / ms * 1e3:.0f} tokens/s for this mixer alone")
