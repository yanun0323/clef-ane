"""Whole layer 0 (Gated DeltaNet mixer + MLP) on the ANE over the sample, against the PyTorch reference and a bf16 baseline."""
import sys, time
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
layer, _ = reference.load_layer()
x = reference.sample_hidden()
p = reference.pieces(layer, x)
w = {k[len("model.language_model.layers.0."):]: v.float().numpy() for k, v in load_file(reference.DATA / "layer0.safetensors").items()}
mixer, mlp = Kernel(layers.gdn_mixer(w, rows=ROWS)), Kernel(layers.mlp(w, rows=ROWS))
print(f"compile: mixer {mixer.compile_seconds:.2f}s, mlp {mlp.compile_seconds:.2f}s")

n = (x.shape[0] // ROWS) * ROWS
conv_state = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16)
rec_state = np.zeros((1, layers.VALUE_HEADS, layers.HEAD_DIM, layers.HEAD_DIM), np.float16)
y = []
for start in range(0, n, ROWS):
    h = x[start:start + ROWS].T[None, :, None, :].numpy()
    h, conv_state, rec_state = mixer(h, conv_state, rec_state)
    y.append(mlp(h)[0][0, :, 0, :].T)
y = np.concatenate(y).astype(np.float64)
want = p["y"][:n].double().numpy()
rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print(f"layer 0 output over {n} tokens: rms rel err {rms(y, want):.5f}   max abs err {np.abs(y - want).max():.3f}   (max |y| {np.abs(want).max():.2f})")
print(f"   same, measured on the change this layer made (y - x): {rms(y - x[:n].double().numpy(), want - x[:n].double().numpy()):.5f}")
layer16, _ = reference.load_layer(); layer16 = layer16.to(torch.bfloat16)
y16 = reference.pieces(layer16, x[:n].to(torch.bfloat16))["y"].double().numpy()
print(f"bf16 model for comparison:    rms rel err {rms(y16, want):.5f}")
a = mixer.time(h, conv_state, rec_state, iters=20); b = mlp.time(h, iters=20)
print(f"per {ROWS} rows: mixer {a:.2f} ms + mlp {b:.2f} ms = {a + b:.2f} ms  -> {ROWS / (a + b) * 1e3:.0f} tokens/s for one GDN layer")
