"""Per layer, how much error does the mixer add and how much does the MLP add (each fed the exact fp32 input)?"""
import sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel

ROWS = 128
x = reference.sample_hidden(); n = (len(x) // ROWS) * ROWS; x = x[:n]
rms = lambda a, b: float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).mean() / (b ** 2).mean()))
ref_in = x
for i in range(3):
    layer, _ = reference.load_layer(i); p = reference.pieces(layer, ref_in)
    prefix = f"model.language_model.layers.{i}."
    w = {k[len(prefix):]: v.float().numpy() for k, v in load_file(reference.layer_file(i)).items() if k.startswith(prefix)}
    mixer, mlp = Kernel(layers.gdn_mixer(w, rows=ROWS)), Kernel(layers.mlp(w, rows=ROWS))
    conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16); rec = np.zeros((1, 32, 128, 128), np.float16)
    h1, mlp_out = [], []
    for s in range(0, n, ROWS):
        h, conv, rec = mixer(ref_in[s:s + ROWS].numpy().T[None, :, None, :], conv, rec)
        h1.append(h[0, :, 0, :].T)
        mlp_out.append(mlp(p["h1"][s:s + ROWS].numpy().T[None, :, None, :])[0][0, :, 0, :].T)
    h1, mlp_out = np.concatenate(h1), np.concatenate(mlp_out)
    d_in = ref_in.double().numpy()
    print(f"layer {i}: mixer  h1 rms err {rms(h1, p['h1'].double().numpy()):.4f}  (on the update only: {rms(h1 - d_in, p['h1'].double().numpy() - d_in):.4f})"
          f"   | mlp given exact h1: y rms err {rms(mlp_out, p['y'].double().numpy()):.4f}")
    ref_in = p["y"]
