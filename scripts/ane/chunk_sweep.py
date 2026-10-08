"""Delta-rule chunk size: mixer time (128 rows) and error against the fp32 reference on real layer-0..2 inputs."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from clef_ane import layer_weights
from ane import Kernel

R = 128
x = reference.sample_hidden(); n = (len(x) // R) * R; x = x[:n]
rms = lambda a, b: float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).mean() / (b ** 2).mean()))
for chunk in (32, 64, 128):
    layers.CHUNK = chunk
    errs, ms = [], []
    ref_in = x
    for i in range(3):
        layer, _ = reference.load_layer(i); p = reference.pieces(layer, ref_in); w = layer_weights(i)
        k = Kernel(layers.gdn_mixer(w, rows=R, chunk=chunk))
        conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16); rec = np.zeros((1, 32, 128, 128), np.float16)
        out = []
        for s in range(0, n, R):
            h, conv, rec = k(ref_in[s:s + R].numpy().T[None, :, None, :], conv, rec); out.append(h[0, :, 0, :].T)
        errs.append(rms(np.concatenate(out), p["h1"].double().numpy()))
        ms.append(k.time(ref_in[:R].numpy().T[None, :, None, :], conv, rec, iters=15)); ref_in = p["y"]
    print(f"chunk {chunk:>3}: mixer {np.mean(ms):.2f} ms  h1 error per layer {[round(e, 4) for e in errs]}", flush=True)
