"""Where does the ANE delta-rule error come from? Compare intermediates of the first chunk against float64."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta, reference
from ane import Kernel

C, H, D = 64, 32, 128
layer, _ = reference.load_layer()
p = reference.pieces(layer, reference.sample_hidden())

def f64(t): return t.double().numpy()
s = slice(0, C)
q, k, v = (f64(p[n][s]) for n in ("q", "k", "v"))            # [C,H,D]
beta, g = f64(p["beta"][s]), f64(p["g"][s])                    # [C,H]

def l2(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
qn, kn = l2(q) / np.sqrt(D), l2(k)
kb, vb = kn * beta[..., None], v * beta[..., None]
cum = np.cumsum(g, axis=0)                                     # [C,H]
dec = np.exp(cum[:, None, :] - cum[None, :, :])                # [i,j,H]
low = np.tril(np.ones((C, C)))
decay = (dec * low[..., None]).transpose(2, 0, 1)              # [H,C,C]
kk = np.einsum("ihd,jhd->hij", kb, kn)
M = kk * decay * np.tril(np.ones((C, C)), -1)
inv64 = np.stack([np.linalg.inv(np.eye(C) + M[h]) for h in range(H)])

def err(name, got, want):
    d = got.astype(np.float64) - want
    print(f"{name:<26} max|.| {np.abs(want).max():9.3f}   rms rel err {np.sqrt((d**2).mean() / (want**2).mean()):.5f}   max abs err {np.abs(d).max():.5f}")

import sys as _s
refine = int(_s.argv[1]) if len(_s.argv) > 1 else 1
kernel = Kernel(delta.build(C, H, D, debug=True, refine=refine))
print("newton refinement steps:", refine)
heads = lambda x: torch.from_numpy(x).permute(1, 0, 2)[None].numpy()
out, st, inv, decay_a = kernel(heads(q), heads(k), heads(v), beta.T[None, :, :, None], g.T[None, :, :, None],
                               np.zeros((1, H, D, D), np.float16))
err("decay mask", decay_a[0], decay)
err("inverse T=(I+M)^-1", inv[0], inv64)
# the same inverse computed in float32 numpy with the same Neumann product, to separate method error from fp16 error
N = -M; eye = np.eye(C); T = eye + N; P = N
for _ in range(5):
    P = P @ P; T = T @ (eye + P)
err("neumann in float64", T, inv64)
