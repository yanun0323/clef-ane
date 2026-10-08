"""Compare every intermediate of one ANE delta-rule chunk (zero state) against float64, for a chosen layer."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta, reference
from ane import Kernel

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
C, H, D = 64, 32, 128
saved = np.load(reference.DATA / "ref" / "invoice.npz")
x = torch.from_numpy(saved["emb"] if LAYER == 0 else saved["layers"][LAYER - 1])[:C]
p = reference.pieces(reference.load_layer(LAYER)[0], x)
f = lambda t: t.double().numpy()
q, k, v, beta, g = (f(p[n]) for n in ("q", "k", "v", "beta", "g"))     # [C,H,D] / [C,H]
l2 = lambda a: a / np.sqrt((a * a).sum(-1, keepdims=True) + 1e-6)
qn, kn = l2(q) / np.sqrt(D), l2(k)
kb, vb = kn * beta[..., None], v * beta[..., None]
cum = np.cumsum(g, 0)
tril = np.tril(np.ones((C, C)))
decay = (np.exp(cum[:, None] - cum[None]) * tril[..., None]).transpose(2, 0, 1)          # [H,i,j]
M = np.einsum("ihd,jhd->hij", kb, kn) * decay * np.tril(np.ones((C, C)), -1)
inv = np.stack([np.linalg.inv(np.eye(C) + M[h]) for h in range(H)])
vbh, kbh = vb.transpose(1, 0, 2), kb.transpose(1, 0, 2)
value = inv @ vbh
kcd = inv @ (kbh * np.exp(cum).T[..., None])
v_new = value                                                                              # zero state
inter = np.zeros_like(value)
attn = np.einsum("ihd,jhd->hij", qn, kn) * decay
intra = attn @ v_new
want = dict(decay=decay, inv=inv, value=value, k_cumdecay=kcd, v_new=v_new, inter=inter, intra=intra, cum=cum.T[..., None])

heads = lambda t: t.permute(1, 0, 2)[None].numpy()
kernel = Kernel(delta.build(C, H, D, debug=True))
outs = kernel(heads(p["q"]), heads(p["k"]), heads(p["v"]), p["beta"].T[None, :, :, None].numpy(), p["g"].T[None, :, :, None].numpy(),
              np.zeros((1, H, D, D), np.float16))
got = {t.name[4:]: o[0].astype(np.float64) for t, o in zip(kernel.outputs, outs) if t.name.startswith("tap_")}
print(f"layer {LAYER}: |g| max {np.abs(g).max():.2f}, cumulative decay over chunk reaches {cum.min():.1f}")
for name in want:
    a, b = got[name], want[name]
    rel = np.sqrt(((a - b) ** 2).mean() / max((b ** 2).mean(), 1e-30))
    print(f"{name:<11} rms |ref| {np.sqrt((b**2).mean()):10.5f}  rms rel err {rel:8.4f}   max abs err {np.abs(a-b).max():.5f}")
out_ref = intra
out = outs[0][0].astype(np.float64)
print(f"{'core out':<11} rms |ref| {np.sqrt((out_ref**2).mean()):10.5f}  rms rel err {np.sqrt(((out-out_ref)**2).mean()/(out_ref**2).mean()):8.4f}")
