"""Compare every intermediate of the ANE Gated DeltaNet mixer (first 64 tokens, zero state) with the reference."""
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel

R = 64
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
saved = np.load(reference.DATA / "ref" / "invoice.npz")            # bf16 model's hidden states, per layer
x = torch.from_numpy(saved["emb"] if LAYER == 0 else saved["layers"][LAYER - 1])[:R]   # hidden state entering the chosen layer
layer, _ = reference.load_layer(LAYER)
p = reference.pieces(layer, x)
prefix = f"model.language_model.layers.{LAYER}."
w = {k[len(prefix):]: v.float().numpy() for k, v in load_file(reference.layer_file(LAYER)).items() if k.startswith(prefix)}
print(f"=== layer {LAYER}, first {R} tokens")
kernel = Kernel(layers.gdn_mixer(w, rows=R, taps=True))
outs = kernel(x[:R].T[None, :, None, :].numpy(), np.zeros((1, layers.CONV_DIM, 1, 3), np.float16),
              np.zeros((1, 32, 128, 128), np.float16))
names = [t.name for t in kernel.outputs]
got = dict(zip(names, outs))

def err(name, a, ref):
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    print(f"{name:<12} max|ref| {np.abs(ref).max():9.3f}   rms rel err {np.sqrt(((a - ref) ** 2).mean() / (ref ** 2).mean()):.5f}   max abs err {np.abs(a - ref).max():.4f}")

T = lambda t: t[:R].double().numpy()
err("xn", got["tap_xn"][0, :, 0, :].T, T(p["xn"]))
err("qkv", got["tap_qkv"][0, :, 0, :].T, T(p["mixed"]))
err("conv+silu", got["tap_conv"][0, :, 0, :].T, T(p["conv"]))
err("q (rep,norm)", got["tap_q"][0].transpose(1, 0, 2), T(p["q"]) / np.sqrt((T(p["q"]) ** 2).sum(-1, keepdims=True) + 1e-6) / np.sqrt(128))
err("k (rep,norm)", got["tap_k"][0].transpose(1, 0, 2), T(p["k"]) / np.sqrt((T(p["k"]) ** 2).sum(-1, keepdims=True) + 1e-6))
err("v", got["tap_v"][0].transpose(1, 0, 2), T(p["v"]))
err("beta", got["tap_beta"][0, :, :, 0].T, T(p["beta"]))
err("g", got["tap_g"][0, :, :, 0].T, T(p["g"]))
err("core", got["tap_core"][0].transpose(1, 0, 2), T(p["core"]))
err("normed", got["tap_normed"][0, :, 0, :].T, T(p["normed"]))
err("h1", got["hid_out"][0, :, 0, :].T, T(p["h1"]))

print("\n--- g details (token 0..3, head 0..3)")
print("ANE:", got["tap_g"][0, :4, :4, 0].T.round(3).tolist())
print("ref:", T(p["g"])[:4, :4].round(3).tolist())
print("A_log[:4]", w["linear_attn.A_log"][:4], " dt_bias[:4]", w["linear_attn.dt_bias"][:4])

print("\n--- conv+silu error by slice (rms rel)")
c_got, c_ref = got["tap_conv"][0, :, 0, :].T.astype(np.float64), T(p["conv"])
for name, sl in (("q channels", slice(0, 2048)), ("k channels", slice(2048, 4096)), ("v channels", slice(4096, 8192))):
    a, r = c_got[:, sl], c_ref[:, sl]
    print(f"{name:<11} rms ref {np.sqrt((r**2).mean()):.4f}  rms err {np.sqrt(((a-r)**2).mean()):.5f}  rel {np.sqrt(((a-r)**2).mean()/(r**2).mean()):.5f}")
qn = T(p["q"]) / np.sqrt((T(p["q"]) ** 2).sum(-1, keepdims=True) + 1e-6) / np.sqrt(128)
qa = got["tap_q"][0].transpose(1, 0, 2).astype(np.float64)
per_head = np.sqrt(((qa - qn) ** 2).mean((0, 2)) / (qn ** 2).mean((0, 2)))
print("per-head q rel err (32 heads):", per_head.round(3).tolist())
norms = np.sqrt((T(p["q"]) ** 2).sum(-1))                      # [T, H]
print("q vector norms: min %.4f  median %.3f  max %.2f" % (norms.min(), np.median(norms), norms.max()))
worst = np.unravel_index(np.argsort(np.abs(qa - qn).max(-1).ravel())[-3:], qa.shape[:2])
print("worst (token,head) pairs and their input norm:", [(int(t), int(h), round(float(norms[t, h]), 4)) for t, h in zip(*worst)])

print("\n--- normed error vs size of the core vector it normalises")
core_ref = T(p["core"])                                         # [R,32,128]
nrm_ref = T(p["normed"]).reshape(R, 32, 128)
nrm_got = got["tap_normed"][0, :, 0, :].T.astype(np.float64).reshape(R, 32, 128)
core_rms = np.sqrt((core_ref ** 2).mean(-1))                    # [R,32]
vec_err = np.sqrt(((nrm_got - nrm_ref) ** 2).mean(-1) / ((nrm_ref ** 2).mean(-1) + 1e-12))
order = np.argsort(core_rms.ravel())
for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 10)):
    sel = (core_rms >= lo) & (core_rms < hi)
    if sel.any():
        print(f"core rms in [{lo:5.2f},{hi:5.2f}): {sel.sum():5d} vectors, median vec err {np.median(vec_err[sel]):.4f}, mean {vec_err[sel].mean():.4f}, share of total sq err {((nrm_got-nrm_ref)**2)[sel].sum() / ((nrm_got-nrm_ref)**2).sum():.2f}")
print("by token (rms err of normed, tokens 0..7):", np.sqrt(((nrm_got - nrm_ref) ** 2).mean((1, 2)))[:8].round(3).tolist(), "...median over tokens", float(np.median(np.sqrt(((nrm_got - nrm_ref) ** 2).mean((1, 2))))))

print("\n--- pipeline core on small vectors")
core_got = got["tap_core"][0].transpose(1, 0, 2).astype(np.float64)
small = core_rms < 0.01
ve_core = np.sqrt(((core_got - core_ref) ** 2).mean(-1) / ((core_ref ** 2).mean(-1) + 1e-12))
print(f"small vectors: {small.sum()}  median core rel err {np.median(ve_core[small]):.3f}  abs err rms {np.sqrt(((core_got-core_ref)[small]**2).mean()):.5f} (ref rms {np.sqrt((core_ref[small]**2).mean()):.5f})")
# does the error come from the normalisation step itself? re-normalise both cores in float64 with the same z
zref = T(p["normed"]).reshape(R, 32, 128) / (core_ref / np.sqrt((core_ref ** 2).mean(-1, keepdims=True) + 1e-6) * w["linear_attn.norm.weight"].astype(np.float64) + 1e-30)
renorm = lambda c: c / np.sqrt((c ** 2).mean(-1, keepdims=True) + 1e-6)
d = renorm(core_got) - renorm(core_ref)
print(f"float64 renorm of pipeline core vs ref core: rms rel err {np.sqrt((d**2).mean()/(renorm(core_ref)**2).mean()):.4f}  (so error carried by core alone)")

print("\n--- is the in-graph gated norm faithful to its own inputs?")
import torch
with torch.no_grad():
    z_ref = layer.linear_attn.in_proj_z(p["xn"][None])[0][:R].double().numpy().reshape(R, 32, 128)
nw = w["linear_attn.norm.weight"].astype(np.float64)
mine = renorm(core_got) * nw * (z_ref / (1 + np.exp(-z_ref)))
d = nrm_got - mine
print(f"ANE normed vs float64(norm of ANE core, ref z): rms rel err {np.sqrt((d**2).mean()/(mine**2).mean()):.4f}")
big = core_rms >= 0.03
print(f"   on big-core vectors: {np.sqrt((d[big]**2).mean()/(mine[big]**2).mean()):.4f}   on small-core vectors: {np.sqrt((d[small]**2).mean()/(mine[small]**2).mean()):.4f}")
print("example small vector (token,head)=", np.argwhere(small)[0].tolist(), " core ref[:6]", core_ref[tuple(np.argwhere(small)[0])][:6].round(5).tolist(), " core ANE[:6]", core_got[tuple(np.argwhere(small)[0])][:6].round(5).tolist())
