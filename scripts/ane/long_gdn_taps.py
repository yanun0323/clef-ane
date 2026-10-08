"""Stage-by-stage error of the Gated DeltaNet mixer over a long input (exact fp32 input to the chosen layer)."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from clef_ane import layer_weights
from ane import Kernel
from ref_backbone import Head
from records import long_record
from safetensors.torch import load_file

chars, L = int(sys.argv[1]), int(sys.argv[2])
head = Head(); ids = list(head.encode(long_record(chars)).input_ids)
x = load_file(reference.DATA / "weights" / "embed.safetensors")["model.language_model.embed_tokens.weight"][torch.tensor(ids)].float()
for i in range(L):
    layer, cfg = reference.load_layer(i, attention="sdpa"); x = reference.layer_forward(layer, cfg, x)
layer, _ = reference.load_layer(L); p = reference.pieces(layer, x)
R = 128; n = (len(ids) // R) * R
k = Kernel(layers.gdn_mixer(layer_weights(L), rows=R, taps=True))
names = [t.name for t in k.outputs]
conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16); rec = np.zeros((1, 32, 128, 128), np.float16)
acc = {}
for s in range(0, n, R):
    outs = k(x[s:s + R].numpy().T[None, :, None, :], conv, rec)
    d = dict(zip(names, outs)); conv, rec = d["conv_out"], d["rec_out"]
    for name in ("tap_xn", "tap_qkv", "tap_conv", "tap_normed", "hid_out"):
        acc.setdefault(name, []).append(d[name][0, :, 0, :].T)
    acc.setdefault("tap_core", []).append(d["tap_core"][0].transpose(1, 0, 2))
    acc.setdefault("tap_g", []).append(d["tap_g"][0, :, :, 0].T); acc.setdefault("tap_beta", []).append(d["tap_beta"][0, :, :, 0].T)
f = lambda t: t[:n].double().numpy()
want = {"tap_xn": f(p["xn"]), "tap_qkv": f(p["mixed"]), "tap_conv": f(p["conv"]), "tap_core": f(p["core"]), "tap_normed": f(p["normed"]),
        "tap_g": f(p["g"]), "tap_beta": f(p["beta"]), "hid_out": f(p["h1"])}
print(f"layer {L}, {n} tokens")
for name, ref in want.items():
    got = np.concatenate(acc[name]).astype(np.float64)
    print(f"{name:<11} rms rel err {np.sqrt(((got - ref) ** 2).mean() / (ref ** 2).mean()):.4f}   max|ref| {np.abs(ref).max():.2f}")
core, nrm = np.concatenate(acc["tap_core"]).astype(np.float64), np.concatenate(acc["tap_normed"]).astype(np.float64).reshape(n, 32, 128)
cr = want["tap_core"]; rms_vec = np.sqrt((cr ** 2).mean(-1)); small = rms_vec < 0.003
nr = want["tap_normed"].reshape(n, 32, 128)
print(f"vectors with core rms < 0.003: {small.mean() * 100:.0f}% of all; they carry {((nrm - nr) ** 2)[small].sum() / ((nrm - nr) ** 2).sum() * 100:.0f}% of the squared error of the normed output")
