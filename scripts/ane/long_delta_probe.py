"""Delta-rule error versus position and head, over a long input, for one layer fed the exact fp32 input."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta, reference
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
n = (len(ids) // 64) * 64
C, H, D = 64, 32, 128
kernel = Kernel(delta.build(C, H, D))
hf = lambda t: t.permute(1, 0, 2)[None].numpy()
state = np.zeros((1, H, D, D), np.float16); outs = []
for s in range(0, n, C):
    sl = slice(s, s + C)
    o, state = kernel(hf(p["q"][sl]), hf(p["k"][sl]), hf(p["v"][sl]), p["beta"][sl].T[None, :, :, None].numpy(), p["g"][sl].T[None, :, :, None].numpy(), state)
    outs.append(o[0].transpose(1, 0, 2))
got = np.concatenate(outs).astype(np.float64); ref = p["core"][:n].double().numpy()             # [T,H,D]
err = (got - ref) ** 2
print(f"layer {L}, {n} tokens; overall core rms rel err {np.sqrt(err.mean() / (ref ** 2).mean()):.4f}")
print("by position (tenths of the sequence):", [round(float(np.sqrt(e.mean() / (r ** 2).mean())), 4) for e, r in zip(np.array_split(err, 10), np.array_split(ref, 10))])
per_head = np.sqrt(err.mean((0, 2)) / (ref ** 2).mean((0, 2)))
print("worst heads:", [(int(h), round(float(per_head[h]), 3)) for h in np.argsort(per_head)[-5:][::-1]])
g = p["g"][:n].double().numpy()
print("per-head mean decay exp(g) of worst heads:", [(int(h), round(float(np.exp(g[:, h]).mean()), 4)) for h in np.argsort(per_head)[-5:][::-1]])
print(f"final state: |max| ref {p['state'].abs().max():.1f}, our state max {np.abs(state).max():.1f}, rms rel err {np.sqrt(((state[0].astype(np.float64) - p['state'].double().numpy()) ** 2).mean() / (p['state'].double().numpy() ** 2).mean()):.4f}" if n == len(ids) else "")
