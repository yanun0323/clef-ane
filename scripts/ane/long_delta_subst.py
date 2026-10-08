"""Which ANE-computed input of the delta rule causes the extra error? Swap them in one at a time (layer L, long input)."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import delta, layers, reference
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
R = 128; n = (len(ids) // R) * R; C = 64
mixer = Kernel(layers.gdn_mixer(layer_weights(L), rows=R, taps=True)); names = [t.name for t in mixer.outputs]
conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16); rec = np.zeros((1, 32, 128, 128), np.float16)
taps = {k: [] for k in ("tap_q", "tap_k", "tap_v", "tap_beta", "tap_g")}
for s in range(0, n, R):
    d = dict(zip(names, mixer(x[s:s + R].numpy().T[None, :, None, :], conv, rec))); conv, rec = d["conv_out"], d["rec_out"]
    for k in taps: taps[k].append(d[k][0])
ane = {k: np.concatenate(v, axis=1) for k, v in taps.items()}      # q,k,v: [H, n, D]; beta,g: [H, n, 1]
hf = lambda t: t[:n].permute(1, 0, 2).numpy()
ref = {"tap_q": hf(p["q"]), "tap_k": hf(p["k"]), "tap_v": hf(p["v"]), "tap_beta": p["beta"][:n].T[..., None].numpy(), "tap_g": p["g"][:n].T[..., None].numpy()}
want = p["core"][:n].double().numpy()
kernel = Kernel(delta.build(C, 32, 128))
def run(use_ane):
    state = np.zeros((1, 32, 128, 128), np.float16); outs = []
    for s in range(0, n, C):
        a = [(ane if name in use_ane else ref)[name][None, :, s:s + C] for name in ("tap_q", "tap_k", "tap_v", "tap_beta", "tap_g")]
        o, state = kernel(*a, state); outs.append(o[0].transpose(1, 0, 2))
    got = np.concatenate(outs).astype(np.float64)
    return float(np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean()))
print(f"layer {L}, {n} tokens, core rms rel err when these inputs come from the ANE (the rest from fp32):")
for label, use in [("none", set()), ("q", {"tap_q"}), ("k", {"tap_k"}), ("v", {"tap_v"}), ("beta", {"tap_beta"}), ("g", {"tap_g"}), ("all", set(ref))]:
    print(f"  {label:<5} {run(use):.4f}", flush=True)

# --- which heads? -------------------------------------------------------------------------------------------------
a_g, r_g = ane["tap_g"][..., 0].astype(np.float64), ref["tap_g"][..., 0].astype(np.float64)        # [H, n]
state = np.zeros((1, 32, 128, 128), np.float16); outs = []
for s in range(0, n, C):
    args = [ref[name][None, :, s:s + C] for name in ("tap_q", "tap_k", "tap_v", "tap_beta")] + [ane["tap_g"][None, :, s:s + C]]
    o, state = kernel(*args, state); outs.append(o[0].transpose(1, 0, 2))
got = np.concatenate(outs).astype(np.float64)
core_err = np.sqrt(((got - want) ** 2).mean((0, 2)) / (want ** 2).mean((0, 2)))
print("head  mean|g| ref  g rel err   core rel err   mean decay  A_log   dt_bias")
w = layer_weights(L)
for h in np.argsort(core_err)[-6:][::-1]:
    print(f"{h:>4}  {np.abs(r_g[h]).mean():10.5f}  {np.sqrt(((a_g[h] - r_g[h]) ** 2).mean() / (r_g[h] ** 2).mean()):9.4f}  {core_err[h]:12.4f}  {np.exp(r_g[h]).mean():10.4f}  {w['linear_attn.A_log'][h]:6.2f}  {w['linear_attn.dt_bias'][h]:7.2f}")
print("example head", int(np.argmax(core_err)), "g ref vs ane (tokens 100-105):", r_g[int(np.argmax(core_err)), 100:106].round(6).tolist(), a_g[int(np.argmax(core_err)), 100:106].round(6).tolist())
