"""Does scaling the SwiGLU activation before the down projection reduce MLP error?"""
import sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel
from mil import Graph

ROWS = 128
x = reference.sample_hidden(); n = (len(x) // ROWS) * ROWS; x = x[:n]
rms = lambda a, b: float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).mean() / (b ** 2).mean()))

def mlp_scaled(w, scale):
    g = Graph(); hid = g.input("hid_in", (1, layers.HIDDEN, 1, ROWS))
    xn = layers.rms_norm(g, hid, w["post_attention_layernorm.weight"])
    gate = g.conv1x1(xn, "w_gate", w["mlp.gate_proj.weight"]); up = g.conv1x1(xn, "w_up", w["mlp.up_proj.weight"])
    act = g.mul(g.silu(gate), g.mul(up, scale))
    down = g.mul(g.conv1x1(act, "w_down", w["mlp.down_proj.weight"]), 1.0 / scale)
    g.output(g.rename(g.add(hid, down), "hid_out")); return g

ref_in = x
for i in range(3):
    layer, _ = reference.load_layer(i); p = reference.pieces(layer, ref_in)
    prefix = f"model.language_model.layers.{i}."
    w = {k[len(prefix):]: v.float().numpy() for k, v in load_file(reference.layer_file(i)).items() if k.startswith(prefix)}
    with torch.no_grad():
        m = layer.mlp; a = m.act_fn(m.gate_proj(p["mlp_in"])) * m.up_proj(p["mlp_in"])
    print(f"layer {i}: activation entering down_proj: rms {a.pow(2).mean().sqrt():.5f}  max {a.abs().max():.3f}")
    want = p["y"].double().numpy()
    for s in (1, 16, 64, 256):
        k = Kernel(mlp_scaled(w, float(s)))
        y = np.concatenate([k(p["h1"][t:t + ROWS].numpy().T[None, :, None, :])[0][0, :, 0, :].T for t in range(0, n, ROWS)])
        print(f"   scale {s:<4} y rms err {rms(y, want):.5f}   finite {np.isfinite(y).all()}")
    ref_in = p["y"]
