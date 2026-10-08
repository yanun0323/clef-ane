"""Speed of MLP layouts at 128 rows: baseline, down projection split in 3, gate+up fused, both."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers
from layers import HIDDEN, MLP_PEAK, MLP_SCALE, rms_norm
from clef_ane import layer_weights
from ane import Kernel
from mil import Graph, T

def mlp(w, rows, split_down, fuse_gu, scale_fixed=True):
    g = Graph(); hid = g.input("hid_in", (1, HIDDEN, 1, rows))
    xn = rms_norm(g, hid, w["post_attention_layernorm.weight"])
    F = w["mlp.gate_proj.weight"].shape[0]
    if fuse_gu:
        gu = g.conv1x1(xn, "w_gu", np.concatenate([w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]]))
        gate, up = g.slice(gu, (0, 0, 0, 0), (1, F, 1, rows)), g.slice(gu, (0, F, 0, 0), (1, F, 1, rows))
    else:
        gate = g.conv1x1(xn, "w_gate", w["mlp.gate_proj.weight"]); up = g.conv1x1(xn, "w_up", w["mlp.up_proj.weight"])
    act = g.mul(g.silu(gate), up)
    peak = g.reduce_max(g.abs(act), 1)
    factor = g.div(T(g.scalar(MLP_PEAK), (), "fp16"), g.maximum(peak, MLP_PEAK / MLP_SCALE))
    a = g.mul(act, factor); wd = w["mlp.down_proj.weight"]
    if split_down:
        parts = [g.conv1x1(g.slice(a, (0, i * HIDDEN, 0, 0), (1, HIDDEN, 1, rows)), f"w_down{i}", wd[:, i * HIDDEN:(i + 1) * HIDDEN]) for i in range(F // HIDDEN)]
        down = parts[0]
        for p in parts[1:]: down = g.add(down, p)
    else:
        down = g.conv1x1(a, "w_down", wd)
    down = g.mul(down, g.div(g.reciprocal_of_one(factor), factor))
    g.output(g.rename(g.add(hid, down), "hid_out")); return g

w = layer_weights(0)
for rows in (128, 192):
    h = np.zeros((1, HIDDEN, 1, rows), np.float16); h[...] = 0.1
    out = {}
    for name, sd, fg in [("baseline", False, False), ("split down x3", True, False), ("fused gate+up", False, True), ("both", True, True)]:
        k = Kernel(mlp(w, rows, sd, fg)); ms = k.time(h, iters=15); out[name] = k(h)[0]
        print(f"rows {rows} {name:<15} {ms:.2f} ms  ({ms * 128 / rows:.2f} ms per 128 tokens)")
    print("   outputs match baseline:", all(np.allclose(out[n], out["baseline"], atol=5e-3) for n in out))
