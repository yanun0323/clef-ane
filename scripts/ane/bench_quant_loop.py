"""Keep the ANE busy with MLP-shaped programs whose weights are fp16 or 4-bit lookup tables, to compare power.

Usage: bench_quant_loop.py fp16|lut4|lut6 SECONDS [LAYERS]
The structure is the same for both (gate+up, SiLU*up, down in three parts); only the weight format differs. Weights are
random Gaussians (lutN: indices from a uniform table over +-3 sigma), one program per layer, cycled so the
weights cannot stay in a cache. Prints "running" once warm, then calls per second.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel
from mil import Graph

kind, seconds = sys.argv[1], float(sys.argv[2])
layers_n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
H, F, R = 4096, 12288, 128
rng = np.random.default_rng(0)
BITS = int(kind[3:]) if kind.startswith("lut") else 0
LUT = (np.linspace(-3, 3, 1 << BITS) * 0.02).astype(np.float16) if BITS else None

def weight(g, x, name, out_ch, in_ch):
    w = rng.standard_normal((out_ch, in_ch)).astype(np.float32) * 0.02
    if kind == "fp16":
        return g.conv1x1(x, name, w)
    idx = np.clip(np.round((w / 0.02 + 3) / 6 * ((1 << BITS) - 1)), 0, (1 << BITS) - 1).astype(np.uint8)
    return g.conv1x1_lut(x, name, idx, LUT, BITS)

def mlp():
    g = Graph(); hid = g.input("hid_in", (1, H, 1, R))
    gate_up = weight(g, hid, "w_gu", 2 * F, H)
    act = g.mul(g.silu(g.slice(gate_up, (0, 0, 0, 0), (1, F, 1, R))), g.slice(gate_up, (0, F, 0, 0), (1, F, 1, R)))
    parts = [weight(g, g.slice(act, (0, i * H, 0, 0), (1, H, 1, R)), f"w_d{i}", H, H) for i in range(F // H)]
    down = parts[0]
    for p in parts[1:]: down = g.add(down, p)
    g.output(g.rename(g.add(hid, down), "hid_out")); return g

kernels = [Kernel(mlp()) for _ in range(layers_n)]
x = (rng.standard_normal((1, H, 1, R)) * 0.5).astype(np.float16)
for k in kernels: k(x)
print("running", flush=True)
start, calls = time.time(), 0
while time.time() - start < seconds:
    kernels[calls % layers_n](x); calls += 1
print(f"{calls} calls in {time.time() - start:.1f}s = {calls / (time.time() - start):.1f} calls/s ({kind}, {layers_n} layers)")
