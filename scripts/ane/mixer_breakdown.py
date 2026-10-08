"""Where does a Gated DeltaNet mixer spend its 128-row call? (cumulative cuts of the same graph)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers
from clef_ane import layer_weights
from ane import Kernel
import bisect_compile

w = layer_weights(0); R = 128
g = layers.gdn_mixer(w, rows=R)
h = np.zeros((1, 4096, 1, R), np.float16); h[...] = 0.1
conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16); rec = np.zeros((1, 32, 128, 128), np.float16)
full = Kernel(g); print(f"full mixer {full.time(h, conv, rec, iters=15):.2f} ms")
by_name = {t.name: t for t, _ in g.defined}
marks = {}
for t, _ in g.defined:
    if t.name.startswith(("conv_",)): marks.setdefault("projections", []).append(t)
print("ops:", len(g.defined))
# cumulative cuts: output after the projections, after the conv+silu, after the delta loop (core), after the gated norm
for label, pick in [("up to in_proj convs + conv1d", lambda ts: [t for t in ts if t.shape == (1, 8192, 1, R)][-1]),
                    ("up to q,k,v prepared", lambda ts: [t for t in ts if t.shape == (1, 32, R, 128)][3])]:
    try:
        t = pick([t for t, _ in g.defined]); sub = g.upto(t)
        ins = [x for x in sub.inputs]; args = [{"hid_in": h, "conv_in": conv, "rec_in": rec}[i.name] for i in ins]
        print(f"{label:<32} {Kernel(sub).time(*args, iters=15):.2f} ms")
    except Exception as e: print(label, "failed", str(e)[:80])
