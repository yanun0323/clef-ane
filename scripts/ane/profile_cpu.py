"""Where does host CPU time go per request? Reports wall and CPU seconds (all threads) per stage."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ane as ane_mod
from clef_ane import Backbone
from ref_backbone import Head
from records import RECORDS

stages = {}
def timed(name, fn):
    def wrapper(*a, **k):
        w, c = time.perf_counter(), time.process_time()
        r = fn(*a, **k)
        s = stages.setdefault(name, [0.0, 0.0]); s[0] += time.perf_counter() - w; s[1] += time.process_time() - c
        return r
    return wrapper
lib = ane_mod._LIB
for fn in ("ane_write_input", "ane_eval", "ane_read_output"):
    setattr(lib, fn, timed(fn, getattr(lib, fn)))

head, bb = Head(), Backbone(rows=128, verbose=False)
enc = [head.encode(r) for r in RECORDS]
bb.hidden(list(enc[0].input_ids)); head.probabilities(torch.from_numpy(bb.hidden(list(enc[0].input_ids))), enc[0])
stages.clear()
N = len(enc); w0, c0 = time.perf_counter(), time.process_time(); hw = hc = 0
for e in enc:
    w, c = time.perf_counter(), time.process_time()
    h = bb.hidden(list(e.input_ids)); hw += time.perf_counter() - w; hc += time.process_time() - c
bw, bc = time.perf_counter() - w0, time.process_time() - c0
pw, pc = time.perf_counter(), time.process_time()
for e, r in zip(enc, RECORDS):
    head.probabilities(torch.from_numpy(bb.hidden(list(e.input_ids))), e)
tw, tc = time.perf_counter() - pw, time.process_time() - pc
print(f"threads: torch {torch.get_num_threads()}")
print(f"backbone only      : wall {bw / N:.3f} s/request, CPU {bc / N:.3f} s/request ({bc / bw * 100:.0f}% of one core)")
print(f"backbone + head    : wall {tw / N:.3f} s/request, CPU {tc / N:.3f} s/request  -> head adds CPU {(tc - bc) / N:.3f} s, wall {(tw - bw) / N:.3f} s")
print("within backbone, per request (wall / cpu seconds):")
for k, (w, c) in stages.items():
    print(f"   {k:<16} {w / (2 * N):.3f} / {c / (2 * N):.3f}")
