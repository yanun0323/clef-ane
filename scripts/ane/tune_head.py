"""CPU seconds and answer drift of the joint head for different thread counts and dtypes (same hidden states)."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_backbone import Head
from records import RECORDS

head32 = None
enc = [Head.encode(Head.__new__(Head), r) if False else None for r in RECORDS]
base = Head(dtype=torch.bfloat16)
enc = [base.encode(r) for r in RECORDS]
hidden = [torch.from_numpy(np.load(Path(__file__).resolve().parents[2] / "data" / "ref" / f"{r['id']}.npz")["final"]) for r in RECORDS]
ref = [base.probabilities(h, e) for h, e in zip(hidden, enc)]       # bf16, default threads = the current behaviour

def run(dtype, threads):
    torch.set_num_threads(threads)
    h = Head(dtype=dtype)
    for hh, e in zip(hidden, enc): h.probabilities(hh, e)             # warm-up
    w, c = time.perf_counter(), time.process_time()
    out = [h.probabilities(hh, e) for hh, e in zip(hidden, enc) for _ in range(3)][::3]
    w, c = time.perf_counter() - w, time.process_time() - c
    drift = max(float(np.abs(o[q] - r[q]).max()) for o, r in zip(out, ref) for q in r)
    print(f"{str(dtype):<16} threads {threads}: wall {w / 18:.3f} s, CPU {c / 18:.3f} s per request, max prob diff vs current {drift:.4f}")
for dtype in (torch.bfloat16, torch.float32):
    for threads in (6, 2, 1):
        run(dtype, threads)
