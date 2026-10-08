"""Whole-model latency of the ANE backbone for different rows-per-call (tokens processed per weight pass)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_ane import Backbone
from ref_backbone import Head
from records import RECORDS

head = Head(); enc = [head.encode(r) for r in RECORDS]
for rows in [int(a) for a in sys.argv[1:]]:
    ctx = -(-1024 // rows) * rows
    bb = Backbone(rows=rows, buckets=(ctx,), verbose=False)
    bb.hidden(list(enc[0].input_ids))
    t, n, calls = 0.0, 0, 0
    for e in enc * 2:
        ids = list(e.input_ids); w = time.perf_counter(); bb.hidden(ids); t += time.perf_counter() - w
        n += len(ids); calls += -(-len(ids) // rows)
    print(f"rows {rows:>3} ctx {ctx}: {t / (2 * len(enc)):.3f} s/request, {n / t:.0f} tok/s, {calls / (2 * len(enc)):.1f} weight passes/request", flush=True)
    del bb
