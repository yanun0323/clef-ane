"""Warm latency of one Clef-Flash request on the ANE backbone, split into backbone and head, with CPU time used."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_ane import Backbone
from ref_backbone import Head
from records import RECORDS

head = Head(); ane = Backbone(rows=128, verbose=False)
enc = [head.encode(r) for r in RECORDS]
ane.hidden(list(enc[0].input_ids))                      # warm-up
print(f"{'record':<10}{'tokens':>7}{'backbone s':>12}{'head s':>9}{'total s':>9}{'tok/s':>8}{'cpu s (backbone)':>18}")
tot_wall = tot_cpu = tot_tokens = 0
for r, e in zip(RECORDS, enc):
    ids = list(e.input_ids)
    w0, c0 = time.perf_counter(), time.process_time()
    h = ane.hidden(ids)
    w1, c1 = time.perf_counter(), time.process_time()
    head.probabilities(torch.from_numpy(h), e)
    w2 = time.perf_counter()
    print(f"{r['id']:<10}{len(ids):>7}{w1 - w0:>12.2f}{w2 - w1:>9.3f}{w2 - w0:>9.2f}{len(ids) / (w2 - w0):>8.0f}{c1 - c0:>18.2f}")
    tot_wall += w1 - w0; tot_cpu += c1 - c0; tot_tokens += len(ids)
print(f"backbone: {tot_tokens / tot_wall:.0f} tokens/s; host CPU time {tot_cpu / tot_wall * 100:.0f}% of one core while the ANE works")
