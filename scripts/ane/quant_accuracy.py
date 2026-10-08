"""Final answers of the whole ANE backbone with 4-bit lookup-table weights vs the bf16 reference (data/ref/golden.json).

Usage: quant_accuracy.py [BITS] [GROUP_CHANNELS]   BITS 4 or 6 (default 4), 0 = keep fp16 weights; GROUP_CHANNELS default 16
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mil, lutq
bits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
group = int(sys.argv[2]) if len(sys.argv) > 2 else 16
if bits:
    mil.Graph.default_quantize = lambda w: lutq.kmeans_lut_cached(w, bits, group)   # bits 0 = keep fp16 weights (the baseline)
from service import Service
from records import RECORDS, long_record

golden = json.loads((Path(__file__).resolve().parents[2] / "data" / "ref" / "golden.json").read_text())
cases = {r["id"]: r for r in RECORDS + [long_record(20000), long_record(60000)]}
t = time.time(); service = Service.load(buckets=(1024, 2048, 4096, 8192, 16384), verbose=False, lut_bits=bits, lut_group=group); print(f"loaded model ({bits}-bit, {group} channels per table) in {time.time() - t:.0f}s", flush=True)
answers = {}
for rid, record in cases.items():
    r = service.systemone({k: v for k, v in record.items() if k != "id"})
    answers[rid] = {q: (a["probabilities"] if a["type"] != "noul" else {"true": a["noul"], "false": 1 - a["noul"]}) for q, a in r["answers"].items()}
out = Path(__file__).resolve().parents[2] / "data" / "ref" / (f"lut{bits}_g{group}_answers.json" if bits else "ane_fp16_answers.json")
out.write_text(json.dumps(answers))
