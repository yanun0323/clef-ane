"""GPU baseline: the same Clef-Flash requests through mlx-community/clef-flash-4bit (MLX, GPU) for N seconds.

Run with .venv-mlx/bin/python -I. Prints "running" once warm, then a throughput summary.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "clef-flash-4bit"))   # clef_mlx.py ships inside the downloaded model
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clef_mlx
from records import RECORDS

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60
model = clef_mlx.load(str(ROOT / "models" / "clef-flash-4bit"))
records = [{k: v for k, v in r.items() if k != "id"} for r in RECORDS]
answers = [model.predict(r) for r in records]               # warm-up; also the answers for correctness checks
if "--answers" in sys.argv:
    for r, a in zip(RECORDS, answers):
        print(r["id"], {q: {k: round(v, 3) for k, v in p.items()} for q, p in a.items()})
print("running", flush=True)
start, done, tokens = time.time(), 0, 0
while time.time() - start < seconds:
    model.predict(records[done % len(records)]); done += 1
elapsed = time.time() - start
print(f"{done} requests in {elapsed:.1f}s = {elapsed / done:.2f} s/request")
