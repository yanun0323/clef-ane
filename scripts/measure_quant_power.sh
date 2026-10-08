#!/bin/bash
# Whole-machine power (macmon sys_power) for MLP-shaped ANE loops with fp16, 6-bit and 4-bit lookup-table weights. No sudo.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/quant_$(date +%Y%m%d_%H%M%S)"; SECS="${SECS:-40}"
command -v macmon >/dev/null || { echo "macmon not found: brew install macmon"; exit 1; }
mkdir -p "$OUT"
sample() { macmon pipe -s "$SECS" -i 1000 > "$OUT/$1.jsonl" 2>/dev/null; }
phase() { name=$1; kind=$2
  "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_quant_loop.py" "$kind" $((SECS + 20)) > "$OUT/$name.log" 2>&1 &
  pid=$!; until grep -q running "$OUT/$name.log" 2>/dev/null; do kill -0 $pid 2>/dev/null || { cat "$OUT/$name.log"; return 1; }; sleep 1; done
  echo "[$name] sampling $SECS s ..."; sample "$name"; wait $pid; tail -1 "$OUT/$name.log"; }
echo "[idle1]"; sleep 5; sample idle1
for round in a b; do
  for kind in fp16 lut6 lut4; do phase ${kind}_$round $kind; sleep 15; done
done
echo "[idle2]"; sample idle2
"$ROOT/.venv/bin/python" -I - "$OUT" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
w = {p.stem: [json.loads(l)["sys_power"] for l in p.read_text().splitlines() if l.startswith("{")] for p in out.glob("*.jsonl")}
rate = lambda n: float(re.search(r"= ([\d.]+) calls/s", (out / f"{n}.log").read_text()).group(1))
idle = st.mean([st.median(w["idle1"]), st.median(w["idle2"])])
print(f"\nidle {idle:.1f} W (medians {st.median(w['idle1']):.1f} / {st.median(w['idle2']):.1f})")
for n in ("fp16_a", "lut6_a", "lut4_a", "fp16_b", "lut6_b", "lut4_b"):
    m = st.mean(w[n]); r = rate(n)
    print(f"{n:<7} {m:5.1f} W ({m - idle:+5.1f} over idle)  {r:5.1f} calls/s  -> {(m - idle) / r * 1000:6.1f} mJ per MLP call over idle")
PY
echo "raw files: $OUT"
