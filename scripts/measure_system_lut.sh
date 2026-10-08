#!/bin/bash
# Whole-machine power (macmon sys_power) for the real Clef-Flash request loop with fp16, 6-bit and 4-bit lookup-table
# weights on the ANE. No sudo. The first run quantises every layer (about 7 minutes each, cached in data/lutcache).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/systemlut_$(date +%Y%m%d_%H%M%S)"; SECS="${SECS:-40}"
command -v macmon >/dev/null || { echo "macmon not found: brew install macmon"; exit 1; }
mkdir -p "$OUT"
sample() { macmon pipe -s "$SECS" -i 1000 > "$OUT/$1.jsonl" 2>/dev/null; }
phase() { name=$1; bits=$2
  CLEF_LUT_BITS=$bits "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_loop.py" $((SECS + 25)) > "$OUT/$name.log" 2>&1 &
  pid=$!; until grep -q running "$OUT/$name.log" 2>/dev/null; do kill -0 $pid 2>/dev/null || { tail -5 "$OUT/$name.log"; return 1; }; sleep 2; done
  echo "[$name] sampling $SECS s ..."; sample "$name"; wait $pid; tail -1 "$OUT/$name.log"; }
echo "[idle1]"; sleep 5; sample idle1
phase fp16_a 0; sleep 20
phase lut6_a 6; sleep 20
phase lut4_a 4; sleep 20
phase fp16_b 0; sleep 20
phase lut6_b 6; sleep 20
echo "[idle2]"; sample idle2
"$ROOT/.venv/bin/python" -I - "$OUT" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
w = {p.stem: [json.loads(l)["sys_power"] for l in p.read_text().splitlines() if l.startswith("{")] for p in out.glob("*.jsonl")}
rate = lambda n: 1 / float(re.search(r"([\d.]+) s/request", (out / f"{n}.log").read_text()).group(1))
idle = st.mean([st.median(w["idle1"]), st.median(w["idle2"])])
print(f"\nidle {idle:.1f} W (medians {st.median(w['idle1']):.1f} / {st.median(w['idle2']):.1f})")
for n in ("fp16_a", "lut6_a", "lut4_a", "fp16_b", "lut6_b"):
    m = st.mean(w[n]); r = rate(n)
    print(f"{n:<7} {m:5.1f} W ({m - idle:+5.1f} over idle)  {r:4.2f} requests/s  -> {(m - idle) / r:5.1f} J per request over idle")
PY
echo "raw files: $OUT"
