#!/bin/bash
# Alternating A/B whole-machine power (macmon sys_power) of the real request loop: fp16 vs a lookup-table width.
# Usage: measure_ab.sh [BITS] [ROUNDS]     defaults 6 and 3. No sudo. Speeds are equal, so absolute watts are compared directly.
set -u
BITS="${1:-6}"; ROUNDS="${2:-3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/ab_lut${BITS}_$(date +%Y%m%d_%H%M%S)"; SECS="${SECS:-45}"
command -v macmon >/dev/null || { echo "macmon not found: brew install macmon"; exit 1; }
mkdir -p "$OUT"
phase() { name=$1; bits=$2
  CLEF_LUT_BITS=$bits "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_loop.py" $((SECS + 25)) > "$OUT/$name.log" 2>&1 &
  pid=$!; until grep -q running "$OUT/$name.log" 2>/dev/null; do kill -0 $pid 2>/dev/null || { tail -5 "$OUT/$name.log"; return 1; }; sleep 2; done
  macmon pipe -s "$SECS" -i 1000 > "$OUT/$name.jsonl" 2>/dev/null; wait $pid; echo "[$name] $(tail -1 "$OUT/$name.log" | grep -o '[0-9.]* s/request')"; sleep 15; }
for r in $(seq 1 "$ROUNDS"); do phase fp16_$r 0; phase lut${BITS}_$r "$BITS"; done
"$ROOT/.venv/bin/python" -I - "$OUT" "$BITS" "$ROUNDS" <<'PY'
import json, re, statistics as st, sys
from pathlib import Path
out, bits, rounds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
watts = lambda n: [json.loads(l)["sys_power"] for l in (out / f"{n}.jsonl").read_text().splitlines() if l.startswith("{")]
rate = lambda n: 1 / float(re.search(r"([\d.]+) s/request", (out / f"{n}.log").read_text()).group(1))
print(f"\n{'round':<6}{'fp16 W':>9}{f'lut{bits} W':>9}{'diff W':>9}{'fp16 J/req':>12}{f'lut{bits} J/req':>12}")
diffs, jd = [], []
for r in range(1, rounds + 1):
    a, b = st.mean(watts(f"fp16_{r}")), st.mean(watts(f"lut{bits}_{r}")); ja, jb = a / rate(f"fp16_{r}"), b / rate(f"lut{bits}_{r}")
    diffs.append(a - b); jd.append(ja - jb)
    print(f"{r:<6}{a:>9.1f}{b:>9.1f}{a - b:>9.1f}{ja:>12.1f}{jb:>12.1f}")
print(f"fp16 minus lut{bits}: mean {st.mean(diffs):.1f} W (range {min(diffs):.1f} to {max(diffs):.1f}), {st.mean(jd):.1f} J per request")
PY
echo "raw files: $OUT"
