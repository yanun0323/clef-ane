#!/bin/bash
# Whole-machine power (macmon's sys_power, from the SMC) while Clef-Flash requests run. No sudo needed.
# Phases: idle1, ane (requests), ane_hold (loaded, no requests), gpu (MLX requests), idle2.
# Keep the Mac on AC power and leave it alone for ~8 minutes; macmon must be installed (brew install macmon).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/system_$(date +%Y%m%d_%H%M%S)"
SECS="${SECS:-45}"                 # seconds sampled per phase (the workload runs a bit longer)
command -v macmon >/dev/null || { echo "macmon not found: brew install macmon"; exit 1; }
mkdir -p "$OUT"

sample() { macmon pipe -s "$SECS" -i 1000 > "$OUT/$1.jsonl" 2>"$OUT/$1.err"; }
run_phase() {  # run_phase <name> <command...>: start the workload, wait until warm, sample, wait for it to finish
  name=$1; shift
  "$@" > "$OUT/$name.log" 2>&1 &
  pid=$!
  until grep -q running "$OUT/$name.log" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "$name workload died:"; tail -5 "$OUT/$name.log"; return 1; }
    sleep 1
  done
  echo "[$name] warm, sampling $SECS s ..."
  sample "$name"
  wait $pid
  tail -1 "$OUT/$name.log"
}

echo "[idle1] sampling $SECS s with nothing running ..."
sleep 5; sample idle1
run_phase ane "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_loop.py" $((SECS + 25))
sleep 20
run_phase ane_hold "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_loop.py" $((SECS + 10)) --hold
sleep 20
run_phase gpu "$ROOT/.venv-mlx/bin/python" -I "$ROOT/scripts/ane/bench_mlx_loop.py" $((SECS + 25))
sleep 20
echo "[idle2] sampling $SECS s with nothing running ..."
sample idle2
echo
"$ROOT/.venv/bin/python" -I "$ROOT/scripts/system_power_summary.py" "$OUT"
echo "raw files: $OUT"
