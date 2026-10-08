#!/bin/bash
# Measure idle / ANE / GPU power while Clef-Flash requests run, then print energy per request.
# Run from anywhere:  scripts/measure_power.sh      (asks for your sudo password once, for powermetrics)
# Keep the Mac on AC power, close heavy apps, and leave it alone for ~4 minutes.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/power_$(date +%Y%m%d_%H%M%S)"
PM_CMD="${PM_CMD:-sudo powermetrics}"
SECS="${SECS:-45}"                 # seconds sampled per phase (workload runs a bit longer)
mkdir -p "$OUT"
[ "$PM_CMD" = "sudo powermetrics" ] && sudo -v

sample() {  # sample <name>: 1 s samples of CPU/GPU/ANE rails
  $PM_CMD --samplers cpu_power,gpu_power,ane_power -i 1000 -n "$SECS" > "$OUT/$1.pm.txt" 2>"$OUT/$1.pm.err"
}
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
sleep 20   # let the machine settle between phases
run_phase ane_hold "$ROOT/.venv/bin/python" -I "$ROOT/scripts/ane/bench_loop.py" $((SECS + 10)) --hold
sleep 20
run_phase gpu "$ROOT/.venv-mlx/bin/python" -I "$ROOT/scripts/ane/bench_mlx_loop.py" $((SECS + 25))
sleep 20
echo "[idle2] sampling $SECS s with nothing running ..."
sample idle2
echo
"$ROOT/.venv/bin/python" -I "$ROOT/scripts/power_summary.py" "$OUT"
echo "raw files: $OUT"
