"""Summarise macmon samples (idle1, ane, ane_hold, gpu, idle2) of a measure_system_power.sh run.

Usage: system_power_summary.py RESULTS_DIR
sys_power is the whole machine (SMC). gpu_power is macmon's GPU reading. macmon's cpu/ane/ram power read 0 on macOS 27
(counters unavailable), so they are not reported.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

PHASES = ("idle1", "ane", "ane_hold", "gpu", "idle2")


def load(directory: Path) -> dict[str, list[dict]]:
    out = {}
    for phase in PHASES:
        path = directory / f"{phase}.jsonl"
        if path.exists():
            out[phase] = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    return out


def requests_per_second(log: Path):
    match = re.search(r"(\d+) requests.*?([\d.]+) s/request", log.read_text()) if log.exists() else None
    return 1.0 / float(match.group(2)) if match else None


def main(out: Path) -> None:
    data = load(out)
    if not data:
        sys.exit(f"no samples in {out}")
    watts = {p: [s["sys_power"] for s in rows] for p, rows in data.items()}
    print(f"{'phase':<10}{'mean W':>8}{'median W':>10}{'min W':>8}{'max W':>8}{'GPU W':>8}{'samples':>9}")
    for phase, w in watts.items():
        gpu = st.mean(s["gpu_power"] for s in data[phase])
        print(f"{phase:<10}{st.mean(w):>8.1f}{st.median(w):>10.1f}{min(w):>8.1f}{max(w):>8.1f}{gpu:>8.2f}{len(w):>9}")
    idles = [st.median(watts[p]) for p in ("idle1", "idle2") if p in watts]
    idle = st.mean(idles)
    if len(idles) == 2 and abs(idles[0] - idles[1]) > 0.2 * max(idles):
        print(f"\nWARNING: the two idle medians differ ({idles[0]:.1f} vs {idles[1]:.1f} W): the machine's own idle draw moved "
              "(display, background apps), so differences of a few watts are not reliable.")
    print(f"\nIdle baseline (mean of the idle medians): {idle:.1f} W")
    for phase, label in (("ane", "ANE"), ("gpu", "GPU")):
        if phase not in watts:
            continue
        rate = requests_per_second(out / f"{phase}.log")
        mean, med = st.mean(watts[phase]), st.median(watts[phase])
        line = f"{label}: {mean:.1f} W (median {med:.1f}), {mean - idle:+.1f} W over idle"
        if rate:
            line += f", {rate:.2f} requests/s -> {(mean - idle) / rate:.1f} J/request over idle ({mean / rate:.1f} J including the idle baseline)"
        print(line)
    if "ane_hold" in watts:
        print(f"ANE loaded, no requests: {st.median(watts['ane_hold']):.1f} W median ({st.median(watts['ane_hold']) - idle:+.1f} W over idle)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
