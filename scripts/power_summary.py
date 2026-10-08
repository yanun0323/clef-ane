"""Summarise powermetrics output (idle1, ane, ane_hold, gpu, idle2) in a results directory.

Usage: power_summary.py RESULTS_DIR            prints the table
Importable: load(dir) -> {phase: {rail: [mW per second]}}, requests_per_second(log).
"""
import re
import sys
from pathlib import Path

RAILS = ("CPU", "GPU", "ANE", "Combined")
PHASES = ("idle1", "ane", "ane_hold", "gpu", "idle2")
# powermetrics prints the three rails, then their sum, on four consecutive lines; the GPU section later repeats a
# "GPU Power" line of its own with a slightly different value, so match the block rather than single lines.
BLOCK = re.compile(r"^CPU Power: (\d+) mW\nGPU Power: (\d+) mW\nANE Power: (\d+) mW\nCombined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", re.M)


def rails(path: Path) -> dict[str, list[int]]:
    samples = [tuple(int(v) for v in m) for m in BLOCK.findall(path.read_text())]
    return {name: [s[i] for s in samples] for i, name in enumerate(RAILS)}


def load(directory: Path) -> dict[str, dict[str, list[int]]]:
    return {p: rails(directory / f"{p}.pm.txt") for p in PHASES if (directory / f"{p}.pm.txt").exists()}


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def requests_per_second(log: Path):
    match = re.search(r"(\d+) requests.*?([\d.]+) s/request", log.read_text()) if log.exists() else None
    return 1.0 / float(match.group(2)) if match else None


def main(out: Path) -> None:
    data = load(out)
    if not data or not any(v for r in data.values() for v in r.values()):
        sys.exit(f"no power readings parsed from {out}; check the *.pm.err files")
    print(f"{'phase':<10}" + "".join(f"{n + ' mW':>13}" for n in RAILS) + f"{'samples':>9}")
    for phase, r in data.items():
        print(f"{phase:<10}" + "".join(f"{mean(r[n]):>13.0f}" for n in RAILS) + f"{len(r['Combined']):>9}")

    idles = [mean(data[p]["Combined"]) for p in ("idle1", "idle2") if p in data]
    idle = mean(idles)
    if len(idles) == 2 and abs(idles[0] - idles[1]) > 0.2 * max(idles):
        print(f"\nWARNING: the two idle readings differ ({idles[0]:.0f} vs {idles[1]:.0f} mW): something else is running, "
              "so 'above idle' numbers are unreliable. Use the absolute rails.")
    print("\n(Combined = CPU + GPU + ANE rails only; it leaves out DRAM, display and the rest of the machine.)")
    for phase in ("ane", "gpu"):
        if phase not in data:
            continue
        rate = requests_per_second(out / f"{phase}.log")
        total, active = mean(data[phase]["Combined"]), mean(data[phase]["Combined"]) - idle
        line = f"{phase.upper()}: {total:.0f} mW total, {active:.0f} mW above idle"
        if rate:
            line += f", {rate:.2f} requests/s -> {total / rate / 1000:.2f} J/request ({active / rate / 1000:.2f} J above idle)"
        print(line)
        if rate:
            dev = "ANE" if phase == "ane" else "GPU"
            print(f"      {dev} rail alone {mean(data[phase][dev]) / rate / 1000:.2f} J/request; CPU rail {mean(data[phase]['CPU']):.0f} mW")
    if "ane_hold" in data and idles:
        held = mean(data["ane_hold"]["Combined"])
        print(f"ANE model loaded, no requests: {held:.0f} mW total vs {idle:.0f} mW idle (difference {held - idle:+.0f} mW), ANE rail {mean(data['ane_hold']['ANE']):.0f} mW")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
