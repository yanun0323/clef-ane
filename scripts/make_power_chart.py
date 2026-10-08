"""Draw docs/power-chart.svg from a powermetrics results directory. Usage: make_power_chart.py RESULTS_DIR OUT.svg"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_summary as ps

COLORS = {"CPU": "#8e9aaf", "GPU": "#e07a5f", "ANE": "#3d85c6"}


def main(results: Path, out: Path) -> None:
    data = ps.load(results)
    # Phases with no workload use the median: a background burst of a few seconds should not move them.
    mean = lambda phase, rail: statistics.median(data[phase][rail]) if phase in ("idle1", "ane_hold") else ps.mean(data[phase][rail])
    rate = {p: ps.requests_per_second(results / f"{p}.log") for p in ("ane", "gpu")}
    idle = mean("idle1", "Combined")   # the first idle reading, before anything else ran
    w, h = 880, 640
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="-apple-system, Helvetica, Arial, sans-serif" font-size="13">',
         f'<rect width="{w}" height="{h}" fill="white"/>']

    def text(x, y, t, size=13, anchor="start", weight="normal", fill="#222"):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{t}</text>')

    # --- panel 1: power by rail, horizontal stacked bars -------------------------------------------------------
    text(30, 32, "Power while running Clef-Flash (average of 45 one-second samples)", 16, weight="bold")
    rows = [("GPU (MLX 4-bit)", "gpu"), ("ANE (fp16, private API)", "ane"), ("ANE loaded, no requests (median)", "ane_hold"), ("Idle (median)", "idle1")]
    x0, scale = 270, 520 / 56000.0
    for i, (label, phase) in enumerate(rows):
        y = 62 + i * 46
        text(x0 - 10, y + 18, label, anchor="end")
        x = x0
        for rail in ("CPU", "GPU", "ANE"):
            v = mean(phase, rail)
            if v * scale >= 0.5:
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{v * scale:.1f}" height="26" fill="{COLORS[rail]}"/>')
            x += v * scale
        total = mean(phase, "Combined")
        text(x + 8, y + 18, f"{total / 1000:.1f} W" if total >= 1000 else f"{total:.0f} mW", weight="bold")
    for j, rail in enumerate(("CPU", "GPU", "ANE")):
        s.append(f'<rect x="{x0 + j * 90}" y="{62 + 4 * 46}" width="14" height="14" fill="{COLORS[rail]}"/>')
        text(x0 + j * 90 + 20, 62 + 4 * 46 + 12, rail)

    # --- panel 2: energy per request -----------------------------------------------------------------------------
    top = 340
    text(30, top, "Energy per request (rail power / requests per second)", 16, weight="bold")
    bars = [("GPU", "gpu", "Combined"), ("ANE", "ane", "Combined")]
    x0, scale = 230, 560 / 18.0
    for i, (label, phase, rail) in enumerate(bars):
        y = top + 30 + i * 80
        total = mean(phase, "Combined") / rate[phase] / 1000
        dev = mean(phase, label) / rate[phase] / 1000
        text(x0 - 10, y + 18, f"{label}: CPU+GPU+ANE rails", anchor="end")
        s.append(f'<rect x="{x0}" y="{y}" width="{total * scale:.1f}" height="26" fill="{COLORS[label]}"/>')
        text(x0 + total * scale + 8, y + 18, f"{total:.1f} J", weight="bold")
        text(x0 - 10, y + 50, f"{label} rail only", anchor="end", fill="#555")
        s.append(f'<rect x="{x0}" y="{y + 32}" width="{dev * scale:.1f}" height="20" fill="{COLORS[label]}" opacity="0.55"/>')
        text(x0 + dev * scale + 8, y + 47, f"{dev:.1f} J", fill="#555")

    # --- panel 3: speed ---------------------------------------------------------------------------------------------
    top = 530
    text(30, top, "Speed on the same six requests (360 to 520 tokens each)", 16, weight="bold")
    for i, (label, phase) in enumerate((("GPU", "gpu"), ("ANE", "ane"))):
        text(30 + i * 400, top + 28, f"{label}: {rate[phase]:.2f} requests/s ({1 / rate[phase]:.2f} s each)")
    text(30, top + 56, "Rails only: powermetrics does not report DRAM, display or the rest of the machine. "
         f"Idle (first reading, median): {idle:.0f} mW.", 12, fill="#666")
    text(30, top + 76, "Measured on Mac17,8 (M5 Pro, 64 GB), macOS 27, on AC power.", 12, fill="#666")
    s.append("</svg>")
    out.write_text("\n".join(s))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
