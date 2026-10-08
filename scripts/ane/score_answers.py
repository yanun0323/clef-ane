"""Compare an answers file (record -> question -> option -> probability) with data/ref/golden.json."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[2] / "data" / "ref"
golden = json.loads((root / "golden.json").read_text())
for name in sys.argv[1:]:
    got = json.loads(Path(name).read_text()); flips = clear_flips = n = 0; worst = 0.0; kls = []; diffs = []
    for rid, qs in got.items():
        for q, p in qs.items():
            ref = golden[rid][q]; opts = list(ref)
            a = np.array([ref[o] for o in opts]); b = np.array([float(p[o]) for o in opts]); b = b / b.sum()
            n += 1; diffs.append(np.abs(a - b).max()); kls.append(float((a * (np.log(a + 1e-9) - np.log(b + 1e-9))).sum()))
            if a.argmax() != b.argmax():
                flips += 1; clear_flips += int(np.sort(a)[-1] - np.sort(a)[-2] > 0.05)
    print(f"{Path(name).name:<28} {n} questions: {flips} different top answers ({clear_flips} where the reference was not a near-tie), "
          f"mean max |dp| {np.mean(diffs):.3f}, worst {np.max(diffs):.3f}, mean KL {np.mean(kls):.4f}")
