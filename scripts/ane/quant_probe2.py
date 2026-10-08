"""Time Core ML-generated quantised/palettised conv programs through the private ANE path (fp16 vs int8 vs int4 vs LUT4).

The programs come from /tmp/cmt/gen.py (coremltools + coremlcompiler); this loads each compiled model.mil and weight.bin as is.
Usage: quant_probe2.py WORKDIR ROWS
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel, CompileError
from mil import Graph, T

work, rows = Path(sys.argv[1]), int(sys.argv[2])
IN, OUT = 4096, 12288

class Raw:
    """A pre-built program: MIL text plus the weight blob, with the same interface Kernel expects from a Graph."""
    def __init__(self, mil: str, blob: bytes):
        self._mil, self.weights = mil, {"@model_path/weights/weight.bin": blob}
        self.inputs, self.outputs = [T("x", (1, IN, 1, rows))], [T("out", (1, OUT, 1, rows))]
    def text(self): return self._mil

rng = np.random.default_rng(0)
x = rng.standard_normal((1, IN, 1, rows)).astype(np.float16)
W = (rng.standard_normal((OUT, IN)) * 0.02).astype(np.float32)
g = Graph(); t = g.input("x", x.shape); g.output(g.rename(g.conv1x1(t, "w", W), "out")); base = Kernel(g)
ref = base(x)[0].astype(np.float64)
print(f"rows {rows:>3} fp16 (ours)   {base.time(x, iters=40):6.3f} ms   weights {len(g.blob.data) / 1e6:6.1f} MB")
for name in ("int8ch", "int4block", "lut4"):
    d = work / f"{name}.mlmodelc"
    if not d.exists(): print(f"rows {rows:>3} {name:<12} (not generated)"); continue
    try:
        k = Kernel(Raw((d / "model.mil").read_text(), (d / "weights" / "weight.bin").read_bytes()))
        y = k(x)[0].astype(np.float64)
        print(f"rows {rows:>3} {name:<12} {k.time(x, iters=40):6.3f} ms   weights {(d / 'weights' / 'weight.bin').stat().st_size / 1e6:6.1f} MB   output differs from fp16 by {np.sqrt(((y - ref) ** 2).mean() / (ref ** 2).mean()):.4f} (random weights; this is quantisation error)")
    except CompileError as e:
        print(f"rows {rows:>3} {name:<12} COMPILE FAIL {str(e)[:100]}")
