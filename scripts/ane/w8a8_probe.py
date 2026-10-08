"""Is INT8 activation caching (quantize/dequantize between convs) faster on this ANE? Speed only; accuracy is a separate question."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel, CompileError
from mil import Graph
rng = np.random.default_rng(0)
ch, rows, depth = 4096, 128, 8
ws = [(rng.standard_normal((ch, ch)) * 0.02).astype(np.float32) for _ in range(depth)]
x = rng.standard_normal((1, ch, 1, rows)).astype(np.float16)
for quant in (False, True):
    g = Graph(); t = g.input("hid_in", x.shape); h = t
    for i, w in enumerate(ws):
        h = g.conv1x1(h, f"w{i}", w)
        if quant and i < depth - 1:
            q = g.op("quantize", h.shape, {"input": h.name, "output_dtype": g.string("int8"), "scale": g.scalar(0.125)}, "int8")
            h = g.op("dequantize", h.shape, {"input": q.name, "scale": g.scalar(0.125)}, "fp16")
    g.output(h)
    try:
        k = Kernel(g); print(f"{'int8 activations' if quant else 'fp16':<17} {depth} x conv {ch}: {k.time(x, iters=20):.3f} ms")
    except CompileError as e: print("int8 activations COMPILE FAIL", str(e)[:120])
