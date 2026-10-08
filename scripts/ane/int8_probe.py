"""Does an int8-weight conv (constexpr_affine_dequantize) compile on this ANE, how accurate, and is it faster?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel, CompileError
from mil import Graph
rng = np.random.default_rng(0)
for out_ch, in_ch, rows in [(4096, 4096, 128), (12288, 4096, 128), (4096, 12288, 128)]:
    w = (rng.standard_normal((out_ch, in_ch)) * 0.02).astype(np.float32)
    x = (rng.standard_normal((1, in_ch, 1, rows))).astype(np.float16)
    want = np.einsum("oc,bcxs->boxs", w.astype(np.float64), x.astype(np.float64))
    for kind in ("fp16", "int8"):
        g = Graph(); t = g.input("hid_in", x.shape)
        g.output(g.conv1x1(t, "w", w) if kind == "fp16" else g.conv1x1_int8(t, "w", w))
        try:
            k = Kernel(g); y = k(x)[0].astype(np.float64)
            print(f"{out_ch:>5}x{in_ch:<5} {kind:<5} rms rel err {np.sqrt(((y - want) ** 2).mean() / (want ** 2).mean()):.4f}  {k.time(x, iters=30):.3f} ms  compile {k.compile_seconds:.2f}s")
        except CompileError as e:
            print(f"{out_ch:>5}x{in_ch:<5} {kind:<5} COMPILE FAIL {str(e)[:150]}")
