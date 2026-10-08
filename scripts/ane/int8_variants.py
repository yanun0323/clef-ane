import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel, CompileError
from mil import Graph, WEIGHT_PATH, T
rng = np.random.default_rng(0)
out_ch, in_ch, rows = 1024, 1024, 128
w = (rng.standard_normal((out_ch, in_ch)) * 0.02).astype(np.float32)
x = rng.standard_normal((1, in_ch, 1, rows)).astype(np.float16)
want = np.einsum("oc,bcxs->boxs", w.astype(np.float64), x.astype(np.float64))
def build(scale_kind, axis):
    g = Graph(); t = g.input("hid_in", x.shape)
    s = np.abs(w).max() / 127; q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    off = g.blob.add(q.reshape(out_ch, in_ch, 1, 1))
    if scale_kind == "scalar": scale = f"fp16({float(np.float16(s)):.8f})"; ws = float(np.float16(s))
    elif scale_kind == "literal_vec":
        sv = np.full(out_ch, s, np.float16); scale = f"tensor<fp16, [{out_ch}]>([{', '.join(f'{v:.8f}' for v in sv)}])"
    else: scale = None
    wt = T("wdeq", (out_ch, in_ch, 1, 1))
    ax = f"axis = int32({axis}), " if axis is not None else ""
    g.lines.append(f'        {wt.type} wdeq = constexpr_affine_dequantize()[{ax}name = string("wdeq"), quantized_data = tensor<int8, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({off}))), scale = {scale}, zero_point = int8(0)];')
    a = {"dilations": g.ints([1, 1]), "groups": g.int_scalar(1), "pad": g.ints([0, 0, 0, 0]), "pad_type": g.string("valid"), "strides": g.ints([1, 1]), "weight": "wdeq", "x": t.name}
    g.output(g.op("conv", (1, out_ch, 1, rows), a)); return g
for kind, axis in [("scalar", 0), ("scalar", None), ("literal_vec", 0)]:
    try:
        k = Kernel(build(kind, axis)); y = k(x)[0].astype(np.float64)
        print(f"{kind:<12} axis={axis}: OK rms rel err {np.sqrt(((y - want) ** 2).mean() / (want ** 2).mean()):.4f}  {k.time(x, iters=30):.3f} ms")
    except CompileError as e: print(f"{kind:<12} axis={axis}: COMPILE FAIL")
