"""Do compressed weights (int8, int4, blockwise int4, 4-bit lookup table) reduce the weight traffic on the ANE?

At few columns per call a 4096x12288 conv is limited by reading its weights, not by arithmetic, so time per call
shows whether the compressed form is read from memory as is (faster) or expanded when the program is compiled (same).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import Kernel, CompileError
from mil import Graph, WEIGHT_PATH, T

rng = np.random.default_rng(0)
OUT, IN = 12288, 4096
W = (rng.standard_normal((OUT, IN)) * 0.02).astype(np.float32)

def add_raw(g: Graph, payload: bytes, bits: int) -> int:
    import struct
    offset = len(g.blob.data)
    header = bytearray(64)
    header[0:4] = b"\xef\xbe\xad\xde"; header[4] = 1; header[10] = bits
    struct.pack_into("<I", header, 8, len(payload)); struct.pack_into("<I", header, 16, offset + 64)
    g.blob.data += header + payload
    g.blob.data += bytes(-len(g.blob.data) % 64)
    return offset

def conv(g, x, weight_name):
    a = {"dilations": g.ints([1, 1]), "groups": g.int_scalar(1), "pad": g.ints([0, 0, 0, 0]), "pad_type": g.string("valid"),
         "strides": g.ints([1, 1]), "weight": weight_name, "x": x.name}
    return g.op("conv", (1, OUT, 1, x.shape[-1]), a)

def graph(kind, rows):
    g = Graph(); x = g.input("hid_in", (1, IN, 1, rows))
    if kind == "fp16":
        return g, g.conv1x1(x, "w", W), W
    if kind == "int8":
        s = np.abs(W).max(1) / 127; q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
        off = add_raw(g, q.tobytes(), 8)
        lit = f"tensor<fp16, [{OUT}]>([{', '.join(f'{float(v):.9g}' for v in s.astype(np.float16))}])"
        g.lines.append(f'        tensor<fp16, [{OUT}, {IN}, 1, 1]> wd = constexpr_affine_dequantize()[axis = int32(0), name = string("wd"), quantized_data = tensor<int8, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({off}))), scale = {lit}, zero_point = int8(0)];')
        return g, conv(g, x, "wd"), q * s[:, None]
    if kind == "int4":
        s = np.abs(W).max(1) / 7; q = np.clip(np.round(W / s[:, None]), -7, 7).astype(np.int8)
        flat = (q.reshape(-1) & 0xF).astype(np.uint8); packed = (flat[0::2] | (flat[1::2] << 4)).tobytes()
        off = add_raw(g, packed, 4)
        lit = f"tensor<fp16, [{OUT}]>([{', '.join(f'{float(v):.9g}' for v in s.astype(np.float16))}])"
        g.lines.append(f'        tensor<fp16, [{OUT}, {IN}, 1, 1]> wd = constexpr_affine_dequantize()[axis = int32(0), name = string("wd"), quantized_data = tensor<int4, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({off}))), scale = {lit}, zero_point = int4(0)];')
        return g, conv(g, x, "wd"), q * s[:, None]
    if kind == "lut4":
        # one 16-entry table per weight matrix (k-means is better; uniform levels are enough to test the speed)
        levels = np.linspace(-1, 1, 16) * np.abs(W).max()
        idx = np.abs(W[..., None] - levels).argmin(-1).astype(np.uint8)
        flat = idx.reshape(-1); packed = (flat[0::2] | (flat[1::2] << 4)).tobytes()
        off = add_raw(g, packed, 4)
        lut = f"tensor<fp16, [1, 1, 1, 1, 16, 1]>([{', '.join(f'{float(v):.9g}' for v in levels.astype(np.float16))}])"
        g.lines.append(f'        tensor<fp16, [{OUT}, {IN}, 1, 1]> wd = constexpr_lut_to_dense()[name = string("wd"), indices = tensor<uint4, [{OUT}, {IN}, 1, 1]>(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({off}))), lut = {lut}, shape = tensor<uint32, [4]>([{OUT}, {IN}, 1, 1])];')
        return g, conv(g, x, "wd"), levels[idx]
    raise ValueError(kind)

for rows in (8, 32, 128):
    x = rng.standard_normal((1, IN, 1, rows)).astype(np.float16)
    for kind in ("fp16", "int8", "int4", "lut4"):
        try:
            g, out, eff = graph(kind, rows); g.output(g.rename(out, "hid_out")); k = Kernel(g)
            y = k(x)[0].astype(np.float64); want = np.einsum("oc,bcxs->boxs", eff.astype(np.float64), x.astype(np.float64))
            err = np.sqrt(((y - want) ** 2).mean() / (want ** 2).mean())
            print(f"rows {rows:>3} {kind:<5} {k.time(x, iters=40):6.3f} ms   err vs its own dequantised weights {err:.4f}", flush=True)
        except CompileError as e:
            print(f"rows {rows:>3} {kind:<5} COMPILE FAIL", flush=True)
        except Exception as e:
            print(f"rows {rows:>3} {kind:<5} ERROR {str(e)[:80]}", flush=True)
