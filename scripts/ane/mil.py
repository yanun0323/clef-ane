"""A small MIL text builder with shape tracking, for programs that run on the ANE.

Tensors are named SSA values. Every helper returns a `T` carrying its name, shape and dtype so
later ops can derive result shapes. Weights become separate BLOBFILE entries.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HEADER = (
    'program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n{\n'
)


@dataclass(frozen=True)
class T:
    name: str
    shape: tuple[int, ...]
    dtype: str = "fp16"

    @property
    def type(self) -> str:
        return f"tensor<{self.dtype}, [{', '.join(map(str, self.shape))}]>"


WEIGHT_PATH = "@model_path/weights/weight.bin"


class WeightBlob:
    """One weight.bin holding every constant of a program: a 64-byte file header, then 64-byte-aligned chunks.

    Programs with several separate weight files compile or fail depending on the file names, so we use one file.
    Each chunk is a 64-byte chunk header followed by fp16 data; a BLOBFILE offset points at the chunk header.
    """

    def __init__(self) -> None:
        self.data = bytearray(64)
        self.data[0], self.data[4] = 1, 2

    # Core ML's chunk type codes for sub-byte index tensors
    INDEX_CODES = {4: 0x0B, 6: 0x0D}

    def add_indices(self, indices: np.ndarray, bits: int) -> int:
        """Lookup-table indices (0..2^bits-1) packed as a little-endian bit stream, the layout Core ML writes."""
        flat = np.ascontiguousarray(indices, dtype=np.uint8).reshape(-1)
        assert flat.max() < (1 << bits)
        if bits == 4:
            assert flat.size % 2 == 0
            payload = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
        elif bits == 6:
            assert flat.size % 4 == 0
            v = flat.reshape(-1, 4).astype(np.uint32)
            word = v[:, 0] | (v[:, 1] << 6) | (v[:, 2] << 12) | (v[:, 3] << 18)           # 4 indices -> 24 bits -> 3 bytes
            payload = np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], axis=1).astype(np.uint8)
        else:
            raise ValueError(f"unsupported index width {bits}")
        payload = payload.tobytes()
        offset = len(self.data)
        header = bytearray(64)
        header[0:4] = b"\xef\xbe\xad\xde"
        header[4] = self.INDEX_CODES[bits]
        struct.pack_into("<I", header, 8, len(payload))
        struct.pack_into("<I", header, 16, offset + 64)
        self.data += header + payload
        self.data += bytes(-len(self.data) % 64)
        return offset

    def add(self, array: np.ndarray) -> int:
        if array.dtype == np.int8:
            payload, bits = np.ascontiguousarray(array).tobytes(), 0x08
        else:
            payload, bits = np.ascontiguousarray(array, dtype=np.float16).tobytes(), 0x10
        offset = len(self.data)
        header = bytearray(64)
        header[0:4] = b"\xef\xbe\xad\xde"
        header[4] = 1
        header[10] = bits
        struct.pack_into("<I", header, 8, len(payload))
        struct.pack_into("<I", header, 16, offset + 64)
        self.data += header + payload
        self.data += bytes(-len(self.data) % 64)
        return offset


def _broadcast(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(np.broadcast_shapes(a, b))


class Graph:
    default_quantize = None   # set to a function (weight -> (indices, tables, bits)) to store big linear layers as lookup tables

    def __init__(self) -> None:
        self.quantize = Graph.default_quantize
        self.lines: list[str] = []
        self.inputs: list[T] = []
        self.outputs: list[T] = []
        self.blob = WeightBlob()
        self._n = 0
        self._consts: dict[tuple, str] = {}
        self.defined: list[tuple[T, int]] = []  # (op result, number of lines up to and including it)

    # -- plumbing ---------------------------------------------------------------------------
    def _fresh(self, hint: str) -> str:
        self._n += 1
        return f"{hint}_{self._n}"

    def _emit(self, name: str, type_: str, expr: str) -> None:
        self.lines.append(f'        {type_} {name} = {expr}[name = string("{name}")];')

    def _const(self, key: tuple, type_: str, literal: str, hint: str) -> str:
        if key not in self._consts:
            name = self._fresh(hint)
            self.lines.append(f'        {type_} {name} = const()[name = string("{name}"), val = {literal}];')
            self._consts[key] = name
        return self._consts[key]

    def scalar(self, value: float, dtype: str = "fp16") -> str:
        return self._const(("s", value, dtype), dtype, f"{dtype}({value:.12f})", "c")

    def int_scalar(self, value: int) -> str:
        return self._const(("i", value), "int32", f"int32({value})", "i")

    def ints(self, values) -> str:
        values = list(values)
        return self._const(("v", tuple(values)), f"tensor<int32, [{len(values)}]>",
                           f"tensor<int32, [{len(values)}]>([{', '.join(map(str, values))}])", "iv")

    def boolean(self, value: bool) -> str:
        return self._const(("b", value), "bool", f"bool({str(value).lower()})", "b")

    def string(self, value: str) -> str:
        return self._const(("str", value), "string", f'string("{value}")', "str")

    def op(self, kind: str, shape, args: dict, dtype: str = "fp16", name: str | None = None) -> T:
        rendered = ", ".join(f"{k}={v}" for k, v in args.items())
        out = T(name or self._fresh(kind), tuple(shape), dtype)
        self._emit(out.name, out.type, f"{kind}({rendered})")
        self.defined.append((out, len(self.lines)))
        return out

    # -- inputs, weights, outputs -------------------------------------------------------------
    def input(self, name: str, shape, dtype: str = "fp16") -> T:
        t = T(name, tuple(shape), dtype)
        self.inputs.append(t)
        return t

    def output(self, *tensors: T) -> None:
        self.outputs.extend(tensors)

    @property
    def weights(self) -> dict[str, bytes]:
        return {WEIGHT_PATH: bytes(self.blob.data)}

    def const_tensor(self, name: str, array: np.ndarray) -> T:
        """Constant fp16 tensor stored in the program's weight blob."""
        array = np.asarray(array)
        offset = self.blob.add(array)
        t = T(self._fresh(name), array.shape)
        shape = ", ".join(map(str, array.shape))
        self.lines.append(
            f'        {t.type} {t.name} = const()[name = string("{t.name}"), val = tensor<fp16, [{shape}]>'
            f'(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({offset})))];'
        )
        return t

    # -- elementwise ------------------------------------------------------------------------
    def _binary(self, kind: str, x: T, y) -> T:
        if not isinstance(y, T):
            y = T(self.scalar(float(y), x.dtype), (), x.dtype)
        return self.op(kind, _broadcast(x.shape, y.shape), {"x": x.name, "y": y.name}, x.dtype)

    def add(self, x, y): return self._binary("add", x, y)
    def sub(self, x, y): return self._binary("sub", x, y)
    def mul(self, x, y): return self._binary("mul", x, y)
    def maximum(self, x, y): return self._binary("maximum", x, y)
    def minimum(self, x, y): return self._binary("minimum", x, y)

    def _unary(self, kind: str, x: T, **extra) -> T:
        return self.op(kind, x.shape, {"x": x.name, **extra}, x.dtype)

    def exp(self, x): return self._unary("exp", x)
    def div(self, x, y): return self._binary("real_div", x, y)

    # The ANE fuses x*sigmoid(x) (and its native sigmoid/silu/softplus) into approximate hardware activations,
    # which cost 1-2% accuracy. Building them from exp and a divide is ~30x more accurate and still runs on the ANE.
    def sigmoid(self, x):
        return self.div(self.reciprocal_of_one(x), self.add(self.exp(self.mul(x, -1.0)), 1.0))

    def reciprocal_of_one(self, like: T) -> T:
        return T(self.scalar(1.0, like.dtype), (), like.dtype)

    def softplus(self, x, cutoff: float = 8.0, split: float = -2.0):
        """log(1 + exp(x)) accurate over the whole range.

        The native op is only good to about 1e-3 absolute near x = -7, which zeroes out the tiny decay rates of the
        slowest Gated DeltaNet heads (softplus ~ 5e-4). Below `split` the series e^x (1 - e^x/2 + e^2x/3 - e^3x/4)
        is used instead (exp is accurate down to fp16's smallest values); above it, the native op. Above `cutoff`
        the result is x itself, which also keeps exp inside fp16.
        """
        e = self.exp(self.minimum(x, split))
        t = self.sub_from(0.333333333, self.mul(e, 0.25))
        t = self.sub_from(0.5, self.mul(e, t))
        t = self.sub_from(1.0, self.mul(e, t))
        low = self.mul(e, t)
        clipped = self.maximum(self.minimum(x, cutoff), split)
        high = self.add(self._unary("softplus", clipped), self.maximum(self.sub(x, cutoff), 0.0))
        use_low = self.minimum(self.maximum(self.mul(self.sub_from(split, x), 64.0), 0.0), 1.0)
        return self.add(self.mul(use_low, low), self.mul(self.sub_from(1.0, use_low), high))

    def sub_from(self, constant: float, x: T) -> T:
        """constant - x"""
        return self.add(self.mul(x, -1.0), constant)

    def sqrt(self, x): return self._unary("sqrt", x)
    def abs(self, x): return self._unary("abs", x)

    def pow(self, x, exponent: float):
        return self.op("pow", x.shape, {"x": x.name, "y": self.scalar(exponent, x.dtype)}, x.dtype)

    def square(self, x):
        # The compiler rejects an op that takes the same tensor for both operands, so mul(x, x) is not an option.
        return self.pow(x, 2.0)

    def silu(self, x):
        return self.div(x, self.add(self.exp(self.mul(x, -1.0)), 1.0))

    def rsqrt(self, x, eps: float = 1e-6):
        return self._unary("rsqrt", x, epsilon=self.scalar(eps, x.dtype))

    def rename(self, x: T, name: str) -> T:
        """Give a value a fixed name (output names decide the ANE's output buffer order)."""
        return self.op("identity", x.shape, {"x": x.name}, x.dtype, name=name)

    def cast(self, x: T, dtype: str) -> T:
        return self.op("cast", x.shape, {"dtype": self.string(dtype), "x": x.name}, dtype)

    # -- reductions and shape ops ---------------------------------------------------------------
    def reduce_sum(self, x: T, axis: int, keep: bool = True) -> T:
        shape = list(x.shape)
        shape[axis] = 1 if keep else None
        shape = [s for s in shape if s is not None]
        return self.op("reduce_sum", shape, {"x": x.name, "axes": self.ints([axis]), "keep_dims": self.boolean(keep)}, x.dtype)

    def reduce_max(self, x: T, axis: int) -> T:
        shape = list(x.shape)
        shape[axis] = 1
        return self.op("reduce_max", shape, {"x": x.name, "axes": self.ints([axis]), "keep_dims": self.boolean(True)}, x.dtype)

    def reshape(self, x: T, shape) -> T:
        return self.op("reshape", shape, {"shape": self.ints(shape), "x": x.name}, x.dtype)

    def transpose(self, x: T, perm) -> T:
        return self.op("transpose", [x.shape[p] for p in perm], {"perm": self.ints(perm), "x": x.name}, x.dtype)

    def slice(self, x: T, begin, size) -> T:
        return self.op("slice_by_size", size, {"x": x.name, "begin": self.ints(begin), "size": self.ints(size)}, x.dtype)

    def concat(self, values: list[T], axis: int) -> T:
        shape = list(values[0].shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        names = ", ".join(v.name for v in values)
        return self.op("concat", shape, {"axis": self.int_scalar(axis), "interleave": self.boolean(False), "values": f"({names})"}, values[0].dtype)

    def matmul(self, x: T, y: T, transpose_x: bool = False, transpose_y: bool = False) -> T:
        m, k = (x.shape[-1], x.shape[-2]) if transpose_x else (x.shape[-2], x.shape[-1])
        n = y.shape[-2] if transpose_y else y.shape[-1]
        batch = _broadcast(x.shape[:-2], y.shape[:-2])
        return self.op("matmul", (*batch, m, n), {
            "transpose_x": self.boolean(transpose_x), "transpose_y": self.boolean(transpose_y),
            "x": x.name, "y": y.name}, x.dtype)

    def conv1x1(self, x: T, name: str, weight: np.ndarray) -> T:
        """Linear layer on a [1, C_in, 1, S] tensor; weight is [C_out, C_in]."""
        out_ch, in_ch = weight.shape
        if self.quantize is not None and weight.size >= 1 << 20:
            indices, tables, bits = self.quantize(weight)
            return self.conv1x1_lut(x, name, indices, tables, bits)
        w = self.const_tensor(name, weight.reshape(out_ch, in_ch, 1, 1))
        args = {"dilations": self.ints([1, 1]), "groups": self.int_scalar(1), "pad": self.ints([0, 0, 0, 0]),
                "pad_type": self.string("valid"), "strides": self.ints([1, 1]), "weight": w.name, "x": x.name}
        return self.op("conv", (1, out_ch, 1, x.shape[-1]), args, x.dtype)

    def conv1x1_int8(self, x: T, name: str, weight: np.ndarray) -> T:
        """Like conv1x1, but the weight is stored as int8 with one fp16 scale per output channel (dequantised on chip)."""
        out_ch, in_ch = weight.shape
        scale = np.abs(weight).max(axis=1) / 127.0
        scale = np.maximum(scale, 1e-8).astype(np.float16)
        q = np.clip(np.round(weight / scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        offset = self.blob.add(q.reshape(out_ch, in_ch, 1, 1))
        literal = f"tensor<fp16, [{out_ch}]>([{', '.join(f'{float(v):.9g}' for v in scale)}])"   # must be inline, not a blob
        w = T(self._fresh(name), (out_ch, in_ch, 1, 1))
        self.lines.append(
            f'        {w.type} {w.name} = constexpr_affine_dequantize()[axis = int32(0), name = string("{w.name}"), '
            f'quantized_data = tensor<int8, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({offset}))), '
            f'scale = {literal}, zero_point = int8(0)];')
        args = {"dilations": self.ints([1, 1]), "groups": self.int_scalar(1), "pad": self.ints([0, 0, 0, 0]),
                "pad_type": self.string("valid"), "strides": self.ints([1, 1]), "weight": w.name, "x": x.name}
        return self.op("conv", (1, out_ch, 1, x.shape[-1]), args, x.dtype)

    def conv1x1_lut(self, x: T, name: str, indices: np.ndarray, lut: np.ndarray, bits: int) -> T:
        """Like conv1x1, but the weight is stored as `bits`-bit indices into fp16 lookup tables.

        `lut` is [2^bits] (one table) or [G, 2^bits]: one table per group of C_out/G consecutive output channels.
        """
        out_ch, in_ch = indices.shape
        lut = np.asarray(lut, np.float16)
        groups = 1 if lut.ndim == 1 else lut.shape[0]
        offset = self.blob.add_indices(indices.reshape(out_ch, in_ch, 1, 1), bits)
        table = self.const_tensor(name + "_lut", lut.reshape(groups, 1, 1, 1, 1 << bits, 1))
        w = T(self._fresh(name), (out_ch, in_ch, 1, 1))
        self.lines.append(
            f'        {w.type} {w.name} = constexpr_lut_to_dense(indices = tensor<uint{bits}, [{out_ch}, {in_ch}, 1, 1]>'
            f'(BLOBFILE(path = string("{WEIGHT_PATH}"), offset = uint64({offset}))), lut = {table.name})[name = string("{w.name}")];')
        args = {"dilations": self.ints([1, 1]), "groups": self.int_scalar(1), "pad": self.ints([0, 0, 0, 0]),
                "pad_type": self.string("valid"), "strides": self.ints([1, 1]), "weight": w.name, "x": x.name}
        return self.op("conv", (1, out_ch, 1, x.shape[-1]), args, x.dtype)

    def conv1x1_lut4(self, x: T, name: str, indices: np.ndarray, lut: np.ndarray) -> T:
        return self.conv1x1_lut(x, name, indices, lut, 4)

    def depthwise_conv(self, x: T, name: str, weight: np.ndarray) -> T:
        """Valid (unpadded) depthwise conv over the last axis. weight is [C, K]; x is [1, C, 1, S+K-1]."""
        ch, k = weight.shape
        w = self.const_tensor(name, weight.reshape(ch, 1, 1, k))
        args = {"dilations": self.ints([1, 1]), "groups": self.int_scalar(ch), "pad": self.ints([0, 0, 0, 0]),
                "pad_type": self.string("valid"), "strides": self.ints([1, 1]), "weight": w.name, "x": x.name}
        return self.op("conv", (1, ch, 1, x.shape[-1] - k + 1), args, x.dtype)

    def upto(self, tensor: T, cuts: tuple[T, ...] = ()) -> "Graph":
        """A copy that keeps only what `tensor` depends on and outputs it. Tensors in `cuts` become inputs.

        Used for bisecting compile failures: the compiler only says "InvalidMILProgram", not where.
        """
        import re

        count = next(n for t, n in self.defined if t.name == tensor.name)
        defs: dict[str, tuple[int, set[str]]] = {}
        for i, line in enumerate(self.lines[:count]):
            m = re.match(r"\s+(?:tensor<[^>]*>|\w+) (\w+) = (.*)$", line)
            defs[m.group(1)] = (i, set(re.findall(r"[A-Za-z_]\w*", m.group(2))) - {m.group(1)})
        cut_names = {c.name for c in cuts}
        needed, leaves, stack = set(), set(), [tensor.name]
        while stack:
            name = stack.pop()
            if name in needed or name in leaves:
                continue
            if name in cut_names and name != tensor.name:
                leaves.add(name)
                continue
            if name not in defs:
                continue
            needed.add(name)
            stack.extend(defs[name][1])
        copy = Graph()
        copy.lines = [self.lines[defs[n][0]] for n in sorted(needed, key=lambda n: defs[n][0])]
        copy.defined = [(t, 0) for t, _ in self.defined if t.name in needed]
        used = set().union(*(defs[n][1] for n in needed)) if needed else set()
        copy.inputs = [t for t in self.inputs if t.name in used] + [c for c in cuts if c.name in leaves]
        copy.blob = self.blob
        copy.outputs = [tensor]
        return copy

    # -- output -----------------------------------------------------------------------------
    def text(self) -> str:
        sig = ", ".join(f"{t.type} {t.name}" for t in self.inputs)
        outs = ", ".join(t.name for t in self.outputs)
        return f"{HEADER}    func main<ios18>({sig}) {{\n" + "\n".join(self.lines) + f"\n    }} -> ({outs});\n}}\n"
