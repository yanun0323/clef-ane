"""Python wrapper around libane_bridge: compile a mil.Graph for the ANE and run it with numpy arrays."""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import numpy as np

if __package__:
    from .mil import Graph, T
else:
    from mil import Graph, T

_LIB = ctypes.CDLL(os.environ.get("CLEF_ANE_BRIDGE") or str(Path(__file__).resolve().parents[2] / "build" / "libane_bridge.dylib"))
_PTR = ctypes.c_void_p
_SIZE = ctypes.c_size_t
_LIB.ane_init.argtypes = []
_LIB.ane_init.restype = ctypes.c_int
_LIB.ane_compile.argtypes = [ctypes.c_char_p, _SIZE, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                             ctypes.POINTER(_PTR), ctypes.POINTER(_SIZE), ctypes.c_int, ctypes.POINTER(_SIZE),
                             ctypes.c_int, ctypes.POINTER(_SIZE), ctypes.c_char_p, _SIZE]
_LIB.ane_compile.restype = _PTR
_LIB.ane_write_input.argtypes = [_PTR, ctypes.c_int, _PTR, _SIZE]
_LIB.ane_write_input.restype = ctypes.c_int
_LIB.ane_write_input_at.argtypes = [_PTR, ctypes.c_int, _SIZE, _PTR, _SIZE]
_LIB.ane_write_input_at.restype = ctypes.c_int
_LIB.ane_read_output.argtypes = [_PTR, ctypes.c_int, _PTR, _SIZE]
_LIB.ane_read_output.restype = ctypes.c_int
_LIB.ane_eval.argtypes = [_PTR, ctypes.c_char_p, _SIZE]
_LIB.ane_eval.restype = ctypes.c_int
_LIB.ane_free.argtypes = [_PTR]
_LIB.ane_free.restype = None
if _LIB.ane_init() != 0:
    raise RuntimeError("private ANE classes are not available on this system")

NP = {"fp16": np.float16, "fp32": np.float32}


def _padded(t: T) -> tuple[int, ...]:
    """The ANE stores each row (last axis) padded to 64 bytes, so I/O buffers use the padded width."""
    per_row = 64 // np.dtype(NP[t.dtype]).itemsize
    return (*t.shape[:-1], -(-t.shape[-1] // per_row) * per_row)


def _bytes(t: T) -> int:
    return int(np.prod(_padded(t))) * np.dtype(NP[t.dtype]).itemsize


class CompileError(RuntimeError):
    pass


class Kernel:
    """A compiled and loaded ANE program."""

    def __init__(self, graph: Graph) -> None:
        self.inputs, self.outputs = graph.inputs, graph.outputs
        # The ANE numbers its I/O buffers by tensor name in alphabetical order, not in declaration order.
        self._in_slot = {t.name: i for i, t in enumerate(sorted(self.inputs, key=lambda t: t.name))}
        self._out_slot = {t.name: i for i, t in enumerate(sorted(self.outputs, key=lambda t: t.name))}
        weights = list(graph.weights.items())
        names = (ctypes.c_char_p * len(weights))(*[n.encode() for n, _ in weights])
        keep = [ctypes.create_string_buffer(b, len(b)) for _, b in weights]
        datas = (ctypes.c_void_p * len(weights))(*[ctypes.addressof(b) for b in keep])
        lens = (ctypes.c_size_t * len(weights))(*[len(b) for _, b in weights])
        in_bytes = (ctypes.c_size_t * len(self.inputs))(*[_bytes(t) for t in sorted(self.inputs, key=lambda t: t.name)])
        out_bytes = (ctypes.c_size_t * len(self.outputs))(*[_bytes(t) for t in sorted(self.outputs, key=lambda t: t.name)])
        mil = graph.text().encode()
        err = ctypes.create_string_buffer(4096)
        start = time.perf_counter()
        self._handle = _LIB.ane_compile(mil, len(mil), len(weights), names, datas, lens, len(self.inputs), in_bytes,
                                        len(self.outputs), out_bytes, err, len(err))
        self.compile_seconds = time.perf_counter() - start
        if not self._handle:
            raise CompileError(err.value.decode())

    def write_region(self, name: str, offset_elems: int, array: np.ndarray) -> None:
        """Overwrite part of input `name` in place. offset_elems counts fp16/fp32 elements from the start of the buffer.

        Only valid for inputs whose last axis is already 64-byte aligned (no row padding)."""
        t = next(t for t in self.inputs if t.name == name)
        assert _padded(t) == t.shape, f"{name} has padded rows; region writes need aligned rows"
        a = np.ascontiguousarray(array, dtype=NP[t.dtype])
        if _LIB.ane_write_input_at(ctypes.c_void_p(self._handle), self._in_slot[name], offset_elems * a.itemsize,
                                   a.ctypes.data_as(ctypes.c_void_p), a.nbytes):
            raise RuntimeError(f"region write to {name} out of range")

    def __call__(self, *arrays: np.ndarray | None) -> list[np.ndarray]:
        """Run the program. An input given as None keeps whatever the buffer already holds."""
        for i, (t, a) in enumerate(zip(self.inputs, arrays)):
            if a is None:
                continue
            if a.shape != t.shape:
                raise ValueError(f"input {t.name}: expected {t.shape}, got {a.shape}")
            if _padded(t) != t.shape:                       # only rows that are not 64-byte aligned need padding
                padded = np.zeros(_padded(t), dtype=NP[t.dtype])
                padded[..., : t.shape[-1]] = a
                a = padded
            else:
                a = np.ascontiguousarray(a, dtype=NP[t.dtype])
            if _LIB.ane_write_input(ctypes.c_void_p(self._handle), self._in_slot[t.name], a.ctypes.data_as(ctypes.c_void_p), a.nbytes):
                raise RuntimeError("write failed")
        err = ctypes.create_string_buffer(4096)
        if _LIB.ane_eval(ctypes.c_void_p(self._handle), err, len(err)):
            raise RuntimeError(err.value.decode())
        results = []
        for i, t in enumerate(self.outputs):
            out = np.empty(_padded(t), dtype=NP[t.dtype])
            if _LIB.ane_read_output(ctypes.c_void_p(self._handle), self._out_slot[t.name], out.ctypes.data_as(ctypes.c_void_p), out.nbytes):
                raise RuntimeError("read failed")
            results.append(out if out.shape == t.shape else out[..., : t.shape[-1]].copy())
        return results

    def time(self, *arrays: np.ndarray, iters: int = 20) -> float:
        self(*arrays)
        start = time.perf_counter()
        for _ in range(iters):
            self(*arrays)
        return (time.perf_counter() - start) / iters * 1e3

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _LIB.ane_free(ctypes.c_void_p(self._handle))
            self._handle = None

    def __del__(self) -> None:
        self.close()
