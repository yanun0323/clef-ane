"""Which MIL ops does the ANE compiler accept on this machine, and how accurate are they in fp16?

Run: .venv/bin/python -I scripts/ane/survey_ops.py   (from the project root; adds scripts/ane to the path)
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import CompileError, Kernel  # noqa: E402
from mil import Graph  # noqa: E402

rng = np.random.default_rng(0)
RESULTS = []


def case(name, build, reference, *arrays, tol=2e-2):
    """build(graph, *input_tensors) -> output tensor(s); reference(*arrays) -> expected arrays."""
    g = Graph()
    ins = [g.input(f"in{i}", a.shape) for i, a in enumerate(arrays)]
    outs = build(g, *ins)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    g.output(*outs)
    try:
        k = Kernel(g)
    except CompileError as e:
        RESULTS.append((name, "COMPILE FAIL", str(e)[:140].replace("\n", " ")))
        return
    except Exception:
        RESULTS.append((name, "ERROR", traceback.format_exc().splitlines()[-1][:140]))
        return
    try:
        got = k(*arrays)
    except Exception as e:
        RESULTS.append((name, "RUN FAIL", str(e)[:140].replace("\n", " ")))
        return
    want = reference(*[a.astype(np.float32) for a in arrays])
    want = want if isinstance(want, (list, tuple)) else [want]
    err = max(float(np.max(np.abs(g_.astype(np.float32) - w))) / max(1.0, float(np.max(np.abs(w)))) for g_, w in zip(got, want))
    RESULTS.append((name, "ok" if err < tol else "WRONG", f"rel err {err:.4f}, {k.compile_seconds * 1e3:.0f} ms compile"))


def r(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float16)


x = r(1, 256, 1, 64)
W = r(256, 256, scale=0.05)
case("conv1x1 numerics", lambda g, a: g.conv1x1(a, "w", W),
     lambda a: np.einsum("oc,bcxs->boxs", W.astype(np.float32), a), x)

for kind, fn in [("exp", np.exp), ("sigmoid", lambda v: 1 / (1 + np.exp(-v))), ("softplus", lambda v: np.log1p(np.exp(v))),
                 ("sqrt", np.sqrt), ("rsqrt", lambda v: 1 / np.sqrt(v + 1e-6))]:
    data = np.abs(r(1, 64, 1, 64)) + 0.1 if kind in ("sqrt", "rsqrt") else r(1, 64, 1, 64)
    case(f"unary {kind}", lambda g, a, kind=kind: getattr(g, kind)(a), fn, data)

a, b = r(1, 32, 64, 1), r(1, 32, 1, 64)
case("broadcast sub [H,C,1]-[H,1,C]", lambda g, p, q: g.sub(p, q), lambda p, q: p - q, a, b)
case("broadcast mul scalar", lambda g, p: g.mul(p, 0.125), lambda p: p * 0.125, a)
case("maximum/minimum", lambda g, p, q: [g.maximum(p, q), g.minimum(p, q)], lambda p, q: [np.maximum(p, q), np.minimum(p, q)], a, b)
case("reduce_sum channel keepdims", lambda g, p: g.reduce_sum(p, 1), lambda p: p.sum(1, keepdims=True), x)
case("reduce_sum last axis", lambda g, p: g.reduce_sum(p, 3), lambda p: p.sum(3, keepdims=True), r(1, 32, 64, 128))
case("reshape [1,2048,1,64]->[1,16,128,64]", lambda g, p: g.reshape(p, (1, 16, 128, 64)), lambda p: p.reshape(1, 16, 128, 64), r(1, 2048, 1, 64))
case("transpose [0,1,3,2]", lambda g, p: g.transpose(p, (0, 1, 3, 2)), lambda p: p.transpose(0, 1, 3, 2), r(1, 16, 128, 64))
case("slice channels", lambda g, p: g.slice(p, (0, 2048, 0, 0), (1, 2048, 1, 64)), lambda p: p[:, 2048:4096], r(1, 8192, 1, 64))
case("concat channels", lambda g, p, q: g.concat([p, q], 1), lambda p, q: np.concatenate([p, q], 1), r(1, 128, 1, 64), r(1, 64, 1, 64))
case("concat last axis", lambda g, p, q: g.concat([p, q], 3), lambda p, q: np.concatenate([p, q], 3), r(1, 8192, 1, 3), r(1, 8192, 1, 64))

# matmul family, shapes from the delta rule (32 heads, chunk 64, head dim 128)
A, B, C = r(1, 32, 64, 128, scale=0.3), r(1, 32, 128, 128, scale=0.1), r(1, 32, 128, 64, scale=0.1)
case("matmul input x input", lambda g, p, q: g.matmul(p, q), lambda p, q: p @ q, A, B)
case("matmul transpose_y (q k^T)", lambda g, p, q: g.matmul(p, q, transpose_y=True), lambda p, q: p @ q.transpose(0, 1, 3, 2), A, r(1, 32, 64, 128, scale=0.3))
case("matmul transpose_x (k^T v)", lambda g, p, q: g.matmul(p, q, transpose_x=True), lambda p, q: p.transpose(0, 1, 3, 2) @ q, A, r(1, 32, 64, 128, scale=0.3))
case("chain: (A@B)@C computed x input", lambda g, p, q, s: g.matmul(g.matmul(p, q), s), lambda p, q, s: (p @ q) @ s, A, B, C)
S64 = r(1, 32, 64, 64, scale=0.1)
case("square 64x64 (computed x computed)", lambda g, p: g.matmul(p, p), lambda p: p @ p, S64)
case("square twice (chain)", lambda g, p: g.matmul(g.matmul(p, p), g.matmul(p, p)), lambda p: (p @ p) @ (p @ p), S64)
case("matmul by constant tril", lambda g, p: g.matmul(g.const_tensor("tril", np.tril(np.ones((64, 64)))), p),
     lambda p: np.tril(np.ones((64, 64))) @ p, r(1, 32, 64, 1))

# depthwise causal conv, kernel 4, state of 3 columns in front
Wd = r(8192, 4, scale=0.3)
def dw_ref(p):
    p = p[:, :, 0]
    return sum(Wd[None, :, i:i + 1].astype(np.float32) * p[:, :, i:i + 64] for i in range(4))[:, :, None, :]
case("depthwise conv k=4 (8192 ch)", lambda g, p: g.depthwise_conv(p, "dw", Wd), dw_ref, r(1, 8192, 1, 67))

case("two inputs, two outputs", lambda g, p, q: [g.add(p, q), g.mul(p, q)], lambda p, q: [p + q, p * q], r(1, 64, 1, 64), r(1, 64, 1, 64))
case("odd width I/O [1,8192,1,3]", lambda g, p: g.mul(p, 2.0), lambda p: p * 2, r(1, 8192, 1, 3))
case("head repeat via concat+reshape", lambda g, p: g.reshape(g.concat([g.reshape(p, (1, 16, 1, 64 * 128))] * 2, 2), (1, 32, 64, 128)),
     lambda p: np.repeat(p, 2, axis=1), r(1, 16, 64, 128))
case("fp16 input into sigmoid(x)*x (silu)", lambda g, p: g.silu(p), lambda p: p / (1 + np.exp(-p)), r(1, 8192, 1, 64))

width = max(len(n) for n, *_ in RESULTS)
for name, status, detail in RESULTS:
    print(f"{name:<{width}}  {status:<13} {detail}")
