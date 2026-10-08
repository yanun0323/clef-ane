"""Gated DeltaNet chunk (the "delta rule") as an ANE program.

Mirrors transformers' torch_chunk_gated_delta_rule for one chunk of C tokens, with the recurrent state passed in
and out. Per head: q, k, v are [C, D]; beta and g are [C]; the state is [D_k, D_v].

The only awkward step is T = (I + M)^-1 for the strictly lower-triangular M (C x C). Since N = -M is nilpotent,
(I - N)^-1 = (I + N)(I + N^2)(I + N^4)...(I + N^(C/2)), which is a short chain of batched matmuls.
"""
from __future__ import annotations

import math

import numpy as np

if __package__:
    from .mil import Graph, T
else:
    from mil import Graph, T


def l2_normalize(g: Graph, x: T, scale: float = 1.0) -> T:
    inv = g.rsqrt(g.reduce_sum(g.square(x), axis=3), eps=1e-6)
    return g.mul(g.mul(x, inv), scale) if scale != 1.0 else g.mul(x, inv)


def inverse_unit_lower(g: Graph, m: T, chunk: int, refine: int = 1) -> T:
    """(I + m)^-1 for strictly lower-triangular m, by merging diagonal blocks of size 1, 2, 4, ... chunk/2.

    With T the inverse of the block-diagonal part at one level, merging pairs of blocks is
    T <- T - T @ (m restricted to the pairs' sub-diagonal blocks) @ T. Every intermediate is an exact block inverse,
    so nothing grows: a Neumann product (I+N)(I+N^2)... overflows fp16 when keys are similar and decay is slow.
    """
    ident = g.const_tensor("eye", np.eye(chunk)[None, None])
    result, size, level = ident, 1, 0
    while size < chunk:
        rows, cols = np.arange(chunk)[:, None], np.arange(chunk)[None, :]
        pair = rows // (2 * size) == cols // (2 * size)
        mask = (pair & ((rows // size) % 2 == 1) & ((cols // size) % 2 == 0)).astype(np.float64)[None, None]
        sub = g.mul(m, g.const_tensor(f"merge{level}", mask))
        result = g.sub(result, g.matmul(g.matmul(result, sub), result))
        size, level = size * 2, level + 1
    for _ in range(refine):
        # Newton step: T <- T + T (I - (I + m) T), removes the remaining fp16 rounding error of the products.
        residual = g.sub(ident, g.matmul(g.add(ident, m), result))
        result = g.add(result, g.matmul(result, residual))
    return result


QK = 16.0   # q and k are carried as 16 * unit vectors
VU = 8.0    # v-like quantities are carried in units of (per-head max)/8
FLOOR = 2.0 ** -6


def amax(g: Graph, x: T) -> T:
    """Largest |x| over the last two axes, per head: [1, H, 1, 1]."""
    return g.reduce_max(g.reduce_max(g.abs(x), 3), 2)


def delta_chunk(g: Graph, q: T, k: T, v: T, beta: T, log_decay: T, state: T, chunk: int, debug: bool = False, refine: int = 1,
                normalize: bool = True, taps: dict | None = None):
    """q, k, v: [1, H, C, D]; beta, log_decay: [1, H, C, 1]; state: [1, H, D, D]. Returns (out, new_state).

    q and k must be unit vectors (normalize=False) or are normalised here. Every matmul operand is rescaled by an
    exact power of two (or a per-head max) so its entries are O(1): the ANE loses precision on products near fp16's
    small end (a 1/sqrt(128) scaling of q alone cost 1-3%). The results are rescaled back to true units.
    """
    d = q.shape[-1]
    tril = g.const_tensor("tril", np.tril(np.ones((chunk, chunk)))[None, None])
    strict = g.const_tensor("tril_strict", np.tril(np.ones((chunk, chunk)), -1)[None, None])

    if normalize:
        q, k = l2_normalize(g, q), l2_normalize(g, k)
    qs, ks = g.mul(q, QK), g.mul(k, QK)
    kbs = g.mul(ks, beta)
    v_beta = g.mul(v, beta)
    c = g.maximum(amax(g, v_beta), FLOOR)                       # per-head scale of this chunk's values
    c_inv = g.div(g.reciprocal_of_one(c), c)
    vh = g.mul(v_beta, g.mul(c_inv, VU))                        # values in units of c/VU

    cum = g.matmul(tril, log_decay)                             # inclusive cumsum over the chunk: [1,H,C,1]
    cum_row = g.transpose(cum, (0, 1, 3, 2))                    # [1,H,1,C]
    decay = g.mul(g.exp(g.mul(g.sub(cum, cum_row), tril)), tril)   # exp(cum_i - cum_j) for i >= j, else 0

    m = g.mul(g.mul(g.mul(g.matmul(kbs, ks, transpose_y=True), 1.0 / (QK * QK)), decay), strict)
    inv = inverse_unit_lower(g, m, chunk, refine)

    exp_cum = g.exp(cum)
    value = g.matmul(inv, vh)                                   # in c/VU units
    k_cumdecay = g.matmul(inv, g.mul(kbs, exp_cum))             # QK * k_cumdecay

    s_max = g.maximum(amax(g, state), FLOOR)
    s_n = g.mul(state, g.div(g.reciprocal_of_one(s_max), s_max))   # state / max|state|: entries <= 1
    v_prime = g.mul(g.matmul(k_cumdecay, s_n), g.mul(g.mul(s_max, c_inv), VU / QK))
    v_new = g.sub(value, v_prime)                               # c/VU units

    inter = g.mul(g.matmul(g.mul(qs, exp_cum), s_n), g.mul(s_max, 1.0 / (QK * math.sqrt(d))))
    attn = g.mul(g.mul(g.matmul(qs, ks, transpose_y=True), decay), 1.0 / 16.0)   # 16 * (q.k * decay)
    intra = g.mul(g.matmul(attn, v_new), g.mul(c, 1.0 / (VU * 16.0 * math.sqrt(d))))
    out = g.add(inter, intra)                                   # true units, including the 1/sqrt(d) on q
    if taps is not None:
        taps.update(decay=decay, inv=inv, value=value, k_cumdecay=k_cumdecay, v_new=v_new, inter=inter, intra=intra, cum=cum)

    last = g.slice(cum, (0, 0, chunk - 1, 0), (1, cum.shape[1], 1, 1))   # [1,H,1,1]
    k_decayed = g.mul(ks, g.exp(g.sub(last, cum)))
    update = g.mul(g.matmul(k_decayed, v_new, transpose_x=True), g.mul(c, 1.0 / (QK * VU)))
    new_state = g.add(g.mul(state, g.exp(last)), update)
    return (out, new_state, inv, decay) if debug else (out, new_state)


def build(chunk: int = 64, heads: int = 32, dim: int = 128, debug: bool = False, refine: int = 1) -> Graph:
    g = Graph()
    q = g.input("q", (1, heads, chunk, dim))
    k = g.input("k", (1, heads, chunk, dim))
    v = g.input("v", (1, heads, chunk, dim))
    beta = g.input("beta", (1, heads, chunk, 1))
    log_decay = g.input("g", (1, heads, chunk, 1))
    state = g.input("rec_state", (1, heads, dim, dim))
    taps: dict = {}
    result = delta_chunk(g, q, k, v, beta, log_decay, state, chunk, False, refine, taps=taps if debug else None)
    g.output(*result)
    if debug:
        g.output(*(g.rename(t, f"tap_{n}") for n, t in taps.items()))
    return g
