"""Clef decoder layers as ANE programs.

Activations are [1, channels, 1, rows] fp16 (channels-first, so a linear layer is a 1x1 conv). `rows` is the number
of tokens handled per call. Recurrent state (conv window and delta-rule matrix) enters and leaves as I/O tensors.

Widths and head counts are read from the weights, so the same builders serve Clef-Flash (hidden 4096) and Clef
(hidden 5120). The constants below are Clef-Flash's sizes; only the head sizes and CONV_K are shared by both models.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from . import delta
    from .mil import Graph, T
else:
    import delta
    from mil import Graph, T

HIDDEN, KEY_HEADS, VALUE_HEADS, HEAD_DIM, CONV_K = 4096, 16, 32, 128, 4
KEY_DIM, VALUE_DIM = KEY_HEADS * HEAD_DIM, VALUE_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CHUNK = 64
MLP_SCALE = 16.0   # normal boost of the activation entering down_proj
MLP_PEAK = 4096.0  # ... but never let the boosted peak exceed this


def check_config(config: dict) -> None:
    """Refuse a text config that needs anything these builders or the host's rotary tables do not implement."""
    kinds = config["layer_types"]
    expected = {"linear_key_head_dim": HEAD_DIM, "linear_value_head_dim": HEAD_DIM, "linear_conv_kernel_dim": CONV_K,
                "head_dim": ATT_DIM, "hidden_act": "silu", "rms_norm_eps": 1e-6, "attn_output_gate": True}
    rope = config.get("rope_parameters", {})
    if (any(config.get(key) != value for key, value in expected.items())
            or kinds != ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(len(kinds))]
            or config["linear_num_value_heads"] % config["linear_num_key_heads"]
            or config["num_attention_heads"] % config["num_key_value_heads"]
            or rope.get("rope_theta") != ROPE_THETA or rope.get("mrope_section") != [11, 11, 10]
            or rope.get("partial_rotary_factor") * ATT_DIM != ROT_DIM or not rope.get("mrope_interleaved")):
        raise ValueError("This checkpoint's text architecture is not supported by the ANE layer builders.")


def unit_rms(g: Graph, x: T, axis: int, eps: float = 1e-6) -> T:
    """x / sqrt(mean(x^2) + eps) along `axis`, safe in fp16 whatever the magnitude of x.

    Squaring raw activations underflows when they are tiny and overflows when they are huge, so the vector is first
    divided by its largest absolute value: x / sqrt(mean(x^2) + eps) == xs / sqrt(mean(xs^2) + eps / m^2), xs = x / m.
    """
    n = x.shape[axis]
    m = g.maximum(g.reduce_max(g.abs(x), axis), 2.0 ** -4)  # floor keeps 1/m^2 inside fp16
    recip = g.div(g.reciprocal_of_one(m), m)                                     # 1 / m
    xs = g.mul(x, recip)
    mean = g.mul(g.reduce_sum(g.square(xs), axis), 1.0 / n)
    return g.mul(xs, g.rsqrt(g.add(mean, g.mul(g.square(recip), eps)), eps=0.0))


def rms_norm(g: Graph, x: T, weight: np.ndarray, eps: float = 1e-6) -> T:
    """Qwen3.5 RMSNorm over the channel axis of [1, C, 1, R]: x / rms(x) * (1 + weight)."""
    w = g.const_tensor("norm_w", (1.0 + weight.astype(np.float32)).reshape(1, x.shape[1], 1, 1))
    return g.mul(unit_rms(g, x, 1, eps), w)


def heads_first(g: Graph, x: T, heads: int, rows: int) -> T:
    """[1, heads*128, 1, R] -> [1, heads, R, 128]."""
    return g.transpose(g.reshape(x, (1, heads, HEAD_DIM, rows)), (0, 1, 3, 2))


def repeat_heads(g: Graph, x: T, times: int, rows: int) -> T:
    """[1, H, R, D] -> [1, H*times, R, D] with each head repeated consecutively (torch.repeat_interleave)."""
    h = x.shape[1]
    flat = g.reshape(x, (1, h, 1, rows * HEAD_DIM))
    return g.reshape(g.concat([flat] * times, 2), (1, h * times, rows, HEAD_DIM))


def gdn_state_shapes(w: dict[str, np.ndarray]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shapes of a Gated DeltaNet layer's recurrent state: the conv window and the delta-rule matrices."""
    conv_dim = w["linear_attn.in_proj_qkv.weight"].shape[0]
    value_heads = w["linear_attn.in_proj_z.weight"].shape[0] // HEAD_DIM
    return (1, conv_dim, 1, CONV_K - 1), (1, value_heads, HEAD_DIM, HEAD_DIM)


def add_gdn(g: Graph, hid: T, conv_in: T, rec_in: T, w: dict[str, np.ndarray], refine: int = 1,
            chunk: int | None = None) -> tuple[T, T, T, dict[str, T]]:
    """Append input_layernorm -> Gated DeltaNet -> residual add to g; returns (hidden, conv state, delta state, taps)."""
    chunk = chunk or CHUNK
    rows = hid.shape[-1]
    assert rows % chunk == 0
    conv_dim, hidden = w["linear_attn.in_proj_qkv.weight"].shape
    value_dim = w["linear_attn.in_proj_z.weight"].shape[0]
    key_dim = (conv_dim - value_dim) // 2
    key_heads, value_heads = key_dim // HEAD_DIM, value_dim // HEAD_DIM
    assert value_heads % key_heads == 0 and w["linear_attn.conv1d.weight"].shape[-1] == CONV_K

    xn = rms_norm(g, hid, w["input_layernorm.weight"])
    qkv = g.conv1x1(xn, "w_qkv", w["linear_attn.in_proj_qkv.weight"])            # [1, 8192, 1, R]
    z = g.conv1x1(xn, "w_z", w["linear_attn.in_proj_z.weight"])                  # [1, 4096, 1, R]
    b = g.conv1x1(xn, "w_b", w["linear_attn.in_proj_b.weight"])                  # [1, 32, 1, R]
    a = g.conv1x1(xn, "w_a", w["linear_attn.in_proj_a.weight"])

    # causal depthwise conv (kernel 4) over [previous 3 columns | this call's columns], then SiLU
    window = g.concat([conv_in, qkv], 3)                                         # [1, 8192, 1, R+3]
    conv_out = g.slice(window, (0, 0, 0, rows), (1, conv_dim, 1, CONV_K - 1))
    mixed = g.silu(g.depthwise_conv(window, "w_conv", w["linear_attn.conv1d.weight"][:, 0, :]))

    q = heads_first(g, g.slice(mixed, (0, 0, 0, 0), (1, key_dim, 1, rows)), key_heads, rows)
    k = heads_first(g, g.slice(mixed, (0, key_dim, 0, 0), (1, key_dim, 1, rows)), key_heads, rows)
    v = heads_first(g, g.slice(mixed, (0, 2 * key_dim, 0, 0), (1, value_dim, 1, rows)), value_heads, rows)
    q = repeat_heads(g, delta.l2_normalize(g, q), value_heads // key_heads, rows)
    k = repeat_heads(g, delta.l2_normalize(g, k), value_heads // key_heads, rows)

    # per-token, per-head gates: beta = sigmoid(b); log decay = -exp(A_log) * softplus(a + dt_bias)
    beta = g.transpose(g.sigmoid(b), (0, 1, 3, 2))                               # [1, 32, R, 1]
    dt_bias = g.const_tensor("dt_bias", w["linear_attn.dt_bias"].astype(np.float32).reshape(1, value_heads, 1, 1))
    neg_a = g.const_tensor("neg_a", (-np.exp(w["linear_attn.A_log"].astype(np.float32))).reshape(1, value_heads, 1, 1))
    log_decay = g.transpose(g.mul(g.softplus(g.add(a, dt_bias)), neg_a), (0, 1, 3, 2))

    outs, state = [], rec_in
    for c in range(rows // chunk):
        sl = lambda t, last: g.slice(t, (0, 0, c * chunk, 0), (1, t.shape[1], chunk, last))
        out, state = delta.delta_chunk(g, sl(q, HEAD_DIM), sl(k, HEAD_DIM), sl(v, HEAD_DIM), sl(beta, 1), sl(log_decay, 1),
                                       state, chunk, refine=refine, normalize=False)
        outs.append(out)
    core = outs[0] if len(outs) == 1 else g.concat(outs, 2)                      # [1, 32, R, 128]

    # gated RMSNorm per head: rms_norm(core) * weight * silu(z)
    norm_w = g.const_tensor("gate_norm_w", w["linear_attn.norm.weight"].astype(np.float32).reshape(1, 1, 1, HEAD_DIM))
    normed = g.mul(g.mul(unit_rms(g, core, 3), norm_w), g.silu(heads_first(g, z, value_heads, rows)))
    flat = g.reshape(g.transpose(normed, (0, 1, 3, 2)), (1, value_dim, 1, rows))

    hid_out = g.add(hid, g.conv1x1(flat, "w_out", w["linear_attn.out_proj.weight"]))
    taps = {"tap_xn": xn, "tap_qkv": qkv, "tap_conv": mixed, "tap_core": core, "tap_normed": flat, "tap_beta": beta,
            "tap_g": log_decay, "tap_q": q, "tap_k": k, "tap_v": v}
    return hid_out, conv_out, state, taps


def gdn_mixer(w: dict[str, np.ndarray], rows: int = CHUNK, refine: int = 1, taps: bool = False, chunk: int | None = None) -> Graph:
    """input_layernorm -> Gated DeltaNet -> residual add.

    w holds the layer's tensors as float arrays under their checkpoint names (without the layer prefix).
    """
    conv_shape, rec_shape = gdn_state_shapes(w)
    g = Graph()
    hid = g.input("hid_in", (1, w["linear_attn.in_proj_qkv.weight"].shape[1], 1, rows))
    conv_in = g.input("conv_in", conv_shape)
    rec_in = g.input("rec_in", rec_shape)
    hid_out, conv_out, state, tapped = add_gdn(g, hid, conv_in, rec_in, w, refine, chunk)
    outputs = [(hid_out, "hid_out"), (conv_out, "conv_out"), (state, "rec_out")]
    if taps:  # intermediates for debugging against the PyTorch reference
        outputs += [(t, n) for n, t in tapped.items()]
    g.output(*(g.rename(t, n) for t, n in outputs))
    return g


def add_mlp(g: Graph, hid: T, w: dict[str, np.ndarray]) -> T:
    """Append post_attention_layernorm -> SwiGLU MLP -> residual add to g."""
    inter, hidden = w["mlp.gate_proj.weight"].shape
    rows = hid.shape[-1]
    xn = rms_norm(g, hid, w["post_attention_layernorm.weight"])
    gate_up = g.conv1x1(xn, "w_gate_up", np.concatenate([w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]]))
    gate = g.slice(gate_up, (0, 0, 0, 0), (1, inter, 1, rows))
    up = g.slice(gate_up, (0, inter, 0, 0), (1, inter, 1, rows))
    # The activation entering down_proj is small (rms ~0.02) and the ANE matmul loses precision there, so scale it up
    # and scale the result back (error 0.7% -> 0.07%). The factor is per token: 16 normally, less when a token has a
    # huge activation (up to ~1200 in the last layer), so that the 12288-term sum stays inside fp16.
    act = g.mul(g.silu(gate), up)
    peak = g.reduce_max(g.abs(act), 1)                                           # [1, 1, 1, R]
    factor = g.div(T(g.scalar(MLP_PEAK), (), "fp16"), g.maximum(peak, MLP_PEAK / MLP_SCALE))   # = min(16, 4096 / peak)
    scaled = g.mul(act, factor)
    # One 12288-input conv runs at about 2/3 the speed of three 4096-input ones summed, so the input is cut into
    # equal slices no wider than the hidden size (Clef: 17408 = 4 x 4352).
    wd = w["mlp.down_proj.weight"]
    count = -(-inter // hidden)
    assert inter % count == 0
    width = inter // count
    parts = [g.conv1x1(g.slice(scaled, (0, i * width, 0, 0), (1, width, 1, rows)), f"w_down{i}", wd[:, i * width:(i + 1) * width])
             for i in range(count)]
    down = parts[0]
    for part in parts[1:]:
        down = g.add(down, part)
    down = g.mul(down, g.div(g.reciprocal_of_one(factor), factor))
    return g.add(hid, down)


def mlp(w: dict[str, np.ndarray], rows: int = CHUNK) -> Graph:
    """post_attention_layernorm -> SwiGLU MLP -> residual add."""
    g = Graph()
    hid = g.input("hid_in", (1, w["mlp.gate_proj.weight"].shape[1], 1, rows))
    g.output(g.rename(add_mlp(g, hid, w), "hid_out"))
    return g


def linear_block(ws: list[dict[str, np.ndarray]], rows: int = CHUNK, lead: dict[str, np.ndarray] | None = None) -> Graph:
    """Consecutive Gated DeltaNet layers (mixer + MLP each) as one program, after the previous layer's MLP if `lead`.

    The ANE holds at most 127 loaded programs on the whole Mac. One program per mixer and per MLP fits Clef-Flash's
    32 layers but not Clef's 64, so Clef's three linear-attention layers between two full-attention layers are one
    program, together with the MLP of the full-attention layer before them. State tensors are numbered per layer.
    """
    g = Graph()
    hid = g.input("hid_in", (1, ws[0]["mlp.gate_proj.weight"].shape[1], 1, rows))
    states = []
    for index, w in enumerate(ws):
        conv_shape, rec_shape = gdn_state_shapes(w)
        states.append((g.input(f"conv_in{index}", conv_shape), g.input(f"rec_in{index}", rec_shape)))
    if lead is not None:
        hid = add_mlp(g, hid, lead)
    outputs = []
    for index, (w, (conv_in, rec_in)) in enumerate(zip(ws, states)):
        hid, conv_out, rec_out, _ = add_gdn(g, hid, conv_in, rec_in, w)
        hid = add_mlp(g, hid, w)
        outputs += [(conv_out, f"conv_out{index}"), (rec_out, f"rec_out{index}")]
    g.output(*(g.rename(t, n) for t, n in [(hid, "hid_out"), *outputs]))
    return g


# ---- full attention layer (every 4th layer) ---------------------------------------------------------------------
HEADS, KV_HEADS, ATT_DIM, ROT_DIM = 16, 4, 256, 64
ROPE_THETA = 1.0e7
MASK_NEG = -30000.0


def rope_tables(start: int, rows: int, position_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Host cos/sin tables [1, 1, rows, 64]; optional Qwen3.5 temporal/height/width positions.

    The text path deliberately keeps its original arithmetic. Multimodal frequencies are
    interleaved T/H/W with sections [11, 11, 10], not concatenated in three blocks.
    """
    if position_ids is None:
        inv = 1.0 / ROPE_THETA ** (np.arange(0, ROT_DIM, 2) / ROT_DIM)
        ang = np.arange(start, start + rows)[:, None] * inv[None]
    else:
        if position_ids.shape != (3, rows) or not np.issubdtype(position_ids.dtype, np.integer) or np.any(position_ids < 0):
            raise ValueError("position_ids must contain three nonnegative integer positions per token")
        inv = 1.0 / np.float32(ROPE_THETA) ** (np.arange(0, ROT_DIM, 2, dtype=np.float32) / np.float32(ROT_DIM))
        freqs = position_ids.astype(np.float32)[..., None] * inv
        ang = freqs[0].copy()
        ang[:, 1:33:3] = freqs[1, :, 1:33:3]
        ang[:, 2:30:3] = freqs[2, :, 2:30:3]
    full = np.concatenate([ang, ang], axis=-1)
    return np.cos(full)[None, None].astype(np.float16), np.sin(full)[None, None].astype(np.float16)


def cache_mask(past: int, ctx: int) -> np.ndarray:
    """Additive mask over the KV cache, [1, 1, 1, ctx]: 0 for the `past` rows that hold real keys, very negative after."""
    row = np.full(ctx, MASK_NEG, np.float16)
    row[:past] = 0
    return row.reshape(1, 1, 1, ctx)


def own_mask(rows: int, group: int = HEADS // KV_HEADS) -> np.ndarray:
    """Causal mask among this call's own keys, [1, 1, group*rows, rows]; the query heads of a kv group are stacked on rows."""
    causal = np.where(np.tril(np.ones((rows, rows), bool)), 0.0, MASK_NEG)
    return np.tile(causal, (group, 1))[None, None].astype(np.float16)


def _rope(g: Graph, x: T, cos: T, sin: T) -> T:
    """Rotate the first 64 of 256 dims (rotate_half convention) of x [1, H, R, 256]."""
    h, r = x.shape[1], x.shape[2]
    part = lambda lo, size: g.slice(x, (0, 0, 0, lo), (1, h, r, size))
    x1, x2 = part(0, ROT_DIM // 2), part(ROT_DIM // 2, ROT_DIM // 2)
    rotated = g.concat([g.mul(x2, -1.0), x1], 3)
    head = g.add(g.mul(g.concat([x1, x2], 3), cos), g.mul(rotated, sin))
    return g.concat([head, part(ROT_DIM, ATT_DIM - ROT_DIM)], 3)


def full_attention(w: dict[str, np.ndarray], rows: int = 128, ctx: int = 512) -> Graph:
    """input_layernorm -> grouped-query attention with output gate -> residual add.

    The KV cache (k_cache, v_cache: [1, 4, ctx, 256]) is an input buffer that the host updates in place; this call's own
    keys and values are attended to separately and returned as k_out / v_out for the host to append to the cache.
    """
    hidden = w["self_attn.q_proj.weight"].shape[1]
    heads = w["self_attn.q_proj.weight"].shape[0] // (2 * ATT_DIM)
    kv_heads = w["self_attn.k_proj.weight"].shape[0] // ATT_DIM
    assert heads % kv_heads == 0
    g = Graph()
    hid = g.input("hid_in", (1, hidden, 1, rows))
    k_cache = g.input("k_cache", (1, kv_heads, ctx, ATT_DIM))
    v_cache = g.input("v_cache", (1, kv_heads, ctx, ATT_DIM))
    cos = g.input("cos_in", (1, 1, rows, ROT_DIM))
    sin = g.input("sin_in", (1, 1, rows, ROT_DIM))
    mask_row = g.input("mask_in", (1, 1, 1, ctx))

    xn = rms_norm(g, hid, w["input_layernorm.weight"])
    # q_proj interleaves [query(256), gate(256)] per head; regroup its rows as all queries, then all gates
    qg = w["self_attn.q_proj.weight"].reshape(heads, 2, ATT_DIM, hidden)
    q_rows = np.concatenate([qg[:, 0].reshape(-1, hidden), qg[:, 1].reshape(-1, hidden)])
    qgate = g.conv1x1(xn, "w_q", q_rows)                                         # [1, 8192, 1, R]
    q = heads_first_n(g, g.slice(qgate, (0, 0, 0, 0), (1, heads * ATT_DIM, 1, rows)), heads, rows)
    gate = heads_first_n(g, g.slice(qgate, (0, heads * ATT_DIM, 0, 0), (1, heads * ATT_DIM, 1, rows)), heads, rows)
    k = heads_first_n(g, g.conv1x1(xn, "w_k", w["self_attn.k_proj.weight"]), kv_heads, rows)
    v = heads_first_n(g, g.conv1x1(xn, "w_v", w["self_attn.v_proj.weight"]), kv_heads, rows)

    qw = g.const_tensor("q_norm_w", (1.0 + w["self_attn.q_norm.weight"].astype(np.float32)).reshape(1, 1, 1, ATT_DIM))
    kw = g.const_tensor("k_norm_w", (1.0 + w["self_attn.k_norm.weight"].astype(np.float32)).reshape(1, 1, 1, ATT_DIM))
    q = _rope(g, g.mul(unit_rms(g, q, 3), qw), cos, sin)
    k = _rope(g, g.mul(unit_rms(g, k, 3), kw), cos, sin)

    q_grouped = g.reshape(q, (1, kv_heads, heads // kv_heads * rows, ATT_DIM))   # the query heads of a kv head, stacked
    scale = ATT_DIM ** -0.5
    s_cache = g.add(g.mul(g.matmul(q_grouped, k_cache, transpose_y=True), scale), mask_row)   # [1, 4, 4R, ctx]
    s_own = g.add(g.mul(g.matmul(q_grouped, k, transpose_y=True), scale), g.const_tensor("own_mask", own_mask(rows, heads // kv_heads)))
    top = g.maximum(g.reduce_max(s_cache, 3), g.reduce_max(s_own, 3))
    e_cache, e_own = g.exp(g.sub(s_cache, top)), g.exp(g.sub(s_own, top))
    inv_total = g.div(g.reciprocal_of_one(top), g.add(g.reduce_sum(e_cache, 3), g.reduce_sum(e_own, 3)))
    boost = 16.0 * 1.0                                                           # probabilities are small; scale for the matmul
    scale_p = g.mul(inv_total, boost)
    ctx_out = g.add(g.matmul(g.mul(e_cache, scale_p), v_cache), g.matmul(g.mul(e_own, scale_p), v))
    ctx_out = g.reshape(g.mul(ctx_out, 1.0 / boost), (1, heads, rows, ATT_DIM))

    gated = g.mul(ctx_out, g.sigmoid(gate))
    flat = g.reshape(g.transpose(gated, (0, 1, 3, 2)), (1, heads * ATT_DIM, 1, rows))
    hid_out = g.add(hid, g.conv1x1(flat, "w_o", w["self_attn.o_proj.weight"]))
    g.output(*(g.rename(t, n) for t, n in ((hid_out, "hid_out"), (k, "k_out"), (v, "v_out"))))
    return g


def heads_first_n(g: Graph, x: T, heads: int, rows: int) -> T:
    """[1, heads*256, 1, R] -> [1, heads, R, 256]."""
    return g.transpose(g.reshape(x, (1, heads, ATT_DIM, rows)), (0, 1, 3, 2))
