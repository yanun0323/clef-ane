"""One layer at long context, fed the exact fp32 input: error of its mixer and of its MLP separately.

Usage: long_layer_isolate.py CHARS LAYER [LAYER ...]
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from clef_ane import Layer, DEFAULT_BUCKETS
from ref_backbone import Head
from records import long_record
from safetensors.torch import load_file

chars, wanted = int(sys.argv[1]), [int(a) for a in sys.argv[2:]]
head = Head(); ids = list(head.encode(long_record(chars)).input_ids); n = len(ids)
x = load_file(reference.DATA / "weights" / "embed.safetensors")["model.language_model.embed_tokens.weight"][torch.tensor(ids)].float()
R = 128; padded = -(-n // R) * R
rms = lambda a, b: float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).mean() / (b ** 2).mean()))
for i in range(max(wanted) + 1):
    layer, cfg = reference.load_layer(i, attention="sdpa")
    with torch.no_grad():
        if layer.layer_type == "linear_attention":
            h1 = reference.pieces(layer, x)["h1"]
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
            pos = torch.arange(n)[None]; cos, sin = Qwen3_5TextRotaryEmbedding(cfg)(x[None], pos)
            mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
            h1 = x + layer.self_attn(layer.input_layernorm(x[None]), position_embeddings=(cos, sin), attention_mask=mask)[0][0]
        y = h1 + layer.mlp(layer.post_attention_layernorm(h1))
    if i in wanted:
        ane = Layer(i, R, (padded,) if False else DEFAULT_BUCKETS)
        xin = np.zeros((1, 4096, 1, padded), np.float16); xin[0, :, 0, :n] = x.numpy().T
        h1_ane, y_ane = [], []
        h1_in = np.zeros_like(xin); h1_in[0, :, 0, :n] = h1.numpy().T
        for s in range(0, padded, R):
            h = xin[..., s:s + R]
            if ane.kind == "linear_attention":
                m, ane.conv, ane.rec = ane.mixer(h, ane.conv, ane.rec)
            else:
                stop = s + R; bucket = next(c for c in ane.buckets if c >= stop)
                if bucket != ane.active: ane._write_cache(bucket, 0, s); ane.active = bucket
                cs, sn = layers.rope_tables(s, R)
                m, k, v = ane.mixers[bucket](h, None, None, cs, sn, layers.cache_mask(s, bucket))
                ane.master_k[:, :, s:stop], ane.master_v[:, :, s:stop] = k, v; ane._write_cache(bucket, s, stop)
            h1_ane.append(m[0, :, 0, :]); y_ane.append(ane.mlp(h1_in[..., s:s + R])[0][0, :, 0, :])
        h1_ane = np.concatenate(h1_ane, 1)[:, :n].T; y_ane = np.concatenate(y_ane, 1)[:, :n].T
        print(f"layer {i:>2} ({layer.layer_type:<16}) n={n}: mixer h1 err {rms(h1_ane, h1.double().numpy()):.4f} | mlp y err (given exact h1) {rms(y_ane, y.double().numpy()):.4f} "
              f"| max|h1| {h1.abs().max():.0f}, max|y| {y.abs().max():.0f}", flush=True)
        del ane
    x = y.detach()
