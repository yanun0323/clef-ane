"""Clef-Flash text backbone on the ANE (private API), plus the joint schema head on the CPU.

hidden(token_ids) runs embedding lookup (host) -> 32 decoder layers (ANE, 24 Gated DeltaNet + 8 full attention)
-> final RMSNorm (host) and returns the [T, 4096] states the head reads.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

if __package__:
    from . import layers
    from .ane import Kernel
    from .runtime import BackboneRunner, LayerState
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import layers
    from ane import Kernel
    from runtime import BackboneRunner, LayerState

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "data" / "weights"
CONFIG = json.loads((ROOT / "data" / "config.json").read_text())["text_config"]
LAYER_TYPES = CONFIG["layer_types"]


def layer_weights(index: int) -> dict[str, np.ndarray]:
    prefix = f"model.language_model.layers.{index}."
    return {k[len(prefix):]: v.float().numpy() for k, v in load_file(WEIGHTS / f"layer_{index:02d}.safetensors").items()}


DEFAULT_BUCKETS = (1024, 2048, 4096, 8192, 16384)


class Layer(LayerState):
    """One decoder layer: a token mixer program plus an MLP program.

    Gated DeltaNet layers keep their recurrent state in program I/O. Full-attention layers keep a KV cache that lives in
    the program's input buffer; one program per context bucket is compiled and each chunk uses the smallest bucket that
    holds the tokens seen so far, so short requests never pay for the long-context attention.
    """

    def __init__(self, index: int, rows: int, buckets: tuple[int, ...]) -> None:
        self.index, self.rows, self.buckets = index, rows, buckets
        self.kind = LAYER_TYPES[index]
        w = layer_weights(index)
        self.mlp = Kernel(layers.mlp(w, rows=rows))
        if self.kind == "linear_attention":
            self.mixer = Kernel(layers.gdn_mixer(w, rows=rows))
        else:
            self.mixers = {c: Kernel(layers.full_attention(w, rows=rows, ctx=c)) for c in buckets}
            self.allocate_cache()
        self.reset()


class Backbone(BackboneRunner):
    def __init__(self, rows: int = 128, buckets: tuple[int, ...] = DEFAULT_BUCKETS, num_layers: int = 32, verbose: bool = True) -> None:
        self.config = CONFIG
        self.rows, self.buckets = rows, tuple(sorted(buckets))
        self.max_tokens = self.buckets[-1]
        t0 = time.time()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(Layer(i, rows, self.buckets))
            if verbose and i % 8 == 7:
                print(f"  loaded {i + 1} layers on the ANE ({time.time() - t0:.0f}s)", flush=True)
        self.embed = load_file(WEIGHTS / "embed.safetensors")["model.language_model.embed_tokens.weight"]
        self.norm_w = load_file(WEIGHTS / "norm.safetensors")["model.language_model.norm.weight"].float().numpy()
        self.trace: list[float] = []

