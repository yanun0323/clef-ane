"""Load exported LUT-6 programs, using the same request execution as the research model."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from .bundle import BUCKETS, BundleError, read_json
from .engine import layers
from .engine.ane import Kernel
from .engine.runtime import BackboneRunner, LayerState
from .program import StoredProgram


class BundleLayer(LayerState):
    def __init__(self, root: Path, manifest: dict, index: int, buckets: tuple[int, ...]) -> None:
        self.index, self.rows, self.buckets = index, manifest["rows"], buckets
        spec = manifest["layers"][index]
        self.kind = spec["kind"]
        self.mixers = {}
        try:
            self.mlp = Kernel(StoredProgram(root, spec["mlp"], manifest))
            if self.kind == "linear_attention":
                self.mixer = Kernel(StoredProgram(root, spec["mixer"], manifest))
            else:
                for size in buckets:
                    self.mixers[size] = Kernel(StoredProgram(root, spec["mixers"][str(size)], manifest))
                self.master_k = np.zeros((1, layers.KV_HEADS, buckets[-1], layers.ATT_DIM), np.float16)
                self.master_v = np.zeros_like(self.master_k)
            self.reset()
        except BaseException:
            self.close()
            raise


class BundleBackbone(BackboneRunner):
    def __init__(self, root: Path, manifest: dict, max_tokens: int = 16384, *, verbose: bool = True) -> None:
        if max_tokens not in BUCKETS:
            raise BundleError(f"Unsupported input length: {max_tokens}")
        self.rows = manifest["rows"]
        self.buckets = tuple(n for n in manifest["buckets"] if n <= max_tokens)
        self.max_tokens = max_tokens
        self.config = read_json(root / "config.json")["text_config"]
        if self.config["hidden_size"] != layers.HIDDEN or self.config["layer_types"] != [v["kind"] for v in manifest["layers"]]:
            raise BundleError("Model configuration does not match the exported ANE programs.")
        self.layers = []
        started = time.monotonic()
        try:
            for index in range(len(manifest["layers"])):
                self.layers.append(BundleLayer(root, manifest, index, self.buckets))
                if verbose and index % 8 == 7:
                    print(f"Loaded {index + 1}/32 layers on the ANE ({time.monotonic() - started:.0f}s)", flush=True)
            self.embed = load_file(root / "weights" / "embed.safetensors")["model.language_model.embed_tokens.weight"]
            self.norm_w = load_file(root / "weights" / "norm.safetensors")["model.language_model.norm.weight"].float().numpy()
            self.trace = []
        except BaseException:
            self.close()
            raise
