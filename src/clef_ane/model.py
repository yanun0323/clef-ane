"""Load exported LUT-6 programs, using the same request execution as the research model."""
from __future__ import annotations

import time
from pathlib import Path

from safetensors.torch import load_file

from .bundle import BUCKETS, BundleError, read_json
from .engine.ane import Kernel
from .engine.runtime import BackboneRunner, BlockState, LayerState
from .program import StoredProgram


class BundleLayer(LayerState):
    def __init__(self, root: Path, manifest: dict, index: int, buckets: tuple[int, ...]) -> None:
        self.index, self.rows, self.buckets = index, manifest["rows"], buckets
        spec = manifest["layers"][index]
        self.kind = spec["kind"]
        self.mixers = {}
        self.mlp = None
        try:
            if "mlp" in spec:
                self.mlp = Kernel(StoredProgram(root, spec["mlp"], manifest))
            if self.kind == "linear_attention":
                self.mixer = Kernel(StoredProgram(root, spec["mixer"], manifest))
            else:
                for size in buckets:
                    self.mixers[size] = Kernel(StoredProgram(root, spec["mixers"][str(size)], manifest))
                self.allocate_cache()
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
        if self.config["layer_types"] != [v["kind"] for v in manifest["layers"]]:
            raise BundleError("Model configuration does not match the exported ANE programs.")
        self.layers = []    # one entry per program group: a layer, or a block of consecutive layers
        started = time.monotonic()
        specs = manifest["layers"]
        index, count = 0, len(specs)
        try:
            while index < count:
                block = specs[index].get("block")
                if block:
                    size = next((n for n in range(1, count - index) if specs[index + n].get("block") != block), count - index)
                    layer = BlockState(Kernel(StoredProgram(root, block, manifest)))
                    self.layers.append(layer)
                    first = layer.kernel.inputs[0]
                    if len(layer.state) != 2 * size:
                        raise BundleError("Model manifest does not match the exported ANE programs.")
                else:
                    size = 1
                    layer = BundleLayer(root, manifest, index, self.buckets)
                    self.layers.append(layer)
                    first = (layer.mixer if layer.kind == "linear_attention" else layer.mixers[self.buckets[0]]).inputs[0]
                if first.shape != (1, self.config["hidden_size"], 1, self.rows):
                    raise BundleError("Model configuration does not match the exported ANE programs.")
                index += size
                if verbose and index // 8 > (index - size) // 8:
                    print(f"Loaded {index}/{count} layers on the ANE ({time.monotonic() - started:.0f}s)", flush=True)
            self.embed = load_file(root / "weights" / "embed.safetensors")["model.language_model.embed_tokens.weight"]
            self.norm_w = load_file(root / "weights" / "norm.safetensors")["model.language_model.norm.weight"].float().numpy()
            self.trace = []
        except BaseException:
            self.close()
            raise
