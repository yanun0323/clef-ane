"""Serialize MIL programs and packed weights without requiring source tensors at runtime."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .bundle import BundleError, read_json, safe_path
from .engine.mil import T, WEIGHT_PATH, Graph


def save_program(root: Path, name: str, graph: Graph, weight_name: str) -> None:
    path, weights_path = safe_path(root, name), safe_path(root, weight_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.weights[WEIGHT_PATH]
    if weights_path.exists():
        if weights_path.read_bytes() != payload:
            raise BundleError(f"Programs do not share identical weights: {weight_name}")
    else:
        weights_path.write_bytes(payload)
    descriptor = {"format_version": 1, "text": graph.text(), "weight_file": weight_name,
                  "inputs": [asdict(t) for t in graph.inputs], "outputs": [asdict(t) for t in graph.outputs]}
    path.write_text(json.dumps(descriptor, separators=(",", ":")) + "\n")


class StoredProgram:
    """The read-only subset of Graph required by the native Kernel wrapper."""

    def __init__(self, root: Path, name: str, manifest: dict) -> None:
        if name not in manifest["files"]:
            raise BundleError(f"Program is not listed in the manifest: {name}")
        spec = read_json(safe_path(root, name))
        if spec.get("format_version") != 1 or not isinstance(spec.get("text"), str):
            raise BundleError(f"Unsupported program descriptor: {name}")
        weight_file = spec.get("weight_file")
        if not isinstance(weight_file, str) or weight_file not in manifest["files"] or not weight_file.endswith(".bin"):
            raise BundleError(f"Missing packed weights for {name}")
        self._weight_file = safe_path(root, weight_file)
        self._text = spec["text"]
        self.inputs = self._tensors(spec.get("inputs"), name)
        self.outputs = self._tensors(spec.get("outputs"), name)

    @staticmethod
    def _tensors(values, program: str) -> list[T]:
        if not isinstance(values, list) or not 1 <= len(values) <= 16:
            raise BundleError(f"Invalid input/output count in {program}")
        tensors = []
        for value in values:
            if (not isinstance(value, dict) or not isinstance(value.get("name"), str)
                    or value.get("dtype") not in ("fp16", "fp32") or not isinstance(value.get("shape"), list)
                    or len(value["shape"]) != 4 or any(type(n) is not int or n <= 0 for n in value["shape"])):
                raise BundleError(f"Invalid tensor descriptor in {program}")
            item_size = 2 if value["dtype"] == "fp16" else 4
            shape = value["shape"]
            padded_width = -(-shape[-1] // (64 // item_size)) * (64 // item_size)
            if math.prod(shape[:-1]) * padded_width * item_size > 256 * 1024 * 1024:
                raise BundleError(f"Input/output buffer exceeds 256 MiB in {program}")
            tensors.append(T(value["name"], tuple(shape), value["dtype"]))
        if len({t.name for t in tensors}) != len(tensors):
            raise BundleError(f"Duplicate input/output names in {program}")
        return tensors

    def text(self) -> str:
        return self._text

    @property
    def weights(self) -> dict[str, bytes]:
        return {WEIGHT_PATH: self._weight_file.read_bytes()}

    @property
    def weight_files(self) -> dict[str, Path]:
        """Where the packed weights are on disk, so loading can map and clone the file instead of reading it."""
        return {WEIGHT_PATH: self._weight_file}
