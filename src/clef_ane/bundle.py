"""The portable, data-only Clef ANE bundle format (no model-repository Python)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

MANIFEST = "clef-ane.json"
FORMAT_VERSION = 1
IMAGE_FORMAT_VERSION = 2
VISION_FILES = ("processor_config.json", "weights/vision.safetensors")
BUCKETS = (1024, 2048, 4096, 8192, 16384)
MODEL_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "weights/embed.safetensors", "weights/norm.safetensors", "weights/lm_head.safetensors",
    "weights/joint_head.safetensors", "weights/joint_head_config.json",
    "LICENSE", "NOTICE",
)


class BundleError(ValueError):
    """An incomplete, incompatible, or damaged model bundle."""


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, glob characters, and symlinks outside the model directory."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        raise BundleError(f"Invalid model file path: {name!r}")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(p in (".", "..") for p in name.split("/")) or str(parts) != name:
        raise BundleError(f"Invalid model file path: {name!r}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise BundleError(f"Model file points outside the model directory: {name}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise BundleError(f"Cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path.name} must contain a JSON object")
    return value


def validate_manifest(data: dict) -> dict:
    version = data.get("format_version")
    if data.get("format") != "clef-ane" or type(version) is not int or version not in (FORMAT_VERSION, IMAGE_FORMAT_VERSION):
        raise BundleError("Unsupported model format. Download a clef-ane LUT-6 bundle, not a Transformers or MLX checkpoint.")
    model_files = MODEL_FILES + (VISION_FILES if version == IMAGE_FORMAT_VERSION else ())
    if data.get("quantization") != {"bits": 6, "group_size": 16, "method": "kmeans-lut"}:
        raise BundleError("This runtime requires a 6-bit, group-16 lookup-table bundle.")
    if data.get("rows") != 128 or data.get("buckets") != list(BUCKETS):
        raise BundleError("Unsupported ANE program sizes in the model bundle.")
    files = data.get("files")
    if not isinstance(files, dict) or not files or len(files) > 1024:
        raise BundleError("The model manifest has no valid file inventory.")
    for name, info in files.items():
        safe_path(Path("/clef-ane-model"), name)
        if name not in model_files and not re.fullmatch(r"programs/\d{2}/[a-z0-9_-]+\.(json|bin)", name):
            raise BundleError(f"Unexpected file in the model manifest: {name}")
        if (not isinstance(info, dict) or type(info.get("size")) is not int or info["size"] <= 0
                or not isinstance(info.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])):
            raise BundleError(f"Invalid size or SHA-256 for {name}")
    for name in model_files:
        if name not in files:
            raise BundleError(f"Model manifest is missing {name}")
    layers = data.get("layers")
    if not isinstance(layers, list) or len(layers) != 32:
        raise BundleError("The model bundle must contain all 32 text layers.")
    for index, layer in enumerate(layers):
        kind = "full_attention" if index % 4 == 3 else "linear_attention"
        if not isinstance(layer, dict) or layer.get("kind") != kind:
            raise BundleError(f"Invalid layer type for layer {index}")
        programs = [layer.get("mlp")]
        if kind == "linear_attention":
            programs.append(layer.get("mixer"))
        else:
            mixers = layer.get("mixers")
            if not isinstance(mixers, dict) or set(mixers) != {str(n) for n in BUCKETS}:
                raise BundleError(f"Missing context programs for layer {index}")
            programs.extend(mixers.values())
        for name in programs:
            if not isinstance(name, str) or not name.startswith("programs/") or not name.endswith(".json") or name not in files:
                raise BundleError(f"Missing program descriptor for layer {index}")
    return data


def load_bundle(root: Path, *, verify: bool = True) -> dict:
    if not (root / MANIFEST).is_file():
        raise BundleError(f"No model bundle in {root}. Run `clef-ane download` first, or select one with --model-dir.")
    data = validate_manifest(read_json(safe_path(root, MANIFEST)))
    for name, info in data["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != info["size"]:
            raise BundleError(f"Missing or incomplete model file: {name}. Run `clef-ane download` again.")
        if verify and sha256(path) != info["sha256"]:
            raise BundleError(f"Model file failed its SHA-256 check: {name}. Download it again with --force.")
    return data
