"""Fetch only a pinned Clef model's vision weights and processor settings (about 0.85 GiB).

Maintainer command; serving never downloads. HTTP ranges avoid fetching the text weights
that share the upstream shard. No code from the model repository is executed.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from clef_ane.bundle import sha256
from clef_ane.models import DEFAULT_MODEL, MODELS, ModelInfo, data_dir

PREFIX = "model.visual."
BLOCK_BYTES = 64 * 1024 * 1024
# The encoders are the same size; only the merger's output matches each text model's width.
PARAMETERS = {"clef-flash": 456_010_480, "clef": 460_730_096}


def get_json(base: str, name: str) -> dict:
    with urllib.request.urlopen(f"{base}/{name}", timeout=60) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError(f"Unexpectedly large metadata: {name}")
    return json.loads(raw)


def get_range(url: str, start: int, stop: int) -> bytes:
    """Read [start, stop), rejecting servers that ignore Range or return a different slice."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{stop - 1}/"):
            raise ValueError("The model host did not return the requested byte range.")
        raw = response.read(stop - start + 1)
    if len(raw) != stop - start:
        raise ValueError("Incomplete model byte range; run fetch-vision again.")
    return raw


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fetch(output: Path, model: ModelInfo = MODELS[DEFAULT_MODEL]) -> None:
    output = output.expanduser().resolve()
    base = f"https://huggingface.co/{model.source_repo}/resolve/{model.source_revision}"
    origin = {"repo_id": model.source_repo, "revision": model.source_revision}
    source_config = get_json(base, "config.json")
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != source_config:
        raise ValueError(f"Local config.json differs from the pinned {model.title} revision.")
    output.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        write_json(config_path, source_config)
    processor = get_json(base, "processor_config.json")
    target = output / "weights/vision.safetensors"
    receipt = output / "weights/vision.source.json"
    if target.exists():
        if receipt.is_file() and json.loads(receipt.read_text()) == {**origin, "sha256": sha256(target)}:
            write_json(output / "processor_config.json", processor)
            print(f"Vision weights already verified: {target}")
            return
        raise ValueError("Existing vision weights cannot be verified. Move them aside before fetching again.")
    index = get_json(base, "model.safetensors.index.json")["weight_map"]
    names = [name for name in index if name.startswith(PREFIX)]
    if len(names) != 333:
        raise ValueError("Expected 333 vision tensors in the pinned checkpoint.")
    tensors = {}
    downloaded = 0
    for shard in sorted({index[name] for name in names}):
        url = f"{base}/{shard}"
        header_size = struct.unpack("<Q", get_range(url, 0, 8))[0]
        if not 1 <= header_size <= 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size.")
        header = json.loads(get_range(url, 8, 8 + header_size))
        selected = sorted((name for name in names if index[name] == shard), key=lambda name: header[name]["data_offsets"][0])
        for name in selected:
            spec = header[name]
            lo, hi = spec["data_offsets"]
            if spec["dtype"] != "BF16" or not 0 <= lo < hi or hi - lo != math.prod(spec["shape"]) * 2:
                raise ValueError(f"Unexpected vision tensor: {name}")
        # Adjacent tensors share requests, but no request can swallow the entire multi-GB shard.
        groups = []
        for name in selected:
            lo, hi = header[name]["data_offsets"]
            if groups and lo == header[groups[-1][-1]]["data_offsets"][1] and hi - header[groups[-1][0]]["data_offsets"][0] <= BLOCK_BYTES:
                groups[-1].append(name)
            else:
                groups.append([name])
        for group in groups:
            lo = header[group[0]]["data_offsets"][0]
            hi = header[group[-1]]["data_offsets"][1]
            raw = bytearray(get_range(url, 8 + header_size + lo, 8 + header_size + hi))
            for name in group:
                spec = header[name]
                start, stop = spec["data_offsets"]
                tensor = torch.frombuffer(raw, dtype=torch.bfloat16, count=(stop - start) // 2, offset=start - lo)
                tensors[name] = tensor.reshape(spec["shape"]).clone()
            downloaded += len(raw)
            print(f"Fetched vision weights: {downloaded / 1024**2:.0f} MiB", flush=True)
    if sum(t.numel() for t in tensors.values()) != PARAMETERS[model.name]:
        raise ValueError("Unexpected vision parameter count.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    try:
        save_file(tensors, temporary)
        # Publish the receipt first: a completed weight file is never left without its provenance.
        write_json(receipt, {**origin, "sha256": sha256(temporary)})
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output / "processor_config.json", processor)
    print(f"Vision weights ready: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, help="data directory (default: the model's research data directory)")
    args = parser.parse_args()
    fetch(args.output or data_dir(Path(__file__).resolve().parents[1], MODELS[args.model]), MODELS[args.model])
