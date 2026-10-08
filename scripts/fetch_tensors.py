"""Fetch only some tensors of a Hugging Face safetensors repo using HTTP range requests.

Usage: fetch_tensors.py REPO OUTPUT.safetensors PREFIX [PREFIX ...]
Every tensor whose name starts with one of the prefixes is downloaded and saved, unchanged, to OUTPUT.
"""
import json
import struct
import sys
import urllib.request

import torch
from safetensors.torch import save_file

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def get(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def main(repo: str, output: str, prefixes: list[str]) -> None:
    base = f"https://huggingface.co/{repo}/resolve/main"
    index = json.load(urllib.request.urlopen(f"{base}/model.safetensors.index.json"))["weight_map"]
    wanted: dict[str, list[str]] = {}
    for name, shard in index.items():
        if any(name.startswith(p) for p in prefixes):
            wanted.setdefault(shard, []).append(name)

    tensors = {}
    for shard, names in wanted.items():
        url = f"{base}/{shard}"
        header_size = struct.unpack("<Q", get(url, 0, 7))[0]
        header = json.loads(get(url, 8, 8 + header_size - 1))
        data_start = 8 + header_size
        for name in names:
            info = header[name]
            begin, end = info["data_offsets"]
            raw = bytearray(get(url, data_start + begin, data_start + end - 1))
            tensors[name] = torch.frombuffer(raw, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).clone()
            print(f"{name} {info['shape']} {info['dtype']}")
    save_file(tensors, output)
    print(f"saved {len(tensors)} tensors, {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6:.1f} MB -> {output}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
