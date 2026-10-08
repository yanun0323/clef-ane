"""Download Clef-Flash's language model one layer at a time with HTTP range requests (resumable).

Writes data/weights/layer_NN.safetensors (NN = 00..31), norm.safetensors, embed.safetensors, lm_head.safetensors and
joint_head.{safetensors,json}. Vision weights are not fetched (text-only for now).
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "Cloudflare/clef-flash"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
OUT = Path(__file__).resolve().parents[1] / "data" / "weights"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
OUT.mkdir(parents=True, exist_ok=True)


def get(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # transient network errors: retry
            if attempt == 4:
                raise
            print("retry", error, flush=True)


index = json.load(urllib.request.urlopen(f"{BASE}/model.safetensors.index.json"))["weight_map"]
headers = {}


def header(shard):
    if shard not in headers:
        url = f"{BASE}/{shard}"
        size = struct.unpack("<Q", get(url, 0, 7))[0]
        headers[shard] = (json.loads(get(url, 8, 8 + size - 1)), 8 + size)
    return headers[shard]


def fetch(names, target):
    if (OUT / target).exists():
        return
    tensors = {}
    for name in names:
        shard = index[name]
        meta, start = header(shard)
        info = meta[name]
        lo, hi = info["data_offsets"]
        raw = bytearray(get(f"{BASE}/{shard}", start + lo, start + hi - 1))
        tensors[name] = torch.frombuffer(raw, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).clone()
    save_file(tensors, OUT / target)
    print(f"{target}: {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6:.0f} MB", flush=True)


worker, workers = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
for layer in range(worker, 32, workers):
    fetch([n for n in index if n.startswith(f"model.language_model.layers.{layer}.")], f"layer_{layer:02d}.safetensors")
extras = [
    lambda: fetch(["model.language_model.norm.weight"], "norm.safetensors"),
    lambda: fetch(["lm_head.weight"], "lm_head.safetensors"),
    lambda: fetch(["model.language_model.embed_tokens.weight"], "embed.safetensors"),
]
for i, extra in enumerate(extras):
    if i % workers == worker:
        extra()
if worker == 0:
    for name in ("joint_head.safetensors", "joint_head_config.json"):
        if not (OUT / name).exists():
            (OUT / name).write_bytes(urllib.request.urlopen(f"{BASE}/{name}").read())
print("done", worker, flush=True)
