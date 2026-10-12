"""Download a Clef language model one layer at a time with HTTP range requests (resumable).

Writes weights/layer_NN.safetensors, norm.safetensors, embed.safetensors, lm_head.safetensors and
joint_head.{safetensors,json} under the model's data directory. Vision weights come from fetch_vision.py.

Run:  fetch_all.py [WORKER WORKERS] [--model clef]     (WORKER of WORKERS splits the layers across processes)
"""
import argparse
import json
import struct
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from clef_ane.models import DEFAULT_MODEL, MODELS, data_dir

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("worker", type=int, nargs="?", default=0)
parser.add_argument("workers", type=int, nargs="?", default=1)
parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
parser.add_argument("--layers", help="comma-separated layer numbers; only these layers are fetched")
args = parser.parse_args()

MODEL = MODELS[args.model]
BASE = f"https://huggingface.co/{MODEL.source_repo}/resolve/{MODEL.source_revision}"
DATA = data_dir(Path(__file__).resolve().parents[1], MODEL)
OUT = DATA / "weights"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
OUT.mkdir(parents=True, exist_ok=True)


def get(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
            if response.status != 206 or len(raw) != end - start + 1:
                raise ValueError("the model host did not return the requested byte range")
            return raw
        except Exception as error:  # transient network errors: retry
            if attempt == 4:
                raise
            print("retry", error, flush=True)


def small(name, target):
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(urllib.request.urlopen(f"{BASE}/{name}", timeout=120).read())
        temporary.replace(target)


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
    temporary = OUT / (target + ".tmp")      # an interrupted run never leaves a short file under the final name
    save_file(tensors, temporary)
    temporary.replace(OUT / target)
    print(f"{target}: {sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6:.0f} MB", flush=True)


count = 1 + max(int(name.split(".")[3]) for name in index if name.startswith("model.language_model.layers."))
selected = [int(n) for n in args.layers.split(",")] if args.layers else range(count)
for layer in selected:
    if layer % args.workers == args.worker:
        fetch([n for n in index if n.startswith(f"model.language_model.layers.{layer}.")], f"layer_{layer:02d}.safetensors")
extras = [] if args.layers else [
    lambda: fetch(["model.language_model.norm.weight"], "norm.safetensors"),
    lambda: fetch(["lm_head.weight"], "lm_head.safetensors"),
    lambda: fetch(["model.language_model.embed_tokens.weight"], "embed.safetensors"),
]
for i, extra in enumerate(extras):
    if i % args.workers == args.worker:
        extra()
if args.worker == 0:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        small(name, DATA / name)
    if not args.layers:
        for name in ("joint_head.safetensors", "joint_head_config.json"):
            small(name, OUT / name)
print("done", args.worker, flush=True)
