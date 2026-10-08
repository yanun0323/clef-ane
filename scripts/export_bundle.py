"""Export existing research weights + LUT-6 cache to a self-contained clef-ane model bundle.

This is a maintainer command, not part of the download path. Install the package first.
No ANE, network access, or credentials are needed; only the local source checkpoint/cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from importlib.resources import files
from pathlib import Path

import torch
from safetensors.torch import load_file

from clef_ane.bundle import BUCKETS, FORMAT_VERSION, MANIFEST, load_bundle, sha256, validate_manifest
from clef_ane.engine import layers, lutq, mil
from clef_ane.program import save_program

SOURCE_REPO = "Cloudflare/clef-flash"
SOURCE_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"

MODEL_CARD = """---
license: apache-2.0
base_model: Cloudflare/clef-flash
base_model_relation: quantized
tags:
- clef-ane
- apple-neural-engine
- text-classification
- quantized
---
# Clef-Flash ANE LUT-6

A text-only export of Cloudflare's Clef-Flash for the `clef-ane` runtime on Apple Silicon Macs.
The 32 text layers run on the Apple Neural Engine; the decision head runs on the CPU.
Large linear-layer weights use packed 6-bit lookup tables with 16 output channels per table.
Embeddings, small weights, and the CPU decision head are not 6-bit quantized.

## Usage

Install `clef-ane` from its source checkout (`python -m pip install .`) or its supplied wheel, then:

```sh
clef-ane download --repo OWNER/clef-flash-ane-lut6
clef-ane serve
```

Serving requires an Apple Silicon Mac, a native Python 3.12+ installation, and Xcode Command Line Tools.
Validation has been on an M5 Pro running macOS 27; compatibility with other systems is not yet established.
The server listens on `127.0.0.1:8080` by default and has no authentication. It exposes
`POST /v1/systemone`, `GET /health`, and `GET /v1/models`.

Supports text/JSON state with yes/no, choice, or score questions, up to 64 questions and 16,384 input tokens.
Long state text is truncated to leave room for the questions. Images, videos, and text generation are not supported.
Requests are processed one at a time.

## Format and limitations

This is a **clef-ane bundle**, not a Transformers, MLX, or Core ML `.mlpackage` checkpoint.
It contains MIL program descriptions, packed weights, CPU tensors, tokenizer files, and the
checksummed `clef-ane.json` manifest. It does not require the original decoder checkpoint,
local LUT cache, or Python code from this repository. No conversion is needed after download,
but ANE programs still compile on the destination machine when the server starts.

The runtime uses Apple's private ANE API, which may change with macOS updates.
This release does not establish a general accuracy or energy-efficiency guarantee.
See the source model for the original model's capabilities and training information.

## Source and license

Derived from [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash),
revision `17f0b0ad64efb65d273590632833508766b2aae6`, based on Qwen3.5-9B.
Released under the upstream Apache-2.0 model license; see `LICENSE` and `NOTICE`.
The large text-layer linear weights were converted to LUT-6; vision weights were omitted.
This export is not an official Cloudflare or Apple release.
"""


def export(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "data" / "config.json").read_text())["text_config"]
    if len(config["layer_types"]) != 32 or config["hidden_size"] != 4096:
        raise ValueError("Expected the Clef-Flash 32-layer text checkpoint.")
    original_cache, original_quantize = lutq.CACHE, mil.Graph.default_quantize
    lutq.CACHE = source / "data" / "lutcache"
    mil.Graph.default_quantize = lambda w: lutq.kmeans_lut_cached(w, 6, 16)
    manifest = {"format": "clef-ane", "format_version": FORMAT_VERSION,
                "source": {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION},
                "quantization": {"bits": 6, "group_size": 16, "method": "kmeans-lut"},
                "rows": 128, "buckets": list(BUCKETS), "layers": [], "files": {}}
    started = time.monotonic()
    try:
        for index, kind in enumerate(config["layer_types"]):
            prefix = f"model.language_model.layers.{index}."
            weights = {k[len(prefix):]: v.float().numpy() for k, v in load_file(source / "data" / "weights" / f"layer_{index:02d}.safetensors").items()}
            base = f"programs/{index:02d}"
            spec = {"kind": kind, "mlp": f"{base}/mlp.json"}
            save_program(output, spec["mlp"], layers.mlp(weights, rows=128), f"{base}/mlp.bin")
            if kind == "linear_attention":
                spec["mixer"] = f"{base}/mixer.json"
                save_program(output, spec["mixer"], layers.gdn_mixer(weights, rows=128), f"{base}/mixer.bin")
            else:
                spec["mixers"] = {}
                for context in BUCKETS:
                    name = f"{base}/attention_{context}.json"
                    save_program(output, name, layers.full_attention(weights, rows=128, ctx=context), f"{base}/attention.bin")
                    spec["mixers"][str(context)] = name
            manifest["layers"].append(spec)
            print(f"Exported {index + 1}/32 layers ({time.monotonic() - started:.0f}s)", flush=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copyfile(source / "data" / name, output / name)
        (output / "weights").mkdir()
        for name in ("embed.safetensors", "norm.safetensors", "lm_head.safetensors", "joint_head.safetensors", "joint_head_config.json"):
            shutil.copyfile(source / "data" / "weights" / name, output / "weights" / name)
        (output / "LICENSE").write_bytes(files("clef_ane._vendor").joinpath("LICENSE").read_bytes())
        (output / "NOTICE").write_text(
            f"Derived from {SOURCE_REPO}, revision {SOURCE_REVISION}.\n"
            "Upstream model: Cloudflare Clef-Flash, based on Qwen3.5-9B; Apache-2.0.\n"
            "Changes: text-only export to MIL programs; large linear-layer weights converted to packed 6-bit lookup tables.\n"
            "Small weights, embeddings, and CPU head tensors retain their original stored precision.\n"
            "This is not an official Cloudflare or Apple release.\n")
        (output / "README.md").write_text(MODEL_CARD)
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "README.md":
                manifest["files"][path.relative_to(output).as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}
        validate_manifest(manifest)
        (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        load_bundle(output)
        size = sum(f["size"] for f in manifest["files"].values()) / 1024**3
        print(f"Bundle ready: {output} ({size:.2f} GiB)", flush=True)
    finally:
        lutq.CACHE, mil.Graph.default_quantize = original_cache, original_quantize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
