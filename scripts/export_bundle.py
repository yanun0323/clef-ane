"""Export local research weights + LUT-6 cache, or add vision to an existing text bundle.

Maintainer command; install the package first. No ANE, network, or credentials are needed.
With --text-bundle, only the vision weights/settings are needed from the source directory;
the existing text programs are copied byte-for-byte, without the original decoder checkpoint.
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

from clef_ane.bundle import BUCKETS, FORMAT_VERSION, IMAGE_FORMAT_VERSION, MANIFEST, VISION_FILES, load_bundle, sha256, validate_manifest
from clef_ane.engine import layers, lutq, mil
from clef_ane.program import save_program

SOURCE_REPO = "Cloudflare/clef-flash"
SOURCE_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"

def model_card(files: dict, with_images: bool) -> str:
    """Hugging Face model card describing the converted weights; sizes come from the manifest."""
    gib = lambda prefix: sum(info["size"] for name, info in files.items() if name.startswith(prefix)) / 1024**3
    sizes = {"total": gib(""), "programs": gib("programs/"), "embeddings": gib("weights/embed.") + gib("weights/lm_head."),
             "head": gib("weights/joint_head"), "vision": gib("weights/vision.")}
    tags = ["apple-neural-engine", "quantized", "lut", "decision-model", "custom-format"] + (["multimodal"] if with_images else [])
    vision_row = ("| Vision encoder | 27 layers; hidden size 1,152; 16 heads; 16×16 patches with two-frame temporal patches; "
                  "2×2 merge into the 4,096-wide text embedding space. Shared by images and video |\n") if with_images else ""
    vision_size = f", vision encoder {sizes['vision']:.2f} GiB" if with_images else ""
    vision_execution = ("| Vision transformer (27 layers) | ANE, FP16. One fixed program per layer over 1,024 patches; "
                        "images and video frame pairs are packed with a block-diagonal attention mask |\n"
                        "| Patch embedding, vision positions and 2×2 merger | CPU |\n") if with_images else ""
    vision_precision = ("| Vision encoder | Original **BF16**; converted to **FP16** when loaded onto the ANE. Not LUT-quantized |\n"
                        if with_images else "")
    vision_section = """
### Vision weights

- LUT-6 was tested on the vision layers and rejected: per-layer error was 5–14× higher than FP16, with only about 10% speedup.
- In FP16 on the ANE, per-layer relative RMSE against float32 was 0.08–0.22%, compared with 0.30–0.55% for BF16 on the Metal GPU.
- The largest residual activation observed was about 12,400, within the FP16 limit of 65,504. Inputs that overflow are rejected rather than answered.
- The vision programs are built from these weights on the destination Mac when the first image or video arrives, which took about 10 seconds on an M5 Pro.
""" if with_images else ""
    validation = """## Validation

On a fixed regression set of 17 synthetic text, image and video requests (27 questions), the original BF16 multimodal model was the reference, with identical preprocessing:

- The selected answer matched on **27 of 27** questions; the largest probability difference was 0.065.
- Both models met 26 of the 27 synthetic expectations. The shared miss is a mixed request in which a separate green image is read as the start of a red-to-blue video.
- During those requests, repeated twice, the serving process recorded **0 ns of GPU time** on an M5 Pro running macOS 27.

This is a regression check, not a general accuracy benchmark. Images were resized with Pillow to at most 262,144 pixels (256 visual tokens). Video was sampled about twice per second by presentation timestamp, up to 64 frames of at most 65,536 pixels each. Audio is not used.

The earlier text-only conversion is available at revision `text-v1`.
""" if with_images else """## Validation

No general accuracy benchmark is claimed. Answers were spot-checked against the original BF16 text model on internal regression records.
"""
    return f"""---
base_model: Cloudflare/clef-flash
base_model_relation: quantized
license: apache-2.0
pipeline_tag: {"image-text-to-text" if with_images else "text-classification"}
tags:
{chr(10).join("- " + tag for tag in tags)}
---

# Clef-Flash-ANE-LUT6

## Model architecture

A mixed-precision conversion of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) for the Apple Neural Engine (ANE) on Apple Silicon Macs. {"It answers questions about text, images and short videos" if with_images else "It answers questions about text"} with yes/no, choice and score outputs; it does not generate text.

| Component | Structure |
|---|---|
| Text backbone | 32 layers; hidden size 4,096; MLP intermediate size 12,288 |
| Attention layout | 24 Gated DeltaNet linear-attention layers (16 key and 32 value heads, head size 128) and 8 full-attention layers (16 query and 4 key-value heads, head size 256), with full attention every fourth layer |
| Decision head | 4 layers; width 1,024; 16 heads; 2 routing layers |
| Vocabulary | 248,320 tokens |
{vision_row}| Context | Up to 16,384 tokens, processed 128 tokens per ANE call; full-attention programs for 1,024, 2,048, 4,096, 8,192 and 16,384 tokens |

The payload is approximately **{sizes["total"]:.2f} GiB**: ANE programs {sizes["programs"]:.2f} GiB, token embeddings and LM head {sizes["embeddings"]:.2f} GiB, decision head {sizes["head"]:.2f} GiB{vision_size}. These are on-disk sizes, not RAM requirements.

| Component | Runs on |
|---|---|
| Text backbone (32 layers) | ANE |
{vision_execution}| Token embedding lookup and decision head | CPU |

No GPU is used.

**Format compatibility:** this is a custom artifact, not a standard Transformers, MLX or Core ML `.mlpackage` checkpoint. The text layers are stored as MIL program descriptions with packed weights, compiled on the destination Mac when loaded. Other tensors are safetensors. Loading uses Apple's private ANE interface, which may change with macOS updates; only an M5 Pro running macOS 27 has been tested.

## Weight precision

| Component | Stored precision |
|---|---|
| Large text linear layers: MLP gate/up/down, linear-attention QKV/Z/output, full-attention Q/K/V/O | **LUT-6**: 6-bit indices with one 64-entry **FP16** table per 16 output channels |
| Other text-program weights: normalization, short convolution, small gate projections and decay parameters | **FP16**, converted from BF16 |
| Token embeddings, LM head and final normalization | Original **BF16** |
| Decision head | Original **BF16** |
{vision_precision}
### LUT-6 text weights

- Every linear layer with at least 1,048,576 weights is quantized. Each group of 16 output channels has its own 64-value table, found by 40 rounds of one-dimensional k-means. Half of the starting values are quantiles; the other half are spread evenly between the group's minimum and maximum, so that outlying weights also get table values.
- Four 6-bit indices are packed into three bytes in Core ML's little-endian layout. Including the tables, a layer with 4,096 inputs stores about **6.02 bits per weight**.
- Activations run in FP16 on the ANE. The MLP down-projection input is scaled up by as much as 16× and attention probabilities by 16× to keep precision in FP16; both are scaled back afterwards.
- Quantization is lossy; the original BF16 weights cannot be reconstructed exactly.
{vision_section}
The original tokenizer, chat template, configuration{" and image/video processor settings" if with_images else ""} are retained. File sizes and SHA-256 checksums are recorded in the bundle manifest.

{validation}
## Source and license

Source revision: `{SOURCE_REVISION}`.

Original model **Clef-Flash** by **Cloudflare**, post-trained from **Qwen3.5-9B** by Qwen. Conversion by **Yanun**. Released under **Apache-2.0**, the license of the source model. Redistributions must retain `LICENSE` and `NOTICE`.

This is not an official Cloudflare, Qwen or Apple release.
"""


def finish_bundle(source: Path, output: Path, manifest: dict, with_images: bool) -> None:
    if with_images:
        for name in VISION_FILES:
            shutil.copyfile(source / "data" / name, output / name)
    vision_notice = "Vision weights retain BF16 storage; the runtime converts them to fp16 ANE programs (not LUT-6)." if with_images else "Vision weights were omitted."
    (output / "LICENSE").write_bytes(files("clef_ane._vendor").joinpath("LICENSE").read_bytes())
    (output / "NOTICE").write_text(
        f"Derived from {SOURCE_REPO}, revision {SOURCE_REVISION}.\n"
        "Upstream model: Cloudflare Clef-Flash, based on Qwen3.5-9B; Apache-2.0.\n"
        "Changes: text backbone exported to MIL; large linear-layer weights converted to packed 6-bit lookup tables.\n"
        "Small weights, embeddings, and CPU head tensors retain their original stored precision.\n"
        f"{vision_notice}\nThis is not an official Cloudflare or Apple release.\n")
    manifest["files"] = {path.relative_to(output).as_posix(): {"size": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"README.md", MANIFEST}}
    (output / "README.md").write_text(model_card(manifest["files"], with_images))
    validate_manifest(manifest)
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    load_bundle(output)
    size = sum(f["size"] for f in manifest["files"].values()) / 1024**3
    print(f"Bundle ready: {output} ({size:.2f} GiB)", flush=True)


def export(source: Path, output: Path, *, with_images: bool = False, text_bundle: Path | None = None) -> None:
    source, output = source.resolve(), output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if with_images:
        if any(not (source / "data" / name).is_file() for name in VISION_FILES):
            raise ValueError("Vision files are missing. Run make fetch-vision first.")
        from clef_ane.vision import image_processor_settings
        settings = json.loads((source / "data/processor_config.json").read_text())
        image_processor_settings(settings)
        if "video_processor" in settings:
            from clef_ane.video import video_processor_settings
            video_processor_settings(settings)
        receipt = source / "data/weights/vision.source.json"
        expected = {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION, "sha256": sha256(source / "data/weights/vision.safetensors")}
        if not receipt.is_file() or json.loads(receipt.read_text()) != expected:
            raise ValueError("Vision weight provenance could not be verified. Use weights from make fetch-vision.")
    source_config = json.loads((source / "data" / "config.json").read_text())
    config = source_config["text_config"]
    if len(config["layer_types"]) != 32 or config["hidden_size"] != 4096:
        raise ValueError("Expected the Clef-Flash 32-layer text checkpoint.")
    if text_bundle is not None:
        if not with_images:
            raise ValueError("--text-bundle requires --with-images")
        text_bundle = text_bundle.expanduser().resolve()
        manifest = load_bundle(text_bundle)
        if (manifest["format_version"] != FORMAT_VERSION
                or manifest.get("source") != {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION}
                or json.loads((text_bundle / "config.json").read_text()) != source_config):
            raise ValueError("Expected a text bundle from the same pinned Clef-Flash revision.")
        output.mkdir(parents=True, exist_ok=True)
        for name in manifest["files"]:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(text_bundle / name, destination)
        manifest["format_version"] = IMAGE_FORMAT_VERSION
        finish_bundle(source, output, manifest, with_images=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    original_cache, original_quantize = lutq.CACHE, mil.Graph.default_quantize
    lutq.CACHE = source / "data" / "lutcache"
    mil.Graph.default_quantize = lambda w: lutq.kmeans_lut_cached(w, 6, 16)
    manifest = {"format": "clef-ane", "format_version": IMAGE_FORMAT_VERSION if with_images else FORMAT_VERSION,
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
        finish_bundle(source, output, manifest, with_images)
    finally:
        lutq.CACHE, mil.Graph.default_quantize = original_cache, original_quantize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-images", action="store_true", help="include the original vision weights and image/video processor settings")
    parser.add_argument("--text-bundle", type=Path, help="reuse a verified text bundle without re-exporting its ANE programs (requires --with-images)")
    args = parser.parse_args()
    export(args.source, args.output, with_images=args.with_images, text_bundle=args.text_bundle)
