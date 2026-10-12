"""Export local research weights as a LUT-6 bundle, or add vision to an existing text bundle.

Maintainer command; install the package first. No ANE, network, or credentials are needed.
--model picks Clef-Flash (weights in data/) or Clef (weights in data/clef/).
With --text-bundle, only the vision weights/settings are needed from the source directory;
the existing text programs are copied byte-for-byte, without the original decoder checkpoint.
A LUT cache in the data directory is reused when present; it is never created (Clef's would be 24 GiB).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path

import torch
from safetensors.torch import load_file

from clef_ane.bundle import (BLOCK_FORMAT_VERSION, BUCKETS, FORMAT_VERSION, IMAGE_FORMAT_VERSION, MANIFEST, VISION_FILES,
                             load_bundle, sha256, validate_manifest)
from clef_ane.engine import layers, lutq, mil
from clef_ane.models import DEFAULT_MODEL, MODELS, ModelInfo, data_dir
from clef_ane.program import save_program

SOURCE_REPO = MODELS[DEFAULT_MODEL].source_repo
SOURCE_REVISION = MODELS[DEFAULT_MODEL].source_revision

# What was measured for each published conversion; a model card never borrows another model's numbers.
VISION_NOTES = {"clef-flash": """
- LUT-6 was tested on the vision layers and rejected: per-layer error was 5–14× higher than FP16, with only about 10% speedup.
- In FP16 on the ANE, per-layer relative RMSE against float32 was 0.08–0.22%, compared with 0.30–0.55% for BF16 on the Metal GPU.
- The largest residual activation observed was about 12,400, within the FP16 limit of 65,504. Inputs that overflow are rejected rather than answered.
- The vision programs are built from these weights on the destination Mac when the first image or video arrives, which took about 10 seconds on an M5 Pro.
""", "clef": """
- The vision layers are not LUT-quantized. LUT-6 was rejected for them on Clef-Flash, whose encoder has the same shape; it was not measured again for this model.
- Inputs that overflow FP16 in the vision encoder are rejected rather than answered.
- The vision programs are built from these weights on the destination Mac when the first image or video arrives, which took about 11 seconds on an M5 Pro.
"""}
# Measured load cost, added to the size paragraph.
RESOURCES = {"clef": " On an M5 Pro with 64 GB of memory, loading every program took about 7 minutes, and the loaded model held about "
                     "24 GiB of wired memory, plus about 3 GiB in the serving process."}
VALIDATION = {"clef-flash": """## Validation

On a fixed regression set of 17 synthetic text, image and video requests (27 questions), the original BF16 multimodal model was the reference, with identical preprocessing:

- The selected answer matched on **27 of 27** questions; the largest probability difference was 0.065.
- Both models met 26 of the 27 synthetic expectations. The shared miss is a mixed request in which a separate green image is read as the start of a red-to-blue video.
- During those requests, repeated twice, the serving process recorded **0 ns of GPU time** on an M5 Pro running macOS 27.

This is a regression check, not a general accuracy benchmark. Images were resized with Pillow to at most 262,144 pixels (256 visual tokens). Video was sampled about twice per second by presentation timestamp, up to 64 frames of at most 65,536 pixels each. Audio is not used.

The earlier text-only conversion is available at revision `text-v1`.
""", "clef": """## Validation

The original BF16 multimodal model was the reference, with identical preprocessing. It does not fit in the test Mac's memory at once, so it was run one decoder layer at a time; on Clef-Flash, that procedure reproduces the whole-model result bit for bit.

- On a fixed regression set of 17 synthetic text, image and video requests (27 questions), the selected answer matched on **27 of 27** questions; the largest probability difference was 0.004. Both models met all 27 synthetic expectations.
- On 8 text-only requests (29 questions), two of them about 4,900 and 15,300 tokens long, the selected answer matched on **28 of 29** questions; the largest probability difference was 0.038. The one differing question is a near-tie in the original model, whose top two options are 0.019 apart.
- During the 17 requests, repeated twice, the serving process recorded **0 ns of GPU time** on an M5 Pro running macOS 27.

This is a regression check, not a general accuracy benchmark. Images were resized with Pillow to at most 262,144 pixels (256 visual tokens). Video was sampled about twice per second by presentation timestamp, up to 64 frames of at most 65,536 pixels each. Audio is not used.
"""}
# macOS keeps at most this many programs loaded on the ANE, counted over every process on the Mac.
ANE_PROGRAM_LIMIT = 127
TEXT_VALIDATION = """## Validation

No general accuracy benchmark is claimed. Answers were spot-checked against the original BF16 text model on internal regression records.
"""
UNMEASURED = """## Validation

No accuracy measurements have been recorded for this conversion.
"""


def copy(source: Path, target: Path) -> None:
    """Copy-on-write on APFS, so multi-gigabyte embedding and vision files are not stored twice on the maintainer's disk."""
    if sys.platform != "darwin" or subprocess.run(["cp", "-c", str(source), str(target)], capture_output=True).returncode:
        shutil.copyfile(source, target)


def model_card(files: dict, with_images: bool, model: ModelInfo = MODELS[DEFAULT_MODEL], config: dict | None = None,
               head: dict | None = None, layout: list | None = None) -> str:
    """Hugging Face model card describing the converted weights; sizes come from the manifest, shapes from the config.

    layout: the manifest's layers, to describe a bundle whose linear-attention layers are compiled in blocks."""
    gib = lambda prefix: sum(info["size"] for name, info in files.items() if name.startswith(prefix)) / 1024**3
    sizes = {"total": gib(""), "programs": gib("programs/"), "embeddings": gib("weights/embed.") + gib("weights/lm_head."),
             "head": gib("weights/joint_head"), "vision": gib("weights/vision.")}
    # Clef-Flash's shapes, for callers that only have a manifest.
    text = {"layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8, "hidden_size": 4096, "intermediate_size": 12288,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32, "num_attention_heads": 16, "num_key_value_heads": 4,
            "vocab_size": 248320, **(config or {}).get("text_config", {})}
    vision = {"depth": 27, "hidden_size": 1152, "num_heads": 16, **(config or {}).get("vision_config", {})}
    head = {"layers": 4, "width": 1024, "heads": 16, "routing_layers": 2, **(head or {})}
    count, hidden = len(text["layer_types"]), text["hidden_size"]
    linear, full = text["layer_types"].count("linear_attention"), text["layer_types"].count("full_attention")
    tags = ["apple-neural-engine", "quantized", "lut", "decision-model", "custom-format"] + (["multimodal"] if with_images else [])
    vision_row = (f"| Vision encoder | {vision['depth']} layers; hidden size {vision['hidden_size']:,}; {vision['num_heads']} heads; "
                  "16×16 patches with two-frame temporal patches; "
                  f"2×2 merge into the {hidden:,}-wide text embedding space. Shared by images and video |\n") if with_images else ""
    vision_size = f", vision encoder {sizes['vision']:.2f} GiB" if with_images else ""
    vision_execution = (f"| Vision transformer ({vision['depth']} layers) | ANE, FP16. One fixed program per layer over 1,024 patches; "
                        "images and video frame pairs are packed with a block-diagonal attention mask |\n"
                        "| Patch embedding, vision positions and 2×2 merger | CPU |\n") if with_images else ""
    vision_precision = ("| Vision encoder | Original **BF16**; converted to **FP16** when loaded onto the ANE. Not LUT-quantized |\n"
                        if with_images else "")
    programs = {name for layer in layout or [] for key in ("block", "mlp") if (name := layer.get(key))}
    programs |= {name for layer in layout or [] for name in layer.get("mixers", {}).values()}
    block_row = (f"| ANE programs | {len(programs)} for text. Each run of three linear-attention layers is one program, together with "
                 "the MLP of the full-attention layer before it, because macOS keeps at most "
                 f"{ANE_PROGRAM_LIMIT} programs on the ANE across all apps. The vision encoder adds {vision['depth']} |\n"
                 ) if any("block" in layer for layer in layout or []) else ""
    vision_section = f"""
### Vision weights
{VISION_NOTES[model.name]}""" if with_images and model.name in VISION_NOTES else ""
    validation = VALIDATION.get(model.name, UNMEASURED) if with_images else TEXT_VALIDATION
    return f"""---
base_model: {model.source_repo}
base_model_relation: quantized
license: apache-2.0
pipeline_tag: {"image-text-to-text" if with_images else "text-classification"}
tags:
{chr(10).join("- " + tag for tag in tags)}
---

# {model.title}-ANE-LUT6

## Model architecture

A mixed-precision conversion of [{model.source_repo}](https://huggingface.co/{model.source_repo}) for the Apple Neural Engine (ANE) on Apple Silicon Macs. {"It answers questions about text, images and short videos" if with_images else "It answers questions about text"} with yes/no, choice and score outputs; it does not generate text.

| Component | Structure |
|---|---|
| Text backbone | {count} layers; hidden size {hidden:,}; MLP intermediate size {text["intermediate_size"]:,} |
| Attention layout | {linear} Gated DeltaNet linear-attention layers ({text["linear_num_key_heads"]} key and {text["linear_num_value_heads"]} value heads, head size 128) and {full} full-attention layers ({text["num_attention_heads"]} query and {text["num_key_value_heads"]} key-value heads, head size 256), with full attention every fourth layer |
| Decision head | {head["layers"]} layers; width {head["width"]:,}; {head["heads"]} heads; {head["routing_layers"]} routing layers |
| Vocabulary | {text["vocab_size"]:,} tokens |
{vision_row}| Context | Up to 16,384 tokens, processed 128 tokens per ANE call; full-attention programs for 1,024, 2,048, 4,096, 8,192 and 16,384 tokens |
{block_row}
The payload is approximately **{sizes["total"]:.2f} GiB**: ANE programs {sizes["programs"]:.2f} GiB, token embeddings and LM head {sizes["embeddings"]:.2f} GiB, decision head {sizes["head"]:.2f} GiB{vision_size}. These are on-disk sizes, not RAM requirements.{RESOURCES.get(model.name, "") if with_images else ""}

| Component | Runs on |
|---|---|
| Text backbone ({count} layers) | ANE |
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
- Four 6-bit indices are packed into three bytes in Core ML's little-endian layout. Including the tables, a layer with {hidden:,} inputs stores about **{6 + 64 / hidden:.2f} bits per weight**.
- Activations run in FP16 on the ANE. The MLP down-projection input is scaled up by as much as 16× and attention probabilities by 16× to keep precision in FP16; both are scaled back afterwards.
- Quantization is lossy; the original BF16 weights cannot be reconstructed exactly.
{vision_section}
The original tokenizer, chat template, configuration{" and image/video processor settings" if with_images else ""} are retained. File sizes and SHA-256 checksums are recorded in the bundle manifest.

{validation}
## Source and license

Source revision: `{model.source_revision}`.

Original model **{model.title}** by **Cloudflare**, post-trained from **{model.base}** by Qwen. Conversion by **Yanun**. Released under **Apache-2.0**, the license of the source model. Redistributions must retain `LICENSE` and `NOTICE`.

This is not an official Cloudflare, Qwen or Apple release.
"""


def finish_bundle(data: Path, output: Path, manifest: dict, with_images: bool, model: ModelInfo = MODELS[DEFAULT_MODEL]) -> None:
    if with_images:
        for name in VISION_FILES:
            copy(data / name, output / name)
    vision_notice = "Vision weights retain BF16 storage; the runtime converts them to fp16 ANE programs (not LUT-6)." if with_images else "Vision weights were omitted."
    (output / "LICENSE").write_bytes(files("clef_ane._vendor").joinpath("LICENSE").read_bytes())
    (output / "NOTICE").write_text(
        f"Derived from {model.source_repo}, revision {model.source_revision}.\n"
        f"Upstream model: Cloudflare {model.title}, based on {model.base}; Apache-2.0.\n"
        "Changes: text backbone exported to MIL; large linear-layer weights converted to packed 6-bit lookup tables.\n"
        "Small weights, embeddings, and CPU head tensors retain their original stored precision.\n"
        f"{vision_notice}\nThis is not an official Cloudflare or Apple release.\n")
    manifest["files"] = {path.relative_to(output).as_posix(): {"size": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"README.md", MANIFEST}}
    (output / "README.md").write_text(model_card(manifest["files"], with_images, model, json.loads((output / "config.json").read_text()),
                                                 json.loads((output / "weights/joint_head_config.json").read_text()), manifest["layers"]))
    validate_manifest(manifest)
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    load_bundle(output)
    size = sum(f["size"] for f in manifest["files"].values()) / 1024**3
    print(f"Bundle ready: {output} ({size:.2f} GiB)", flush=True)


def export(source: Path, output: Path, *, with_images: bool = False, text_bundle: Path | None = None,
           model: ModelInfo = MODELS[DEFAULT_MODEL]) -> None:
    data, output = data_dir(source.resolve(), model), output.expanduser().resolve()
    origin = {"repo_id": model.source_repo, "revision": model.source_revision}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if with_images:
        if any(not (data / name).is_file() for name in VISION_FILES):
            raise ValueError("Vision files are missing. Run make fetch-vision first.")
        from clef_ane.vision import image_processor_settings
        settings = json.loads((data / "processor_config.json").read_text())
        image_processor_settings(settings)
        if "video_processor" in settings:
            from clef_ane.video import video_processor_settings
            video_processor_settings(settings)
        receipt = data / "weights/vision.source.json"
        expected = {**origin, "sha256": sha256(data / "weights/vision.safetensors")}
        if not receipt.is_file() or json.loads(receipt.read_text()) != expected:
            raise ValueError("Vision weight provenance could not be verified. Use weights from make fetch-vision.")
    source_config = json.loads((data / "config.json").read_text())
    config = source_config["text_config"]
    if text_bundle is not None:
        if not with_images:
            raise ValueError("--text-bundle requires --with-images")
        text_bundle = text_bundle.expanduser().resolve()
        manifest = load_bundle(text_bundle)
        if (manifest["format_version"] != FORMAT_VERSION
                or manifest.get("source") != origin
                or json.loads((text_bundle / "config.json").read_text()) != source_config):
            raise ValueError(f"Expected a text bundle from the same pinned {model.title} revision.")
        output.mkdir(parents=True, exist_ok=True)
        for name in manifest["files"]:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(text_bundle / name, destination)
        manifest["format_version"] = IMAGE_FORMAT_VERSION
        finish_bundle(data, output, manifest, True, model)
        return
    layers.check_config(config)
    count = len(config["layer_types"])
    missing = [name for name in ("embed", "norm", "lm_head", "joint_head", *(f"layer_{i:02d}" for i in range(count)))
               if not (data / "weights" / f"{name}.safetensors").is_file()]
    if missing:
        raise ValueError(f"Source weights are missing ({missing[0]}.safetensors and {len(missing) - 1} more). Run make fetch first.")
    # One program per mixer and per MLP, plus the vision encoder's, fits Clef-Flash (123) but not Clef (219).
    kinds = config["layer_types"]
    separate = count + kinds.count("linear_attention") + kinds.count("full_attention") * len(BUCKETS)
    blocks = separate + source_config["vision_config"]["depth"] > ANE_PROGRAM_LIMIT
    if blocks and not with_images:
        raise ValueError(f"{model.title} is exported with images and video only. Add --with-images.")
    output.mkdir(parents=True, exist_ok=True)
    original_cache, original_quantize = lutq.CACHE, mil.Graph.default_quantize
    lutq.CACHE = data / "lutcache"
    quantize = lutq.kmeans_lut_cached if lutq.CACHE.is_dir() else lutq.kmeans_lut
    mil.Graph.default_quantize = lambda w: quantize(w, 6, 16)
    manifest = {"format": "clef-ane", "format_version": BLOCK_FORMAT_VERSION if blocks else IMAGE_FORMAT_VERSION if with_images else FORMAT_VERSION,
                "source": origin,
                "quantization": {"bits": 6, "group_size": 16, "method": "kmeans-lut"},
                "rows": 128, "buckets": list(BUCKETS), "layers": [], "files": {}}
    started = time.monotonic()

    def load(index: int) -> dict:
        prefix = f"model.language_model.layers.{index}."
        return {k[len(prefix):]: v.float().numpy() for k, v in load_file(data / "weights" / f"layer_{index:02d}.safetensors").items()}

    def attention(index: int, weights: dict) -> dict:
        """One program per context size; they share one weight file."""
        mixers = {}
        for context in BUCKETS:
            mixers[str(context)] = f"programs/{index:02d}/attention_{context}.json"
            save_program(output, mixers[str(context)], layers.full_attention(weights, rows=128, ctx=context),
                         f"programs/{index:02d}/attention.bin")
        return mixers

    try:
        index, lead = 0, None       # lead: the previous full-attention layer's MLP weights, which open the next block
        while index < count:
            base = f"programs/{index:02d}"
            if blocks and kinds[index] == "linear_attention":
                size = next((n for n in range(1, count - index) if kinds[index + n] != "linear_attention"), count - index)
                save_program(output, f"{base}/block.json",
                             layers.linear_block([load(index + n) for n in range(size)], rows=128, lead=lead), f"{base}/block.bin")
                manifest["layers"] += [{"kind": "linear_attention", "block": f"{base}/block.json"} for _ in range(size)]
                index, lead = index + size, None
                print(f"Exported {index}/{count} layers ({time.monotonic() - started:.0f}s)", flush=True)
                continue
            weights = load(index)
            spec = {"kind": kinds[index]}
            if not blocks or index == count - 1:
                spec["mlp"] = f"{base}/mlp.json"
                save_program(output, spec["mlp"], layers.mlp(weights, rows=128), f"{base}/mlp.bin")
            else:
                lead = {name: value for name, value in weights.items() if name.startswith(("mlp.", "post_attention_layernorm."))}
            if kinds[index] == "linear_attention":
                spec["mixer"] = f"{base}/mixer.json"
                save_program(output, spec["mixer"], layers.gdn_mixer(weights, rows=128), f"{base}/mixer.bin")
            else:
                spec["mixers"] = attention(index, weights)
            manifest["layers"].append(spec)
            index += 1
            print(f"Exported {index}/{count} layers ({time.monotonic() - started:.0f}s)", flush=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copyfile(data / name, output / name)
        (output / "weights").mkdir()
        for name in ("embed.safetensors", "norm.safetensors", "lm_head.safetensors", "joint_head.safetensors", "joint_head_config.json"):
            copy(data / "weights" / name, output / "weights" / name)
        finish_bundle(data, output, manifest, with_images, model)
    finally:
        lutq.CACHE, mil.Graph.default_quantize = original_cache, original_quantize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="which model's research weights to export")
    parser.add_argument("--with-images", action="store_true", help="include the original vision weights and image/video processor settings")
    parser.add_argument("--text-bundle", type=Path, help="reuse a verified text bundle without re-exporting its ANE programs (requires --with-images)")
    args = parser.parse_args()
    export(args.source, args.output, with_images=args.with_images, text_bundle=args.text_bundle, model=MODELS[args.model])
