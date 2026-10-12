"""Small synthetic bundle fixtures; no real weights, network, or ANE required."""
import json
from pathlib import Path

from clef_ane.bundle import BLOCK_FORMAT_VERSION, BUCKETS, IMAGE_FORMAT_VERSION, MANIFEST, MODEL_FILES, VISION_FILES, sha256


def make_bundle(root: Path, *, with_images: bool = False, layer_count: int = 32, blocks: bool = False) -> dict:
    """blocks: Clef's layout, with each run of linear-attention layers in one program."""
    with_images = with_images or blocks
    root.mkdir(parents=True, exist_ok=True)
    names = [*MODEL_FILES, *(VISION_FILES if with_images else ()), "README.md", "programs/00/test.json", "programs/00/test.bin"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    spec = {"format_version": 1, "text": "test MIL", "weight_file": "programs/00/test.bin",
            "inputs": [{"name": "x", "shape": [1, 32, 1, 32], "dtype": "fp16"}],
            "outputs": [{"name": "y", "shape": [1, 32, 1, 32], "dtype": "fp16"}]}
    (root / "programs/00/test.json").write_text(json.dumps(spec))
    layers = []
    for index in range(layer_count):
        kind = "full_attention" if index % 4 == 3 else "linear_attention"
        entry = {"kind": kind, "mlp": "programs/00/test.json"}
        if kind == "linear_attention":
            entry["mixer"] = "programs/00/test.json"
        else:
            entry["mixers"] = {str(n): "programs/00/test.json" for n in BUCKETS}
        if blocks:
            name = f"programs/{index - index % 4:02d}/block.json"
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(json.dumps(spec))
            names.append(name) if name not in names else None
            entry = ({"kind": kind, "block": name} if kind == "linear_attention" else
                     {key: value for key, value in entry.items() if key != "mlp" or index == layer_count - 1})
        layers.append(entry)
    version = BLOCK_FORMAT_VERSION if blocks else IMAGE_FORMAT_VERSION if with_images else 1
    manifest = {"format": "clef-ane", "format_version": version, "rows": 128, "buckets": list(BUCKETS),
                "quantization": {"bits": 6, "group_size": 16, "method": "kmeans-lut"}, "layers": layers,
                "files": {name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)} for name in names if name != "README.md"}}
    write_manifest(root, manifest)
    return manifest


def write_manifest(root: Path, manifest: dict) -> None:
    (root / MANIFEST).write_text(json.dumps(manifest))


def processor_config() -> dict:
    return {"processor_class": "Qwen3VLProcessor", "image_processor": {
        "image_processor_type": "Qwen2VLImageProcessor", "do_convert_rgb": True, "do_normalize": True,
        "do_rescale": True, "do_resize": True, "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5],
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "resample": 3,
        "rescale_factor": 1 / 255, "size": {"shortest_edge": 65536, "longest_edge": 16777216}}}


def image_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tokens = ["[UNK]", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=tokens[1:])
