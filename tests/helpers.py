"""Small synthetic bundle fixtures; no real weights, network, or ANE required."""
import json
from pathlib import Path

from clef_ane.bundle import BUCKETS, MANIFEST, MODEL_FILES, sha256


def make_bundle(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    names = [*MODEL_FILES, "README.md", "programs/00/test.json", "programs/00/test.bin"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    spec = {"format_version": 1, "text": "test MIL", "weight_file": "programs/00/test.bin",
            "inputs": [{"name": "x", "shape": [1, 32, 1, 32], "dtype": "fp16"}],
            "outputs": [{"name": "y", "shape": [1, 32, 1, 32], "dtype": "fp16"}]}
    (root / "programs/00/test.json").write_text(json.dumps(spec))
    layers = []
    for index in range(32):
        kind = "full_attention" if index % 4 == 3 else "linear_attention"
        entry = {"kind": kind, "mlp": "programs/00/test.json"}
        if kind == "linear_attention":
            entry["mixer"] = "programs/00/test.json"
        else:
            entry["mixers"] = {str(n): "programs/00/test.json" for n in BUCKETS}
        layers.append(entry)
    manifest = {"format": "clef-ane", "format_version": 1, "rows": 128, "buckets": list(BUCKETS),
                "quantization": {"bits": 6, "group_size": 16, "method": "kmeans-lut"}, "layers": layers,
                "files": {name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)} for name in names if name != "README.md"}}
    write_manifest(root, manifest)
    return manifest


def write_manifest(root: Path, manifest: dict) -> None:
    (root / MANIFEST).write_text(json.dumps(manifest))
