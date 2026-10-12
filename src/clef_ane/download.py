"""Download a single pinned Hub snapshot, then verify the complete model."""
from __future__ import annotations

import json
import os
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from . import paths
from .bundle import MANIFEST, BundleError, load_bundle, read_json, safe_path, validate_manifest


def download_model(model_dir: Path, repo_id: str | None = None, revision: str = "main", *, force: bool = False,
                   default_repo: str | None = None) -> Path:
    """default_repo is the published bundle of the model chosen with --model; without it, the default model's."""
    model_dir = model_dir.expanduser().resolve()
    saved = model_dir / ".source.json"
    repo_id = repo_id or os.environ.get("CLEF_ANE_REPO")
    if not repo_id and saved.is_file():
        repo_id = read_json(saved).get("repo_id")
    repo_id = repo_id or default_repo or paths.DEFAULT_REPO_ID
    if not repo_id:
        raise BundleError("The public model repository has not been configured yet. Use --repo OWNER/clef-flash-ane-lut6.")
    model_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(model_dir / ".clef-ane.lock"), timeout=0):
        # Pin first: the manifest and every large file must come from the same commit,
        # even if the publisher updates `main` while the download is in progress.
        commit = HfApi().model_info(repo_id, revision=revision).sha
        if not commit:
            raise BundleError("Hugging Face did not return a model commit. Try the download again.")
        manifest_path = hf_hub_download(repo_id, MANIFEST, revision=commit, force_download=force)
        manifest = validate_manifest(read_json(Path(manifest_path)))
        for name in [MANIFEST, "README.md", *manifest["files"]]:
            safe_path(model_dir, name)
        size = sum(info["size"] for info in manifest["files"].values()) / 1024**3
        print(f"Downloading {repo_id} ({commit[:12]}, {size:.2f} GiB) to {model_dir}", flush=True)
        snapshot_download(repo_id, revision=commit, local_dir=model_dir,
                          allow_patterns=[MANIFEST, "README.md", *manifest["files"]], force_download=force)
        print("Checking model files...", flush=True)
        actual = load_bundle(model_dir)
        if actual != manifest:
            raise BundleError("Downloaded manifest differs from the selected commit. Run download again.")
        temporary = model_dir / ".source.json.tmp"
        temporary.write_text(json.dumps({"repo_id": repo_id, "revision": commit}, indent=2) + "\n")
        temporary.replace(saved)
    print(f"Model ready in {model_dir}", flush=True)
    return model_dir
