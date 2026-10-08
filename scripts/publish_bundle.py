"""Publish a verified model bundle to the currently authenticated user's public HF repository.

Uses an explicit file list: research files, download metadata, locks and credentials are never uploaded.
Run `hf auth login` first. This command creates/updates a PUBLIC model repository.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import LocalTokenNotFoundError

from clef_ane.bundle import MANIFEST, BundleError, load_bundle, read_json, safe_path
from clef_ane.paths import model_dir


def publish(root: Path, name: str = "clef-flash-ane-lut6") -> tuple[str, str]:
    root = root.expanduser().resolve()
    executable = shutil.which("hf")
    if not executable:
        raise RuntimeError("The hf command is missing. Install huggingface-hub in the active environment.")
    api = HfApi()
    try:
        owner = api.whoami()["name"]
    except LocalTokenNotFoundError as error:
        raise RuntimeError("Hugging Face is not logged in. Run `hf auth login`, then publish again.") from error
    if "/" in name:
        raise ValueError("Use a repository name without an owner; the owner comes from hf auth login.")
    repo_id = f"{owner}/{name}"
    with FileLock(str(root / ".clef-ane.lock"), timeout=0):
        manifest = load_bundle(root)
        card = safe_path(root, "README.md")
        if not card.is_file():
            raise BundleError("The model card README.md is missing.")
        api.create_repo(repo_id, private=False, exist_ok=True)
        info = api.model_info(repo_id)
        if info.private:
            raise RuntimeError(f"{repo_id} already exists as a private repository; its visibility was not changed.")
        existing = {entry.rfilename for entry in info.siblings or []}
        if existing - {".gitattributes", "README.md", "LICENSE", "NOTICE"} and MANIFEST not in existing:
            raise RuntimeError(f"{repo_id} contains another model format; refusing to overwrite it.")
        if MANIFEST in existing:
            old = read_json(Path(hf_hub_download(repo_id, MANIFEST, revision=info.sha)))
            if old.get("format") != "clef-ane" or old.get("source") != manifest.get("source"):
                raise RuntimeError(f"{repo_id} contains another model; refusing to overwrite it.")
        card.write_text(card.read_text().replace("OWNER/clef-flash-ane-lut6", repo_id))
        command = [executable, "upload", repo_id, str(root), ".", "--no-private",
                   "--commit-message", "Publish Clef-Flash ANE LUT-6 bundle"]
        for filename in [MANIFEST, "README.md", *manifest["files"]]:
            command.extend(["--include", filename])
        print(f"Publishing PUBLIC model {repo_id} ({len(manifest['files'])} runtime files)", flush=True)
        subprocess.run(command, check=True)
        # Verify public visibility and a fresh, anonymous manifest download after the upload.
        published = HfApi(token=False).model_info(repo_id)
        if published.private or not published.sha:
            raise RuntimeError("The uploaded repository is not publicly available.")
        remote = hf_hub_download(repo_id, MANIFEST, revision=published.sha, token=False, force_download=True)
        if read_json(Path(remote)) != manifest:
            raise RuntimeError("The published model manifest does not match the local bundle.")
        temporary = root / ".source.json.tmp"
        temporary.write_text(json.dumps({"repo_id": repo_id, "revision": published.sha}, indent=2) + "\n")
        temporary.replace(root / ".source.json")
    return repo_id, published.sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_dir())
    parser.add_argument("--name", default="clef-flash-ane-lut6")
    args = parser.parse_args()
    try:
        repo, revision = publish(args.model_dir, args.name)
    except Exception as error:
        parser.exit(1, f"publish: {error}\n")
    print(f"Published {repo} at {revision}")
    print("Before sharing a wheel, set DEFAULT_REPO_ID in src/clef_ane/paths.py to this repository and rebuild it.")
