"""Writable files live in the user's cache, never beside the installed package."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .models import DEFAULT_MODEL, MODELS

# Public LUT-6 bundle of the default model. --model, --repo and CLEF_ANE_REPO can override it;
# download also remembers an explicitly selected repository.
DEFAULT_REPO_ID: str | None = MODELS[DEFAULT_MODEL].repo_id


def home() -> Path:
    if value := os.environ.get("CLEF_ANE_HOME"):
        return Path(value).expanduser().resolve()
    base = (Path.home() / "Library" / "Caches") if sys.platform == "darwin" else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "clef-ane"


def model_dir(model: str = DEFAULT_MODEL) -> Path:
    return home() / "models" / MODELS[model].directory
