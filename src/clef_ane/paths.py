"""Writable files live in the user's cache, never beside the installed package."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Public LUT-6 bundle. --repo and CLEF_ANE_REPO can override it;
# download also remembers an explicitly selected repository.
DEFAULT_REPO_ID: str | None = "Yanun/clef-flash-ane-lut6"


def home() -> Path:
    if value := os.environ.get("CLEF_ANE_HOME"):
        return Path(value).expanduser().resolve()
    base = (Path.home() / "Library" / "Caches") if sys.platform == "darwin" else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "clef-ane"


def model_dir() -> Path:
    return home() / "models" / "clef-flash-ane-lut6"
