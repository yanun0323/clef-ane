"""Build the small native bridge once, in a writable per-user cache."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

from filelock import FileLock

from . import paths


def require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Serving requires an Apple Silicon Mac and a native arm64 Python installation.")


def build_bridge() -> Path:
    require_apple_silicon()
    source = files("clef_ane.engine").joinpath("ane_bridge.m").read_bytes()
    digest = hashlib.sha256(source + platform.release().encode() + platform.machine().encode()).hexdigest()[:20]
    directory = paths.home() / "native" / digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "libane_bridge.dylib"
    with FileLock(str(directory / ".build.lock"), timeout=120):
        if target.is_file():
            return target
        print("Building the ANE bridge...", flush=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            src = Path(temporary) / "ane_bridge.m"
            output = Path(temporary) / target.name
            src.write_bytes(source)
            command = ["xcrun", "--sdk", "macosx", "clang", "-O2", "-fno-objc-arc", "-dynamiclib",
                       "-framework", "Foundation", "-framework", "IOSurface", str(src), "-o", str(output)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            except FileNotFoundError as error:
                raise RuntimeError("Install Xcode Command Line Tools with `xcode-select --install`, then run serve again.") from error
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Building the ANE bridge timed out. Check Xcode Command Line Tools and try again.") from error
            if result.returncode:
                raise RuntimeError("Could not build the ANE bridge. Check `xcode-select --install`.\n" + result.stderr.strip())
            output.replace(target)
    return target
