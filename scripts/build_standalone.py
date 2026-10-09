"""Build the standalone macOS arm64 bundle used by install.sh.

The bundle contains a pinned python-build-standalone Python 3.12, the exact dependencies in
packaging/requirements.lock (installed with --require-hashes), clef-ane itself, and a prebuilt
ANE bridge, so users need neither Python nor Xcode Command Line Tools. Models are not included.

Needs network access to GitHub and PyPI, Xcode Command Line Tools, and an Apple Silicon Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from importlib.metadata import PathDistribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261003/"
              "cpython-3.12.15%2B20261003-aarch64-apple-darwin-install_only_stripped.tar.gz")
PYTHON_SHA256 = "ad8d0c637c0a36b967b310e2c07254f4d2ca8cabaa7699e55ed6290aceb481a2"
ASSET = "clef-ane-macos-arm64.tar.gz"
MINIMUM_MACOS = "14.0"  # the oldest macOS any locked wheel supports
LAUNCHER = """#!/bin/sh
# clef-ane standalone launcher: runs the bundled Python in isolated mode.
here=$(cd "$(dirname "$0")/.." && pwd -P)
exec "$here/python/bin/python3" -I -m clef_ane "$@"
"""


def run(*command: str | Path) -> None:
    subprocess.run([str(part) for part in command], check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_python(cache: Path) -> Path:
    archive = cache / "python-3.12.15-aarch64-apple-darwin.tar.gz"
    if not archive.is_file() or sha256(archive) != PYTHON_SHA256:
        cache.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".partial")
        print("Downloading Python 3.12.15 (python-build-standalone)...", flush=True)
        with urllib.request.urlopen(PYTHON_URL, timeout=120) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        if sha256(partial) != PYTHON_SHA256:
            partial.unlink()
            raise RuntimeError("The downloaded Python archive does not match its pinned SHA-256.")
        partial.replace(archive)
    return archive


def notices(site: Path, ffmpeg: str) -> str:
    """Third-party components and where their license texts are inside the bundle."""
    rows = []
    for info in sorted(site.glob("*.dist-info")):
        meta = PathDistribution(info).metadata
        license_name = meta.get("License-Expression") or meta.get("License") or "see license files"
        files = sorted(str(p.relative_to(site.parent.parent.parent.parent)) for p in info.rglob("*") if p.is_file()
                       and ("licen" in p.name.lower() or "copying" in p.name.lower() or "notice" in p.name.lower()
                            or p.parent.name == "licenses"))
        rows.append(f"- {meta['Name']} {meta['Version']}: {license_name.splitlines()[0][:120]}"
                    + (f"\n  License files: {', '.join(files)}" if files else ""))
    return f"""clef-ane standalone bundle: third-party components
====================================================

This bundle redistributes unmodified upstream binaries. clef-ane's own license is in LICENSE.

Python 3.12.15 from python-build-standalone (release 20261003):
  python/lib/python3.12/LICENSE.txt and https://github.com/astral-sh/python-build-standalone

FFmpeg and codec libraries bundled inside PyAV (python/lib/python3.12/site-packages/av/.dylibs):
  FFmpeg {ffmpeg} libraries (libavcodec, libavformat, ...), configured as LGPL version 3 or later.
  libx264 and libx265 are licensed under the GNU GPL version 2 or later; libvpx, libdav1d,
  libSvtAv1Enc, libopus, libwebp and libvmaf use BSD-style licenses.
  clef-ane only decodes video and does not call the x264/x265 encoders, but their libraries are
  included because PyAV links them. Corresponding source code:
    https://github.com/PyAV-Org/pyav-ffmpeg (build scripts and pinned source versions)
    https://ffmpeg.org/download.html, https://code.videolan.org/videolan/x264,
    https://bitbucket.org/multicoreware/x265_git
  The GPL version 2 text is at https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.

Python packages (python/lib/python3.12/site-packages):
{chr(10).join(rows)}
"""


def build(output: Path, cache: Path) -> Path:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Build the standalone bundle on an Apple Silicon Mac.")
    stage = cache / "stage"
    bundle = stage / "clef-ane"
    shutil.rmtree(stage, ignore_errors=True)
    bundle.mkdir(parents=True)
    with tarfile.open(fetch_python(cache)) as archive:
        archive.extractall(bundle, filter="tar")  # creates bundle/python
    python = bundle / "python/bin/python3"
    site = bundle / "python/lib/python3.12/site-packages"

    print("Installing locked dependencies...", flush=True)
    run(python, "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--only-binary=:all:", "--require-hashes",
        "--no-compile", "-r", ROOT / "packaging/requirements.lock")
    wheels = cache / "wheel"
    shutil.rmtree(wheels, ignore_errors=True)
    shutil.rmtree(ROOT / "build/lib", ignore_errors=True)  # setuptools reuses stale in-tree build output
    run(python, "-I", "-m", "pip", "wheel", "--quiet", "--no-deps", "--no-build-isolation", "-w", wheels, ROOT)
    wheel, = wheels.glob("clef_ane-*.whl")
    run(python, "-I", "-m", "pip", "install", "--quiet", "--no-deps", "--no-compile", wheel)
    for module in ("clef_ane.server", "clef_ane.vision", "clef_ane.video", "clef_ane.engine.service", "clef_ane.engine.vision_encoder"):
        run(python, "-I", "-c", f"import {module}")
    run(python, "-I", "-m", "pip", "check")

    print("Building the ANE bridge...", flush=True)
    bridge = site / "clef_ane/engine/libane_bridge.dylib"
    run("xcrun", "--sdk", "macosx", "clang", "-O2", "-fno-objc-arc", "-dynamiclib", f"-mmacosx-version-min={MINIMUM_MACOS}",
        "-framework", "Foundation", "-framework", "IOSurface", site / "clef_ane/engine/ane_bridge.m", "-o", bridge)
    run("codesign", "--force", "--sign", "-", bridge)

    # Headers and CMake files are only for compiling against torch; pip is not needed at run time.
    for path in (site / "torch/include", site / "torch/share", site / "pip", bundle / "python/bin/pip",
                 bundle / "python/bin/pip3", bundle / "python/bin/pip3.12"):
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    for path in site.glob("pip-*.dist-info"):
        shutil.rmtree(path)
    run(python, "-I", "-m", "compileall", "-q", "-j", "0", site)

    (bundle / "bin").mkdir()
    launcher = bundle / "bin/clef-ane"
    launcher.write_text(LAUNCHER)
    launcher.chmod(0o755)
    shutil.copyfile(ROOT / "LICENSE", bundle / "LICENSE")
    ffmpeg = subprocess.run([python, "-I", "-c", "import av; print(av.ffmpeg_version_info)"], check=True,
                            capture_output=True, text=True).stdout.strip()
    (bundle / "THIRD_PARTY_NOTICES.txt").write_text(notices(site, ffmpeg))
    version = subprocess.run([launcher, "--version"], check=True, capture_output=True, text=True).stdout.split()[-1]
    (bundle / "VERSION").write_text(version + "\n")
    (bundle / "build.json").write_text(json.dumps({
        "version": version, "python": "3.12.15+20261003", "minimum_macos": MINIMUM_MACOS,
        "requirements_lock_sha256": sha256(ROOT / "packaging/requirements.lock")}, indent=2) + "\n")

    output.mkdir(parents=True, exist_ok=True)
    target = output / ASSET
    print(f"Compressing {target.name}...", flush=True)
    run("tar", "-czf", target, "-C", stage, "clef-ane")
    (output / f"{ASSET}.sha256").write_text(f"{sha256(target)}  {ASSET}\n")
    shutil.copyfile(ROOT / "install.sh", output / "install.sh")
    print(f"Bundle ready: {target} ({target.stat().st_size / 1024**2:.0f} MiB), version {version}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--cache", type=Path, default=ROOT / "build/standalone")
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.cache.resolve())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"build_standalone: {error}")
