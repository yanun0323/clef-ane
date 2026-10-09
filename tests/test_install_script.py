import hashlib
import io
import os
import platform
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = "clef-ane-macos-arm64.tar.gz"


def fake_release(directory: Path, version: str, *, damaged: bool = False, broken: bool = False) -> None:
    """A tiny release with the same layout as the standalone bundle."""
    launcher = "#!/bin/sh\nexit 3\n" if broken else f"#!/bin/sh\necho clef-ane {version}\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text, mode in (("clef-ane/bin/clef-ane", launcher, 0o755), ("clef-ane/VERSION", version + "\n", 0o644)):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    (directory / ASSET).write_bytes(raw)
    digest = hashlib.sha256(raw if not damaged else raw + b"x").hexdigest()
    (directory / f"{ASSET}.sha256").write_text(f"{digest}  {ASSET}\n")


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "installer targets Apple Silicon")
class InstallScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.release, self.install, self.bin = root / "release", root / "home/share/clef-ane", root / "home/bin"
        self.release.mkdir()

    def run_installer(self, on_path=True):
        prefix = f"{self.bin}:" if on_path else ""
        env = {"PATH": f"{prefix}/usr/bin:/bin:/usr/sbin:/sbin", "HOME": self.temp.name,
               "CLEF_ANE_DOWNLOAD_URL": self.release.as_uri(), "CLEF_ANE_INSTALL_DIR": str(self.install),
               "CLEF_ANE_BIN_DIR": str(self.bin)}
        script = (ROOT / "install.sh").read_text()
        return subprocess.run(["/bin/sh"], input=script, capture_output=True, text=True, env=env, timeout=60)

    def command(self):
        return subprocess.run([self.bin / "clef-ane", "--version"], capture_output=True, text=True).stdout.strip()

    def test_install_then_upgrade_through_a_pipe(self):
        fake_release(self.release, "1.0.0")
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.command(), "clef-ane 1.0.0")
        self.assertIn("clef-ane download", result.stdout)
        fake_release(self.release, "1.1.0")
        self.assertEqual(self.run_installer().returncode, 0)
        self.assertEqual(self.command(), "clef-ane 1.1.0")
        self.assertEqual(sorted(p.name for p in self.install.parent.iterdir()), ["clef-ane"])  # no staging leftovers

    def test_bad_checksum_or_broken_program_keeps_the_existing_install(self):
        fake_release(self.release, "1.0.0")
        self.assertEqual(self.run_installer().returncode, 0)
        for options in ({"damaged": True}, {"broken": True}):
            with self.subTest(**options):
                fake_release(self.release, "2.0.0", **options)
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("clef-ane install:", result.stderr)
                self.assertEqual(self.command(), "clef-ane 1.0.0")
                self.assertEqual(sorted(p.name for p in self.install.parent.iterdir()), ["clef-ane"])

    def test_an_unrelated_command_is_not_overwritten(self):
        fake_release(self.release, "1.0.0")
        self.bin.mkdir(parents=True)
        (self.bin / "clef-ane").write_text("#!/bin/sh\necho someone else\n")
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not created by this installer", result.stderr)
        self.assertFalse(self.install.exists())
        self.assertIn("someone else", (self.bin / "clef-ane").read_text())

    def test_path_hint_when_the_command_folder_is_not_on_path(self):
        fake_release(self.release, "1.0.0")
        result = self.run_installer(on_path=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Add {self.bin} to your PATH", result.stdout)
        self.assertNotIn("to your PATH", self.run_installer().stdout)


if __name__ == "__main__":
    unittest.main()
