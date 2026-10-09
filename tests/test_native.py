import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clef_ane.native import build_bridge, require_apple_silicon


class NativeTests(unittest.TestCase):
    def test_unsupported_platform_is_actionable(self):
        for system, machine in (("Linux", "x86_64"), ("Darwin", "x86_64")):
            with self.subTest(system=system, machine=machine), patch("platform.system", return_value=system), \
                    patch("platform.machine", return_value=machine), self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                require_apple_silicon()

    def test_bridge_is_built_atomically_and_reused(self):
        def compile(command, **kwargs):
            Path(command[-1]).write_bytes(b"test dylib")
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"CLEF_ANE_HOME": home}), \
                patch("clef_ane.native.require_apple_silicon"), patch("subprocess.run", side_effect=compile) as run:
            first = build_bridge()
            self.assertTrue(first.is_relative_to(Path(home).resolve()))
            self.assertEqual(first.read_bytes(), b"test dylib")
            self.assertEqual(build_bridge(), first)
            run.assert_called_once()
            self.assertIn("ane_bridge.m", run.call_args.args[0][-3])

    def test_prebuilt_bridge_is_used_without_a_compiler(self):
        with tempfile.TemporaryDirectory() as package:
            (Path(package) / "libane_bridge.dylib").write_bytes(b"prebuilt")
            with patch("clef_ane.native.files", return_value=Path(package)), \
                    patch("clef_ane.native.require_apple_silicon"), patch("subprocess.run") as run:
                self.assertEqual(build_bridge(), Path(package) / "libane_bridge.dylib")
            run.assert_not_called()

    def test_failed_compiler_does_not_leave_a_cached_library(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"CLEF_ANE_HOME": home}), \
                patch("clef_ane.native.require_apple_silicon"), \
                patch("subprocess.run", return_value=SimpleNamespace(returncode=1, stderr="compiler error")):
            with self.assertRaisesRegex(RuntimeError, "xcode-select"):
                build_bridge()
            self.assertFalse(list(Path(home).rglob("*.dylib")))
