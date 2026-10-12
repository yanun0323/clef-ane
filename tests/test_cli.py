import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clef_ane import paths
from clef_ane.cli import main, parser


class CLITests(unittest.TestCase):
    def test_installed_command_works_outside_the_checkout(self):
        command = Path(sys.executable).parent / "clef-ane"
        with tempfile.TemporaryDirectory() as cwd:
            for args in (["--version"], ["--help"], ["serve", "--help"], ["download", "--help"]):
                with self.subTest(args=args):
                    result = subprocess.run([str(command), *args], cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("clef-ane", result.stdout)

    def test_help_does_not_import_torch_or_native_runtime(self):
        code = """import sys
from clef_ane.cli import main
try:
    main(['serve', '--help'])
except SystemExit as e:
    assert e.code == 0
assert 'torch' not in sys.modules
assert 'clef_ane.engine.ane' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd="/", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_download_arguments_are_forwarded(self):
        with patch("clef_ane.download.download_model") as download:
            self.assertEqual(main(["download", "--repo", "person/model", "--revision", "v1", "--model-dir", "/tmp/model", "--force"]), 0)
            download.assert_called_once_with(Path("/tmp/model"), "person/model", "v1", force=True,
                                             default_repo="Yanun/clef-flash-ane-lut6")

    def test_model_option_selects_the_repository_and_directory(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"CLEF_ANE_HOME": home}):
            models = Path(home).resolve() / "models"
            with patch("clef_ane.download.download_model") as download:
                self.assertEqual(main(["download", "--model", "clef"]), 0)
                download.assert_called_once_with(models / "clef-ane-lut6", None, "main", force=False,
                                                 default_repo="Yanun/clef-ane-lut6")
            with patch("clef_ane.server.serve") as serve:
                self.assertEqual(main(["serve", "--model", "clef"]), 0)
                self.assertEqual(serve.call_args.args, (models / "clef-ane-lut6",))
                self.assertIsNone(serve.call_args.kwargs["name"])
                self.assertEqual(main(["serve"]), 0)
                self.assertEqual(serve.call_args.args, (models / "clef-flash-ane-lut6",))

    def test_serve_arguments_are_forwarded_without_download(self):
        with patch("clef_ane.server.serve") as serve, patch("clef_ane.download.download_model") as download:
            self.assertEqual(main(["serve", "--model-dir", "/tmp/model", "--port", "9999", "--max-tokens", "4096", "--name", "test"]), 0)
            serve.assert_called_once_with(Path("/tmp/model"), host="127.0.0.1", port=9999, max_tokens=4096, name="test", max_body_mb=32, vision_device="ane")
            download.assert_not_called()

    def test_invalid_arguments_fail_before_loading(self):
        for args in (["serve", "--port", "-1"], ["serve", "--port", "65536"], ["serve", "--max-tokens", "123"],
                     ["serve", "--model", "clef-pro"], ["download", "--model", "unknown"],
                     ["serve", "--max-body-mb", "0"], ["serve", "--vision-device", "cuda"], ["serve", "--port", "not-a-port"], [], ["unknown"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parser().parse_args(args)
            self.assertEqual(error.exception.code, 2)

    def test_errors_are_concise_and_nonzero(self):
        stream = io.StringIO()
        with patch("clef_ane.download.download_model", side_effect=RuntimeError("download interrupted")), contextlib.redirect_stderr(stream):
            self.assertEqual(main(["download", "--repo", "person/model"]), 1)
        self.assertIn("download interrupted", stream.getvalue())
        self.assertNotIn("Traceback", stream.getvalue())

    def test_keyboard_interrupt_returns_130(self):
        with patch("clef_ane.server.serve", side_effect=KeyboardInterrupt):
            self.assertEqual(main(["serve"]), 130)

    def test_home_override(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"CLEF_ANE_HOME": home}):
            self.assertEqual(paths.home(), Path(home).resolve())
            self.assertEqual(paths.model_dir(), paths.home() / "models" / "clef-flash-ane-lut6")
            self.assertEqual(paths.model_dir("clef"), paths.home() / "models" / "clef-ane-lut6")
