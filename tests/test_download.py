import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from filelock import FileLock, Timeout

from clef_ane.bundle import MANIFEST, BundleError, load_bundle
from clef_ane.download import download_model
from helpers import make_bundle, write_manifest


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.remote, self.local = Path(self.temp.name) / "remote", Path(self.temp.name) / "local"
        self.manifest = make_bundle(self.remote)
        self.commit = "a" * 40
        self.info = self.enterContext(patch("clef_ane.download.HfApi"))
        self.info.return_value.model_info.return_value = SimpleNamespace(sha=self.commit)
        self.file = self.enterContext(patch("clef_ane.download.hf_hub_download", return_value=str(self.remote / MANIFEST)))
        self.snapshot = self.enterContext(patch("clef_ane.download.snapshot_download", side_effect=self.copy_snapshot))
        self.enterContext(patch("clef_ane.paths.DEFAULT_REPO_ID", None))
        self.enterContext(patch.dict(os.environ, {"CLEF_ANE_REPO": ""}))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def copy_snapshot(self, repo_id, **kwargs):
        for name in kwargs["allow_patterns"]:
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / name, path)
        return str(kwargs["local_dir"])

    def test_download_pins_one_commit_and_only_manifest_files(self):
        download_model(self.local, "owner/model", "release")
        self.info.return_value.model_info.assert_called_once_with("owner/model", revision="release")
        self.file.assert_called_once_with("owner/model", MANIFEST, revision=self.commit, force_download=False)
        self.assertEqual(self.snapshot.call_args.kwargs["revision"], self.commit)
        self.assertEqual(set(self.snapshot.call_args.kwargs["allow_patterns"]), {MANIFEST, "README.md", *self.manifest["files"]})
        self.assertEqual(load_bundle(self.local), self.manifest)
        self.assertEqual(json.loads((self.local / ".source.json").read_text()), {"repo_id": "owner/model", "revision": self.commit})

    def test_image_bundle_download_includes_and_verifies_vision_files(self):
        self.manifest = make_bundle(self.remote, with_images=True)
        download_model(self.local, "owner/images")
        self.assertEqual(load_bundle(self.local), self.manifest)
        included = self.snapshot.call_args.kwargs["allow_patterns"]
        self.assertIn("weights/vision.safetensors", included)
        self.assertIn("processor_config.json", included)

    def test_fresh_install_uses_the_default_repository(self):
        with patch("clef_ane.paths.DEFAULT_REPO_ID", "owner/release"):
            download_model(self.local)
        self.info.return_value.model_info.assert_called_once_with("owner/release", revision="main")
        self.assertEqual(self.snapshot.call_args.args[0], "owner/release")

    def test_explicit_repository_overrides_the_release_default(self):
        with patch("clef_ane.paths.DEFAULT_REPO_ID", "owner/release"):
            download_model(self.local, "owner/other")
        self.assertEqual(self.snapshot.call_args.args[0], "owner/other")

    def test_selected_model_repository_is_used_unless_one_is_named(self):
        with patch("clef_ane.paths.DEFAULT_REPO_ID", "owner/release"):
            download_model(self.local, default_repo="owner/larger")
            self.assertEqual(self.snapshot.call_args.args[0], "owner/larger")
            download_model(self.local, "owner/other", default_repo="owner/larger")
        self.assertEqual(self.snapshot.call_args.args[0], "owner/other")

    def test_second_download_remembers_the_explicit_repository(self):
        download_model(self.local, "owner/model")
        with patch("clef_ane.paths.DEFAULT_REPO_ID", "someone/default"):
            download_model(self.local)
        self.assertEqual(self.snapshot.call_args.args[0], "owner/model")

    def test_force_is_forwarded_to_both_downloads(self):
        download_model(self.local, "owner/model", force=True)
        self.assertTrue(self.file.call_args.kwargs["force_download"])
        self.assertTrue(self.snapshot.call_args.kwargs["force_download"])

    def test_unknown_model_format_stops_before_large_download(self):
        self.manifest["format"] = "other-runtime"
        write_manifest(self.remote, self.manifest)
        with self.assertRaises(BundleError):
            download_model(self.local, "owner/model")
        self.snapshot.assert_not_called()

    def test_interrupted_download_does_not_mark_model_ready_and_can_resume(self):
        def partial(*args, **kwargs):
            shutil.copyfile(self.remote / MANIFEST, self.local / MANIFEST)
            raise OSError("connection lost")
        self.snapshot.side_effect = partial
        with self.assertRaisesRegex(OSError, "connection lost"):
            download_model(self.local, "owner/model")
        self.assertFalse((self.local / ".source.json").exists())
        with self.assertRaises(BundleError):
            load_bundle(self.local)
        self.snapshot.side_effect = self.copy_snapshot
        download_model(self.local, "owner/model")
        self.assertTrue((self.local / ".source.json").exists())

    def test_corrupt_download_is_rejected(self):
        def corrupt(*args, **kwargs):
            self.copy_snapshot(*args, **kwargs)
            (self.local / "weights/embed.safetensors").write_bytes(b"broken!\n")
        self.snapshot.side_effect = corrupt
        with self.assertRaisesRegex(BundleError, "SHA-256"):
            download_model(self.local, "owner/model")
        self.assertFalse((self.local / ".source.json").exists())

    def test_model_lock_prevents_replacing_files_during_serving(self):
        self.local.mkdir()
        with FileLock(str(self.local / ".clef-ane.lock")), self.assertRaises(Timeout):
            download_model(self.local, "owner/model")
        self.info.return_value.model_info.assert_not_called()

    def test_missing_default_repository_is_actionable(self):
        with self.assertRaisesRegex(BundleError, "--repo"):
            download_model(self.local)
        self.info.assert_not_called()

    def test_repository_environment_override(self):
        with patch.dict(os.environ, {"CLEF_ANE_REPO": "owner/environment"}):
            download_model(self.local)
        self.assertEqual(self.snapshot.call_args.args[0], "owner/environment")
