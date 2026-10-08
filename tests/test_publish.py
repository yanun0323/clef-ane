import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.errors import LocalTokenNotFoundError

from clef_ane.bundle import MANIFEST, BundleError
from helpers import make_bundle

spec = importlib.util.spec_from_file_location("publish_bundle", Path(__file__).resolve().parents[1] / "scripts" / "publish_bundle.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = make_bundle(self.root)
        (self.root / "README.md").write_text("clef-ane download --repo OWNER/clef-flash-ane-lut6\n")
        (self.root / ".source.json").write_text("private local metadata")
        (self.root / "do-not-upload.txt").write_text("private local notes")
        self.api = self.enterContext(patch.object(publisher, "HfApi"))
        self.api.return_value.whoami.return_value = {"name": "actual-user"}
        self.api.return_value.model_info.return_value = SimpleNamespace(private=False, sha="b" * 40, siblings=[])
        self.enterContext(patch.object(publisher.shutil, "which", return_value="/test/bin/hf"))
        self.run = self.enterContext(patch.object(publisher.subprocess, "run"))
        self.file = self.enterContext(patch.object(publisher, "hf_hub_download", return_value=str(self.root / MANIFEST)))

    def test_publish_uses_authenticated_owner_and_an_explicit_file_list(self):
        repo, commit = publisher.publish(self.root)
        self.assertEqual(repo, "actual-user/clef-flash-ane-lut6")
        self.api.return_value.create_repo.assert_called_once_with(repo, private=False, exist_ok=True)
        command = self.run.call_args.args[0]
        included = {command[i + 1] for i, arg in enumerate(command) if arg == "--include"}
        self.assertEqual(included, {MANIFEST, "README.md", *self.manifest["files"]})
        self.assertNotIn(".source.json", included)
        self.assertNotIn(".clef-ane.lock", included)
        self.assertNotIn("do-not-upload.txt", included)
        self.assertIn("--no-private", command)
        self.file.assert_called_with(repo, MANIFEST, revision=commit, token=False, force_download=True)
        self.assertIn(repo, (self.root / "README.md").read_text())
        self.assertEqual(json.loads((self.root / ".source.json").read_text())["repo_id"], repo)

    def test_no_authentication_does_not_create_or_upload(self):
        self.api.return_value.whoami.side_effect = LocalTokenNotFoundError("no token")
        with self.assertRaisesRegex(RuntimeError, "hf auth login"):
            publisher.publish(self.root)
        self.api.return_value.create_repo.assert_not_called()
        self.run.assert_not_called()

    def test_existing_private_repository_is_not_changed(self):
        self.api.return_value.model_info.return_value.private = True
        with self.assertRaisesRegex(RuntimeError, "visibility was not changed"):
            publisher.publish(self.root)
        self.run.assert_not_called()

    def test_unrelated_model_is_not_overwritten(self):
        self.api.return_value.model_info.return_value.siblings = [SimpleNamespace(rfilename="model.safetensors")]
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            publisher.publish(self.root)
        self.run.assert_not_called()

    def test_incomplete_bundle_is_not_uploaded(self):
        (self.root / "weights/embed.safetensors").unlink()
        with self.assertRaises(BundleError):
            publisher.publish(self.root)
        self.api.return_value.create_repo.assert_not_called()
        self.run.assert_not_called()
