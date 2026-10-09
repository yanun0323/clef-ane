import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from clef_ane.bundle import IMAGE_FORMAT_VERSION, VISION_FILES, load_bundle, sha256
from helpers import make_bundle, processor_config, write_manifest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = script("export_bundle")
fetcher = script("fetch_vision")


class ImageExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.base, self.output = root / "source", root / "base", root / "output"
        self.manifest = make_bundle(self.base)
        self.manifest["source"] = {"repo_id": exporter.SOURCE_REPO, "revision": exporter.SOURCE_REVISION}
        config = {"text_config": {"hidden_size": 4096, "layer_types": [entry["kind"] for entry in self.manifest["layers"]]}}
        (self.source / "data/weights").mkdir(parents=True)
        for path in (self.source / "data/config.json", self.base / "config.json"):
            path.write_text(json.dumps(config))
        self.manifest["files"]["config.json"] = {"size": (self.base / "config.json").stat().st_size, "sha256": sha256(self.base / "config.json")}
        write_manifest(self.base, self.manifest)
        (self.source / "data/processor_config.json").write_text(json.dumps(processor_config()))
        (self.source / "data/weights/vision.safetensors").write_bytes(b"vision fixture\n")
        (self.source / "data/weights/vision.source.json").write_text(json.dumps({
            "repo_id": exporter.SOURCE_REPO, "revision": exporter.SOURCE_REVISION,
            "sha256": sha256(self.source / "data/weights/vision.safetensors")}))
        (self.base / ".source.json").write_text("private download metadata")
        (self.base / "credentials.txt").write_text("private")

    def test_upgrade_reuses_exact_programs_and_only_copies_allowlisted_files(self):
        with patch.object(exporter.layers, "mlp") as compile_program:
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        compile_program.assert_not_called()
        manifest = load_bundle(self.output)
        self.assertEqual(manifest["format_version"], IMAGE_FORMAT_VERSION)
        self.assertEqual(manifest["layers"], self.manifest["layers"])
        for name in self.manifest["files"]:
            if name.startswith("programs/"):
                self.assertEqual(manifest["files"][name], self.manifest["files"][name])
        self.assertEqual(set(manifest["files"]), set(self.manifest["files"]) | set(VISION_FILES))
        self.assertFalse((self.output / ".source.json").exists())
        self.assertFalse((self.output / "credentials.txt").exists())
        card = (self.output / "README.md").read_text()
        self.assertIn("| Vision encoder |", card)
        self.assertNotIn("clef-ane", card.lower())
        self.assertIn("Vision weights retain BF16 storage", (self.output / "NOTICE").read_text())
        self.assertEqual(load_bundle(self.base), self.manifest)

    def test_missing_vision_files_fails_before_copying(self):
        (self.source / "data/weights/vision.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "fetch-vision"):
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        self.assertFalse(self.output.exists())

    def test_wrong_base_revision_is_rejected(self):
        self.manifest["source"]["revision"] = "a" * 40
        write_manifest(self.base, self.manifest)
        with self.assertRaisesRegex(ValueError, "same pinned"):
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        self.assertFalse(self.output.exists())

    def test_unverified_vision_weights_are_rejected(self):
        (self.source / "data/weights/vision.safetensors").write_bytes(b"replaced weights\n")
        with self.assertRaisesRegex(ValueError, "provenance"):
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        self.assertFalse(self.output.exists())

    def test_processor_cannot_disable_the_resize_limit(self):
        config = processor_config()
        config["image_processor"]["do_resize"] = False
        (self.source / "data/processor_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "processor settings"):
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        self.assertFalse(self.output.exists())

    def test_nonempty_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            exporter.export(self.source, self.output, with_images=True, text_bundle=self.base)
        self.assertEqual(marker.read_bytes(), b"keep")


class RangeDownloadTests(unittest.TestCase):
    def response(self, raw, status=206, content_range="bytes 8-11/100"):
        response = Mock(status=status, headers={"Content-Range": content_range})
        response.read = io.BytesIO(raw).read
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_exact_range(self):
        with patch.object(fetcher.urllib.request, "urlopen", return_value=self.response(b"abcd")) as opened:
            self.assertEqual(fetcher.get_range("https://example.com/shard", 8, 12), b"abcd")
        self.assertEqual(opened.call_args.args[0].headers["Range"], "bytes=8-11")

    def test_ignored_or_wrong_range_is_rejected_before_body_read(self):
        for status, content_range in ((200, "bytes 8-11/100"), (206, "bytes 0-3/100"), (206, "")):
            response = self.response(b"abcd", status, content_range)
            response.read = Mock()
            with patch.object(fetcher.urllib.request, "urlopen", return_value=response), self.assertRaisesRegex(ValueError, "requested byte range"):
                fetcher.get_range("https://example.com/shard", 8, 12)
            response.read.assert_not_called()

    def test_short_or_extra_body_is_rejected(self):
        for raw in (b"ab", b"abcde"):
            with patch.object(fetcher.urllib.request, "urlopen", return_value=self.response(raw)), self.assertRaisesRegex(ValueError, "Incomplete"):
                fetcher.get_range("https://example.com/shard", 8, 12)
