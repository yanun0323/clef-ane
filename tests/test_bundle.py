import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clef_ane.bundle import BundleError, load_bundle, safe_path, sha256, validate_manifest
from clef_ane.engine.mil import Graph, WeightBlob
from clef_ane.program import StoredProgram, save_program
from helpers import make_bundle, write_manifest


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = make_bundle(self.root)

    def test_complete_bundle_is_valid(self):
        self.assertEqual(load_bundle(self.root), self.manifest)

    def test_partial_download_is_rejected(self):
        (self.root / "weights/embed.safetensors").unlink()
        with self.assertRaisesRegex(BundleError, "Missing or incomplete"):
            load_bundle(self.root)

    def test_same_size_corruption_is_detected(self):
        (self.root / "weights/embed.safetensors").write_bytes(b"broken!\n")
        with self.assertRaisesRegex(BundleError, "SHA-256"):
            load_bundle(self.root)

    def test_manifest_cannot_contain_code_or_glob_patterns(self):
        for name in ("model.py", "../token", "/tmp/token", "programs/00/*.bin", "programs/00/a[1].bin", "a\\b"):
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][name] = {"size": 1, "sha256": "0" * 64}
                with self.assertRaises(BundleError):
                    validate_manifest(manifest)

    def test_symlinks_outside_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "private"
            external.write_text("not a model file")
            path = self.root / "weights/embed.safetensors"
            path.unlink()
            path.symlink_to(external)
            with self.assertRaisesRegex(BundleError, "outside"):
                load_bundle(self.root)

    def test_missing_context_program_is_rejected(self):
        self.manifest["layers"][3]["mixers"].pop("16384")
        with self.assertRaisesRegex(BundleError, "context"):
            validate_manifest(self.manifest)

    def test_unknown_format_is_rejected(self):
        self.manifest["format_version"] = 99
        with self.assertRaisesRegex(BundleError, "Unsupported model format"):
            validate_manifest(self.manifest)

    def test_six_bit_program_roundtrip_preserves_every_byte(self):
        graph = Graph()
        x = graph.input("input", (1, 32, 1, 32))
        indices = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32).astype(np.uint8) % 64
        tables = np.arange(2 * 64, dtype=np.float16).reshape(2, 64) * 0.001
        graph.output(graph.rename(graph.conv1x1_lut(x, "w", indices, tables, 6), "output"))
        name, weight = "programs/00/roundtrip.json", "programs/00/roundtrip.bin"
        save_program(self.root, name, graph, weight)
        for path in (name, weight):
            self.manifest["files"][path] = {"size": (self.root / path).stat().st_size, "sha256": sha256(self.root / path)}
        loaded = StoredProgram(self.root, name, self.manifest)
        self.assertEqual(loaded.text(), graph.text())
        self.assertEqual(loaded.inputs, graph.inputs)
        self.assertEqual(loaded.outputs, graph.outputs)
        self.assertEqual(loaded.weights, graph.weights)
        save_program(self.root, "programs/00/other.json", graph, weight)
        (self.root / weight).write_bytes(b"bad")
        with self.assertRaisesRegex(BundleError, "identical weights"):
            save_program(self.root, name, graph, weight)

    def test_six_bit_indices_are_packed_not_stored_as_bytes(self):
        blob = WeightBlob()
        offset = blob.add_indices(np.array([1, 2, 3, 63], dtype=np.uint8), 6)
        packed = int.from_bytes(blob.data[offset + 64:offset + 67], "little")
        self.assertEqual([(packed >> (i * 6)) & 63 for i in range(4)], [1, 2, 3, 63])

    def test_program_cannot_reference_unlisted_weight_file(self):
        path = self.root / "programs/00/test.json"
        spec = json.loads(path.read_text())
        spec["weight_file"] = "../../private.bin"
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(BundleError, "Missing packed weights"):
            StoredProgram(self.root, "programs/00/test.json", self.manifest)

    def test_model_card_can_change_without_invalidating_weights(self):
        (self.root / "README.md").write_text("Updated usage instructions")
        self.assertEqual(load_bundle(self.root), self.manifest)

    def test_image_bundle_requires_both_vision_files(self):
        manifest = make_bundle(self.root, with_images=True)
        self.assertEqual(load_bundle(self.root), manifest)
        for name in ("weights/vision.safetensors", "processor_config.json"):
            changed = copy.deepcopy(manifest)
            changed["files"].pop(name)
            with self.subTest(name=name), self.assertRaisesRegex(BundleError, "missing"):
                validate_manifest(changed)

    def test_vision_files_are_not_allowed_in_legacy_manifests(self):
        manifest = make_bundle(self.root, with_images=True)
        manifest["format_version"] = 1
        with self.assertRaisesRegex(BundleError, "Unexpected file"):
            validate_manifest(manifest)

    def test_image_bundle_detects_vision_weight_corruption(self):
        make_bundle(self.root, with_images=True)
        (self.root / "weights/vision.safetensors").write_bytes(b"broken!\n")
        with self.assertRaisesRegex(BundleError, "SHA-256"):
            load_bundle(self.root)

    def test_native_buffer_size_is_bounded(self):
        path = self.root / "programs/00/test.json"
        spec = json.loads(path.read_text())
        spec["inputs"][0]["shape"] = [1, 4096, 1, 2**30]
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(BundleError, "256 MiB"):
            StoredProgram(self.root, "programs/00/test.json", self.manifest)

    def test_native_buffer_count_is_bounded(self):
        path = self.root / "programs/00/test.json"
        spec = json.loads(path.read_text())
        spec["inputs"] *= 17
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(BundleError, "count"):
            StoredProgram(self.root, "programs/00/test.json", self.manifest)
