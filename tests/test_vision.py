import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.video_processing_utils import BaseVideoProcessor

from clef_ane.bundle import BundleError
from clef_ane.engine.errors import RequestError
from clef_ane.server import load_service
from clef_ane.vision import ImageProcessor, Vision, image_processor_settings
from helpers import image_tokenizer, processor_config


class ProcessorTests(unittest.TestCase):
    def test_image_only_adapter_matches_stock_processor(self):
        tokenizer = image_tokenizer()
        settings = image_processor_settings(processor_config())
        # Skip only the unused video's type guard when optional torchvision is absent.
        # All image preprocessing and token expansion still execute in the stock processor.
        video = Mock(spec=BaseVideoProcessor, model_input_names=[], valid_kwargs=None)
        check = Qwen3VLProcessor.check_argument_for_proper_class
        def check_image_and_tokenizer(self, name, argument):
            return BaseVideoProcessor if name == "video_processor" else check(self, name, argument)
        with patch.object(Qwen3VLProcessor, "check_argument_for_proper_class", check_image_and_tokenizer):
            stock = Qwen3VLProcessor(Qwen2VLImageProcessorPil(**copy.deepcopy(settings)), tokenizer, video)
        image_only = ImageProcessor(Qwen2VLImageProcessorPil(**copy.deepcopy(settings)), tokenizer)
        images = [Image.new("RGB", (800, 400), "red"), Image.new("RGB", (256, 256), "blue")]
        for image in images:
            self.addCleanup(image.close)
        text = ["<|vision_start|><|image_pad|><|vision_end|>" * 2 + "\n"]
        expected = stock(text=text, images=images, return_tensors="pt")
        actual = image_only(text=text, images=images, return_tensors="pt")
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        video.assert_not_called()

    def test_processor_settings_are_bounded_without_mutating_the_source(self):
        config = processor_config()
        original = copy.deepcopy(config)
        settings = image_processor_settings(config)
        self.assertEqual(config, original)
        self.assertEqual(settings["size"]["longest_edge"], 512 * 512)
        self.assertTrue(settings["do_resize"])

    def test_invalid_settings_and_cap_bypasses_are_rejected(self):
        for key, value in (("do_resize", False), ("do_rescale", False), ("do_normalize", False),
                           ("patch_size", 14), ("image_mean", [0, 0, 0]), ("max_pixels", 100000000)):
            config = processor_config()
            config["image_processor"][key] = value
            with self.subTest(key=key), self.assertRaises(BundleError):
                image_processor_settings(config)
        for config in ([], {}, {"processor_class": "Qwen3VLProcessor", "image_processor": []}):
            with self.subTest(config=config), self.assertRaises(BundleError):
                image_processor_settings(config)


class VisionTests(unittest.TestCase):
    def test_device_selection_is_lazy_and_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "image_token_id": 1}))
            (root / "processor_config.json").write_text(json.dumps(processor_config()))
            tokenizer = image_tokenizer()
            for available, requested, expected in ((False, "auto", "ane"), (True, "auto", "ane"), (True, "ane", "ane"),
                                                   (True, "mps", "mps"), (True, "cpu", "cpu")):
                with patch("clef_ane.vision.torch.backends.mps.is_available", return_value=available):
                    vision = Vision(root, tokenizer, requested)
                    self.assertEqual(vision.device, expected)
                    self.assertIsNone(vision.model)  # no weight files even exist in this fixture
            with patch("clef_ane.vision.torch.backends.mps.is_available", return_value=False), self.assertRaisesRegex(BundleError, "--vision-device cpu"):
                Vision(root, tokenizer, "mps")
            with self.assertRaisesRegex(ValueError, "vision device"):
                Vision(root, tokenizer, "cuda")

    def fixture(self, features=None):
        vision = object.__new__(Vision)
        vision.device = "cpu"
        vision.config = SimpleNamespace(image_token_id=1, text_config=SimpleNamespace(hidden_size=4))
        features = torch.tensor([[100., 101., 102., 103.], [-4., -3., -2., -1.]]) if features is None else features
        vision.model = Mock(dtype=torch.float32, return_value=SimpleNamespace(pooler_output=features))
        encoded = SimpleNamespace(input_ids=(0, 1, 0, 1, 0), media={
            "pixel_values": torch.zeros(8, 1536), "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "token_offset": 0, "mm_token_type_ids": [0, 1, 0, 1, 0]})
        table = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        return vision, encoded, table

    def test_features_are_scattered_in_order_without_changing_the_table(self):
        vision, encoded, table = self.fixture()
        original = table.clone()
        embeddings, positions = vision.prepare(encoded, table)
        np.testing.assert_array_equal(embeddings[1], [100, 101, 102, 103])
        np.testing.assert_array_equal(embeddings[3], [-4, -3, -2, -1])
        np.testing.assert_array_equal(embeddings[[0, 2, 4]], table[[0, 0, 0]].numpy())
        np.testing.assert_array_equal(positions, np.tile(np.arange(5), (3, 1)))
        torch.testing.assert_close(table, original)

    def test_nonfinite_or_wrong_shape_features_are_rejected(self):
        for features in (torch.ones(3, 4), torch.full((2, 4), float("nan"))):
            vision, encoded, table = self.fixture(features)
            with self.assertRaisesRegex(RuntimeError, "invalid visual features"):
                vision.prepare(encoded, table)

    def test_placeholder_in_text_fails_before_loading_or_running_vision(self):
        vision, encoded, table = self.fixture()
        encoded.input_ids = (*encoded.input_ids, 1)
        with self.assertRaises(RequestError):
            vision.prepare(encoded, table)
        vision.model.assert_not_called()


class ImageServiceLoaderTests(unittest.TestCase):
    def test_loader_enables_images_only_for_v2(self):
        for version in (1, 2):
            backbone = Mock()
            module = SimpleNamespace(BundleBackbone=Mock(return_value=backbone))
            with patch.dict(sys.modules, {"clef_ane.model": module}), patch("clef_ane.engine.ref_backbone.Head") as head, \
                    patch("clef_ane.engine.service.Service") as service, patch("clef_ane.vision.Vision") as vision:
                load_service(Path("/model"), {"format_version": version}, 1024, "test", "cpu")
                if version == 2:
                    vision.assert_called_once_with(Path("/model"), head.return_value.tokenizer, device="cpu")
                    self.assertIs(service.call_args.kwargs["vision"], vision.return_value)
                else:
                    vision.assert_not_called()
                    self.assertIsNone(service.call_args.kwargs["vision"])
                backbone.close.assert_not_called()

    def test_loader_closes_ane_if_vision_configuration_fails(self):
        backbone = Mock()
        module = SimpleNamespace(BundleBackbone=Mock(return_value=backbone))
        with patch.dict(sys.modules, {"clef_ane.model": module}), patch("clef_ane.engine.ref_backbone.Head"), \
                patch("clef_ane.vision.Vision", side_effect=BundleError("invalid processor")), self.assertRaises(BundleError):
            load_service(Path("/model"), {"format_version": 2}, 1024, "test")
        backbone.close.assert_called_once()
