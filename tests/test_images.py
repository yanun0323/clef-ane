import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding

from clef_ane.engine.errors import RequestError, TooLong
from clef_ane.engine.layers import rope_tables
from clef_ane.engine.runtime import BackboneRunner, LayerState
from clef_ane.engine.service import Service
from clef_ane.vision import decode_images, image_positions


def data_url(image=None, format="PNG"):
    image = image or Image.new("RGB", (32, 32), "red")
    stream = io.BytesIO()
    image.save(stream, format=format)
    return f"data:image/{'jpeg' if format == 'JPEG' else format.lower()};base64," + base64.b64encode(stream.getvalue()).decode()


class ImageInputTests(unittest.TestCase):
    def test_supported_formats_decode_to_owned_rgb_images(self):
        for format in ("PNG", "JPEG", "WEBP"):
            with self.subTest(format=format):
                images = decode_images([data_url(format=format)])
                self.addCleanup(images[0].close)
                self.assertEqual(images[0].mode, "RGB")
                self.assertEqual(images[0].size, (32, 32))
                self.assertGreater(images[0].getpixel((0, 0))[0], 240)

    def test_empty_images(self):
        self.assertEqual(decode_images(None), [])
        self.assertEqual(decode_images([]), [])

    def test_paths_urls_objects_and_invalid_base64_are_rejected(self):
        for value in ("/etc/passwd", "file:///etc/passwd", "https://example.com/image.png", "http://127.0.0.1/a",
                      "data:image/svg+xml;base64,PHN2Zz4=", {"url": "https://example.com"}, None, 4,
                      "data:image/png;base64,%%%%", "data:image/png;base64,é", "data:image/png;base64,eA=="):
            with self.subTest(value=value), self.assertRaises(RequestError):
                decode_images([value])
        for value in ("not a list", {}, False):
            with self.subTest(value=value), self.assertRaises(RequestError):
                decode_images(value)

    def test_color_modes_are_normalized(self):
        for mode, format in (("RGBA", "PNG"), ("P", "PNG"), ("I;16", "PNG"), ("CMYK", "JPEG")):
            with self.subTest(mode=mode):
                images = decode_images([data_url(Image.new(mode, (32, 32)), format=format)])
                self.addCleanup(images[0].close)
                self.assertEqual(images[0].mode, "RGB")

    def test_truncated_image_is_a_request_error(self):
        image = Image.fromarray(np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
        raw = base64.b64decode(data_url(image).split(",", 1)[1])
        damaged = "data:image/png;base64," + base64.b64encode(raw[:len(raw) // 2]).decode()
        with self.assertRaises(RequestError):
            decode_images([damaged])

    def test_decoder_metadata_errors_are_request_errors(self):
        for error in (EOFError("truncated"), KeyError("bad metadata"), IndexError("bad offset")):
            with patch("clef_ane.vision.Image.open", side_effect=error), self.assertRaises(RequestError):
                decode_images([data_url()])

    def test_mime_must_match_bytes(self):
        with self.assertRaisesRegex(RequestError, "does not match"):
            decode_images([data_url().replace("image/png", "image/jpeg")])

    def test_image_count_is_checked_before_decoding(self):
        with patch("clef_ane.vision.Image.open") as opened, self.assertRaises(TooLong):
            decode_images(["invalid"] * 5)
        opened.assert_not_called()

    def test_compressed_size_is_checked_before_decoding(self):
        with patch("clef_ane.vision.MAX_IMAGE_BYTES", 8), patch("clef_ane.vision.Image.open") as opened, self.assertRaises(TooLong):
            decode_images([data_url()])
        opened.assert_not_called()

    def test_combined_compressed_size(self):
        url = data_url()
        size = len(base64.b64decode(url.split(",", 1)[1]))
        with patch("clef_ane.vision.MAX_TOTAL_BYTES", size), self.assertRaises(TooLong):
            decode_images([url, url])

    def test_pixel_limits_apply_before_loading_pixels(self):
        with patch("clef_ane.vision.MAX_IMAGE_PIXELS", 100), self.assertRaises(TooLong):
            decode_images([data_url()])
        with patch("clef_ane.vision.MAX_TOTAL_PIXELS", 1500), self.assertRaises(TooLong):
            decode_images([data_url(), data_url()])

    def test_animation_is_rejected(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(stream, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (32, 32), "blue")], duration=100, loop=0)
        with self.assertRaisesRegex(RequestError, "still image"):
            decode_images(["data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()])

    def test_extreme_aspect_ratio_is_rejected(self):
        with self.assertRaisesRegex(RequestError, "aspect ratio"):
            decode_images([data_url(Image.new("RGB", (201, 1)))])

    def test_exif_orientation_is_applied(self):
        image = Image.new("RGB", (32, 64), "red")
        exif = image.getexif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        images = decode_images(["data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()])
        self.addCleanup(images[0].close)
        self.assertEqual(images[0].size, (64, 32))

    def test_earlier_images_are_closed_if_a_later_image_fails(self):
        image = Image.new("RGB", (32, 32))
        with patch("clef_ane.vision.ImageOps.exif_transpose") as transpose:
            transpose.return_value.convert.return_value = image
            with patch.object(image, "close") as close, self.assertRaises(RequestError):
                decode_images([data_url(), "invalid"])
            close.assert_called_once()


class ImagePositionTests(unittest.TestCase):
    @staticmethod
    def upstream(ids, grids):
        class Positions:
            config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
            get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
            get_rope_index = Qwen3_5Model.get_rope_index
        tensor = torch.tensor([ids])
        positions, _ = Positions().get_rope_index(tensor, (tensor == 7).long(), image_grid_thw=torch.tensor(grids))
        return positions[:, 0].numpy()

    def test_positions_match_upstream_for_multiple_grids_and_chunk_boundaries(self):
        for prefix, grids, tail in ((12, [[1, 14, 14]], 19), (130, [[1, 16, 20], [1, 8, 32]], 23),
                                    (7, [[1, 64, 64]], 14000), (1, [[1, 2, 2]], 1)):
            ids = [1] * prefix
            for _, height, width in grids:
                ids += [7] * (height * width // 4) + [1, 1]
            ids += [1] * tail
            actual = image_positions(ids, np.array(grids), 7)
            np.testing.assert_array_equal(actual, self.upstream(ids, grids))
            config = Qwen3_5TextConfig(hidden_size=4096, head_dim=256, num_attention_heads=16,
                rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": .25,
                                 "mrope_section": [11, 11, 10]})
            rotary = Qwen3_5TextRotaryEmbedding(config)
            expected = rotary(torch.empty(1, len(ids), 1, dtype=torch.float16), torch.tensor(actual[:, None]))
            tables = rope_tables(0, len(ids), actual)
            for table, reference in zip(tables, expected):
                np.testing.assert_allclose(table[:, 0].astype(np.float32), reference.numpy().astype(np.float32), atol=.002, rtol=0)
            self.assertLess(actual[:, -1].max(), len(ids))

    def test_invalid_grids_and_placeholder_counts(self):
        for ids, grids in (([7] * 3, [[1, 4, 4]]), ([7] * 5, [[1, 4, 4]]), ([7, 1, 7, 7, 7], [[1, 4, 4]]),
                           ([7] * 4, [[2, 4, 4]]), ([7] * 4, [[1, 3, 4]])):
            with self.subTest(ids=ids, grids=grids), self.assertRaises(RequestError):
                image_positions(ids, grids, 7)

    def test_text_tables_remain_byte_identical(self):
        for start in (0, 128, 16256):
            inverse = 1.0 / 1e7 ** (np.arange(0, 64, 2) / 64)
            angle = np.arange(start, start + 128)[:, None] * inverse[None]
            full = np.concatenate([angle, angle], axis=-1)
            for actual, expected in zip(rope_tables(start, 128), (np.cos(full), np.sin(full))):
                np.testing.assert_array_equal(actual, expected[None, None].astype(np.float16))


class RuntimeEmbeddingTests(unittest.TestCase):
    def runner(self):
        runner = BackboneRunner()
        runner.rows, runner.max_tokens = 4, 8
        runner.config = {"hidden_size": 2, "rms_norm_eps": 1e-6}
        runner.embed = torch.tensor([[1., 0.], [0., 1.]])
        runner.norm_w = np.zeros(2, dtype=np.float32)
        runner.layers = []
        return runner

    def test_custom_embeddings_and_then_text_do_not_leak(self):
        runner = self.runner()
        ids = [0, 1, 0, 1, 0]
        baseline = runner.hidden(ids)
        features = np.tile([0., 2.], (len(ids), 1))
        positions = np.tile(np.arange(len(ids)), (3, 1))
        result = runner.hidden(ids, inputs_embeds=features, position_ids=positions)
        self.assertTrue(np.all(result[:, 0] == 0))
        self.assertTrue(np.all(result[:, 1] > 1))
        np.testing.assert_array_equal(runner.hidden(ids), baseline)

    def test_tables_are_padded_and_shared_across_layers(self):
        runner = self.runner()
        calls = []
        class Layer:
            def reset(self): pass
            def __call__(self, h, start, tables):
                calls.append((start, tables))
                return h
        runner.layers = [Layer(), Layer()]
        runner.hidden([0] * 5, inputs_embeds=np.ones((5, 2)), position_ids=np.zeros((3, 5), dtype=np.int64))
        self.assertEqual([start for start, _ in calls], [0, 4, 0, 4])
        self.assertEqual(calls[0][1][0].shape, (1, 1, 8, 64))
        self.assertTrue(all(tables is calls[0][1] for _, tables in calls))

    def test_invalid_embedding_inputs_fail_before_layers(self):
        runner = self.runner()
        for embeddings, positions in ((np.ones((1, 2)), None), (None, np.zeros((3, 1), dtype=int)),
                                      (np.ones((2, 2)), np.zeros((3, 1), dtype=int)),
                                      (np.array([[np.nan, 0.]]), np.zeros((3, 1), dtype=int)),
                                      (np.ones((1, 2)) * 1e8, np.zeros((3, 1), dtype=int)),
                                      (np.ones((1, 2)), np.zeros((3, 1), dtype=float))):
            with self.subTest(embeddings=embeddings, positions=positions), self.assertRaises(ValueError):
                runner.hidden([0], inputs_embeds=embeddings, position_ids=positions)

    def test_layer_slices_image_tables_without_changing_cache_offsets(self):
        layer = LayerState()
        layer.kind, layer.rows, layer.buckets, layer.active = "full_attention", 4, (8,), 8
        layer.master_k = np.zeros((1, 4, 8, 256), dtype=np.float16)
        layer.master_v = np.zeros_like(layer.master_k)
        kernel = Mock(return_value=(np.zeros((1, 2, 1, 4)), np.zeros((1, 4, 4, 256)), np.zeros((1, 4, 4, 256))))
        layer.mixers = {8: kernel}
        layer.mlp = Mock(side_effect=lambda x: [x])
        tables = (np.arange(512).reshape(1, 1, 8, 64), np.arange(512).reshape(1, 1, 8, 64) * -1)
        layer(np.zeros((1, 2, 1, 4)), 4, tables)
        np.testing.assert_array_equal(kernel.call_args.args[3], tables[0][:, :, 4:8])
        np.testing.assert_array_equal(kernel.call_args.args[4], tables[1][:, :, 4:8])
        self.assertEqual(kernel.write_region.call_args_list[0].args[1], 4 * 256)


class ImageServiceTests(unittest.TestCase):
    request = {"state": "test", "questions": {"q": {"type": "noul", "instructions": "Is it red?"}}}

    def test_text_bundle_rejects_images_and_videos_before_model_execution(self):
        head, backbone = Mock(), Mock()
        service = Service(backbone, head)
        for extra in ({"images": [data_url()]}, {"videos": ["movie.mp4"]}, {"media_kwargs": {}}, {"images": "invalid"}):
            with self.subTest(extra=extra), self.assertRaises(RequestError):
                service.systemone({**self.request, **extra})
        head.encode.assert_not_called()
        backbone.hidden.assert_not_called()

    def test_image_request_passes_embeddings_and_positions_then_text_resets(self):
        encoded = SimpleNamespace(input_ids=(1, 2), questions=(SimpleNamespace(question_id="q", option_ids=("true", "false")),))
        head = Mock()
        head.encode.return_value = encoded
        head.probabilities.return_value = {"q": np.array([.9, .1])}
        backbone = Mock(max_tokens=1024)
        backbone.hidden.return_value = np.zeros((2, 2), dtype=np.float32)
        vision = Mock()
        embeddings, positions = np.ones((2, 2)), np.zeros((3, 2), dtype=int)
        vision.prepare.return_value = embeddings, positions
        service = Service(backbone, head, vision=vision)
        request = {**self.request, "images": [data_url()]}
        service.systemone(request)
        self.assertIsInstance(request["images"][0], str)
        self.assertIs(head.encode.call_args.kwargs["processor"], vision.processor)
        self.assertIs(backbone.hidden.call_args.kwargs["inputs_embeds"], embeddings)
        self.assertIs(backbone.hidden.call_args.kwargs["position_ids"], positions)
        service.systemone(self.request)
        self.assertEqual(backbone.hidden.call_args.kwargs, {})
        vision.prepare.assert_called_once()

    def test_processor_errors_are_400_not_context_overflow(self):
        head, backbone, vision = Mock(), Mock(max_tokens=1024), Mock()
        head.encode.side_effect = ValueError("image processor error")
        with self.assertRaises(RequestError) as raised:
            Service(backbone, head, vision=vision).systemone({**self.request, "images": [data_url()]})
        self.assertNotIsInstance(raised.exception, TooLong)
        backbone.hidden.assert_not_called()

    def test_schema_overflow_precedes_vision_and_ane_execution(self):
        head, backbone, vision = Mock(), Mock(max_tokens=1024), Mock()
        head.encode.side_effect = ValueError("schema requires too many tokens")
        with self.assertRaises(TooLong):
            Service(backbone, head, vision=vision).systemone({**self.request, "images": [data_url()]})
        vision.prepare.assert_not_called()
        backbone.hidden.assert_not_called()
