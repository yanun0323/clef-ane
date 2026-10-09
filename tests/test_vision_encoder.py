import unittest
from unittest.mock import patch

import numpy as np
import torch

from clef_ane.engine import vision_encoder
from clef_ane.engine.vision_encoder import MASK_NEG, block_mask, pack, vision_block


def block_weights(seed=0):
    rng = np.random.default_rng(seed)
    shapes = {"norm1.weight": (1152,), "norm1.bias": (1152,), "norm2.weight": (1152,), "norm2.bias": (1152,),
              "attn.qkv.weight": (3456, 1152), "attn.qkv.bias": (3456,), "attn.proj.weight": (1152, 1152),
              "attn.proj.bias": (1152,), "mlp.linear_fc1.weight": (4304, 1152), "mlp.linear_fc1.bias": (4304,),
              "mlp.linear_fc2.weight": (1152, 4304), "mlp.linear_fc2.bias": (1152,)}
    return {name: rng.standard_normal(shape).astype(np.float32) * .02 for name, shape in shapes.items()}


class PackingTests(unittest.TestCase):
    def test_whole_segments_fill_each_call(self):
        self.assertEqual(pack([0, 1024]), [(0, 1024, [(0, 1024)])])
        five_pairs = [0, 256, 512, 768, 1024, 1280]
        self.assertEqual(pack(five_pairs), [(0, 1024, [(0, 256), (256, 512), (512, 768), (768, 1024)]),
                                            (1024, 1280, [(0, 256)])])
        # 600 + 600 cannot share a call; images then a video pair stay in request order
        self.assertEqual(pack([0, 600, 1200, 1456]), [(0, 600, [(0, 600)]), (600, 1456, [(0, 600), (600, 856)])])

    def test_oversized_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 1024"):
            pack([0, 1025])

    def test_mask_isolates_segments_and_padding(self):
        mask = block_mask([(0, 2), (2, 5)], rows=8)[0, 0]
        self.assertEqual(mask.dtype, np.float16)
        allowed = mask == 0
        expected = np.zeros((8, 8), bool)
        expected[:2, :2] = expected[2:5, 2:5] = expected[5:, 5:] = True
        np.testing.assert_array_equal(allowed, expected)
        self.assertTrue(np.all(mask[~allowed] == np.float16(MASK_NEG)))


class GraphTests(unittest.TestCase):
    def test_block_program_has_fixed_io_and_dense_fp16_weights(self):
        with patch.object(vision_encoder.Graph, "default_quantize", lambda w: self.fail("vision weights must not use LUT")):
            graph = vision_block(block_weights(), rows=64)
        self.assertEqual([(t.name, t.shape) for t in graph.inputs],
                         [("hid_in", (1, 1152, 1, 64)), ("cos_in", (1, 1, 64, 72)), ("sin_in", (1, 1, 64, 72)),
                          ("mask_in", (1, 1, 64, 64))])
        self.assertEqual([(t.name, t.shape) for t in graph.outputs], [("hid_out", (1, 1152, 1, 64))])
        text = graph.text()
        self.assertNotIn("constexpr_lut_to_dense", text)
        self.assertIn("matmul", text)


class EncoderTests(unittest.TestCase):
    def test_packs_run_every_block_and_return_only_real_rows(self):
        encoder = object.__new__(vision_encoder.VisionEncoder)
        encoder.rows = 8
        calls = []

        class Kernel:
            def __call__(self, hid, cos, sin, mask):
                calls.append((hid.shape, mask.copy()))
                return [hid + np.float16(1)]
        encoder.kernels = [Kernel(), Kernel()]
        hidden = torch.arange(10 * 1152, dtype=torch.float32).reshape(10, 1152) % 7
        cos = sin = torch.zeros(10, 72)
        result = encoder.blocks(hidden, cos, sin, [0, 4, 8, 10])
        torch.testing.assert_close(result, hidden + 2)
        self.assertEqual(len(calls), 4)  # two packs x two blocks
        self.assertTrue(np.all(calls[2][1][0, 0, :2, :2] == 0))

    def test_overflow_and_nonfinite_inputs_fail_instead_of_returning_garbage(self):
        encoder = object.__new__(vision_encoder.VisionEncoder)
        encoder.rows = 8

        class Overflow:
            def __call__(self, hid, *rest):
                return [np.full_like(hid, np.inf)]
        encoder.kernels = [Overflow()]
        cos = sin = torch.zeros(4, 72)
        with self.assertRaisesRegex(RuntimeError, "--vision-device cpu"):
            encoder.blocks(torch.zeros(4, 1152), cos, sin, [0, 4])
        with self.assertRaisesRegex(RuntimeError, "fp16 range"):
            encoder.blocks(torch.full((4, 1152), 1e6), cos, sin, [0, 4])


if __name__ == "__main__":
    unittest.main()
