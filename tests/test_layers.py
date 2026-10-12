"""The layer builders and request state take their sizes from the weights, so Clef-Flash and Clef share them."""
import unittest
from types import SimpleNamespace

import numpy as np

from clef_ane.engine import layers
from clef_ane.engine.mil import T
from clef_ane.engine.runtime import LayerState


def weights(shapes: dict) -> dict:
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) * 0.02 for name, shape in shapes.items()}


def shapes(graph) -> dict:
    return {t.name: t.shape for t in (*graph.inputs, *graph.outputs)}


def text_config(**changes) -> dict:
    kinds = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
    return {"layer_types": kinds, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "head_dim": 256, "hidden_act": "silu", "rms_norm_eps": 1e-6, "attn_output_gate": True,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "num_attention_heads": 24, "num_key_value_heads": 4,
            "rope_parameters": {"rope_theta": 10000000, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                                "mrope_interleaved": True}, **changes}


class LayerShapeTests(unittest.TestCase):
    """A narrow model with Clef's ratios: 3 value heads per key head, 6 query heads per key-value head, MLP 3.4x wide."""
    hidden = 160

    def test_linear_attention_state_follows_the_head_counts(self):
        graph = layers.gdn_mixer(weights({
            "input_layernorm.weight": (self.hidden,), "linear_attn.in_proj_qkv.weight": (640, self.hidden),
            "linear_attn.in_proj_z.weight": (384, self.hidden), "linear_attn.in_proj_b.weight": (3, self.hidden),
            "linear_attn.in_proj_a.weight": (3, self.hidden), "linear_attn.conv1d.weight": (640, 1, 4),
            "linear_attn.dt_bias": (3,), "linear_attn.A_log": (3,), "linear_attn.norm.weight": (128,),
            "linear_attn.out_proj.weight": (self.hidden, 384)}), rows=64)
        self.assertEqual(shapes(graph), {"hid_in": (1, 160, 1, 64), "conv_in": (1, 640, 1, 3), "rec_in": (1, 3, 128, 128),
                                         "hid_out": (1, 160, 1, 64), "conv_out": (1, 640, 1, 3), "rec_out": (1, 3, 128, 128)})

    def test_mlp_down_projection_is_cut_into_equal_slices_no_wider_than_the_hidden_size(self):
        graph = layers.mlp(weights({"post_attention_layernorm.weight": (self.hidden,), "mlp.gate_proj.weight": (544, self.hidden),
                                    "mlp.up_proj.weight": (544, self.hidden), "mlp.down_proj.weight": (self.hidden, 544)}), rows=64)
        self.assertEqual(shapes(graph), {"hid_in": (1, 160, 1, 64), "hid_out": (1, 160, 1, 64)})
        text = graph.text()
        self.assertEqual([f"w_down{i}_" in text for i in range(5)], [True] * 4 + [False])
        self.assertIn("tensor<fp16, [160, 136, 1, 1]>", text)

    def test_full_attention_cache_follows_the_key_value_heads(self):
        graph = layers.full_attention(weights({
            "input_layernorm.weight": (self.hidden,), "self_attn.q_proj.weight": (2 * 6 * 256, self.hidden),
            "self_attn.k_proj.weight": (256, self.hidden), "self_attn.v_proj.weight": (256, self.hidden),
            "self_attn.q_norm.weight": (256,), "self_attn.k_norm.weight": (256,),
            "self_attn.o_proj.weight": (self.hidden, 6 * 256)}), rows=8, ctx=64)
        found = shapes(graph)
        self.assertEqual((found["k_cache"], found["v_cache"], found["k_out"]), ((1, 1, 64, 256), (1, 1, 64, 256), (1, 1, 8, 256)))
        self.assertEqual(layers.own_mask(8, 6).shape, (1, 1, 48, 8))

    def test_request_state_is_sized_by_the_loaded_programs(self):
        layer = LayerState()
        layer.kind = "linear_attention"
        layer.mixer = SimpleNamespace(inputs=[T("hid_in", (1, 5120, 1, 128)), T("conv_in", (1, 10240, 1, 3)), T("rec_in", (1, 48, 128, 128))])
        layer.reset()
        self.assertEqual((layer.conv.shape, layer.rec.shape), ((1, 10240, 1, 3), (1, 48, 128, 128)))
        layer.kind, layer.buckets = "full_attention", (1024, 2048)
        layer.mixers = {size: SimpleNamespace(inputs=[T("k_cache", (1, 4, size, 256))]) for size in layer.buckets}
        layer.allocate_cache()
        self.assertEqual((layer.master_k.shape, layer.master_v.shape), ((1, 4, 2048, 256),) * 2)


class ConfigTests(unittest.TestCase):
    def test_both_published_architectures_are_supported(self):
        layers.check_config(text_config())
        layers.check_config(text_config(layer_types=text_config()["layer_types"][:32], linear_num_value_heads=32, num_attention_heads=16))

    def test_unsupported_architectures_fail_before_export(self):
        rope = text_config()["rope_parameters"]
        for change in ({"head_dim": 128}, {"linear_key_head_dim": 64}, {"linear_conv_kernel_dim": 3}, {"hidden_act": "gelu"},
                       {"linear_num_value_heads": 40}, {"num_attention_heads": 18}, {"attn_output_gate": False},
                       {"layer_types": ["linear_attention"] * 64}, {"rope_parameters": {**rope, "rope_theta": 10000}},
                       {"rope_parameters": {**rope, "mrope_section": [16, 8, 8]}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not supported"):
                layers.check_config(text_config(**change))


if __name__ == "__main__":
    unittest.main()
