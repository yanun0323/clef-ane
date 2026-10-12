"""Clef's bundle layout: runs of linear-attention layers compiled into one program each, from export to loading."""
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from clef_ane.bundle import BLOCK_FORMAT_VERSION, IMAGE_FORMAT_VERSION, BundleError, load_bundle, sha256
from clef_ane.engine import layers
from clef_ane.engine.mil import T
from clef_ane.engine.runtime import BlockState
from clef_ane.program import StoredProgram
from helpers import processor_config

spec = importlib.util.spec_from_file_location("export_bundle", Path(__file__).resolve().parents[1] / "scripts" / "export_bundle.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

HIDDEN, LAYERS = 160, 8
SHAPES = {
    "linear_attention": {
        "input_layernorm.weight": (HIDDEN,), "linear_attn.in_proj_qkv.weight": (640, HIDDEN),
        "linear_attn.in_proj_z.weight": (384, HIDDEN), "linear_attn.in_proj_b.weight": (3, HIDDEN),
        "linear_attn.in_proj_a.weight": (3, HIDDEN), "linear_attn.conv1d.weight": (640, 1, 4), "linear_attn.dt_bias": (3,),
        "linear_attn.A_log": (3,), "linear_attn.norm.weight": (128,), "linear_attn.out_proj.weight": (HIDDEN, 384)},
    "full_attention": {
        "input_layernorm.weight": (HIDDEN,), "self_attn.q_proj.weight": (3072, HIDDEN), "self_attn.k_proj.weight": (256, HIDDEN),
        "self_attn.v_proj.weight": (256, HIDDEN), "self_attn.q_norm.weight": (256,), "self_attn.k_norm.weight": (256,),
        "self_attn.o_proj.weight": (HIDDEN, 1536)},
}
MLP = {"post_attention_layernorm.weight": (HIDDEN,), "mlp.gate_proj.weight": (544, HIDDEN), "mlp.up_proj.weight": (544, HIDDEN),
       "mlp.down_proj.weight": (HIDDEN, 544)}


def write_source(root: Path) -> dict:
    """A narrow 8-layer checkpoint with Clef's head ratios, in the layout fetch_all.py and fetch_vision.py write."""
    data = root / "data"
    (data / "weights").mkdir(parents=True)
    kinds = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)]
    config = {"text_config": {
        "layer_types": kinds, "hidden_size": HIDDEN, "intermediate_size": 544, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "head_dim": 256, "hidden_act": "silu", "rms_norm_eps": 1e-6, "attn_output_gate": True,
        "linear_num_key_heads": 1, "linear_num_value_heads": 3, "num_attention_heads": 6, "num_key_value_heads": 1, "vocab_size": 16,
        "rope_parameters": {"rope_theta": 10000000, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "mrope_interleaved": True}},
        "vision_config": {"depth": 27, "hidden_size": 1152, "num_heads": 16}}
    (data / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(0)
    for index, kind in enumerate(kinds):
        save_file({f"model.language_model.layers.{index}.{name}": torch.randn(shape, generator=generator) * 0.02
                   for name, shape in {**SHAPES[kind], **MLP}.items()}, data / "weights" / f"layer_{index:02d}.safetensors")
    for name in ("embed", "norm", "lm_head", "joint_head", "vision"):
        save_file({"weight": torch.zeros(2)}, data / "weights" / f"{name}.safetensors")
    (data / "weights/joint_head_config.json").write_text(json.dumps({"layers": 4, "width": 1024, "heads": 16, "routing_layers": 2}))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (data / name).write_text("{}")
    (data / "processor_config.json").write_text(json.dumps(processor_config()))
    (data / "weights/vision.source.json").write_text(json.dumps({
        "repo_id": exporter.SOURCE_REPO, "revision": exporter.SOURCE_REVISION, "sha256": sha256(data / "weights/vision.safetensors")}))
    return config


class FakeKernel:
    """Stands in for the ANE: remembers the program and reports its inputs and outputs."""
    loaded = []

    def __init__(self, program) -> None:
        self.inputs, self.outputs, self.closed = program.inputs, program.outputs, False
        FakeKernel.loaded.append(self)

    def close(self) -> None:
        self.closed = True


class BlockExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.config = write_source(cls.root)
        # 8 layers need 8 + 6 + 2 x 5 = 24 separate programs plus 27 for vision; a limit of 50 forces the block layout.
        with patch.object(exporter, "ANE_PROGRAM_LIMIT", 50):
            exporter.export(cls.root, cls.root / "blocks", with_images=True)
        exporter.export(cls.root, cls.root / "separate", with_images=True)
        cls.manifest = load_bundle(cls.root / "blocks")

    def programs(self, manifest: dict) -> set:
        return {name for name in manifest["files"] if name.startswith("programs/") and name.endswith(".json")}

    def test_a_model_over_the_program_limit_is_exported_in_blocks(self):
        self.assertEqual(self.manifest["format_version"], BLOCK_FORMAT_VERSION)
        first, second = "programs/00/block.json", "programs/04/block.json"
        self.assertEqual([layer.get("block") for layer in self.manifest["layers"]], [first] * 3 + [None] + [second] * 3 + [None])
        self.assertEqual(["mlp" in layer for layer in self.manifest["layers"]], [False] * 7 + [True])
        # 2 blocks, 2 x 5 context programs and the last MLP, against 24 with one program per mixer and MLP.
        self.assertEqual(len(self.programs(self.manifest)), 13)

    def test_a_model_within_the_limit_keeps_one_program_per_mixer_and_mlp(self):
        manifest = load_bundle(self.root / "separate")
        self.assertEqual(manifest["format_version"], IMAGE_FORMAT_VERSION)
        self.assertEqual([sorted(layer) for layer in manifest["layers"][2:4]], [["kind", "mixer", "mlp"], ["kind", "mixers", "mlp"]])
        self.assertEqual(len(self.programs(manifest)), 24)

    def test_the_previous_layers_mlp_opens_the_next_block(self):
        states = [T(f"{kind}_in{i}", shape) for i in range(3) for kind, shape in (("conv", (1, 640, 1, 3)), ("rec", (1, 3, 128, 128)))]
        opening = StoredProgram(self.root / "blocks", "programs/00/block.json", self.manifest)
        following = StoredProgram(self.root / "blocks", "programs/04/block.json", self.manifest)
        for program in (opening, following):
            self.assertEqual(program.inputs, [T("hid_in", (1, HIDDEN, 1, 128)), *states])
            self.assertEqual([t.name for t in program.outputs], ["hid_out", *(t.name.replace("_in", "_out") for t in states)])
        count = lambda program: program.text().count('const()[name = string("w_gate_up_')
        self.assertEqual((count(opening), count(following)), (3, 4))

    def test_blocks_require_the_vision_files(self):
        with patch.object(exporter, "ANE_PROGRAM_LIMIT", 50), self.assertRaisesRegex(ValueError, "--with-images"):
            exporter.export(self.root, self.root / "text-only")
        self.assertFalse((self.root / "text-only").exists())

    def test_missing_layer_weights_stop_the_export_before_it_writes(self):
        with tempfile.TemporaryDirectory() as other:
            write_source(Path(other))
            (Path(other) / "data/weights/layer_05.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "layer_05"):
                exporter.export(Path(other), Path(other) / "out", with_images=True)
            self.assertFalse((Path(other) / "out").exists())


class BlockLoadingTests(unittest.TestCase):
    """BundleBackbone on the exported bundle, with the ANE replaced by FakeKernel."""

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        write_source(cls.root)
        with patch.object(exporter, "ANE_PROGRAM_LIMIT", 50):
            exporter.export(cls.root, cls.root / "blocks", with_images=True)
        cls.bundle = cls.root / "blocks"
        cls.manifest = load_bundle(cls.bundle)
        # clef_ane.model imports the native bridge; a stub lets it load on machines without the ANE.
        saved = {name: sys.modules.pop(name, None) for name in ("clef_ane.model", "clef_ane.engine.ane")}
        sys.modules["clef_ane.engine.ane"] = SimpleNamespace(Kernel=FakeKernel)
        cls.model = importlib.import_module("clef_ane.model")
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    def backbone(self, **changes):
        FakeKernel.loaded = []
        tensors = {"model.language_model.embed_tokens.weight": torch.zeros(4, HIDDEN), "model.language_model.norm.weight": torch.zeros(HIDDEN)}
        manifest = {**self.manifest, **changes}
        with patch.object(self.model, "load_file", return_value=tensors):
            return self.model.BundleBackbone(self.bundle, manifest, max_tokens=2048, verbose=False)

    def test_each_block_is_one_program_and_only_the_last_layer_has_its_own_mlp(self):
        backbone = self.backbone()
        self.assertEqual([type(layer).__name__ for layer in backbone.layers], ["BlockState", "BundleLayer"] * 2)
        self.assertEqual([len(layer.state) for layer in backbone.layers[::2]], [6, 6])
        first, last = backbone.layers[1], backbone.layers[3]
        self.assertIsNone(first.mlp)
        self.assertIsNotNone(last.mlp)
        self.assertEqual((sorted(first.mixers), first.master_k.shape), ([1024, 2048], (1, 1, 2048, 256)))
        self.assertEqual(len(FakeKernel.loaded), 2 + 2 * 2 + 1)
        backbone.close()
        self.assertTrue(all(kernel.closed for kernel in FakeKernel.loaded))

    def test_a_manifest_that_disagrees_with_the_programs_is_rejected_and_unloaded(self):
        layers_ = [dict(layer) for layer in self.manifest["layers"]]
        layers_[2]["block"] = layers_[4]["block"]   # the first block program now covers two layers, the second four
        with self.assertRaisesRegex(BundleError, "does not match"):
            self.backbone(layers=layers_)
        self.assertTrue(FakeKernel.loaded and all(kernel.closed for kernel in FakeKernel.loaded))


class BlockStateTests(unittest.TestCase):
    def test_every_layers_state_is_carried_between_calls_and_cleared_between_requests(self):
        calls = []

        def kernel(hidden, *state):
            calls.append(state)
            return [hidden + 1, *(value + index + 1 for index, value in enumerate(state))]
        kernel.inputs = [T("hid_in", (1, 4, 1, 2)), T("conv_in0", (1, 2, 1, 3)), T("rec_in0", (1, 1, 2, 2)), T("conv_in1", (1, 2, 1, 3))]
        block = BlockState(kernel)
        hidden = block(np.zeros((1, 4, 1, 2), np.float16), 0)
        hidden = block(hidden, 2)
        self.assertEqual(float(hidden.max()), 2.0)
        self.assertEqual([float(value.max()) for value in calls[1]], [1.0, 2.0, 3.0])
        self.assertEqual([value.shape for value in block.state], [(1, 2, 1, 3), (1, 1, 2, 2), (1, 2, 1, 3)])
        block.reset()
        self.assertTrue(all(value.dtype == np.float16 and not value.any() for value in block.state))


if __name__ == "__main__":
    unittest.main()
