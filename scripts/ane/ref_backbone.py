"""Reference: the transformers text model (bf16), loaded from the per-layer files, and the Clef joint head on the CPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "data" / "weights"
if __package__:
    from clef_ane._vendor import joint_schema_model as clef
else:
    sys.path.insert(0, str(ROOT / "hf-meta"))
    import joint_schema_model as clef  # Cloudflare's own record encoder and head


class _LayerByLayer:
    """Stands in for the model's layer list: each decoder layer is read from disk, run, and dropped again.

    A 27B checkpoint (about 50 GiB in bf16) does not fit in this Mac's memory at once; one layer at a time does.
    The model's own forward still drives the layers, so the arithmetic is the upstream code's.
    """

    def __init__(self, config, weights: Path, device: str) -> None:
        self.config, self.weights, self.device = config, weights, device

    def __len__(self) -> int:
        return len(self.config.layer_types)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        return lambda hidden_states, **kwargs: self._run(key, hidden_states, **kwargs)

    def _run(self, index: int, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(self.config, index)
        prefix = f"model.language_model.layers.{index}."
        state = {k[len(prefix):]: v for k, v in load_file(self.weights / f"layer_{index:02d}.safetensors").items()}
        layer.load_state_dict(state, strict=True, assign=True)
        return layer.to(self.device).eval()(hidden_states, **kwargs)


def load_text_model(device: str = "mps", dtype: torch.dtype = torch.bfloat16, attention: str = "eager",
                    data: Path | None = None, layerwise: bool = False) -> Qwen3_5TextModel:
    """data: the model's research data directory (default: Clef-Flash's). layerwise: see _LayerByLayer."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    data = data or ROOT / "data"
    weights = data / "weights"
    config = Qwen3_5TextConfig(**json.loads((data / "config.json").read_text())["text_config"])
    config._attn_implementation = attention
    count = config.num_hidden_layers
    if layerwise:
        config.num_hidden_layers = 0          # build only the embedding table and the final norm
    torch.set_default_dtype(dtype)
    model = Qwen3_5TextModel(config)
    torch.set_default_dtype(torch.float32)
    state = {}
    for i in range(config.num_hidden_layers):
        for k, v in load_file(weights / f"layer_{i:02d}.safetensors").items():
            state[k.replace("model.language_model.", "")] = v
    state["norm.weight"] = load_file(weights / "norm.safetensors")["model.language_model.norm.weight"]
    state["embed_tokens.weight"] = load_file(weights / "embed.safetensors")["model.language_model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if layerwise:
        config.num_hidden_layers = count
        del model.layers
        model.layers = _LayerByLayer(config, weights, device)
    return model


class _Rows:
    """Row lookup into a bf16 embedding table that returns fp32, so the 1.3 GB table is never converted whole."""

    def __init__(self, table: torch.Tensor, dtype: torch.dtype) -> None:
        self.table, self.dtype = table, dtype

    def __getitem__(self, ids):
        return self.table[ids].to(self.dtype)


class Head:
    """The joint schema head, fed hidden states from either backbone.

    fp32 is the default: on the CPU a bf16 head fans out over every core and burns ~15x more CPU time than fp32
    (0.46 s vs 0.03 s per request) for the same answer; fp32 is also closer to the exact result.
    """

    def __init__(self, dtype: torch.dtype = torch.float32, model_dir: Path | None = None) -> None:
        model_dir = model_dir or ROOT / "data"
        weights = model_dir / "weights"
        config = json.loads((weights / "joint_head_config.json").read_text())
        self.module = clef.JointSchemaHead(**config)
        self.module.load_state_dict(load_file(weights / "joint_head.safetensors"), strict=True)
        self.module = self.module.to(dtype).eval()
        self.lm_head = _Rows(load_file(weights / "lm_head.safetensors")["lm_head.weight"], dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.dtype = dtype

    def encode(self, record: dict, max_length: int = 16384, *, processor=None):
        return clef.encode_record(self.tokenizer, record, max_length=max_length, processor=processor)

    @torch.no_grad()
    def probabilities(self, hidden: torch.Tensor, encoded) -> dict[str, np.ndarray]:
        ids = torch.tensor(encoded.input_ids)[None]
        mask = torch.ones_like(ids)
        logits = self.module(hidden.to(self.dtype)[None], ids, mask, [encoded], self.lm_head)[0]
        return {q.question_id: l.float().softmax(-1).numpy() for q, l in zip(encoded.questions, logits)}
