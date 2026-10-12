"""The Clef models this runtime can serve: where each bundle is published and what it was converted from."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str             # --model value, and the model name the API reports by default
    title: str            # upstream product name
    base: str             # the model Cloudflare post-trained it from
    repo_id: str          # published LUT-6 bundle
    source_repo: str      # original checkpoint
    source_revision: str  # pinned commit the bundle was converted from

    @property
    def directory(self) -> str:
        return f"{self.name}-ane-lut6"


DEFAULT_MODEL = "clef-flash"
MODELS = {model.name: model for model in (
    ModelInfo("clef-flash", "Clef-Flash", "Qwen3.5-9B", "Yanun/clef-flash-ane-lut6",
              "Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6"),
    ModelInfo("clef", "Clef", "Qwen3.8-27B", "Yanun/clef-ane-lut6",
              "Cloudflare/clef", "0b331204bb13fbd2ca93a64df1956f5b55478ce5"),
)}


def data_dir(root, model: ModelInfo):
    """Maintainer checkout layout for original weights: data/ for the default model, data/<name>/ for the others."""
    return root / "data" if model.name == DEFAULT_MODEL else root / "data" / model.name
