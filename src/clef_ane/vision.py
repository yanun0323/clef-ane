"""Bounded visual input and the upstream image/video encoder; no downloads or remote model code."""
from __future__ import annotations

import base64
import binascii
import io
import itertools
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import ProcessorMixin

from .bundle import BundleError, read_json
from .engine.errors import RequestError, TooLong

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_TOTAL_PIXELS = 32_000_000
# 16-pixel patches, merged 2x2: at most about 256 language-model tokens per image.
MAX_PROCESSED_PIXELS = 512 * 512
IMAGE_FORMATS = {"data:image/png;base64,": "PNG", "data:image/jpeg;base64,": "JPEG", "data:image/webp;base64,": "WEBP"}


def decode_images(values) -> list[Image.Image]:
    """Only embedded PNG/JPEG/WebP data, never URLs, paths, arbitrary objects, or animation."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise RequestError("images must be a list of base64 image data URLs")
    if len(values) > MAX_IMAGES:
        raise TooLong(f"send at most {MAX_IMAGES} images per request")
    images, total_bytes, total_pixels = [], 0, 0
    try:
        for index, value in enumerate(values):
            prefix = next((prefix for prefix in IMAGE_FORMATS if isinstance(value, str) and value.startswith(prefix)), None)
            if prefix is None:
                raise RequestError(f"images[{index}]: send a PNG, JPEG, or WebP data URL; file paths and remote URLs are not accepted")
            payload = value[len(prefix):]
            if len(payload) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
                raise TooLong(f"images[{index}]: compressed image exceeds 4 MiB")
            try:
                raw = base64.b64decode(payload, validate=True)
            except (ValueError, binascii.Error) as error:
                raise RequestError(f"images[{index}]: invalid base64 image data") from error
            total_bytes += len(raw)
            if len(raw) > MAX_IMAGE_BYTES or total_bytes > MAX_TOTAL_BYTES:
                raise TooLong("images exceed the 4 MiB per-image or 8 MiB combined size limit")
            try:
                with Image.open(io.BytesIO(raw), formats=tuple(IMAGE_FORMATS.values())) as image:
                    if image.format != IMAGE_FORMATS[prefix]:
                        raise RequestError(f"images[{index}]: data URL type does not match the image")
                    pixels = image.width * image.height
                    total_pixels += pixels
                    if pixels > MAX_IMAGE_PIXELS or total_pixels > MAX_TOTAL_PIXELS:
                        raise TooLong("images exceed the 16 megapixel per-image or 32 megapixel combined limit")
                    if getattr(image, "is_animated", False):
                        raise RequestError(f"images[{index}]: send a still image; animated images are not supported")
                    if min(image.size) < 1 or max(image.size) / min(image.size) > 200:
                        raise RequestError(f"images[{index}]: image aspect ratio must not exceed 200:1")
                    # Load while the size checks and error handling are still in effect.
                    image.load()
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            except RequestError:
                raise
            except Image.DecompressionBombError as error:
                raise TooLong(f"images[{index}]: image dimensions exceed the pixel limit") from error
            except MemoryError:
                raise
            except Exception as error:  # malformed EXIF and compressed streams use several decoder exception types
                raise RequestError(f"images[{index}]: invalid or incomplete PNG, JPEG, or WebP image") from error
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


class ImageProcessor(Qwen3VLProcessor):
    """The upstream Qwen3VL image path without its mandatory, unused video processor.

    ProcessorMixin derives the sub-processors from this constructor's arguments.
    Token expansion, token types, and patch processing remain upstream implementations.
    """

    def __init__(self, image_processor, tokenizer):
        ProcessorMixin.__init__(self, image_processor, tokenizer)
        self.image_token = "<|image_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)


class MediaProcessor(ImageProcessor):
    """Keep the upstream token expansion, including the pinned version's outer video wrapper."""

    def __init__(self, image_processor, tokenizer):
        from .video import VideoProcessor
        super().__init__(image_processor, tokenizer)
        self.video_processor = VideoProcessor()
        self.video_token = "<|video_pad|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.vision_start_token, self.vision_end_token = "<|vision_start|>", "<|vision_end|>"

    def _process_videos(self, videos, **kwargs):
        processed = self.video_processor(videos, **kwargs)
        return processed, [self.replace_video_token(processed, index) for index in range(len(videos))]

    @property
    def model_input_names(self):
        return list(dict.fromkeys([*super().model_input_names, *self.video_processor.model_input_names]))


def media_positions(token_types, image_grid_thw=None, video_grid_thw=None, merge_size: int = 2) -> np.ndarray:
    """Qwen3.5 positions: timestamp text separates each temporal pair into its own grid."""
    types = np.asarray(token_types)
    if types.ndim != 1 or not np.issubdtype(types.dtype, np.integer) or np.any((types < 0) | (types > 2)):
        raise RequestError("invalid media token types")
    grids = {}
    for kind, value in ((1, image_grid_thw), (2, video_grid_thw)):
        rows = np.asarray(value) if value is not None else np.empty((0, 3), dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != 3 or not np.issubdtype(rows.dtype, np.integer):
            raise RequestError("invalid media patch grid")
        expanded = []
        for temporal, height, width in rows:
            if (not 1 <= temporal <= len(types) or (kind == 1 and temporal != 1)
                    or min(height, width) < merge_size or height % merge_size or width % merge_size):
                raise RequestError("invalid media patch grid")
            expanded.extend([(int(height // merge_size), int(width // merge_size))] * int(temporal))
        grids[kind] = iter(expanded)
    positions = np.empty((3, len(types)), dtype=np.int64)
    cursor = current = 0
    for kind, group in itertools.groupby(types.tolist()):
        length = sum(1 for _ in group)
        if kind == 0:
            positions[:, cursor:cursor + length] = np.arange(length)[None] + current
            current += length
        else:
            grid = next(grids[kind], None)
            if grid is None or grid[0] * grid[1] != length:
                raise RequestError("media features do not match the placeholders; remove special media tokens from state and questions")
            height, width = grid
            positions[:, cursor:cursor + length] = np.stack((np.zeros(length, dtype=np.int64),
                np.repeat(np.arange(height), width), np.tile(np.arange(width), height))) + current
            current += max(height, width)
        cursor += length
    if any(next(iterator, None) is not None for iterator in grids.values()):
        raise RequestError("media features have no matching placeholders")
    return positions


def image_positions(input_ids, grid_thw, image_token_id: int, merge_size: int = 2) -> np.ndarray:
    return media_positions((np.asarray(input_ids) == image_token_id).astype(np.int64), grid_thw, merge_size=merge_size)


def image_processor_settings(settings: dict) -> dict:
    """Accept the pinned preprocessing recipe, but never configuration that bypasses the pixel cap."""
    if not isinstance(settings, dict):
        raise BundleError("Invalid image processor settings. Use the processor_config.json pinned with the model.")
    image = settings.get("image_processor")
    expected = {"image_processor_type": "Qwen2VLImageProcessor", "do_convert_rgb": True, "do_normalize": True,
                "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "resample": 3, "rescale_factor": 1 / 255}
    if (settings.get("processor_class") != "Qwen3VLProcessor" or not isinstance(image, dict)
            or any(image.get(key) != value for key, value in expected.items())
            or set(image) - set(expected) - {"size"}):
        raise BundleError("Invalid image processor settings. Use the processor_config.json pinned with the model.")
    return {**expected, "size": {"shortest_edge": 256 * 256, "longest_edge": MAX_PROCESSED_PIXELS}}


class Vision:
    def __init__(self, root: Path, tokenizer, device: str = "auto") -> None:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

        if device not in ("auto", "ane", "cpu", "mps"):
            raise ValueError("vision device must be ane, mps, or cpu")
        # The ANE is the default: media requests then leave the GPU free for other work.
        self.device = "ane" if device == "auto" else device
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise BundleError("Metal is not available. Start serve with --vision-device cpu.")
        self.root = root
        self.config = Qwen3_5Config(**read_json(root / "config.json"))
        self.config.vision_config._attn_implementation = "sdpa"
        settings = read_json(root / "processor_config.json")
        image = image_processor_settings(settings)
        self.supports_video = "video_processor" in settings
        processor_class = ImageProcessor
        if self.supports_video:
            from .video import video_processor_settings
            video_processor_settings(settings)
            processor_class = MediaProcessor
        self.processor = processor_class(Qwen2VLImageProcessorPil(**image), tokenizer)
        if (self.processor.image_token_id != self.config.image_token_id
                or self.supports_video and self.processor.video_token_id != self.config.video_token_id):
            raise BundleError("The tokenizer and model configuration use different media tokens.")
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        print(f"Loading vision encoder on {self.device.upper()}...", flush=True)
        state = {key.removeprefix("model.visual."): value for key, value in load_file(self.root / "weights/vision.safetensors").items()}
        if self.device == "ane":
            from .engine.vision_encoder import VisionEncoder
            self.model = VisionEncoder(self.config.vision_config, state)
            return
        model = Qwen3_5VisionModel(self.config.vision_config)
        model.load_state_dict(state, strict=True, assign=True)
        dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.model = model.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @torch.inference_mode()
    def prepare(self, encoded, embedding_table: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        media = encoded.media
        ids = np.asarray(encoded.input_ids)
        # Literal media tokens in user text must never become extra visual placeholders.
        kinds = np.zeros(len(ids), dtype=np.int64)
        offset = media["token_offset"]
        types = media["mm_token_type_ids"]
        kinds[offset:offset + len(types)] = types
        if (not np.array_equal(kinds == 1, ids == self.config.image_token_id)
                or not np.array_equal(kinds == 2, ids == getattr(self.config, "video_token_id", -1))):
            raise RequestError("remove image and video placeholder tokens from state and questions")
        positions = media_positions(kinds, media.get("image_grid_thw"), media.get("video_grid_thw"))
        self._load()
        embeddings = embedding_table[torch.tensor(encoded.input_ids)].float().numpy()
        for kind, pixel_key, grid_key in ((1, "pixel_values", "image_grid_thw"), (2, "pixel_values_videos", "video_grid_thw")):
            if pixel_key not in media:
                continue
            device = "cpu" if self.device == "ane" else self.device  # ANE inputs are prepared on the CPU
            pixels = media[pixel_key].to(device=device, dtype=self.model.dtype)
            grid = media[grid_key].to(device)
            features = self.model(pixels, grid_thw=grid, return_dict=True).pooler_output.float().cpu().numpy()
            mask = kinds == kind
            if features.shape != (int(mask.sum()), self.config.text_config.hidden_size) or not np.isfinite(features).all():
                raise RuntimeError("The vision encoder returned invalid visual features.")
            embeddings[mask] = features
        return embeddings, positions

    def close(self) -> None:
        """Release ANE programs; other devices need no explicit cleanup."""
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        self.model = None
