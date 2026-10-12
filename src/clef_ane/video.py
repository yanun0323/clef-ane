"""Video data URLs, isolated decoding, and Qwen's two-frame temporal patch layout."""
from __future__ import annotations

import base64
import binascii
import json
import subprocess
import sys
import tempfile
from zipfile import BadZipFile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ._video_worker import DECODE_TIMEOUT, MAX_DECODED_FRAMES, MAX_OUTPUT_BYTES, MAX_VIDEO_BYTES, MAX_VIDEO_FRAMES, MAX_VIDEO_SECONDS, VIDEO_PIXELS
from .bundle import BundleError
from .engine.errors import RequestError, TooLong

VIDEO_FORMATS = {"data:video/mp4;base64,": "mov", "data:video/quicktime;base64,": "mov", "data:video/webm;base64,": "matroska"}
MAX_VIDEOS = 1


@dataclass(frozen=True)
class DecodedVideo:
    frames: np.ndarray  # [sampled frames, height, width, RGB], already size-bounded
    timestamps: np.ndarray  # presentation times in seconds relative to the first frame
    duration: float
    decoded_frames: int


def decode_video(raw: bytes, demuxer: str) -> DecodedVideo:
    worker = Path(__file__).with_name("_video_worker.py")
    # File-backed output prevents a faulty native decoder from filling the parent's pipe buffers.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            result = subprocess.run([sys.executable, "-I", str(worker), demuxer], input=raw,
                                    stdout=output, stderr=errors, timeout=DECODE_TIMEOUT, check=False)
        except subprocess.TimeoutExpired as error:
            raise TooLong("video decoding exceeded 45 seconds; reduce resolution, frame rate, or length") from error
        if result.returncode:
            errors.seek(0)
            message = "invalid or incomplete video; re-encode as H.264 MP4 or VP9 WebM"
            status = 400
            try:
                detail = json.loads(errors.read(8192).decode().splitlines()[-1])
                if not isinstance(detail, dict):
                    raise ValueError("invalid decoder report")
                if isinstance(detail.get("message"), str):
                    message = detail["message"][:300]
                status = detail.get("status", 400)
            except (ValueError, IndexError, UnicodeError):
                pass
            raise (TooLong if status == 413 else RequestError)(message)
        if output.seek(0, 2) > MAX_OUTPUT_BYTES:
            raise RequestError("video decoder output exceeds the size limit")
        output.seek(0)
        try:
            with np.load(output, allow_pickle=False) as archive:
                if set(archive.files) != {"frames", "timestamps", "duration", "decoded_frames"}:
                    raise ValueError("unexpected decoder fields")
                frames, timestamps = archive["frames"], archive["timestamps"]
                duration, count = float(archive["duration"].item()), int(archive["decoded_frames"].item())
            if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
                    or not 1 <= frames.shape[0] <= MAX_VIDEO_FRAMES
                    or min(frames.shape[1:3]) < 32 or any(size % 32 for size in frames.shape[1:3])
                    or frames.shape[1] * frames.shape[2] > VIDEO_PIXELS
                    or timestamps.shape != (len(frames),) or not np.isfinite(timestamps).all()
                    or timestamps[0] != 0 or np.any(np.diff(timestamps) <= 0)
                    or timestamps[-1] > MAX_VIDEO_SECONDS + .001
                    or not np.isfinite(duration) or not 0 <= duration <= MAX_VIDEO_SECONDS + .001
                    or not len(frames) <= count <= MAX_DECODED_FRAMES):
                raise ValueError("invalid decoded video")
            return DecodedVideo(frames, timestamps, duration, count)
        except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile) as error:
            raise RequestError("invalid video decoder output; re-encode the clip") from error


def decode_videos(values) -> list[DecodedVideo]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise RequestError("videos must be a list of base64 video data URLs")
    if len(values) > MAX_VIDEOS:
        raise TooLong("send at most one video per request")
    videos = []
    for value in values:
        prefix = next((key for key in VIDEO_FORMATS if isinstance(value, str) and value.startswith(key)), None)
        if prefix is None:
            raise RequestError("send an MP4, MOV, or WebM data URL; file paths and remote URLs are not accepted")
        payload = value[len(prefix):]
        if len(payload) > 4 * ((MAX_VIDEO_BYTES + 2) // 3):
            raise TooLong("video exceeds 16 MiB; send a smaller clip")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as error:
            raise RequestError("invalid base64 video data") from error
        if len(raw) > MAX_VIDEO_BYTES:
            raise TooLong("video exceeds 16 MiB; send a smaller clip")
        videos.append(decode_video(raw, VIDEO_FORMATS[prefix]))
    return videos


def video_processor_settings(settings: dict) -> None:
    video = settings.get("video_processor")
    expected = {"video_processor_type": "Qwen3VLVideoProcessor", "do_convert_rgb": True, "do_normalize": True,
                "do_rescale": True, "do_resize": True, "do_sample_frames": True, "image_mean": [.5, .5, .5],
                "image_std": [.5, .5, .5], "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "resample": 3, "rescale_factor": 1 / 255, "fps": 2, "min_frames": 4, "max_frames": 768,
                "return_metadata": False}
    if (not isinstance(video, dict) or any(video.get(key) != value for key, value in expected.items())
            or set(video) - set(expected) - {"size"}):
        raise BundleError("Invalid video processor settings. Use the processor_config.json pinned with the model.")


class VideoProcessor:
    """Pre-sampled RGB frames: normalize, pad an odd last frame, and pack temporal pairs.

    Sampling and resizing already happened in the worker. Neither is performed a second time.
    The patch layout matches Qwen3VLVideoProcessor._preprocess with resize disabled.
    """
    valid_kwargs = None
    merge_size = temporal_patch_size = 2
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __call__(self, videos: list[DecodedVideo], **kwargs) -> dict:
        import torch

        patches, grids, metadata = [], [], []
        for video in videos:
            frames = video.frames
            if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
                    or not 1 <= len(frames) <= MAX_VIDEO_FRAMES or min(frames.shape[1:3]) < 32
                    or any(size % 32 for size in frames.shape[1:3]) or frames.shape[1] * frames.shape[2] > VIDEO_PIXELS
                    or video.timestamps.shape != (len(frames),) or not np.isfinite(video.timestamps).all()
                    or np.any(np.diff(video.timestamps) <= 0)):
                raise RequestError("invalid sampled video frames or timestamps")
            if len(frames) % 2:
                frames = np.concatenate([frames, frames[-1:]], axis=0)
            temporal, height, width = len(frames) // 2, frames.shape[1], frames.shape[2]
            normalized = frames.astype(np.float32) * np.float32(2 / 255) - np.float32(1)
            packed = normalized.transpose(0, 3, 1, 2).reshape(temporal, 2, 3, height // 32, 2, 16, width // 32, 2, 16)
            packed = packed.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(temporal * (height // 16) * (width // 16), 1536)
            patches.append(torch.from_numpy(packed.copy()))
            grids.append([temporal, height // 16, width // 16])
            # Unit timebase: upstream divides indices by fps, then averages each temporal pair.
            # Feed actual PTS seconds at fps=1 to preserve VFR timing without inventing a nominal fps.
            metadata.append(SimpleNamespace(fps=1.0, frames_indices=video.timestamps.copy()))
        return {"pixel_values_videos": torch.cat(patches), "video_grid_thw": torch.tensor(grids), "video_metadata": metadata}
