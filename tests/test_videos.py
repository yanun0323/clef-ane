"""Video input: generated clips only; no weights, network, or ANE.

Small containers are encoded here with PyAV. Upstream references (Qwen3VLVideoProcessor,
Qwen3VLProcessor, Qwen3_5Model) run on frames that are already resized and normalized,
because their resize/normalize steps need the optional torchvision package.
"""
import base64
import copy
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata

from clef_ane import _video_worker as worker
from clef_ane import video
from clef_ane.bundle import BundleError
from clef_ane.engine.errors import RequestError, TooLong
from clef_ane.video import DecodedVideo, VideoProcessor, decode_video, decode_videos, video_processor_settings
from clef_ane.vision import MediaProcessor, Vision, image_processor_settings, media_positions

CONTAINERS = {"mp4": ("mp4", "libx264", "mov", "video/mp4"), "mov": ("mov", "libx264", "mov", "video/quicktime"),
              "webm": ("webm", "libvpx-vp9", "matroska", "video/webm")}
MILLISECONDS = Fraction(1, 1000)


def gray_frames(count: int, width: int = 64, height: int = 64) -> list[np.ndarray]:
    """Flat frames whose brightness identifies the source frame after lossy coding."""
    return [np.full((height, width, 3), 20 + 30 * (index % 8), dtype=np.uint8) for index in range(count)]


def encode_video(container: str, pts: list[int], *, frames=None, size=(64, 64), codec=None, rotation=None,
                 time_base=MILLISECONDS, tracks: int = 1, hflip: bool = False) -> bytes:
    """Encode RGB frames at explicit presentation timestamps (in time_base units)."""
    muxer, default_codec = CONTAINERS[container][:2]
    frames = gray_frames(len(pts), *size) if frames is None else frames
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=muxer) as output:
        streams = []
        for _ in range(tracks):
            stream = output.add_stream(codec or default_codec, rate=30)
            stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
            stream.time_base = stream.codec_context.time_base = time_base
            if codec == "libsvtav1":
                stream.options = {"preset": "12", "svtav1-params": "lp=2"}
            elif codec == "libx265":
                stream.options = {"x265-params": "pools=1:frame-threads=1:log-level=error"}
            if rotation is not None:
                stream.set_display_rotation(rotation, hflip=hflip)
            streams.append(stream)
        for stream in streams:
            for pixels, timestamp in zip(frames, pts):
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
                frame.pts, frame.time_base = timestamp, time_base
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return buffer.getvalue()


def data_url(raw: bytes, mime: str = "video/mp4") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


@contextmanager
def limits(**values):
    with patch.multiple(worker, **values):
        yield


@contextmanager
def replaced_worker(source: str):
    """Run decode_video against a stand-in worker script; the real one is not started."""
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "worker.py"
        script.write_text(source)
        with patch.object(video, "Path", lambda _: SimpleNamespace(with_name=lambda name: script)):
            yield script


ECHO_WORKER = "import sys\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n"


def archive(**overrides) -> bytes:
    fields = {"frames": np.zeros((2, 32, 64, 3), dtype=np.uint8), "timestamps": np.array([0.0, 0.5]),
              "duration": np.array(1.0), "decoded_frames": np.array(2)}
    fields.update(overrides)
    buffer = io.BytesIO()
    np.savez(buffer, **{key: value for key, value in fields.items() if value is not None})
    return buffer.getvalue()


def media_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tokens = ["[UNK]", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=tokens[1:])


def processor_config() -> dict:
    return {"processor_class": "Qwen3VLProcessor", "image_processor": {
        "image_processor_type": "Qwen2VLImageProcessor", "do_convert_rgb": True, "do_normalize": True,
        "do_rescale": True, "do_resize": True, "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5],
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "resample": 3,
        "rescale_factor": 1 / 255, "size": {"shortest_edge": 65536, "longest_edge": 16777216}},
        "video_processor": {
        "video_processor_type": "Qwen3VLVideoProcessor", "do_convert_rgb": True, "do_normalize": True,
        "do_rescale": True, "do_resize": True, "do_sample_frames": True, "image_mean": [.5, .5, .5],
        "image_std": [.5, .5, .5], "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "resample": 3,
        "rescale_factor": 1 / 255, "fps": 2, "min_frames": 4, "max_frames": 768, "return_metadata": False,
        "size": {"shortest_edge": 4096, "longest_edge": 25165824}}}


def image_processor():
    return Qwen2VLImageProcessorPil(**image_processor_settings(processor_config()))


def stock_processor():
    """The upstream processor with its real video processor.

    Only the constructor's type guard is bypassed: without torchvision the video
    processor class is not registered as a BaseVideoProcessor backend.
    """
    check = Qwen3VLProcessor.check_argument_for_proper_class

    def check_image_and_tokenizer(self, name, argument):
        return BaseVideoProcessor if name == "video_processor" else check(self, name, argument)
    with patch.object(Qwen3VLProcessor, "check_argument_for_proper_class", check_image_and_tokenizer):
        return Qwen3VLProcessor(image_processor(), media_tokenizer(), Qwen3VLVideoProcessor())


def normalized(frames: np.ndarray) -> torch.Tensor:
    """[frames, RGB, height, width] in [-1, 1]: what upstream holds after resize and normalize."""
    return torch.from_numpy(frames.astype(np.float32) * np.float32(2 / 255) - np.float32(1)).permute(0, 3, 1, 2).contiguous()


def sampled_video(count: int, height: int = 64, width: int = 96, timestamps=None, seed: int = 0) -> DecodedVideo:
    frames = np.random.default_rng(seed).integers(0, 256, (count, height, width, 3), dtype=np.uint8)
    timestamps = np.arange(count) / 2 if timestamps is None else np.asarray(timestamps, dtype=np.float64)
    return DecodedVideo(frames, timestamps, float(timestamps[-1]), count)


# Pre-resized, pre-normalized tensors: every remaining upstream step is real.
UPSTREAM_VIDEO_KWARGS = {"do_sample_frames": False, "do_convert_rgb": False, "do_resize": False,
                         "do_rescale": False, "do_normalize": False}


def upstream_rope_index(input_ids, token_types, image_grid=None, video_grid=None) -> np.ndarray:
    model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    model.get_vision_position_ids = lambda *args, **kwargs: Qwen3_5Model.get_vision_position_ids(model, *args, **kwargs)
    positions, _ = Qwen3_5Model.get_rope_index(model, input_ids, token_types, image_grid, video_grid)
    return positions[:, 0].numpy()


class ContainerDecodingTests(unittest.TestCase):
    def test_mp4_mov_and_webm_decode_to_aligned_rgb_frames(self):
        for name, (_, _, demuxer, _) in CONTAINERS.items():
            with self.subTest(container=name):
                result = worker.decode(encode_video(name, [0, 250, 500, 750, 1000], size=(96, 64)), demuxer)
                self.assertEqual(result["frames"].shape, (3, 64, 96, 3))
                self.assertEqual(result["frames"].dtype, np.uint8)
                np.testing.assert_allclose(result["timestamps"], [0, .5, 1.0], atol=1e-9)
                self.assertEqual(int(result["decoded_frames"]), 5)
                self.assertGreaterEqual(float(result["duration"]), 1.0)
                # Frames 0, 2, 4 were sampled; brightness is 20 + 30 * index.
                np.testing.assert_allclose(result["frames"].reshape(3, -1).mean(axis=1), [20, 80, 140], atol=6)

    def test_advertised_codecs_decode(self):
        for container, codec in (("mp4", "mpeg4"), ("mp4", "libx265"), ("mp4", "libsvtav1"), ("webm", "libvpx"),
                                 ("webm", "libsvtav1")):
            with self.subTest(container=container, codec=codec), patch.dict(os.environ, {"SVT_LOG": "0"}):
                result = worker.decode(encode_video(container, [0, 500], codec=codec), CONTAINERS[container][2])
                self.assertEqual(int(result["decoded_frames"]), 2)
                np.testing.assert_allclose(result["timestamps"], [0, .5], atol=1e-6)

    def test_webm_duration_is_relative_to_nonzero_start(self):
        result = worker.decode(encode_video("webm", [30000, 30500]), "matroska")
        self.assertLess(float(result["duration"]), 1)
        np.testing.assert_allclose(result["timestamps"], [0, .5], atol=1e-6)

    def test_variable_frame_rate_keeps_true_presentation_times(self):
        # Irregular gaps and a non-zero first PTS. One frame per half-second bucket, then the final frame.
        pts = [1000, 1100, 1700, 2500, 2600, 3900]
        for name, (_, _, demuxer, _) in CONTAINERS.items():
            with self.subTest(container=name):
                result = worker.decode(encode_video(name, pts), demuxer)
                np.testing.assert_allclose(result["timestamps"], [0, .7, 1.5, 2.9], atol=1e-9)
                np.testing.assert_allclose(result["frames"].reshape(4, -1).mean(axis=1), [20, 80, 110, 170], atol=6)
                self.assertEqual(int(result["decoded_frames"]), 6)

    def test_time_base_does_not_change_timestamps(self):
        coarse = worker.decode(encode_video("mp4", [0, 9, 63, 135], time_base=Fraction(1, 90)), "mov")
        fine = worker.decode(encode_video("mp4", [0, 100, 700, 1500]), "mov")
        np.testing.assert_allclose(coarse["timestamps"], [0, .7, 1.5], atol=1e-9)
        np.testing.assert_allclose(coarse["timestamps"], fine["timestamps"], atol=1e-9)

    def test_last_frame_is_kept_once(self):
        trailing = worker.decode(encode_video("mp4", [0, 500, 900]), "mov")
        np.testing.assert_allclose(trailing["timestamps"], [0, .5, .9], atol=1e-9)
        sampled = worker.decode(encode_video("mp4", [0, 500, 1000]), "mov")
        np.testing.assert_allclose(sampled["timestamps"], [0, .5, 1.0], atol=1e-9)
        single = worker.decode(encode_video("mp4", [0]), "mov")
        np.testing.assert_array_equal(single["timestamps"], [0])
        self.assertEqual(single["frames"].shape[0], 1)

    def test_rotation_metadata_is_applied(self):
        # Upright scene: red top, blue bottom, green on the left half.
        upright = np.zeros((96, 64, 3), dtype=np.uint8)
        upright[:48, :, 0], upright[48:, :, 2], upright[:, :32, 1] = 255, 255, 255
        # A display rotation of N degrees counterclockwise undoes a stored frame turned N degrees clockwise.
        for rotation, quarter_turns in ((90, -1), (-90, 1), (180, 2)):
            stored = np.rot90(upright, quarter_turns)
            for name in ("mp4", "webm"):
                with self.subTest(rotation=rotation, container=name):
                    raw = encode_video(name, [0, 500], frames=[stored, stored], rotation=rotation)
                    with av.open(io.BytesIO(raw)) as container:
                        frame = next(container.decode(video=0))
                        self.assertEqual(abs(frame.rotation), abs(rotation))
                        self.assertEqual((frame.width, frame.height), stored.shape[1::-1])
                    frames = worker.decode(raw, CONTAINERS[name][2])["frames"]
                    self.assertEqual(frames.shape, (2, 96, 64, 3))
                    for (row, column), expected in (((12, 12), (255, 255, 0)), ((12, 52), (255, 0, 0)),
                                                    ((84, 12), (0, 255, 255)), ((84, 52), (0, 0, 255))):
                        np.testing.assert_allclose(frames[0, row, column], expected, atol=24)

    def test_mirrored_display_metadata_is_not_silently_ignored(self):
        raw = encode_video("mp4", [0, 500], rotation=0, hflip=True)
        with self.assertRaisesRegex(ValueError, "mirrored video metadata"):
            worker.decode(raw, "mov")

    def test_mp4_trim_edit_list_ignores_discarded_preroll(self):
        raw = encode_video("mp4", list(range(0, 3000, 100)))
        buffer = io.BytesIO()
        with av.open(io.BytesIO(raw)) as source, av.open(buffer, "w", format="mp4") as output:
            original = source.streams.video[0]
            stream = output.add_stream_from_template(original)
            shift = int(1 / original.time_base)
            for packet in source.demux(original):
                if packet.dts is not None:
                    packet.pts -= shift
                    packet.dts -= shift
                    packet.stream = stream
                    output.mux(packet)
        with av.open(io.BytesIO(buffer.getvalue())) as container:
            stream = container.streams.video[0]
            expected = sum(not entry.is_discard for entry in stream.index_entries)
            self.assertLess(expected, stream.frames)
        result = worker.decode(buffer.getvalue(), "mov")
        self.assertEqual(int(result["decoded_frames"]), expected)
        self.assertEqual(result["timestamps"][0], 0)
        self.assertLess(float(result["duration"]), 2.1)

    def test_mp4_audio_track_is_ignored(self):
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as container:
            visual = container.add_stream("libx264", rate=2)
            visual.width = visual.height = 64
            visual.pix_fmt = "yuv420p"
            audio = container.add_stream("aac", rate=48000)
            audio.layout = "stereo"
            for pixels in gray_frames(4):
                for packet in visual.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in visual.encode():
                container.mux(packet)
            for index in range(48):
                frame = av.AudioFrame.from_ndarray(np.zeros((2, 1024), np.float32), format="fltp", layout="stereo")
                frame.sample_rate, frame.pts, frame.time_base = 48000, index * 1024, Fraction(1, 48000)
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
        result = worker.decode(buffer.getvalue(), "mov")
        self.assertEqual(int(result["decoded_frames"]), 4)
        np.testing.assert_allclose(result["timestamps"], [0, .5, 1, 1.5], atol=1e-6)

    def test_unsupported_rotation_is_rejected(self):
        raw = encode_video("mp4", [0, 500], rotation=45)
        with self.assertRaisesRegex(ValueError, "rotation"):
            worker.decode(raw, "mov")

    def test_frames_are_resized_to_the_token_budget(self):
        for size, expected in (((32, 32), (64, 64)), ((640, 360), (192, 320)), ((360, 640), (320, 192)), ((64, 48), (64, 64))):
            with self.subTest(size=size):
                frames = worker.decode(encode_video("mp4", [0, 500], size=size, codec="mpeg4"), "mov")["frames"]
                self.assertEqual(frames.shape[1:3], expected)
                self.assertLessEqual(frames.shape[1] * frames.shape[2], worker.VIDEO_PIXELS)

    def test_resize_shape_is_aligned_and_bounded(self):
        for width, height in ((32, 32), (33, 47), (640, 360), (4096, 64), (64, 4096), (1920, 1080), (100, 37), (4096, 4096)):
            with self.subTest(width=width, height=height):
                resized_width, resized_height = worker.resize_shape(width, height)
                self.assertEqual((resized_width % 32, resized_height % 32), (0, 0))
                self.assertGreaterEqual(min(resized_width, resized_height), 32)
                self.assertLessEqual(resized_width * resized_height, worker.VIDEO_PIXELS)
                self.assertEqual(resized_width >= resized_height, width >= height)

    def test_small_or_extreme_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 32 pixels"):
            worker.decode(encode_video("mp4", [0, 500], size=(16, 16)), "mov")
        with limits(MAX_FRAME_ASPECT=1.2), self.assertRaisesRegex(ValueError, "aspect ratio"):
            worker.decode(encode_video("mp4", [0, 500], size=(96, 64)), "mov")

    def test_container_and_codec_must_match_the_declared_type(self):
        mp4, webm = encode_video("mp4", [0, 500]), encode_video("webm", [0, 500])
        for raw, demuxer in ((mp4, "matroska"), (webm, "mov"), (mp4, "avi"), (mp4, ""), (mp4, "mov,matroska")):
            with self.subTest(demuxer=demuxer), self.assertRaises(ValueError):
                worker.decode(raw, demuxer)
        with self.assertRaisesRegex(ValueError, "unsupported video codec"):
            worker.decode(encode_video("mp4", [0, 500], codec="libvpx-vp9"), "mov")

    def test_multiple_video_tracks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one video track"):
            worker.decode(encode_video("mp4", [0, 500], tracks=2), "mov")

    def test_malformed_and_truncated_containers_are_rejected(self):
        mp4, webm = encode_video("mp4", list(range(0, 3000, 100))), encode_video("webm", list(range(0, 3000, 100)))
        for name, raw, demuxer in (("empty", b"", "mov"), ("text", b"not a video" * 20, "mov"),
                                   ("text as webm", b"not a video" * 20, "matroska"),
                                   ("half mp4", mp4[:len(mp4) // 2], "mov"), ("mp4 header", mp4[:30], "mov"),
                                   ("webm header", webm[:30], "matroska")):
            with self.subTest(name=name), self.assertRaises(Exception):
                worker.decode(raw, demuxer)


class DecodeLimitTests(unittest.TestCase):
    def setUp(self):
        self.clips = {name: (encode_video(name, [0, 100, 700, 1500, 1600]), values[2]) for name, values in CONTAINERS.items()}

    def rejected(self, pattern: str, **values):
        for name, (raw, demuxer) in self.clips.items():
            with self.subTest(container=name), limits(**values), self.assertRaisesRegex(worker.VideoLimitError, pattern):
                worker.decode(raw, demuxer)

    def test_duration_limit(self):
        self.rejected("exceeds 30 seconds", MAX_VIDEO_SECONDS=1.0)

    def test_duration_limit_without_container_duration(self):
        # Timestamps are checked frame by frame, not only against container metadata.
        raw, demuxer = self.clips["webm"]
        with limits(MAX_VIDEO_SECONDS=1.55), self.assertRaisesRegex(worker.VideoLimitError, "30 seconds"):
            with patch.object(av.container.InputContainer, "duration", None, create=True):
                worker.decode(raw, demuxer)

    def test_sampled_frame_limit(self):
        self.rejected("too many sampled frames", MAX_VIDEO_FRAMES=2)

    def test_source_frame_limit(self):
        self.rejected("source frames|frame or pixel limit", MAX_DECODED_FRAMES=3)

    def test_total_decoded_pixel_limit(self):
        self.rejected("frame or pixel limit", MAX_DECODED_PIXELS=64 * 64 * 2)

    def test_frame_side_limit(self):
        self.rejected("frame or pixel limit", MAX_FRAME_SIDE=63)

    def test_frame_pixel_limit_is_enforced_by_the_decoder(self):
        for name, (raw, demuxer) in self.clips.items():
            with self.subTest(container=name), limits(MAX_FRAME_PIXELS=64 * 64 - 1), self.assertRaises(ValueError):
                worker.decode(raw, demuxer)

    def test_limits_admit_clips_at_the_boundary(self):
        raw, demuxer = self.clips["mp4"]
        with limits(MAX_VIDEO_FRAMES=4, MAX_DECODED_FRAMES=5, MAX_DECODED_PIXELS=64 * 64 * 5, MAX_FRAME_SIDE=64,
                    MAX_FRAME_PIXELS=64 * 64):
            result = worker.decode(raw, demuxer)
        np.testing.assert_allclose(result["timestamps"], [0, .7, 1.5, 1.6], atol=1e-9)

    def test_compressed_size_limit(self):
        with limits(MAX_VIDEO_BYTES=64), self.assertRaisesRegex(worker.VideoLimitError, "16 MiB"):
            worker.decode(self.clips["mp4"][0], "mov")


class ExternalInputTests(unittest.TestCase):
    def test_external_references_cannot_be_opened(self):
        for arguments in (("http://example.invalid/clip.mp4", 0, {}), ("/etc/passwd", 0, {}), ("../clip.mp4",)):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "external video references"):
                worker.reject_external_io(*arguments)

    def test_decoder_reads_only_memory_with_protocols_disabled(self):
        calls = []
        real_open = av.open

        def recording_open(*args, **kwargs):
            calls.append((args, kwargs))
            return real_open(*args, **kwargs)
        raw = encode_video("mp4", [0, 500])
        with patch.object(av, "open", recording_open):
            worker.decode(raw, "mov")
        (args, kwargs), = calls
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(kwargs["mode"], "r")
        self.assertEqual(kwargs["format"], "mov")
        self.assertIs(kwargs["io_open"], worker.reject_external_io)
        self.assertEqual(kwargs["options"]["protocol_whitelist"], "")
        self.assertEqual(kwargs["options"]["enable_drefs"], "0")
        self.assertEqual(kwargs["options"]["use_absolute_path"], "0")
        self.assertEqual(kwargs["options"]["max_pixels"], str(worker.MAX_FRAME_PIXELS))
        self.assertEqual(set(kwargs["options"]["codec_whitelist"].split(",")), {"h264", "hevc", "mpeg4", "av1", "libdav1d"})

    def test_paths_and_urls_never_reach_the_decoder(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
            file.write(encode_video("mp4", [0, 500]))
            file.flush()
            values = (file.name, f"file://{file.name}", "http://127.0.0.1:9/clip.mp4", "https://example.invalid/clip.mp4",
                      "data:video/mp4," + file.name, "data:video/mp4;base64", "data:text/plain;base64,AAAA",
                      "data:video/x-msvideo;base64,AAAA", "data:image/png;base64,AAAA", "DATA:VIDEO/MP4;BASE64,AAAA",
                      " data:video/mp4;base64,AAAA", "", b"data:video/mp4;base64,AAAA", None, 7, ["nested"], {"url": "x"},
                      Path(file.name))
            with patch.object(video, "decode_video") as decode, patch.object(video.subprocess, "run") as run:
                for value in values:
                    with self.subTest(value=value), self.assertRaisesRegex(RequestError, "data URL"):
                        decode_videos([value])
                decode.assert_not_called()
                run.assert_not_called()

    def test_worker_is_started_isolated_with_bytes_on_stdin(self):
        completed = SimpleNamespace(returncode=1)
        with patch.object(video.subprocess, "run", return_value=completed) as run, self.assertRaises(RequestError):
            decode_video(b"clip bytes", "mov")
        (command,), kwargs = run.call_args
        self.assertEqual(command[:2], [sys.executable, "-I"])
        self.assertEqual(Path(command[2]), Path(worker.__file__))
        self.assertEqual(command[3:], ["mov"])  # a demuxer name; never a filename or URL
        self.assertEqual(kwargs["input"], b"clip bytes")
        self.assertEqual(kwargs["timeout"], video.DECODE_TIMEOUT)
        self.assertNotIn("shell", kwargs)


class DataUrlTests(unittest.TestCase):
    def test_absent_and_empty_lists(self):
        self.assertEqual(decode_videos(None), [])
        self.assertEqual(decode_videos([]), [])

    def test_container_types_select_the_demuxer(self):
        for mime, demuxer in (("video/mp4", "mov"), ("video/quicktime", "mov"), ("video/webm", "matroska")):
            with self.subTest(mime=mime), patch.object(video, "decode_video", return_value="decoded") as decode:
                self.assertEqual(decode_videos([data_url(b"\x00\x01\xff", mime)]), ["decoded"])
                decode.assert_called_once_with(b"\x00\x01\xff", demuxer)

    def test_non_list_values_are_rejected(self):
        for value in ("data:video/mp4;base64,AAAA", {"0": "x"}, ("data:video/mp4;base64,AAAA",), 1, b"x"):
            with self.subTest(value=value), self.assertRaisesRegex(RequestError, "list of base64 video data URLs"):
                decode_videos(value)

    def test_more_than_one_video_is_rejected_before_decoding(self):
        with patch.object(video, "decode_video") as decode, self.assertRaisesRegex(TooLong, "at most one video"):
            decode_videos([data_url(b"a"), data_url(b"b")])
        decode.assert_not_called()

    def test_invalid_base64_is_rejected(self):
        prefix = "data:video/mp4;base64,"
        with patch.object(video, "decode_video") as decode:
            for payload in ("AAA", "AA=A", "AAAA\n", "AA AA", "!!!!", "QUJD=", "AAAA,AAAA", "QUJD#"):
                with self.subTest(payload=payload), self.assertRaisesRegex(RequestError, "invalid base64 video data"):
                    decode_videos([prefix + payload])
            decode.assert_not_called()

    def test_oversized_payloads_are_rejected_before_decoding(self):
        with patch.object(video, "MAX_VIDEO_BYTES", 30), patch.object(video, "decode_video", return_value="decoded") as decode:
            self.assertEqual(decode_videos([data_url(b"x" * 30)]), ["decoded"])
            decode.reset_mock()
            for raw in (b"x" * 31, b"x" * 300):
                with self.subTest(size=len(raw)), self.assertRaisesRegex(TooLong, "16 MiB"):
                    decode_videos([data_url(raw)])
            with patch.object(video.base64, "b64decode") as b64decode, self.assertRaises(TooLong):
                decode_videos([data_url(b"x" * 300)])
            b64decode.assert_not_called()  # the encoded length alone is enough
            decode.assert_not_called()

    def test_default_size_limit(self):
        self.assertEqual(video.MAX_VIDEO_BYTES, 16 * 1024 * 1024)
        self.assertEqual(video.MAX_VIDEOS, 1)


class SubprocessDecodingTests(unittest.TestCase):
    def test_data_url_round_trip_through_the_real_worker(self):
        raw = encode_video("mp4", [1000, 1100, 1700, 2500, 2600], size=(96, 64))
        decoded, = decode_videos([data_url(raw)])
        expected = worker.decode(raw, "mov")
        np.testing.assert_array_equal(decoded.frames, expected["frames"])
        np.testing.assert_array_equal(decoded.timestamps, expected["timestamps"])
        np.testing.assert_allclose(decoded.timestamps, [0, .7, 1.5, 1.6], atol=1e-9)
        self.assertEqual(decoded.frames.shape, (4, 64, 96, 3))
        self.assertEqual(decoded.decoded_frames, 5)
        self.assertAlmostEqual(decoded.duration, float(expected["duration"]))

    def test_webm_round_trip_through_the_real_worker(self):
        decoded, = decode_videos([data_url(encode_video("webm", [0, 400, 1300]), "video/webm")])
        np.testing.assert_allclose(decoded.timestamps, [0, 1.3], atol=1e-9)
        self.assertEqual(decoded.decoded_frames, 3)

    def test_invalid_and_truncated_clips_are_request_errors(self):
        mp4 = encode_video("mp4", list(range(0, 3000, 100)))
        webm = encode_video("webm", [0, 500])
        for name, raw, mime in (("empty", b"", "video/mp4"), ("half mp4", mp4[:len(mp4) // 2], "video/mp4"),
                                ("webm as mp4", webm, "video/mp4"), ("mp4 as webm", mp4, "video/webm")):
            with self.subTest(name=name), self.assertRaises(RequestError) as caught:
                decode_videos([data_url(raw, mime)])
            self.assertNotIsInstance(caught.exception, TooLong)

    def test_worker_limit_errors_become_too_long(self):
        source = "import json, sys\nprint('noise', file=sys.stderr)\nprint(json.dumps({'status': %d, 'message': %r}), file=sys.stderr)\nsys.exit(1)\n"
        for status, expected in ((413, TooLong), (400, RequestError)):
            with self.subTest(status=status), replaced_worker(source % (status, "worker says no")):
                with self.assertRaisesRegex(expected, "worker says no") as caught:
                    decode_video(b"clip", "mov")
                self.assertEqual(type(caught.exception), expected)

    def test_worker_crash_without_a_report_is_a_generic_request_error(self):
        for name, source in (("silent", "import sys\nsys.exit(3)\n"),
                             ("traceback", "raise RuntimeError('/private/path secret')\n"),
                             ("binary", "import os, sys\nos.write(2, bytes(range(128, 256)))\nsys.exit(1)\n"),
                             ("killed", "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")):
            with self.subTest(name=name), replaced_worker(source):
                with self.assertRaisesRegex(RequestError, "invalid or incomplete video") as caught:
                    decode_video(b"clip", "mov")
                self.assertNotIn("secret", str(caught.exception))

    def test_worker_messages_are_bounded(self):
        source = "import json, sys\nprint(json.dumps({'status': 400, 'message': 'x' * 5000}), file=sys.stderr)\nsys.exit(1)\n"
        with replaced_worker(source), self.assertRaises(RequestError) as caught:
            decode_video(b"clip", "mov")
        self.assertLessEqual(len(str(caught.exception)), 300)

    def test_deadline_stops_a_stuck_worker(self):
        source = ("import os, sys, time\nopen(__file__ + '.pid', 'w').write(str(os.getpid()))\n"
                  "sys.stdin.buffer.read()\ntime.sleep(120)\n")
        with replaced_worker(source) as script, patch.object(video, "DECODE_TIMEOUT", 1.0):
            started = time.monotonic()
            with self.assertRaisesRegex(TooLong, "video decoding exceeded"):
                decode_video(b"clip", "mov")
            elapsed = time.monotonic() - started
            pid = int(Path(f"{script}.pid").read_text())
        self.assertGreaterEqual(elapsed, 1.0)
        self.assertLess(elapsed, 15)
        with self.assertRaises(ProcessLookupError):  # killed and reaped, not left running
            os.kill(pid, 0)

    def test_worker_has_its_own_deadline_if_the_parent_is_gone(self):
        import signal
        import subprocess
        source = (f"import runpy, time\nn = runpy.run_path({worker.__file__!r})\n"
                  "n['main'].__globals__.update(DECODE_TIMEOUT=1, decode=lambda *args: time.sleep(120))\n"
                  "n['main']()\n")
        result = subprocess.run([sys.executable, "-I", "-c", source], input=b"clip", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, -signal.SIGALRM, result.stderr)

    def test_default_deadline(self):
        self.assertEqual(video.DECODE_TIMEOUT, 45)

    def test_valid_worker_output_is_accepted(self):
        with replaced_worker(ECHO_WORKER):
            decoded = decode_video(archive(), "mov")
        self.assertEqual(decoded.frames.shape, (2, 32, 64, 3))
        np.testing.assert_array_equal(decoded.timestamps, [0, .5])
        self.assertEqual((decoded.duration, decoded.decoded_frames), (1.0, 2))

    def test_invalid_worker_output_is_rejected(self):
        cases = {
            "garbage": b"not an archive",
            "float frames": archive(frames=np.zeros((2, 32, 64, 3), dtype=np.float32)),
            "unaligned frames": archive(frames=np.zeros((2, 48, 64, 3), dtype=np.uint8)),
            "tiny frames": archive(frames=np.zeros((2, 0, 64, 3), dtype=np.uint8)),
            "four channels": archive(frames=np.zeros((2, 32, 64, 4), dtype=np.uint8)),
            "oversized frames": archive(frames=np.zeros((2, 288, 256, 3), dtype=np.uint8)),
            "no frames": archive(frames=np.zeros((0, 32, 64, 3), dtype=np.uint8), timestamps=np.zeros(0)),
            "too many frames": archive(frames=np.zeros((65, 32, 32, 3), dtype=np.uint8), timestamps=np.arange(65) / 4,
                                       decoded_frames=np.array(65), duration=np.array(20.0)),
            "timestamp count": archive(timestamps=np.array([0.0, .5, 1.0])),
            "repeated timestamps": archive(timestamps=np.array([0.0, 0.0])),
            "non-zero start": archive(timestamps=np.array([.25, .5])),
            "nan timestamps": archive(timestamps=np.array([0.0, np.nan])),
            "late timestamps": archive(timestamps=np.array([0.0, 31.0])),
            "infinite duration": archive(duration=np.array(np.inf)),
            "negative duration": archive(duration=np.array(-1.0)),
            "long duration": archive(duration=np.array(31.0)),
            "fewer decoded than sampled": archive(decoded_frames=np.array(1)),
            "too many decoded": archive(decoded_frames=np.array(1801)),
            "missing field": archive(duration=None),
            "extra field": archive(extra=np.array(1)),
            "pickled object": archive(duration=np.array({"a": 1}, dtype=object)),
            "vector duration": archive(duration=np.array([1.0, 2.0])),
        }
        with replaced_worker(ECHO_WORKER):
            for name, raw in cases.items():
                with self.subTest(name=name), self.assertRaisesRegex(RequestError, "invalid video decoder output"):
                    decode_video(raw, "mov")

    def test_empty_or_truncated_worker_output_is_rejected(self):
        # A worker that exits successfully after writing nothing or part of its archive.
        with replaced_worker(ECHO_WORKER):
            for name, raw in (("empty", b""), ("truncated", archive()[:-40]), ("half", archive()[:200])):
                with self.subTest(name=name), self.assertRaisesRegex(RequestError, "invalid video decoder output"):
                    decode_video(raw, "mov")

    def test_worker_output_size_is_bounded(self):
        with replaced_worker(ECHO_WORKER), patch.object(video, "MAX_OUTPUT_BYTES", 1024):
            with self.assertRaisesRegex(RequestError, "exceeds the size limit"):
                decode_video(archive(), "mov")


class TemporalPackingTests(unittest.TestCase):
    """VideoProcessor against the installed Qwen3VLVideoProcessor._preprocess."""

    def upstream(self, videos: list[DecodedVideo]):
        return Qwen3VLVideoProcessor()._preprocess(
            [normalized(item.frames) for item in videos], do_convert_rgb=False, do_resize=False, do_rescale=False,
            do_normalize=False, patch_size=16, temporal_patch_size=2, merge_size=2, return_tensors="pt")

    def test_pairs_match_upstream_layout(self):
        for count, height, width in ((2, 32, 32), (4, 64, 96), (6, 96, 64), (8, 32, 128), (2, 256, 256)):
            with self.subTest(frames=count, height=height, width=width):
                clip = sampled_video(count, height, width, seed=count)
                actual, expected = VideoProcessor()([clip]), self.upstream([clip])
                self.assertEqual(actual["pixel_values_videos"].dtype, torch.float32)
                torch.testing.assert_close(actual["pixel_values_videos"], expected["pixel_values_videos"], rtol=0, atol=0)
                self.assertTrue(torch.equal(actual["video_grid_thw"], expected["video_grid_thw"]))
                self.assertEqual(actual["video_grid_thw"].tolist(), [[count // 2, height // 16, width // 16]])

    def test_odd_frame_count_repeats_the_last_frame_like_upstream(self):
        for count in (1, 3, 5, 7):
            with self.subTest(frames=count):
                clip = sampled_video(count, seed=10 + count)
                actual, expected = VideoProcessor()([clip]), self.upstream([clip])
                torch.testing.assert_close(actual["pixel_values_videos"], expected["pixel_values_videos"], rtol=0, atol=0)
                self.assertEqual(actual["video_grid_thw"].tolist(), expected["video_grid_thw"].tolist())
                self.assertEqual(actual["video_grid_thw"][0, 0].item(), (count + 1) // 2)
                # The padded pair holds the last frame twice: [RGB, 2 frames, 16, 16] per patch.
                last_pair = actual["pixel_values_videos"][-(64 // 16) * (96 // 16):].reshape(-1, 3, 2, 16, 16)
                torch.testing.assert_close(last_pair[:, :, 0], last_pair[:, :, 1], rtol=0, atol=0)
                self.assertEqual(clip.frames.shape[0], count)  # the caller's array is not extended

    def test_patch_contents_and_order(self):
        # Each pixel encodes its frame, so a patch reveals which frames and location it came from.
        frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        for index in range(4):
            frames[index, :, :, 0] = 60 * index
            frames[index, :, :, 1] = (np.arange(64)[:, None] // 16) * 60
            frames[index, :, :, 2] = (np.arange(64)[None, :] // 16) * 60
        clip = DecodedVideo(frames, np.arange(4) / 2, 1.5, 4)
        patches = VideoProcessor()([clip])["pixel_values_videos"].reshape(2, 16, 3, 2, 16, 16)
        values = torch.round((patches[..., 0, 0] + 1) * 255 / 2 / 60).long()  # [pair, patch, RGB, frame in pair]
        self.assertEqual(values[:, :, 0].tolist(), [[[0, 1]] * 16, [[2, 3]] * 16])
        # Within a pair, patches go 2x2 merge block by merge block: rows 0,0,1,1 then columns 0,1,0,1.
        self.assertEqual(values[0, :, 1, 0].tolist(), [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
        self.assertEqual(values[0, :, 2, 0].tolist(), [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3])

    def test_normalization_matches_rescale_then_normalize(self):
        clip = sampled_video(2, 32, 32, seed=3)
        reference = (torch.from_numpy(clip.frames).permute(0, 3, 1, 2).float() * (1 / 255) - .5) / .5
        expected = Qwen3VLVideoProcessor()._preprocess(
            [reference], do_convert_rgb=False, do_resize=False, do_rescale=False, do_normalize=False, patch_size=16,
            temporal_patch_size=2, merge_size=2, return_tensors="pt")["pixel_values_videos"]
        actual = VideoProcessor()([clip])["pixel_values_videos"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertLessEqual(actual.abs().max().item(), 1.0)
        extremes = DecodedVideo(np.stack([np.zeros((32, 32, 3), np.uint8), np.full((32, 32, 3), 255, np.uint8)]),
                                np.array([0.0, .5]), .5, 2)
        packed = VideoProcessor()([extremes])["pixel_values_videos"].reshape(-1, 3, 2, 16, 16)
        self.assertEqual((packed[:, :, 0].unique().tolist(), packed[:, :, 1].unique().tolist()), ([-1.0], [1.0]))

    def test_metadata_carries_true_timestamps_at_unit_rate(self):
        clip = sampled_video(3, timestamps=[0, .7, 1.5])
        metadata, = VideoProcessor()([clip])["video_metadata"]
        self.assertEqual(metadata.fps, 1.0)
        np.testing.assert_array_equal(metadata.frames_indices, [0, .7, 1.5])
        metadata.frames_indices[0] = 9
        self.assertEqual(clip.timestamps[0], 0)

    def test_invalid_sampled_videos_are_rejected(self):
        good = sampled_video(2, 64, 64)
        cases = {
            "float frames": DecodedVideo(good.frames.astype(np.float32), good.timestamps, 1, 2),
            "unaligned": DecodedVideo(np.zeros((2, 48, 64, 3), np.uint8), good.timestamps, 1, 2),
            "too small": DecodedVideo(np.zeros((2, 16, 64, 3), np.uint8), good.timestamps, 1, 2),
            "too large": DecodedVideo(np.zeros((2, 288, 256, 3), np.uint8), good.timestamps, 1, 2),
            "grayscale": DecodedVideo(np.zeros((2, 64, 64, 1), np.uint8), good.timestamps, 1, 2),
            "no frames": DecodedVideo(np.zeros((0, 64, 64, 3), np.uint8), np.zeros(0), 0, 0),
            "too many frames": DecodedVideo(np.zeros((65, 32, 32, 3), np.uint8), np.arange(65) / 2, 32, 65),
            "timestamp count": DecodedVideo(good.frames, np.array([0.0]), 1, 2),
            "repeated timestamps": DecodedVideo(good.frames, np.array([0.0, 0.0]), 1, 2),
            "nan timestamps": DecodedVideo(good.frames, np.array([0.0, np.nan]), 1, 2),
        }
        for name, clip in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(RequestError, "invalid sampled video"):
                VideoProcessor()([clip])


class TimestampTests(unittest.TestCase):
    """Timestamp text against Qwen3VLProcessor, given the same PTS at a unit timebase."""

    @classmethod
    def setUpClass(cls):
        cls.stock = stock_processor()
        cls.media = MediaProcessor(image_processor(), media_tokenizer())

    def upstream_replacement(self, clip: DecodedVideo) -> str:
        inputs = Qwen3VLVideoProcessor()._preprocess(
            [normalized(clip.frames)], do_convert_rgb=False, do_resize=False, do_rescale=False, do_normalize=False,
            patch_size=16, temporal_patch_size=2, merge_size=2, return_tensors="pt")
        inputs["video_metadata"] = [VideoMetadata(total_num_frames=len(clip.frames), fps=1.0,
                                                  frames_indices=clip.timestamps.tolist())]
        return self.stock.replace_video_token(inputs, 0)

    def test_replacement_text_equals_upstream(self):
        cases = {"even": [0, .5, 1.0, 1.5], "odd": [0, .5, 1.0], "single": [0], "variable": [0, .7, 1.5, 2.04, 2.6, 9.99],
                 "long": [0, 12.3, 29.0, 30.0], "odd variable": [0, .1, .9, 1.3, 7.25]}
        for name, timestamps in cases.items():
            with self.subTest(name=name):
                clip = sampled_video(len(timestamps), 64, 96, timestamps=timestamps)
                _, replacements = self.media._process_videos([clip])
                self.assertEqual(replacements, [self.upstream_replacement(clip)])

    def test_each_pair_is_labelled_with_its_mean_presentation_time(self):
        clip = sampled_video(5, 64, 96, timestamps=[0, 1.0, 2.0, 5.0, 8.0])
        _, (replacement,) = self.media._process_videos([clip])
        frame = "<|vision_start|>" + "<|video_pad|>" * 6 + "<|vision_end|>"
        # The unpaired last frame is repeated, so its label is its own time.
        self.assertEqual(replacement, f"<0.5 seconds>{frame}<3.5 seconds>{frame}<8.0 seconds>{frame}")

    def test_timestamps_are_not_derived_from_a_nominal_frame_rate(self):
        regular = sampled_video(4, timestamps=[0, .5, 1.0, 1.5])
        irregular = sampled_video(4, timestamps=[0, 3.0, 10.0, 20.0])
        self.assertIn("<0.2 seconds>", self.media._process_videos([regular])[1][0])
        replacement = self.media._process_videos([irregular])[1][0]
        self.assertIn("<1.5 seconds>", replacement)
        self.assertIn("<15.0 seconds>", replacement)

    def test_processor_output_equals_upstream_for_a_video(self):
        for timestamps in ([0, .7, 1.5, 2.04], [0, .7, 1.5, 2.04, 2.6]):
            with self.subTest(frames=len(timestamps)):
                clip = sampled_video(len(timestamps), 64, 96, timestamps=timestamps)
                text = ["describe <|vision_start|><|video_pad|><|vision_end|> now"]
                actual = self.media(text=copy.copy(text), videos=[clip], return_tensors="pt")
                metadata = VideoMetadata(total_num_frames=len(timestamps), fps=1.0, frames_indices=list(timestamps))
                expected = self.stock(text=copy.copy(text), videos=[normalized(clip.frames)], return_tensors="pt",
                                      videos_kwargs={**UPSTREAM_VIDEO_KWARGS, "video_metadata": [metadata]})
                self.assertEqual(set(actual), set(expected))
                self.assertNotIn("video_metadata", actual)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.assertEqual(int((actual["mm_token_type_ids"] == 2).sum()), (len(timestamps) + 1) // 2 * 6)


class PositionTests(unittest.TestCase):
    """3-axis positions against Qwen3_5Model.get_rope_index."""

    @classmethod
    def setUpClass(cls):
        cls.media = MediaProcessor(image_processor(), media_tokenizer())

    def positions(self, encoded) -> np.ndarray:
        image_grid, video_grid = encoded.get("image_grid_thw"), encoded.get("video_grid_thw")
        return media_positions(encoded["mm_token_type_ids"][0].numpy(),
                               None if image_grid is None else image_grid.numpy(),
                               None if video_grid is None else video_grid.numpy())

    def test_video_positions_equal_upstream(self):
        for count, height, width in ((2, 64, 64), (5, 64, 96), (4, 96, 64), (7, 32, 128), (1, 256, 256)):
            with self.subTest(frames=count, height=height, width=width):
                clip = sampled_video(count, height, width)
                encoded = self.media(text=["a b <|vision_start|><|video_pad|><|vision_end|> c d e"], videos=[clip],
                                     return_tensors="pt")
                expected = upstream_rope_index(encoded["input_ids"], encoded["mm_token_type_ids"],
                                               video_grid=encoded["video_grid_thw"].clone())
                np.testing.assert_array_equal(self.positions(encoded), expected)

    def test_mixed_image_and_video_positions_equal_upstream(self):
        images = [Image.new("RGB", (320, 160), "red"), Image.new("RGB", (256, 256), "blue")]
        for image in images:
            self.addCleanup(image.close)
        image, clip = "<|vision_start|><|image_pad|><|vision_end|>", "<|vision_start|><|video_pad|><|vision_end|>"
        for name, text in (("image first", f"x {image} y {clip} z {image} end"), ("video first", f"{clip}{image}{image}"),
                           ("video last", f"{image} a b c {image} {clip}")):
            with self.subTest(order=name):
                encoded = self.media(text=[text], images=images, videos=[sampled_video(5, 64, 96)], return_tensors="pt")
                self.assertEqual(sorted(encoded["mm_token_type_ids"].unique().tolist()), [0, 1, 2])
                expected = upstream_rope_index(encoded["input_ids"], encoded["mm_token_type_ids"],
                                               encoded["image_grid_thw"].clone(), encoded["video_grid_thw"].clone())
                np.testing.assert_array_equal(self.positions(encoded), expected)

    def test_each_temporal_pair_is_its_own_grid(self):
        # text, then two 2x3 pairs split by one timestamp token, then text
        types = [0, 0, *[2] * 6, 0, *[2] * 6, 0]
        positions = media_positions(types, video_grid_thw=np.array([[2, 4, 6]]))
        np.testing.assert_array_equal(positions[:, :2], [[0, 1]] * 3)
        np.testing.assert_array_equal(positions[:, 2:8], [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]])
        np.testing.assert_array_equal(positions[:, 8], [5, 5, 5])  # advanced by max(height, width) = 3
        np.testing.assert_array_equal(positions[:, 9:15], [[6] * 6, [6, 6, 6, 7, 7, 7], [6, 7, 8, 6, 7, 8]])
        np.testing.assert_array_equal(positions[:, 15], [9, 9, 9])
        expected = upstream_rope_index(torch.zeros(1, len(types), dtype=torch.long), torch.tensor([types]),
                                       video_grid=torch.tensor([[2, 4, 6]]))
        np.testing.assert_array_equal(positions, expected)

    def test_placeholders_must_match_the_grids(self):
        cases = {
            "missing pair": ([0, *[2] * 6, 0], None, [[2, 4, 6]]),
            "extra pair": ([*[2] * 6, 0, *[2] * 6], None, [[1, 4, 6]]),
            "merged pairs": ([2] * 12, None, [[2, 4, 6]]),
            "wrong size": ([2] * 4, None, [[1, 4, 6]]),
            "video without grid": ([2] * 6, None, None),
            "image tokens for a video": ([1] * 6, None, [[1, 4, 6]]),
            "multi-frame image": ([1] * 6, [[2, 4, 6]], None),
            "odd grid": ([2] * 6, None, [[1, 3, 6]]),
            "zero frames": ([0], None, [[0, 4, 6]]),
            "unknown type": ([0, 3], None, None),
            "float grid": ([2] * 6, None, np.array([[1.0, 4.0, 6.0]])),
        }
        for name, (types, image_grid, video_grid) in cases.items():
            with self.subTest(name=name), self.assertRaises(RequestError):
                media_positions(types, None if image_grid is None else np.array(image_grid),
                                None if video_grid is None else np.array(video_grid))


class FeatureScatterTests(unittest.TestCase):
    def test_image_and_video_features_are_scattered_like_upstream(self):
        media = MediaProcessor(image_processor(), media_tokenizer())
        image = Image.new("RGB", (256, 256), "green")
        self.addCleanup(image.close)
        text = "q <|vision_start|><|video_pad|><|vision_end|> r <|vision_start|><|image_pad|><|vision_end|> s"
        encoded = media(text=[text], images=[image], videos=[sampled_video(3, 64, 96)], return_tensors="pt")
        ids, types = encoded["input_ids"][0], encoded["mm_token_type_ids"][0]
        hidden = 4
        image_features = torch.arange(int((types == 1).sum()) * hidden, dtype=torch.float32).reshape(-1, hidden) + 1000
        video_features = -torch.arange(int((types == 2).sum()) * hidden, dtype=torch.float32).reshape(-1, hidden) - 1000
        self.assertEqual((len(image_features), len(video_features)), (64, 12))

        def encoder(pixels, grid_thw, return_dict):
            if pixels is sent["pixel_values"]:
                self.assertTrue(torch.equal(grid_thw, encoded["image_grid_thw"]))
                return SimpleNamespace(pooler_output=image_features)
            self.assertIs(pixels, sent["pixel_values_videos"])
            self.assertTrue(torch.equal(grid_thw, encoded["video_grid_thw"]))
            return SimpleNamespace(pooler_output=video_features)
        sent = {key: encoded[key] for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")}
        vision = object.__new__(Vision)
        vision.device = "cpu"
        vision.config = SimpleNamespace(image_token_id=1, video_token_id=4, text_config=SimpleNamespace(hidden_size=hidden))
        vision.model = Mock(dtype=torch.float32, side_effect=encoder)
        table = torch.arange(5 * hidden, dtype=torch.float32).reshape(5, hidden)
        request = SimpleNamespace(input_ids=tuple(ids.tolist()),
                                  media={**sent, "token_offset": 0, "mm_token_type_ids": types.tolist()})
        embeddings, positions = vision.prepare(request, table.clone())

        # Upstream Qwen3_5Model.forward: masked_scatter images, then videos, by placeholder id.
        expected = table[ids]
        expected = expected.masked_scatter((ids == 1).unsqueeze(-1).expand_as(expected), image_features)
        expected = expected.masked_scatter((ids == 4).unsqueeze(-1).expand_as(expected), video_features)
        np.testing.assert_array_equal(embeddings, expected.numpy())
        self.assertEqual(vision.model.call_count, 2)
        np.testing.assert_array_equal(positions, upstream_rope_index(
            encoded["input_ids"], encoded["mm_token_type_ids"], encoded["image_grid_thw"].clone(),
            encoded["video_grid_thw"].clone()))

    def test_video_placeholder_in_text_fails_before_running_vision(self):
        vision = object.__new__(Vision)
        vision.device = "cpu"
        vision.config = SimpleNamespace(image_token_id=1, video_token_id=4, text_config=SimpleNamespace(hidden_size=4))
        vision.model = Mock(dtype=torch.float32)
        request = SimpleNamespace(input_ids=(0, 4, 4, 4, 4, 0, 4), media={
            "pixel_values_videos": torch.zeros(16, 1536), "video_grid_thw": torch.tensor([[1, 4, 4]]),
            "token_offset": 0, "mm_token_type_ids": [0, 2, 2, 2, 2, 0]})
        with self.assertRaisesRegex(RequestError, "placeholder tokens"):
            vision.prepare(request, torch.zeros(5, 4))
        vision.model.assert_not_called()


class ProcessorSettingsTests(unittest.TestCase):
    def test_pinned_video_settings_are_accepted(self):
        video_processor_settings(processor_config())

    def test_missing_or_changed_video_settings_are_rejected(self):
        for key, value in (("do_resize", False), ("do_sample_frames", False), ("fps", 30), ("max_frames", 100000),
                           ("patch_size", 14), ("temporal_patch_size", 1), ("image_mean", [0, 0, 0]),
                           ("video_processor_type", "Other"), ("max_pixels", 10 ** 9), ("return_metadata", True)):
            config = processor_config()
            config["video_processor"][key] = value
            with self.subTest(key=key), self.assertRaises(BundleError):
                video_processor_settings(config)
        for config in ({}, {"video_processor": []}, {"video_processor": None}):
            with self.subTest(config=config), self.assertRaises(BundleError):
                video_processor_settings(config)

    def test_adapter_shares_the_upstream_patch_constants(self):
        upstream = Qwen3VLVideoProcessor()
        self.assertEqual((VideoProcessor.merge_size, VideoProcessor.temporal_patch_size),
                         (upstream.merge_size, upstream.temporal_patch_size))
        self.assertEqual(VideoProcessor.model_input_names, upstream.model_input_names)
        self.assertEqual(worker.VIDEO_FPS, upstream.fps)


class VideoServiceTests(unittest.TestCase):
    def setUp(self):
        from clef_ane.engine.service import Service
        self.head, self.backbone = Mock(), Mock(max_tokens=1024)
        self.vision = Mock(supports_video=True)
        self.encoded = SimpleNamespace(input_ids=(1, 2), questions=(SimpleNamespace(question_id="q", option_ids=("true", "false")),))
        self.head.encode.return_value = self.encoded
        self.head.probabilities.return_value = {"q": np.array([.9, .1])}
        self.backbone.hidden.return_value = np.zeros((2, 2), np.float32)
        self.vision.prepare.return_value = np.ones((2, 2)), np.zeros((3, 2), dtype=int)
        self.service = Service(self.backbone, self.head, vision=self.vision)
        self.request = {"state": "Watch the clip.", "questions": {"q": {"type": "noul", "instructions": "Does it move?"}},
                        "videos": ["data:video/mp4;base64,AAAA"]}

    def test_video_decode_is_locked_and_then_text_uses_the_original_path(self):
        def decode(values):
            self.assertTrue(self.service.lock.locked())
            self.assertEqual(values, self.request["videos"])
            return [sampled_video(3)]
        with patch.object(video, "decode_videos", side_effect=decode) as decoder:
            response = self.service.systemone(self.request)
            self.assertEqual(response["usage"]["video_frames"], 3)
            self.assertEqual(response["answers"]["q"]["noul"], .9)
            self.assertEqual(self.request["videos"], ["data:video/mp4;base64,AAAA"])
            self.assertIs(self.head.encode.call_args.kwargs["processor"], self.vision.processor)
            self.assertIs(self.backbone.hidden.call_args.kwargs["inputs_embeds"], self.vision.prepare.return_value[0])
            text_response = self.service.systemone({**self.request, "videos": []})
            self.assertNotIn("video_frames", text_response["usage"])
            self.assertEqual(self.backbone.hidden.call_args.kwargs, {})
            decoder.assert_called_once()
            self.vision.prepare.assert_called_once()

    def test_image_only_bundle_rejects_video_before_decoding(self):
        self.vision.supports_video = False
        with patch.object(video, "decode_videos") as decoder, self.assertRaisesRegex(RequestError, "video processor"):
            self.service.systemone(self.request)
        decoder.assert_not_called()
        self.head.encode.assert_not_called()
        self.backbone.hidden.assert_not_called()

    def test_bad_question_precedes_decoding(self):
        with patch.object(video, "decode_videos") as decoder, self.assertRaises(RequestError):
            self.service.systemone({**self.request, "questions": {"q": {"type": "invalid"}}})
        decoder.assert_not_called()

    def test_decode_failure_closes_images_and_releases_lock(self):
        image = Image.new("RGB", (64, 64))
        with patch("clef_ane.vision.decode_images", return_value=[image]), patch.object(image, "close") as close, \
                patch.object(video, "decode_videos", side_effect=RequestError("invalid clip")):
            with self.assertRaisesRegex(RequestError, "invalid clip"):
                self.service.systemone({**self.request, "images": ["test"]})
            close.assert_called_once()
        self.assertFalse(self.service.lock.locked())
        self.vision.prepare.assert_not_called()
        self.backbone.hidden.assert_not_called()
        self.service.systemone({**self.request, "videos": []})

    def test_context_overflow_never_runs_vision_or_ane(self):
        self.head.encode.side_effect = ValueError("schema requires too many tokens")
        with patch.object(video, "decode_videos", return_value=[sampled_video(3)]), self.assertRaises(TooLong):
            self.service.systemone(self.request)
        self.vision.prepare.assert_not_called()
        self.backbone.hidden.assert_not_called()
        self.assertFalse(self.service.lock.locked())


class IncompleteClipTests(unittest.TestCase):
    """Clips cut off in transit must not be answered as if they were whole."""

    def test_truncated_webm_is_rejected(self):
        raw = encode_video("webm", list(range(0, 3000, 100)))
        with av.open(io.BytesIO(raw)) as container:
            self.assertEqual(sum(1 for _ in container.decode(video=0)), 30)
        with self.assertRaises(ValueError):
            worker.decode(raw[:len(raw) // 2], "matroska")

    def test_truncated_webm_data_url_is_rejected(self):
        raw = encode_video("webm", list(range(0, 3000, 100)))
        with self.assertRaises(RequestError):
            decode_videos([data_url(raw[:len(raw) // 2], "video/webm")])


class DecoderErrorMessageTests(unittest.TestCase):
    """Decoder failures reach the client as guidance, not FFmpeg internals."""

    def assert_guidance(self, message: str):
        for internal in ("Errno", "avcodec_open2", "last error log", "<none>", "moov atom"):
            self.assertNotIn(internal, message)

    def test_malformed_clip_message(self):
        with self.assertRaises(RequestError) as caught:
            decode_videos([data_url(b"not a video" * 20)])
        self.assert_guidance(str(caught.exception))
        self.assertIn("re-encode as H.264 MP4 or VP9 WebM", str(caught.exception))

    def test_oversized_frame_is_a_limit_error(self):
        with limits(MAX_FRAME_PIXELS=64 * 64 - 1):
            with self.assertRaises(ValueError) as caught:
                worker.decode(encode_video("mp4", [0, 500]), "mov")
        self.assert_guidance(str(caught.exception))
        self.assertIsInstance(caught.exception, worker.VideoLimitError)


if __name__ == "__main__":
    unittest.main()
