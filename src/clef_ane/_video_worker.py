"""Bounded video decoding in a short-lived process; executable without importing Torch.

The parent supplies bytes through stdin, never a filename or URL. This is process
isolation and a deadline, not an OS security sandbox.
"""
from __future__ import annotations

import io
import json
import math
import sys

import numpy as np
from PIL import Image

MAX_VIDEO_BYTES = 16 * 1024 * 1024
MAX_VIDEO_SECONDS = 30.0
MAX_VIDEO_FRAMES = 64
MAX_FRAME_PIXELS = 4_000_000
MAX_DECODED_FRAMES = 1800
MAX_DECODED_PIXELS = 4_000_000_000
MAX_FRAME_SIDE = 4096
MAX_FRAME_ASPECT = 64
VIDEO_FPS = 2
VIDEO_PIXELS = 256 * 256
MAX_OUTPUT_BYTES = MAX_VIDEO_FRAMES * VIDEO_PIXELS * 3 + 65536
DECODE_TIMEOUT = 45


class VideoLimitError(ValueError):
    pass


def reject_external_io(*args, **kwargs):
    raise ValueError("external video references are not allowed")


def resize_shape(width: int, height: int) -> tuple[int, int]:
    """Pillow bicubic, Qwen's 32-pixel alignment, at most 64 visual tokens per temporal pair."""
    factor, minimum = 32, 64 * 64
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > VIDEO_PIXELS:
        scale = math.sqrt(width * height / VIDEO_PIXELS)
        h, w = max(factor, math.floor(height / scale / factor) * factor), max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < minimum:
        scale = math.sqrt(minimum / (width * height))
        h, w = math.ceil(height * scale / factor) * factor, math.ceil(width * scale / factor) * factor
    if h * w > VIDEO_PIXELS:
        raise ValueError("video aspect ratio cannot fit the processing size limit")
    return w, h


def decode(raw: bytes, demuxer: str) -> dict[str, np.ndarray]:
    import av

    if demuxer not in ("mov", "matroska"):
        raise ValueError("unsupported video container")
    if len(raw) > MAX_VIDEO_BYTES:
        raise VideoLimitError("video exceeds 16 MiB; send a smaller clip")
    av.logging.set_level(av.logging.PANIC)
    allowed = {"h264", "hevc", "mpeg4", "av1"} if demuxer == "mov" else {"vp8", "vp9", "av1"}
    # find_stream_info can open decoders while probing, before av.open returns.
    options = {"protocol_whitelist": "", "enable_drefs": "0", "use_absolute_path": "0", "max_streams": "8",
               "max_pixels": str(MAX_FRAME_PIXELS), "codec_whitelist": ",".join(sorted(allowed | {"libdav1d"}))}
    frames, times = [], []
    with av.open(io.BytesIO(raw), mode="r", format=demuxer, options=options, io_open=reject_external_io,
                 timeout=(5.0, 10.0)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("send a clip with exactly one video track")
        stream = container.streams.video[0]
        if stream.codec_context.codec.canonical_name not in allowed:
            raise ValueError("unsupported video codec; re-encode as H.264 MP4 or VP9 WebM")
        width, height = stream.codec_context.width, stream.codec_context.height
        if width * height > MAX_FRAME_PIXELS or max(width, height) > MAX_FRAME_SIDE:
            raise VideoLimitError("video decoding exceeds the frame or pixel limit; reduce resolution, frame rate, or length")
        if stream.sample_aspect_ratio not in (None, 0, 1):
            raise ValueError("send a video with square pixels")
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else (
            float(container.duration / av.time_base) if container.duration is not None else 0.0)
        # Matroska's container duration is an end timestamp, unlike a track's duration.
        if stream.duration is None and container.duration is not None and demuxer == "matroska":
            start = float(stream.start_time * stream.time_base) if stream.start_time is not None else float((container.start_time or 0) / av.time_base)
            duration -= start
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid video duration")
        if duration > MAX_VIDEO_SECONDS + .001:
            raise VideoLimitError("video exceeds 30 seconds; trim the clip")
        if stream.frames > MAX_DECODED_FRAMES:
            raise VideoLimitError("video exceeds 1800 source frames; reduce its frame rate or length")
        stream.codec_context.thread_count = 2
        stream.codec_context.options = {"max_pixels": str(MAX_FRAME_PIXELS), "err_detect": "explode"}
        initial_size, origin, previous, next_sample, last_frame = None, None, None, 0.0, None
        count = pixels = 0
        rotation, target_size = 0, None

        def capture(frame, timestamp):
            if len(frames) >= MAX_VIDEO_FRAMES:
                raise VideoLimitError("video needs too many sampled frames; trim the clip")
            with frame.to_image() as image:
                if rotation:
                    oriented = image.transpose({90: Image.Transpose.ROTATE_90, -90: Image.Transpose.ROTATE_270,
                                                180: Image.Transpose.ROTATE_180, -180: Image.Transpose.ROTATE_180}[rotation])
                else:
                    oriented = image
                try:
                    with oriented.resize(target_size, resample=Image.Resampling.BICUBIC) as resized:
                        frames.append(np.array(resized.convert("RGB"), dtype=np.uint8))
                finally:
                    if oriented is not image:
                        oriented.close()
            times.append(timestamp)

        for frame in container.decode(stream):
            count += 1
            frame_pixels = frame.width * frame.height
            pixels += frame_pixels
            if (count > MAX_DECODED_FRAMES or pixels > MAX_DECODED_PIXELS or frame_pixels > MAX_FRAME_PIXELS
                    or max(frame.width, frame.height) > MAX_FRAME_SIDE):
                raise VideoLimitError("video decoding exceeds the frame or pixel limit; reduce resolution, frame rate, or length")
            if min(frame.width, frame.height) < 32 or max(frame.width, frame.height) / min(frame.width, frame.height) > MAX_FRAME_ASPECT:
                raise ValueError("video dimensions must be at least 32 pixels and its aspect ratio at most 64:1")
            display = frame.side_data.get("DISPLAYMATRIX")
            if display is not None:
                matrix = np.frombuffer(display, dtype=np.int32)
                if matrix.shape != (9,):
                    raise ValueError("invalid video display matrix; re-encode the clip")
                a, b, c, d = (int(matrix[index]) for index in (0, 1, 3, 4))
                if a * d - b * c < 0:
                    raise ValueError("mirrored video metadata is not supported; re-encode the displayed orientation")
            current_rotation = frame.rotation
            if current_rotation not in (0, 90, -90, 180, -180):
                raise ValueError("unsupported video rotation; re-encode the clip")
            if initial_size is None:
                initial_size = frame.width, frame.height
                rotation = current_rotation
                display_size = initial_size[::-1] if abs(rotation) == 90 else initial_size
                target_size = resize_shape(*display_size)
            elif initial_size != (frame.width, frame.height) or (display is not None and current_rotation != rotation):
                raise ValueError("changing video dimensions or rotation are not supported; re-encode the clip")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("video frames need timestamps; re-encode the clip")
            absolute = float(frame.pts * frame.time_base)
            if not math.isfinite(absolute):
                raise ValueError("invalid video frame timestamp")
            if origin is None:
                origin = absolute
            timestamp = absolute - origin
            if previous is not None and timestamp <= previous:
                raise ValueError("video frame timestamps must increase; re-encode the clip")
            if timestamp > MAX_VIDEO_SECONDS + .001:
                raise VideoLimitError("video exceeds 30 seconds; trim the clip")
            previous, last_frame = timestamp, frame
            if timestamp + 1e-7 >= next_sample:
                capture(frame, timestamp)
                next_sample = (math.floor(timestamp * VIDEO_FPS + 1e-7) + 1) / VIDEO_FPS
        if last_frame is None:
            raise ValueError("the video contains no decodable frames")
        # MOV edit lists mark pre-trim samples as discarded; they are not output frames.
        discarded = sum(entry.is_discard for entry in stream.index_entries) if demuxer == "mov" and stream.frames else 0
        if stream.frames and stream.frames - discarded != count:
            raise ValueError("the video is incomplete; send a complete clip")
        end = previous + float((last_frame.duration or 0) * last_frame.time_base)
        if end > MAX_VIDEO_SECONDS + .001:
            raise VideoLimitError("video exceeds 30 seconds; trim the clip")
        if duration > end + .1:
            raise ValueError("the video ends before its declared duration; send a complete clip or trim its audio-only tail")
        if previous > times[-1] + 1e-7:
            capture(last_frame, previous)
    return {"frames": np.stack(frames), "timestamps": np.array(times, dtype=np.float64),
            "duration": np.array(max(duration, end)), "decoded_frames": np.array(count)}


def main() -> int:
    import av

    try:
        # Keep a deadline even if the parent crashes or is killed during native decoding.
        import signal
        signal.alarm(DECODE_TIMEOUT)
        # Bound file-backed stdout/stderr in addition to the parent's wall-clock timeout.
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        raw = sys.stdin.buffer.read(MAX_VIDEO_BYTES + 1)
        result = decode(raw, sys.argv[1] if len(sys.argv) == 2 else "")
        np.savez(sys.stdout.buffer, **result)
        return 0
    except VideoLimitError as error:
        print(json.dumps({"status": 413, "message": str(error)}), file=sys.stderr)
    except av.FFmpegError:
        print(json.dumps({"status": 400, "message": "invalid or incomplete video; re-encode as H.264 MP4 or VP9 WebM"}), file=sys.stderr)
    except (ValueError, MemoryError) as error:
        print(json.dumps({"status": 400, "message": str(error)}), file=sys.stderr)
    except Exception:
        print(json.dumps({"status": 400, "message": "invalid or incomplete video; re-encode as H.264 MP4 or VP9 WebM"}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
