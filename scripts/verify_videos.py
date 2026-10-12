"""Compare generated clips, mixed media, images, and text against the original BF16 forward.

Needs the original research weights and Apple Silicon. No network or external media.
The shared Pillow preprocessing and bounded PTS sampling are not a general quality benchmark.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_images import fixtures as image_fixtures, png, verify


def clip_url(colors: list[str], *, container: str = "mp4", seconds: int = 4, moving: bool = False) -> str:
    buffer = io.BytesIO()
    fps = 8
    with av.open(buffer, "w", format=container) as output:
        stream = output.add_stream("libvpx-vp9" if container == "webm" else "libx264", rate=fps)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        for index in range(seconds * fps):
            with Image.new("RGB", (256, 256), colors[min(len(colors) - 1, index * len(colors) // (seconds * fps))]) as image:
                if moving:
                    left = int(12 + 176 * index / (seconds * fps - 1))
                    ImageDraw.Draw(image).rectangle((left, 96, left + 56, 152), fill="red")
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    mime = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm"}[container]
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


def fixtures():
    colors = {"red": "Red", "blue": "Blue", "green": "Green", "white": "White"}
    def color(when):
        return {"type": "choice", "instructions": f"What is the main color at the {when} of the video?", "criteria": colors}
    questions = {"start": color("beginning"), "end": color("end")}
    forward, reverse = clip_url(["red", "blue"]), clip_url(["blue", "red"])
    rows = [
        ("video_red_blue", {"videos": [forward], "questions": questions}, {"start": "red", "end": "blue"}),
        ("video_blue_red", {"videos": [reverse], "questions": questions}, {"start": "blue", "end": "red"}),
        ("video_webm", {"videos": [clip_url(["red", "blue"], container="webm")], "questions": questions}, {"start": "red", "end": "blue"}),
        ("video_mov", {"videos": [clip_url(["blue"], container="mov", seconds=1)], "questions": {"color": color("beginning")}}, {"color": "blue"}),
        ("video_motion", {"videos": [clip_url(["white"], moving=True)], "questions": {"direction": {
            "type": "choice", "instructions": "Which direction does the red square move in the video?",
            "criteria": {"right": "Left to right", "left": "Right to left", "still": "It does not move"}}}}, {"direction": "right"}),
        ("mixed_image_video", {"images": [png(Image.new("RGB", (256, 256), "green"))], "videos": [forward], "questions": {
            "image": {"type": "choice", "instructions": "What is the main color of the still image (not the video)?", "criteria": colors},
            "start": color("beginning"), "end": color("end")}}, {"image": "green", "start": "red", "end": "blue"}),
        ("video_bucket_boundary", {"videos": [clip_url(["red", "blue"], seconds=14)], "questions": questions}, {"start": "red", "end": "blue"}),
        ("video_after_long", {"videos": [reverse], "questions": questions}, {"start": "blue", "end": "red"}),
    ]
    images = image_fixtures()
    return images[:-1] + [(name, {"state": "Decide from the attached images and video, in time order.", **record}, expected)
                         for name, record, expected in rows] + images[-1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".tmp/video-support/verification.json"))
    parser.add_argument("--vision-device", choices=("ane", "mps", "cpu"), default="ane")
    args = parser.parse_args()
    # Keep this observed Clef-Flash mistake visible; do not relabel green as the correct video color. Clef does not make it.
    verify(args.model_dir.expanduser().resolve(), args.output, args.vision_device, records=fixtures(), max_tokens=2048,
           known_reference_errors={"clef-flash": {"mixed_image_video": {"start": "green"}}})
