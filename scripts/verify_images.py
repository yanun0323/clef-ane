"""Compare image decisions with the original BF16 multimodal forward pass, then exercise HTTP.

Requires the original text/vision research weights, an image-capable exported bundle, and an
Apple Silicon Mac. Uses generated test images; no network, external image files, or API keys.
This is a regression check, not a general vision-quality benchmark.
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from clef_ane.bundle import load_bundle
from clef_ane.engine.ref_backbone import Head, clef, load_text_model
from clef_ane.engine.service import Service
from clef_ane.native import build_bridge, require_apple_silicon
from clef_ane.server import create_server
from clef_ane.vision import Vision, decode_images
from clef_ane.video import decode_videos


def png(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def fixtures() -> list[tuple[str, dict, dict]]:
    red, blue = png(Image.new("RGB", (256, 256), "red")), png(Image.new("RGB", (256, 256), "blue"))
    color = {"type": "choice", "instructions": "What is the main color of the image?",
             "criteria": {"red": "Red", "blue": "Blue", "green": "Green", "white": "White"}}
    shape = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(shape)
    draw.rectangle((32, 64, 144, 176), fill="red")
    draw.ellipse((224, 64, 336, 176), fill="blue")
    label = Image.new("RGB", (512, 256), "white")
    ImageDraw.Draw(label).text((88, 75), "PAID", font=ImageFont.load_default(size=90), fill="black")
    resized = Image.new("RGB", (1200, 800), "blue")
    records = [
        ("text_before", {"state": "The invoice has been paid.", "questions": {"paid": {"type": "noul", "instructions": "Has the invoice been paid?"}}}, {"paid": "true"}),
        ("red", {"images": [red], "questions": {"color": color}}, {"color": "red"}),
        ("blue", {"images": [blue], "questions": {"color": color}}, {"color": "blue"}),
        ("layout", {"images": [png(shape)], "questions": {
            "left": {"type": "noul", "instructions": "Is the red square to the left of the blue circle?"},
            "circle_color": {"type": "choice", "instructions": "What color is the circle?", "criteria": color["criteria"]}}}, {"left": "true", "circle_color": "blue"}),
        ("label", {"images": [png(label)], "questions": {"word": {"type": "choice", "instructions": "What word is written in the image?",
            "criteria": {"paid": "PAID", "unpaid": "UNPAID", "void": "VOID"}}}}, {"word": "paid"}),
        ("two_images", {"images": [red, blue], "questions": {
            "first": {**color, "instructions": "What is the main color of the FIRST image?"},
            "second": {**color, "instructions": "What is the main color of the SECOND image?"}}}, {"first": "red", "second": "blue"}),
        ("two_images_reversed", {"images": [blue, red], "questions": {
            "first": {**color, "instructions": "What is the main color of the FIRST image?"},
            "second": {**color, "instructions": "What is the main color of the SECOND image?"}}}, {"first": "blue", "second": "red"}),
        ("resized", {"images": [png(resized)], "questions": {"color": color}}, {"color": "blue"}),
        ("text_after", {"state": "The invoice has NOT been paid.", "questions": {"paid": {"type": "noul", "instructions": "Has the invoice been paid?"}}}, {"paid": "false"}),
    ]
    return [(name, {"state": "Decide from the attached images.", **record}, expected) for name, record, expected in records]


def winners(encoded, probabilities):
    return {question.question_id: question.option_ids[int(np.argmax(probabilities[question.question_id]))] for question in encoded.questions}


def verify(model_dir: Path, output: Path, device: str, *, records=None, max_tokens: int = 2048,
           known_reference_errors: dict | None = None) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    require_apple_silicon()
    torch.set_num_threads(4)
    manifest = load_bundle(model_dir)
    head = Head(model_dir=model_dir)
    vision = Vision(model_dir, head.tokenizer, device="mps")
    records, expected = fixtures() if records is None else records, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".requests.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Loading the original BF16 multimodal reference...", flush=True)
    text = load_text_model(device="mps", attention="eager")
    vision._load()
    # Construct the upstream wrapper without allocating a second 9B model. Its forward,
    # visual scatter, and position calculation remain the original Transformers code.
    with torch.device("meta"):
        reference = Qwen3_5Model(vision.config)
    reference.language_model, reference.visual = text, vision.model
    reference.eval()
    assert not any(parameter.is_meta for parameter in reference.parameters())
    for name, record, semantic in records:
        images = decode_images(record.get("images"))
        try:
            videos = decode_videos(record.get("videos"))
            encoded = head.encode({**record, "images": images, "videos": videos}, max_length=max_tokens, processor=vision.processor)
        finally:
            for image in images:
                image.close()
        del videos
        batch = clef.collate_records([encoded], head.tokenizer.eos_token_id, torch.device("mps"))
        started = time.monotonic()
        with torch.inference_mode():
            hidden = reference(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               **batch["media"], use_cache=False).last_hidden_state[0].float().cpu()
        probabilities = head.probabilities(hidden, encoded)
        expected[name] = {"encoded": encoded, "hidden": hidden.numpy(), "probabilities": probabilities,
                          "winners": winners(encoded, probabilities), "reference_seconds": time.monotonic() - started}
        print(f"BF16 {name}: {expected[name]['winners']}", flush=True)
    del reference, text, batch, hidden
    vision.model = None
    gc.collect()
    torch.mps.empty_cache()
    vision = Vision(model_dir, head.tokenizer, device=device)

    os.environ["CLEF_ANE_BRIDGE"] = str(build_bridge())
    from clef_ane.model import BundleBackbone
    backbone = BundleBackbone(model_dir, manifest, max_tokens=max_tokens)
    service = Service(backbone, head, vision=vision)
    summary = {"vision_device": vision.device, "preprocessing": "Pillow; sampled video PTS; no audio",
               "max_tokens": max_tokens, "known_reference_errors": known_reference_errors or {}, "records": [], "http": {}}
    latest = {}
    original_probabilities = head.probabilities
    def capture(hidden, encoded):
        probabilities = original_probabilities(hidden, encoded)
        latest.update(hidden=hidden.numpy(), encoded=encoded, probabilities=probabilities)
        return probabilities
    try:
        with patch.object(head, "probabilities", side_effect=capture):
            for name, record, semantic in records:
                started = time.monotonic()
                response = service.systemone(record)
                elapsed = time.monotonic() - started
                ref = expected[name]
                probabilities = latest["probabilities"]
                actual = winners(latest["encoded"], probabilities)
                assert latest["encoded"].input_ids == ref["encoded"].input_ids
                delta = max(float(np.abs(probabilities[key] - ref["probabilities"][key]).max()) for key in probabilities)
                difference = latest["hidden"] - ref["hidden"]
                relative_rmse = float(np.sqrt(np.mean(difference**2) / np.mean(ref["hidden"]**2)))
                row = {"name": name, "reference": ref["winners"], "ane": actual, "semantic_expected": semantic,
                       "answers_match": actual == ref["winners"], "semantic_match": actual == semantic,
                       "max_probability_error": delta, "hidden_relative_rmse": relative_rmse,
                       "input_tokens": response["usage"]["input_tokens"], "ane_seconds": elapsed,
                       "reference_seconds": ref["reference_seconds"], "response": response}
                summary["records"].append(row)
                print(f"ANE {name}: {actual}, max probability difference {delta:.6f}, {elapsed:.2f}s", flush=True)

        with create_server(service, port=0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            def post(record):
                request = urllib.request.Request(url + "/v1/systemone", data=json.dumps(record).encode(), headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as result:
                        return result.status, json.load(result)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)
            try:
                with urllib.request.urlopen(url + "/health", timeout=10) as result:
                    summary["http"]["health"] = json.load(result)
                status, body = post(records[1][1])
                assert status == 200 and body == summary["records"][1]["response"]
                summary["http"]["media_status"] = status
                invalid = {**records[1][1], "images": ["https://127.0.0.1/private"]}
                assert post(invalid)[0] == 400
                assert post({**records[1][1], "images": [png(Image.new("RGB", (32, 32), "red"))] * 5})[0] == 413
                assert post({**records[0][1], "videos": ["movie.mp4"]})[0] == 400
                assert post({**records[1][1], "media_kwargs": {"max_pixels": 999999999}})[0] == 400
                summary["http"].update(remote_url_status=400, excessive_images_status=413, video_status=400, media_kwargs_status=400)
                for name, record, _ in records:
                    if record.get("videos"):
                        status, body = post(record)
                        assert status == 200 and body == next(row["response"] for row in summary["records"] if row["name"] == name)
                        assert post({**record, "videos": record["videos"] * 2})[0] == 413
                        assert post({**record, "videos": ["https://127.0.0.1/private.mp4"]})[0] == 400
                        summary["http"].update(valid_video_status=200, excessive_videos_status=413, remote_video_status=400)
                        break
            finally:
                server.shutdown()
                thread.join(timeout=10)
    finally:
        vision.close()
        backbone.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
    assert all(row["answers_match"] for row in summary["records"]), f"Answers differ; see {output}"
    unexpected = {}
    correct = total = 0
    for row in summary["records"]:
        for key, answer in row["ane"].items():
            total += 1
            if answer == row["semantic_expected"][key]:
                correct += 1
            elif answer != (known_reference_errors or {}).get(row["name"], {}).get(key):
                unexpected.setdefault(row["name"], {})[key] = answer
    assert not unexpected, f"Unexpected synthetic-answer errors: {unexpected}; see {output}"
    print(f"BF16 option agreement: {total}/{total}; synthetic expected answers: {correct}/{total}.", flush=True)
    print(f"Media regression checks passed (known BF16 errors are retained). Results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".tmp/image-support/verification.json"))
    parser.add_argument("--vision-device", choices=("ane", "mps", "cpu"), default="ane")
    args = parser.parse_args()
    verify(args.model_dir.expanduser().resolve(), args.output, args.vision_device)
