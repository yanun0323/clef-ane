"""Jev / SystemOne `POST /v1/systemone` on the ANE backbone: request dict in, response dict out (no HTTP here)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import torch

if __package__:
    from . import lutq, mil
    from .ref_backbone import Head, clef
    from .errors import RequestError, TooLong
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lutq
    import mil
    from ref_backbone import Head, clef
    from errors import RequestError, TooLong

# 6-bit lookup tables: 0 of 29 answers differ from the bf16 model (fp16: 1 near-tie), at about 3/4 of fp16's weight traffic
DEFAULT_LUT_BITS = 6


class Service:
    def __init__(self, backbone: Backbone, head: Head, name: str = "clef-flash", *, vision=None) -> None:
        self.backbone, self.head, self.name, self.vision = backbone, head, name, vision
        self.lock = threading.Lock()            # one ANE pipeline: requests run one at a time

    @classmethod
    def load(cls, name: str = "clef-flash", buckets: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384), verbose: bool = True,
             lut_bits: int = DEFAULT_LUT_BITS, lut_group: int = 16) -> "Service":
        """lut_bits: store the big linear layers as 4- or 6-bit lookup tables (0 = fp16). The first load quantises every
        layer (about 7 minutes) and caches the result in data/lutcache."""
        if __package__:
            from .clef_ane import Backbone
        else:
            from clef_ane import Backbone
        start = time.time()
        mil.Graph.default_quantize = (lambda w: lutq.kmeans_lut_cached(w, lut_bits, lut_group)) if lut_bits else None
        service = cls(Backbone(rows=128, buckets=buckets, verbose=verbose), Head(), name)
        service.systemone({"state": "warmup", "questions": {"w": {"type": "noul", "instructions": "Is this a warmup?"}}})
        if verbose:
            print(f"ready in {time.time() - start:.0f}s; {'fp16' if not lut_bits else f'{lut_bits}-bit lookup-table'} weights; "
                  f"context up to {service.backbone.max_tokens} tokens", flush=True)
        return service

    def systemone(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise RequestError("request body must be a JSON object")
        questions = request.get("questions")
        if "state" not in request:
            raise RequestError("state is required")
        if "media_kwargs" in request:
            raise RequestError("media_kwargs is not supported; media limits are set by the server")
        for field in ("images", "videos"):
            if request.get(field) is not None and not isinstance(request[field], list):
                raise RequestError(f"{field} must be a list of base64 data URLs")
        has_media = bool(request.get("images") or request.get("videos"))
        if has_media and self.vision is None:
            raise RequestError("this model bundle is text-only; use --model-dir with a media-capable bundle")
        if request.get("videos") and getattr(self.vision, "supports_video", False) is not True:
            raise RequestError("this bundle has no video processor settings; download the combined image/video bundle")
        if not isinstance(questions, dict) or not questions:
            raise RequestError("at least one question is required")
        if len(questions) > 64:
            raise RequestError("at most 64 questions per request")
        for question_id, question in questions.items():
            if not isinstance(question, dict) or not isinstance(question.get("type"), str) or question["type"] not in clef.QUESTION_TYPES:
                raise RequestError(f"{question_id}: type must be noul, choice, or score")
            kind, criteria = question["type"], question.get("criteria")
            if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
                raise RequestError(f"{question_id}: choice criteria must be a non-empty object")
            if kind == "score" and (not isinstance(criteria, list) or not criteria):
                raise RequestError(f"{question_id}: score criteria must be a non-empty list")
            if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
                raise RequestError(f"{question_id}: noul criteria must be an object")
        # Decode and preprocess inside the lock as well: concurrent requests cannot pile up decoded images.
        with self.lock:
            images, videos, video_frames = [], [], 0
            try:
                if request.get("images"):
                    from clef_ane.vision import decode_images
                    images = decode_images(request["images"])
                if request.get("videos"):
                    from clef_ane.video import decode_videos
                    videos = decode_videos(request["videos"])
                    video_frames = sum(len(video.frames) for video in videos)
                try:
                    if has_media:
                        encoded = self.head.encode({**request, "images": images, "videos": videos},
                                                   max_length=self.backbone.max_tokens, processor=self.vision.processor)
                    else:
                        encoded = self.head.encode(request, max_length=self.backbone.max_tokens)
                except ValueError as error:
                    if str(error).startswith("schema requires "):  # includes media, before any state truncation
                        raise TooLong(str(error)) from error
                    raise RequestError("cannot encode the request; check the state, media, and questions") from error
            finally:
                for image in images:
                    image.close()
                images.clear()
                videos.clear()
            if has_media:
                embeddings, positions = self.vision.prepare(encoded, self.backbone.embed)
                hidden = self.backbone.hidden(list(encoded.input_ids), inputs_embeds=embeddings, position_ids=positions)
            else:
                hidden = self.backbone.hidden(list(encoded.input_ids))
            probabilities = self.head.probabilities(torch.from_numpy(hidden), encoded)
        answers = {
            q.question_id: clef.systemone_answer(questions[q.question_id], dict(zip(q.option_ids, map(float, probabilities[q.question_id]))))
            for q in encoded.questions
        }
        usage = {"input_tokens": len(encoded.input_ids), "output_tokens": 0}
        if video_frames:
            usage["video_frames"] = video_frames
        return {"model": request.get("model") or self.name, "answers": answers, "usage": usage}
