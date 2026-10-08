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
    def __init__(self, backbone: Backbone, head: Head, name: str = "clef-flash") -> None:
        self.backbone, self.head, self.name = backbone, head, name
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
        if request.get("images") or request.get("videos"):
            raise RequestError("images and videos are not supported: this server runs the text backbone only")
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
        try:
            encoded = self.head.encode(request, max_length=self.backbone.max_tokens)
        except ValueError as error:                       # the schema by itself is longer than the largest bucket
            raise TooLong(str(error)) from error
        with self.lock:
            hidden = self.backbone.hidden(list(encoded.input_ids))
            probabilities = self.head.probabilities(torch.from_numpy(hidden), encoded)
        answers = {
            q.question_id: clef.systemone_answer(questions[q.question_id], dict(zip(q.option_ids, map(float, probabilities[q.question_id]))))
            for q in encoded.questions
        }
        return {"model": request.get("model") or self.name, "answers": answers,
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}
