import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clef_ane.engine import mil
from clef_ane.engine.service import RequestError, Service


class ServiceValidationTests(unittest.TestCase):
    def test_invalid_questions_fail_before_running_the_model(self):
        head, backbone = Mock(), Mock()
        service = Service(backbone, head)
        for question in ({"type": []}, {"type": "choice", "criteria": ["a"]},
                         {"type": "score", "criteria": {"a": "b"}}, {"type": "noul", "criteria": ["yes"]}):
            with self.subTest(question=question), self.assertRaises(RequestError):
                service.systemone({"state": "test", "questions": {"q": question}})
        head.encode.assert_not_called()
        backbone.hidden.assert_not_called()

    def test_loader_respects_explicit_precision_and_group(self):
        stub = SimpleNamespace(Backbone=Mock())
        original = mil.Graph.default_quantize
        self.addCleanup(setattr, mil.Graph, "default_quantize", original)
        with patch.dict(sys.modules, {"clef_ane.engine.clef_ane": stub}), \
                patch("clef_ane.engine.service.Head"), patch.object(Service, "systemone"), \
                patch("clef_ane.engine.service.lutq.kmeans_lut_cached") as quantize:
            Service.load(verbose=False, lut_bits=4, lut_group=32)
            marker = object()
            mil.Graph.default_quantize(marker)
            quantize.assert_called_once_with(marker, 4, 32)
            Service.load(verbose=False, lut_bits=0)
            self.assertIsNone(mil.Graph.default_quantize)
