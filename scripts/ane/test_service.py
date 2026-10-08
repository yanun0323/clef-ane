"""Regression tests for the ANE Clef-Flash service against the bf16 reference answers in data/ref/golden.json.

Run:  .venv/bin/python -I scripts/ane/test_service.py     (loads the model on the ANE, about 70 s, then ~2 min of requests)
Create the golden file first with make_golden.py if it is missing.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import RECORDS, long_record
from service import DEFAULT_LUT_BITS, RequestError, Service, TooLong

GOLDEN = json.loads((Path(__file__).resolve().parents[2] / "data" / "ref" / "golden.json").read_text())
CASES = {r["id"]: r for r in RECORDS + [long_record(20000), long_record(60000)]}
service = None
# Measured worst probability difference from the bf16 model: 0.012 with fp16 weights, 0.083 with the default 6-bit tables.
TOLERANCE = 0.03 if DEFAULT_LUT_BITS == 0 else 0.10


def option_probabilities(question: dict, answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global service
        service = Service.load(verbose=False)

    def ask(self, record: dict) -> dict:
        return service.systemone({k: v for k, v in record.items() if k != "id"})

    def test_answers_match_the_bf16_reference(self):
        for record_id, record in CASES.items():
            response = self.ask(record)
            for question_id, answer in response["answers"].items():
                got = option_probabilities(record["questions"][question_id], answer)
                want = GOLDEN[record_id][question_id]
                worst = max(abs(got[o] - want[o]) for o in want)
                self.assertLess(worst, TOLERANCE, f"{record_id}/{question_id}: probabilities differ by {worst:.3f}")
                ranked = sorted(want.values(), reverse=True)
                if ranked[0] - ranked[1] > 0.05:     # near-ties may legitimately flip; clear answers must not
                    self.assertEqual(max(got, key=got.get), max(want, key=want.get), f"{record_id}/{question_id}: different answer")

    def test_no_state_leaks_between_requests(self):
        short = CASES["invoice"]
        before = self.ask(short)
        self.ask(CASES["code_60000"])                 # fills every KV bucket and the recurrent state with unrelated data
        self.assertEqual(self.ask(short), before)

    def test_long_state_is_truncated_to_the_context_limit(self):
        response = self.ask(long_record(400000))
        self.assertLessEqual(response["usage"]["input_tokens"], service.backbone.max_tokens)
        self.assertIn("language", response["answers"])

    def test_invalid_requests(self):
        ok = {"type": "noul", "instructions": "Is it?"}
        for request, error in [({"questions": {"q": ok}}, RequestError),
                               ({"state": "x"}, RequestError),
                               ({"state": "x", "questions": {"q": {"type": "bogus"}}}, RequestError),
                               ({"state": "x", "questions": {"q": {"type": "choice"}}}, RequestError),
                               ({"state": "x", "images": ["a"], "questions": {"q": ok}}, RequestError),
                               ({"state": "x", "questions": {f"q{i}": ok for i in range(65)}}, RequestError)]:
            with self.assertRaises(error, msg=str(request)[:60]):
                service.systemone(request)

    def test_schema_too_long_for_the_context(self):
        huge = {f"q{i}": {"type": "choice", "instructions": "x " * 200, "criteria": {f"o{j}": "y " * 400 for j in range(60)}} for i in range(60)}
        with self.assertRaises(TooLong):
            service.systemone({"state": "x", "questions": huge})


if __name__ == "__main__":
    unittest.main(verbosity=2)
