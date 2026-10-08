"""Reference answers from the transformers bf16 model (MPS), saved for the regression tests.

Writes data/ref/golden.json: {record id: {question: {option: probability}}} for the six short records and two long ones.
Run once (needs the reference model, ~20 GB of memory); test_service.py then needs no reference model.
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_backbone import Head, load_text_model
from records import RECORDS, long_record

head, ref = Head(), load_text_model("mps", attention="sdpa")
records = RECORDS + [long_record(20000), long_record(60000)]
golden = {}
for record in records:
    enc = head.encode(record)
    ids = torch.tensor(enc.input_ids)[None].to("mps")
    with torch.no_grad():
        hidden = ref(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0].float().cpu()
    probs = head.probabilities(hidden, enc)
    golden[record["id"]] = {q.question_id: dict(zip(q.option_ids, map(float, probs[q.question_id]))) for q in enc.questions}
    print(record["id"], len(enc.input_ids), "tokens", flush=True)
out = Path(__file__).resolve().parents[2] / "data" / "ref" / "golden.json"
out.write_text(json.dumps(golden, indent=1))
print("wrote", out)
