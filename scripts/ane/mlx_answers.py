"""Answers of the MLX 4-bit GPU model on the same cases, for scoring against golden.json. Run with .venv-mlx/bin/python -I."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "clef-flash-4bit")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import clef_mlx
from records import RECORDS, long_record
model = clef_mlx.load(str(ROOT / "models" / "clef-flash-4bit"))
answers = {}
for record in RECORDS + [long_record(20000), long_record(60000)]:
    request = {k: v for k, v in record.items() if k != "id"}
    answers[record["id"]] = {q: {str(o): float(p) for o, p in probs.items()} for q, probs in model.predict(request).items()}
(ROOT / "data" / "ref" / "mlx4_answers.json").write_text(json.dumps(answers))
print("done")
