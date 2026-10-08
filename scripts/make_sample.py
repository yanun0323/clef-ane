"""Build a realistic Clef prompt, tokenize it, and fetch just the embedding rows it needs (HTTP range requests).

Writes data/sample_tokens.npy and data/sample_embed.npy (float32, [tokens, 4096]).
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "hf-meta"))
from joint_schema_model import encode_record  # noqa: E402  (the model's own record encoder)

tokenizer = AutoTokenizer.from_pretrained(root / "data")
record = {
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue",
                          "notes": "Customer disputes the second line item and asked for a call back on Friday."}},
    "questions": {
        "status": {"type": "choice", "instructions": "What is the invoice status?",
                   "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today"]},
    },
}
encoded = encode_record(tokenizer, record)
ids = np.array(encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"])
print("tokens:", len(ids))

base = "https://huggingface.co/Cloudflare/clef-flash/resolve/main"
shard = f"{base}/model-00001-of-00004.safetensors"
def get(start, end):
    return urllib.request.urlopen(urllib.request.Request(shard, headers={"Range": f"bytes={start}-{end}"})).read()
size = struct.unpack("<Q", get(0, 7))[0]
info = json.loads(get(8, 8 + size - 1))["model.language_model.embed_tokens.weight"]
begin = 8 + size + info["data_offsets"][0]
dim = info["shape"][1]
rows = []
for token in ids:
    raw = bytearray(get(begin + int(token) * dim * 2, begin + (int(token) + 1) * dim * 2 - 1))
    rows.append(torch.frombuffer(raw, dtype=torch.bfloat16).float().numpy())
np.save(root / "data" / "sample_tokens.npy", ids)
np.save(root / "data" / "sample_embed.npy", np.stack(rows))
print("saved", np.stack(rows).shape)
