"""Long inputs: ANE backbone (context buckets) vs the transformers bf16 model on the same tokens, same head.

Usage: long_context_test.py CHARS [CHARS ...]     (e.g. 6000 20000 50000; ~3.3 characters per token for source code)
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_ane import Backbone
from ref_backbone import Head, load_text_model
from records import long_record

head = Head()
print("loading reference (bf16, mps, sdpa)...", flush=True)
ref = load_text_model("mps", attention="sdpa")
ane = Backbone(rows=128, verbose=False)
rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
for chars in map(int, sys.argv[1:]):
    record = long_record(chars)
    enc = head.encode(record); ids = list(enc.input_ids)
    if len(ids) > ane.max_tokens: print(f"{len(ids)} tokens: too long for the largest bucket"); continue
    t = time.time()
    with torch.no_grad():
        tid = torch.tensor(ids)[None].to("mps")
        h_ref = ref(input_ids=tid, attention_mask=torch.ones_like(tid), use_cache=False).last_hidden_state[0].float().cpu()
    ref_s = time.time() - t
    t = time.time(); h_ane = ane.hidden(ids); ane_s = time.time() - t
    p_ref, p_ane = head.probabilities(h_ref, enc), head.probabilities(torch.from_numpy(h_ane), enc)
    same = sum(int(p_ref[q].argmax() == p_ane[q].argmax()) for q in p_ref)
    worst = max(float(np.abs(p_ref[q] - p_ane[q]).max()) for q in p_ref)
    print(f"{len(ids):>6} tokens: ANE {ane_s:6.1f}s ({len(ids) / ane_s:4.0f} tok/s)  mps bf16 {ref_s:5.1f}s | hidden rms err {rms(h_ane, h_ref.numpy()):.4f} "
          f"| same answer {same}/{len(p_ref)}, worst prob diff {worst:.3f}", flush=True)
    for q in p_ref:
        print(f"         {q:<10} bf16 {np.round(p_ref[q], 3)}  ANE {np.round(p_ane[q], 3)}")
