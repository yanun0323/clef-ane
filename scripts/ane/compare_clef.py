"""Clef-Flash decisions: ANE backbone vs the transformers bf16 backbone, same head, several records."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_ane import Backbone
from ref_backbone import Head, load_text_model
from records import RECORDS

BUCKETS = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (1024,)
NUM = int(sys.argv[2]) if len(sys.argv) > 2 else len(RECORDS)
head = Head()
print("loading reference model (bf16, mps)...", flush=True)
ref = load_text_model("mps")
print("loading ANE backbone...", flush=True)
ane = Backbone(rows=128, buckets=BUCKETS)

captured = []
for layer in ref.layers:
    layer.register_forward_hook(lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))

rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
agree, rows = 0, 0
for n, record in enumerate(RECORDS[:NUM]):
    enc = head.encode(record)
    ids = torch.tensor(enc.input_ids)[None]
    captured.clear()
    with torch.no_grad():
        ref_hidden = ref(input_ids=ids.to("mps"), attention_mask=torch.ones_like(ids).to("mps"), use_cache=False).last_hidden_state[0].float().cpu()
    ref_layers = [c[0] for c in captured]
    t0 = time.time()
    ane_hidden, ane_layers = ane.hidden(list(enc.input_ids), return_layers=True)
    ane_secs = time.time() - t0
    p_ref, p_ane = head.probabilities(ref_hidden, enc), head.probabilities(torch.from_numpy(ane_hidden), enc)
    print(f"\n[{record['id']}] {len(enc.input_ids)} tokens, ANE backbone {ane_secs:.2f}s ({len(enc.input_ids) / ane_secs:.0f} tok/s incl. padding), "
          f"final hidden rms err {rms(ane_hidden, ref_hidden.numpy()):.4f}, max |hidden| seen {max(ane.trace):.0f}")
    if n == 0:
        print("  per-layer error (rms rel vs bf16 model):", " ".join(f"{rms(a, b.numpy()):.3f}" for a, b in zip(ane_layers, ref_layers)))
    for q, pr in p_ref.items():
        pa = p_ane[q]
        kl = float((pr * (np.log(pr + 1e-9) - np.log(pa + 1e-9))).sum())
        ok = int(pr.argmax() == pa.argmax()); agree += ok; rows += 1
        print(f"  {q:<22} bf16 {np.round(pr, 3)}  ANE {np.round(pa, 3)}  max|dp| {np.abs(pr - pa).max():.3f}  KL {kl:.4f}  {'same answer' if ok else 'DIFFERENT ANSWER'}")
print(f"\nsame top answer on {agree}/{rows} questions")
