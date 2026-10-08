"""Run the bf16 reference once per record and save every layer's output (float32) for offline diagnosis."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_backbone import Head, load_text_model
from records import RECORDS

head, ref = Head(), load_text_model("mps")
captured = []
for layer in ref.layers:
    layer.register_forward_hook(lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
out = Path(__file__).resolve().parents[2] / "data" / "ref"
for record in RECORDS:
    enc = head.encode(record); ids = torch.tensor(enc.input_ids)[None]; captured.clear()
    with torch.no_grad():
        final = ref(input_ids=ids.to("mps"), attention_mask=torch.ones_like(ids).to("mps"), use_cache=False).last_hidden_state[0].float().cpu()
    with torch.no_grad():
        emb = ref.embed_tokens(ids.to("mps"))[0].float().cpu()
    np.savez(out / f"{record['id']}.npz", ids=np.array(enc.input_ids), emb=emb.numpy(), layers=np.stack([c[0].numpy() for c in captured]), final=final.numpy())
    print(record["id"], len(enc.input_ids), "tokens saved", flush=True)
