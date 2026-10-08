"""Per-layer error at long context, measured against an fp32 reference: ANE vs the bf16 model.

Usage: long_layers_test.py CHARS NUM_LAYERS     (the fp32 reference runs on the CPU, so keep NUM_LAYERS small)
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference
from clef_ane import Backbone
from ref_backbone import Head
from records import long_record

chars, nl = int(sys.argv[1]), int(sys.argv[2])
head = Head(); ids = list(head.encode(long_record(chars)).input_ids)
print(len(ids), "tokens, first", nl, "layers", flush=True)
embed = Backbone.__new__(Backbone)                      # only need the embedding table; avoid compiling all layers twice
from safetensors.torch import load_file
table = load_file(reference.DATA / "weights" / "embed.safetensors")["model.language_model.embed_tokens.weight"]
x32 = table[torch.tensor(ids)].float()
ane = Backbone(rows=128, num_layers=nl, verbose=False)
_, ane_layers = ane.hidden(ids, return_layers=True)
x16, h32, h16 = x32.to(torch.bfloat16).to("mps"), x32, x32.to(torch.bfloat16).to("mps")
rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
print(f"{'layer':<6}{'type':<18}{'ANE vs fp32':>12}{'bf16 vs fp32':>14}")
for i in range(nl):
    l32, cfg = reference.load_layer(i, attention="sdpa")
    t = time.time(); h32 = reference.layer_forward(l32, cfg, h32).detach(); del l32
    l16, cfg16 = reference.load_layer(i, attention="sdpa"); l16 = l16.to(torch.bfloat16).to("mps")
    h16 = reference.layer_forward(l16, cfg16, h16).detach(); del l16
    ref = h32.numpy().astype(np.float64)
    print(f"{i:<6}{l16.layer_type if False else ('full_attention' if i % 4 == 3 else 'linear_attention'):<18}{rms(ane_layers[i], ref):>12.4f}{rms(h16.float().cpu().numpy(), ref):>14.4f}   ({time.time() - t:.0f}s)", flush=True)
