"""How far is the bf16 model (how Clef-Flash is normally served) from the fp32 reference, on the same layer and tokens?"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference

R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
layer, _ = reference.load_layer()
x = reference.sample_hidden(R)
p32 = reference.pieces(layer, x)
layer16, _ = reference.load_layer()
layer16 = layer16.to(torch.bfloat16)
p16 = reference.pieces(layer16, x.to(torch.bfloat16))
rms = lambda a, b: float(torch.sqrt(((a.float() - b.float()) ** 2).mean() / (b.float() ** 2).mean()))
for name in ("xn", "conv", "core", "normed", "h1", "y"):
    print(f"bf16 vs fp32  {name:<7} rms rel err {rms(p16[name], p32[name]):.5f}")
