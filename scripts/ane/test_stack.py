"""Chain the first N Gated DeltaNet layers (0, 1, 2) on the ANE and watch how the error grows layer by layer.

Three numbers per layer, all against the fp32 PyTorch chain:
  chained   - ANE fed its own previous output (what a real run does)
  isolated  - ANE fed the exact fp32 input of that layer (error added by this layer alone)
  bf16      - the bf16 PyTorch model chained the same way (how the model is normally served)
"""
import sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers, reference
from ane import Kernel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ROWS = 128
x0 = reference.sample_hidden()
n = (x0.shape[0] // ROWS) * ROWS
x0 = x0[:n]
rms = lambda a, b: float(np.sqrt(((np.asarray(a, np.float64) - b) ** 2).mean() / (b ** 2).mean()))

torch_layers = [reference.load_layer(i)[0] for i in range(N)]
bf16_layers = [reference.load_layer(i)[0].to(torch.bfloat16) for i in range(N)]
weights = []
for i in range(N):
    prefix = f"model.language_model.layers.{i}."
    weights.append({k[len(prefix):]: v.float().numpy() for k, v in load_file(reference.layer_file(i)).items() if k.startswith(prefix)})
kernels = [(Kernel(layers.gdn_mixer(w, rows=ROWS)), Kernel(layers.mlp(w, rows=ROWS))) for w in weights]

def run_ane(i, x):  # x: [n, 4096] float -> [n, 4096] float64
    mixer, mlp = kernels[i]
    conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16)
    rec = np.zeros((1, layers.VALUE_HEADS, layers.HEAD_DIM, layers.HEAD_DIM), np.float16)
    out = []
    for s in range(0, n, ROWS):
        h, conv, rec = mixer(np.asarray(x[s:s + ROWS], np.float32).T[None, :, None, :], conv, rec)
        out.append(mlp(h)[0][0, :, 0, :].T)
    return np.concatenate(out).astype(np.float64)

ref_in, chained, bf = x0, x0.numpy().astype(np.float64), x0.to(torch.bfloat16)
print(f"{'layer':<6}{'|y| max':>9}{'chained':>10}{'isolated':>10}{'bf16':>9}")
for i in range(N):
    ref_out = reference.pieces(torch_layers[i], ref_in)["y"]
    want = ref_out.double().numpy()
    isolated = run_ane(i, ref_in.numpy())
    chained = run_ane(i, chained)
    bf = reference.pieces(bf16_layers[i], bf)["y"]
    print(f"{i:<6}{np.abs(want).max():9.1f}{rms(chained, want):10.4f}{rms(isolated, want):10.4f}{rms(bf.double().numpy(), want):9.4f}")
    ref_in = ref_out
