"""How many full-size layer programs can one process keep compiled and loaded on the ANE?"""
import sys, time, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers
from ane import Kernel, CompileError

rng = np.random.default_rng(0)
def rand(*s): return (rng.standard_normal(s) * 0.02).astype(np.float32)
def weights():
    H, F = 4096, 12288
    return {"input_layernorm.weight": rand(H), "post_attention_layernorm.weight": rand(H),
            "linear_attn.in_proj_qkv.weight": rand(8192, H), "linear_attn.in_proj_z.weight": rand(H, H),
            "linear_attn.in_proj_b.weight": rand(32, H), "linear_attn.in_proj_a.weight": rand(32, H),
            "linear_attn.conv1d.weight": rand(8192, 1, 4), "linear_attn.dt_bias": rand(32), "linear_attn.A_log": rand(32),
            "linear_attn.norm.weight": rand(128), "linear_attn.out_proj.weight": rand(H, H),
            "mlp.gate_proj.weight": rand(F, H), "mlp.up_proj.weight": rand(F, H), "mlp.down_proj.weight": rand(H, F)}
kept = []
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for i in range(limit):
    w = weights()
    try:
        t0 = time.time(); a = Kernel(layers.gdn_mixer(w, rows=128)); b = Kernel(layers.mlp(w, rows=128))
    except CompileError as e:
        print(f"layer {i}: FAILED after {len(kept) * 2} programs: {str(e)[:200]}"); break
    kept += [a, b]
    if i % 4 == 3 or i == limit - 1:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(__import__('os').getpid())], capture_output=True, text=True).stdout.strip()
        print(f"{len(kept)} programs loaded ({len(kept)//2} layers, ~{len(kept)//2*0.44:.1f} GB weights), last pair {time.time()-t0:.1f}s, process RSS {int(rss)/1e6:.1f} GB", flush=True)
x = np.zeros((1, 4096, 1, 128), np.float16)
print("last program still runs:", kept[-1](x)[0].shape)
