"""Keep the ANE busy with Clef-Flash requests for N seconds (default 60), for measuring power from another terminal:

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 70 | grep -E "ANE Power|CPU Power|GPU Power|Combined"

Start this first, wait for "running", then start powermetrics; compare with an idle baseline (same command, nothing running).
"""
import os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lutq, mil
if int(os.environ.get("CLEF_LUT_BITS", 0)):   # e.g. CLEF_LUT_BITS=6 to run the 6-bit lookup-table weights
    mil.Graph.default_quantize = lambda w: lutq.kmeans_lut_cached(w, int(os.environ["CLEF_LUT_BITS"]), 16)
from clef_ane import Backbone
from ref_backbone import Head
from records import RECORDS

seconds = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 60
head = Head(); ane = Backbone(rows=128, verbose=False)
enc = [head.encode(r) for r in RECORDS]
ane.hidden(list(enc[0].input_ids))
print("running", flush=True)
if "--hold" in sys.argv:        # keep the model loaded on the ANE but send nothing: the "resident and idle" case
    time.sleep(seconds); print("held", seconds, "s with no requests"); sys.exit()
start, done, tokens = time.time(), 0, 0
while time.time() - start < seconds:
    e = enc[done % len(enc)]
    head.probabilities(torch.from_numpy(ane.hidden(list(e.input_ids))), e)
    done += 1; tokens += len(e.input_ids)
print(f"{done} requests, {tokens} tokens in {time.time() - start:.1f}s = {tokens / (time.time() - start):.0f} tokens/s, {(time.time() - start) / done:.2f} s/request")
