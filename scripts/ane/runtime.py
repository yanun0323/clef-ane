"""Request execution shared by source-weight and exported-bundle backbones."""
from __future__ import annotations

import numpy as np
import torch

if __package__:
    from . import layers
else:
    import layers


class LayerState:
    """State management for a mixer + MLP pair, independent of how the programs are loaded."""

    def reset(self) -> None:
        self.active = None
        if self.kind == "linear_attention":
            self.conv = np.zeros((1, layers.CONV_DIM, 1, 3), np.float16)
            self.rec = np.zeros((1, layers.VALUE_HEADS, layers.HEAD_DIM, layers.HEAD_DIM), np.float16)

    def _write_cache(self, bucket: int, start: int, stop: int) -> None:
        kernel = self.mixers[bucket]
        for head in range(layers.KV_HEADS):
            offset = (head * bucket + start) * layers.ATT_DIM
            kernel.write_region("k_cache", offset, self.master_k[0, head, start:stop])
            kernel.write_region("v_cache", offset, self.master_v[0, head, start:stop])

    def __call__(self, h: np.ndarray, start: int) -> np.ndarray:
        if self.kind == "linear_attention":
            h, self.conv, self.rec = self.mixer(h, self.conv, self.rec)
        else:
            stop = start + self.rows
            bucket = next(c for c in self.buckets if c >= stop)
            if bucket != self.active:
                self._write_cache(bucket, 0, start)
                self.active = bucket
            cos, sin = layers.rope_tables(start, self.rows)
            h, k, v = self.mixers[bucket](h, None, None, cos, sin, layers.cache_mask(start, bucket))
            self.master_k[:, :, start:stop], self.master_v[:, :, start:stop] = k, v
            self._write_cache(bucket, start, stop)
        return self.mlp(h)[0]

    def close(self) -> None:
        kernels = [getattr(self, "mlp", None), getattr(self, "mixer", None), *getattr(self, "mixers", {}).values()]
        for kernel in kernels:
            if kernel is not None:
                kernel.close()


class BackboneRunner:
    def hidden(self, ids: list[int], return_layers: bool = False, track_max: bool = False):
        n = len(ids)
        if not n:
            raise ValueError("input must contain at least one token")
        padded = -(-n // self.rows) * self.rows
        if padded > self.max_tokens:
            raise ValueError(f"{n} tokens need a context of {padded}; the largest bucket is {self.max_tokens}")
        x = np.zeros((1, self.config["hidden_size"], 1, padded), np.float16)
        x[0, :, 0, :n] = self.embed[torch.tensor(ids)].float().numpy().T
        for layer in self.layers:
            layer.reset()
        self.trace, per_layer = [], []
        for layer in self.layers:
            x = np.concatenate([layer(x[..., s:s + self.rows], s) for s in range(0, padded, self.rows)], axis=3)
            if track_max or return_layers:
                self.trace.append(float(np.abs(x[..., :n]).max()))
            if return_layers:
                per_layer.append(x[0, :, 0, :n].T.astype(np.float32))
        h = x[0, :, 0, :n].T.astype(np.float32)
        h = h * (1.0 / np.sqrt((h * h).mean(-1, keepdims=True) + self.config["rms_norm_eps"], dtype=np.float32))
        h *= 1.0 + self.norm_w
        return (h, per_layer) if return_layers else h

    def close(self) -> None:
        for layer in self.layers:
            layer.close()
