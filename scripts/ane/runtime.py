"""Request execution shared by source-weight and exported-bundle backbones."""
from __future__ import annotations

import numpy as np
import torch

if __package__:
    from . import layers
else:
    import layers


class BlockState:
    """Several consecutive layers compiled as one program (layers.linear_block), with every layer's recurrent state.

    The program's inputs are the hidden states followed by the state tensors, and its outputs come in the same order.
    """

    def __init__(self, kernel) -> None:
        self.kernel = kernel
        self.reset()

    def reset(self) -> None:
        self.state = [np.zeros(t.shape, np.float16) for t in self.kernel.inputs[1:]]

    def __call__(self, h: np.ndarray, start: int, position_embeddings=None) -> np.ndarray:
        h, *self.state = self.kernel(h, *self.state)
        return h

    def close(self) -> None:
        self.kernel.close()


class LayerState:
    """State management for a mixer + MLP pair, independent of how the programs are loaded.

    State sizes come from the programs' own inputs, so Clef-Flash and Clef share this code.
    A full-attention layer whose MLP was compiled into the next block has mlp = None.
    """

    def _shape(self, kernel, name: str) -> tuple[int, ...]:
        return next(t.shape for t in kernel.inputs if t.name == name)

    def allocate_cache(self) -> None:
        """Host copy of a full-attention layer's keys and values, as long as the largest context program."""
        _, kv_heads, _, width = self._shape(self.mixers[self.buckets[-1]], "k_cache")
        self.master_k = np.zeros((1, kv_heads, self.buckets[-1], width), np.float16)
        self.master_v = np.zeros_like(self.master_k)

    def reset(self) -> None:
        self.active = None
        if self.kind == "linear_attention":
            self.conv = np.zeros(self._shape(self.mixer, "conv_in"), np.float16)
            self.rec = np.zeros(self._shape(self.mixer, "rec_in"), np.float16)

    def _write_cache(self, bucket: int, start: int, stop: int) -> None:
        kernel = self.mixers[bucket]
        _, kv_heads, _, width = self.master_k.shape
        for head in range(kv_heads):
            offset = (head * bucket + start) * width
            kernel.write_region("k_cache", offset, self.master_k[0, head, start:stop])
            kernel.write_region("v_cache", offset, self.master_v[0, head, start:stop])

    def __call__(self, h: np.ndarray, start: int, position_embeddings=None) -> np.ndarray:
        if self.kind == "linear_attention":
            h, self.conv, self.rec = self.mixer(h, self.conv, self.rec)
        else:
            stop = start + self.rows
            bucket = next(c for c in self.buckets if c >= stop)
            if bucket != self.active:
                self._write_cache(bucket, 0, start)
                self.active = bucket
            cos, sin = (layers.rope_tables(start, self.rows) if position_embeddings is None
                        else tuple(table[:, :, start:stop] for table in position_embeddings))
            h, k, v = self.mixers[bucket](h, None, None, cos, sin, layers.cache_mask(start, bucket))
            self.master_k[:, :, start:stop], self.master_v[:, :, start:stop] = k, v
            self._write_cache(bucket, start, stop)
        return h if self.mlp is None else self.mlp(h)[0]

    def close(self) -> None:
        kernels = [getattr(self, "mlp", None), getattr(self, "mixer", None), *getattr(self, "mixers", {}).values()]
        for kernel in kernels:
            if kernel is not None:
                kernel.close()


class BackboneRunner:
    def hidden(self, ids: list[int], return_layers: bool = False, track_max: bool = False, *,
               inputs_embeds: np.ndarray | None = None, position_ids: np.ndarray | None = None):
        n = len(ids)
        if not n:
            raise ValueError("input must contain at least one token")
        padded = -(-n // self.rows) * self.rows
        if padded > self.max_tokens:
            raise ValueError(f"{n} tokens need a context of {padded}; the largest bucket is {self.max_tokens}")
        position_embeddings = None
        if (inputs_embeds is None) != (position_ids is None):
            raise ValueError("custom embeddings and position_ids must be supplied together")
        if inputs_embeds is not None:
            if (inputs_embeds.shape != (n, self.config["hidden_size"]) or not np.isfinite(inputs_embeds).all()
                    or np.any(np.abs(inputs_embeds) > np.finfo(np.float16).max)):
                raise ValueError("custom embeddings must be finite, fp16-safe vectors matching the input tokens")
            if (position_ids.shape != (3, n) or not np.issubdtype(position_ids.dtype, np.integer)
                    or np.any(position_ids < 0) or np.any(position_ids >= self.config.get("max_position_embeddings", 262144))):
                raise ValueError("position_ids must contain three valid integer positions per input token")
            padding = np.broadcast_to(np.arange(padded - n) + int(position_ids.max()) + 1, (3, padded - n))
            positions = np.concatenate([position_ids, padding], axis=1)
            position_embeddings = layers.rope_tables(0, padded, positions)
        x = np.zeros((1, self.config["hidden_size"], 1, padded), np.float16)
        embeddings = self.embed[torch.tensor(ids)].float().numpy() if inputs_embeds is None else inputs_embeds
        x[0, :, 0, :n] = embeddings.T
        for layer in self.layers:
            layer.reset()
        self.trace, per_layer = [], []
        for layer in self.layers:
            if position_embeddings is None:
                x = np.concatenate([layer(x[..., s:s + self.rows], s) for s in range(0, padded, self.rows)], axis=3)
            else:
                x = np.concatenate([layer(x[..., s:s + self.rows], s, position_embeddings)
                                    for s in range(0, padded, self.rows)], axis=3)
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
