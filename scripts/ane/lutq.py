"""4-bit and 6-bit lookup-table quantisation of a linear layer's weight: per group of output channels, 2^bits k-means centroids."""
import hashlib
from pathlib import Path

import numpy as np
import torch


def kmeans_lut(weight: np.ndarray, bits: int = 4, group: int = 16, iters: int = 40) -> tuple[np.ndarray, np.ndarray, int]:
    """weight [out, in] -> (indices uint8 [out, in], tables float16 [out/group, 2^bits], bits). Plain 1-D k-means per group."""
    n = 1 << bits
    out_ch, in_ch = weight.shape
    assert out_ch % group == 0
    v = torch.from_numpy(np.ascontiguousarray(weight, np.float32)).reshape(out_ch // group, group * in_ch)
    sample = v[:, :: max(1, v.shape[1] // 8192)]                      # initialise from a subsample
    # Weights are heavy-tailed (max is ~60x the rms). Half the centres start at quantiles, half spread evenly between the
    # group's min and max so the tails get centres too; that halves the 6-bit error compared with quantiles alone.
    quantile_centers = torch.quantile(sample, torch.linspace(0.5 / (n // 2), 1 - 0.5 / (n // 2), n // 2), dim=1).T
    lo, hi = v.min(1, keepdim=True).values, v.max(1, keepdim=True).values
    even_centers = lo + (hi - lo) * torch.linspace(0.02, 0.98, n // 2)[None]
    centers = torch.cat([quantile_centers, even_centers], 1).sort(dim=1).values.contiguous()   # [G, n]
    for _ in range(iters):
        bounds = (centers[:, 1:] + centers[:, :-1]) / 2
        idx = torch.searchsorted(bounds, v)                            # [G, N] in 0..15
        sums = torch.zeros_like(centers).scatter_add_(1, idx, v)
        counts = torch.zeros_like(centers).scatter_add_(1, idx, torch.ones_like(v))
        centers = torch.where(counts > 0, sums / counts.clamp(min=1), centers)
        centers, _ = centers.sort(dim=1)
    bounds = (centers[:, 1:] + centers[:, :-1]) / 2
    idx = torch.searchsorted(bounds, v)
    return idx.reshape(out_ch, in_ch).to(torch.uint8).numpy(), centers.to(torch.float16).numpy(), bits


def kmeans_lut4(weight: np.ndarray, group: int = 16, iters: int = 40):
    return kmeans_lut(weight, 4, group, iters)


def error(weight: np.ndarray, indices: np.ndarray, tables: np.ndarray, bits: int = 4) -> float:
    group = weight.shape[0] // tables.shape[0]
    deq = tables.astype(np.float32)[np.repeat(np.arange(tables.shape[0]), group)[:, None], indices]
    return float(np.sqrt(((deq - weight) ** 2).mean() / (weight ** 2).mean()))


CACHE = Path(__file__).resolve().parents[2] / "data" / "lutcache"


def kmeans_lut_cached(weight: np.ndarray, bits: int, group: int = 16) -> tuple[np.ndarray, np.ndarray, int]:
    """kmeans_lut, remembered on disk by the weight's content so the 5-minute quantisation runs once per model."""
    key = hashlib.sha1(np.ascontiguousarray(weight, np.float32).tobytes()).hexdigest()[:20]
    path = CACHE / f"lut{bits}_g{group}_{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["indices"], data["tables"], bits
    indices, tables, _ = kmeans_lut(weight, bits, group)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".tmp.npz"), indices=indices, tables=tables)
    path.with_suffix(".tmp.npz").rename(path)
    return indices, tables, bits
