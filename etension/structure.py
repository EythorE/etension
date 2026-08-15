"""Ground-truth structure masks and structure-related measurements.

The structural prior under test: an attention row is described by
  - a triangle anchored on the diagonal (attend within the current segment),
  - a small local band, and
  - a sparse set of pair links (k -> j).
Everything here builds that description as an explicit (N, N) support so it can
be used as an oracle attention mask, as supervision for the learned predictor,
and as the reference when measuring how much of a trained model's attention
mass actually lives inside the structure.
"""

import numpy as np


def triangle_mask(meta: dict, N: int) -> np.ndarray:
    idx = np.arange(N)
    j = idx[None, :]
    i = idx[:, None]
    seg_start = meta["seg_start"][:, None]
    return (j >= seg_start) & (j <= i)


def band_mask(N: int, width: int = 2) -> np.ndarray:
    idx = np.arange(N)
    j = idx[None, :]
    i = idx[:, None]
    return (j <= i) & (i - j <= width)


def pair_mask(meta: dict, N: int) -> np.ndarray:
    m = np.zeros((N, N), dtype=bool)
    for j, k, _name in meta["pairs"]:
        m[k, j] = True
    return m


def union_mask(meta: dict, N: int, band: int = 2) -> np.ndarray:
    """The full structural description: triangle + pair links + local band."""
    return triangle_mask(meta, N) | pair_mask(meta, N) | band_mask(N, band)


def batch_union_mask(metas: list, N: int, band: int = 2) -> np.ndarray:
    return np.stack([union_mask(m, N, band) for m in metas])


def batch_pair_mask(metas: list, N: int) -> np.ndarray:
    return np.stack([pair_mask(m, N) for m in metas])


def attention_mass_in_structure(attn: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total attention mass that falls inside the structure mask.

    attn: (B, H, N, N) row-stochastic; mask: (B, N, N) bool.
    """
    m = mask[:, None].astype(attn.dtype)
    return float((attn * m).sum() / attn.sum())


def source_mass_at_pairs(attn: np.ndarray, metas: list, group: str, halo: int = 1) -> float:
    """At pair-target rows of a distance group: attention mass on the source
    column (+/- halo), best head.  'Did any head find the needle?'"""
    masses = []
    for b, meta in enumerate(metas):
        for j, k, name in meta["pairs"]:
            if name != group:
                continue
            lo, hi = max(0, j - halo), j + halo + 1
            masses.append(attn[b, :, k, lo:hi].sum(axis=-1).max())
    return float(np.mean(masses)) if masses else float("nan")
