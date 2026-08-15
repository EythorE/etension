"""Synthetic seq2seq task with fully controlled ground-truth attention structure.

The output at each position is a sum of three multiplicative interactions plus
noise.  Each component is designed to probe one part of the question "which
compression of the attention map is the right one":

  local     y_i += a_loc * u_i * v_{i-1}
            Trivially within reach of every mechanism.  This is the easy column
            that must never be averaged with the others.

  segment   y_i += a_seg * v_i * (sum_{m=s_i..i} u_m) / sqrt(i - s_i + 1)
            s_i is the start of the segment containing i.  The attention row
            needed is exactly the triangle anchored on the diagonal: uniform
            over [s_i, i], zero before the boundary.  Full rank as a map,
            described completely by the boundary positions.  The *read* is an
            average (smooth) but the *support* is sharp: any mechanism that
            blurs the boundary leaks the previous segment's content in.

  pair      y_k += a_pair * u_k * v_j   for sampled pairs (j, k), j < k
            Source and target carry the same random key vector in dedicated
            input channels; the model must match keys to find j.  v is white
            noise, so the average of any window around j carries almost none of
            v_j: retrieval must be at full resolution, sharpness is enforced by
            construction, not by assumption.  Pair distances are sampled in
            three bands (near / mid / far) so accuracy can be reported per
            distance and never aggregated.

Ground-truth metadata (boundaries, pairs) is returned with every sequence and
is used for (a) oracle-structure attention, (b) supervision of the learned
structure predictor, (c) stratified counterfactual evaluation, and
(d) visualisation.  Targets are a deterministic function of the input channels
given metadata and the frozen noise draw, so counterfactual probes can edit an
input channel and recompute the exact ground-truth change.
"""

from dataclasses import dataclass, field

import numpy as np

# input channel layout
CH_V, CH_U, CH_B, CH_ROLE = 0, 1, 2, 3
KEY_DIM = 4
N_CHANNELS = 4 + KEY_DIM


@dataclass(frozen=True)
class TaskConfig:
    seq_len: int = 256
    seg_len_min: int = 12
    seg_len_max: int = 40
    pairs_per_group: int = 3
    # (name, min_distance, max_distance); max is clamped to seq_len - 2
    pair_groups: tuple = (("near", 2, 16), ("mid", 24, 64), ("far", 96, 248))
    a_loc: float = 1.0
    a_seg: float = 1.0
    a_pair: float = 2.0
    sigma: float = 0.3


def sample_meta(rng: np.random.Generator, cfg: TaskConfig) -> dict:
    """Sample segment boundaries and pair positions (the ground-truth structure)."""
    N = cfg.seq_len
    starts = [0]
    while True:
        step = int(rng.integers(cfg.seg_len_min, cfg.seg_len_max + 1))
        if starts[-1] + step >= N:
            break
        starts.append(starts[-1] + step)
    starts = np.asarray(starts, dtype=np.int64)
    seg_id = np.searchsorted(starts, np.arange(N), side="right") - 1
    seg_start = starts[seg_id]

    used: set = set()
    pairs = []
    for name, lo, hi in cfg.pair_groups:
        hi = min(hi, N - 2)
        for _ in range(cfg.pairs_per_group):
            for _attempt in range(200):
                d = int(rng.integers(lo, hi + 1))
                j = int(rng.integers(0, N - d))
                k = j + d
                if j == k or j in used or k in used:
                    continue
                used.update((j, k))
                pairs.append((j, k, name))
                break
    return {"starts": starts, "seg_start": seg_start, "pairs": pairs}


def sample_inputs(rng: np.random.Generator, meta: dict, cfg: TaskConfig) -> np.ndarray:
    N = cfg.seq_len
    x = np.zeros((N, N_CHANNELS), dtype=np.float32)
    x[:, CH_V] = rng.standard_normal(N)
    x[:, CH_U] = rng.standard_normal(N)
    x[meta["starts"], CH_B] = 1.0
    for j, k, _name in meta["pairs"]:
        key = rng.standard_normal(KEY_DIM)
        key = (key / np.linalg.norm(key)).astype(np.float32)  # unit sphere:
        # matched keys dot to exactly 1, unmatched to ~0 — addressing is sharp
        # by construction, so retrieval failures are the mechanism's, not the
        # address's
        x[j, CH_ROLE] = 1.0
        x[k, CH_ROLE] = -1.0
        x[j, 4:] = key
        x[k, 4:] = key
    return x


def compute_targets(
    x: np.ndarray,
    meta: dict,
    cfg: TaskConfig,
    eps: np.ndarray | None = None,
    pair_window: int = 0,
    seg_grid: int = 0,
) -> np.ndarray:
    """Deterministic targets from input channels + metadata (+ frozen noise).

    pair_window > 0 and seg_grid > 0 are *deliberately corrupted* closed-form
    readouts used for task validation:
      pair_window W : the pair read uses the mean of v over [j-W, j+W] instead
                      of v_j — the best any mechanism that only sees a
                      (2W+1)-wide average around the source can do.
      seg_grid g    : the segment read snaps the boundary to the previous
                      multiple of g — what a fixed pooling grid does to a
                      boundary it cannot align to.
    """
    N = cfg.seq_len
    v = x[:, CH_V].astype(np.float64)
    u = x[:, CH_U].astype(np.float64)

    y_loc = np.zeros(N)
    y_loc[1:] = u[1:] * v[:-1]

    seg_start = meta["seg_start"]
    if seg_grid > 0:
        seg_start = (seg_start // seg_grid) * seg_grid
    cs = np.cumsum(u)
    base = cs[seg_start] - u[seg_start]
    t = np.arange(N) - seg_start + 1
    y_seg = v * (cs - base) / np.sqrt(t)

    y_pair = np.zeros(N)
    for j, k, _name in meta["pairs"]:
        if pair_window > 0:
            lo, hi = max(0, j - pair_window), min(N, j + pair_window + 1)
            src = v[lo:hi].mean()
        else:
            src = v[j]
        y_pair[k] += u[k] * src

    y = cfg.a_loc * y_loc + cfg.a_seg * y_seg + cfg.a_pair * y_pair
    if eps is not None:
        y = y + cfg.sigma * eps
    return y.astype(np.float32)


@dataclass
class Batch:
    x: np.ndarray  # (B, N, N_CHANNELS) float32
    y: np.ndarray  # (B, N) float32
    eps: np.ndarray  # (B, N) float64 frozen noise draw
    metas: list  # list of meta dicts
    cfg: TaskConfig

    def recompute_targets(self, x_new: np.ndarray) -> np.ndarray:
        """Exact targets for edited inputs, same metadata and same noise draw."""
        return np.stack(
            [
                compute_targets(x_new[b], self.metas[b], self.cfg, self.eps[b])
                for b in range(x_new.shape[0])
            ]
        )


def generate_batch(rng: np.random.Generator, batch_size: int, cfg: TaskConfig) -> Batch:
    xs, ys, epss, metas = [], [], [], []
    for _ in range(batch_size):
        meta = sample_meta(rng, cfg)
        x = sample_inputs(rng, meta, cfg)
        eps = rng.standard_normal(cfg.seq_len)
        y = compute_targets(x, meta, cfg, eps)
        xs.append(x)
        ys.append(y)
        epss.append(eps)
        metas.append(meta)
    return Batch(np.stack(xs), np.stack(ys), np.stack(epss), metas, cfg)
