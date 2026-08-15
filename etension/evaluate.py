"""Stratified evaluation via counterfactual probes.

Aggregate error is refused as a headline number: local structure dominates any
average, and the question under test lives entirely in the long-range column.
Instead, each capability is measured by editing one input and checking whether
the model's output tracks the exact ground-truth change:

  pair probe (per distance band)
      Resample v at the pair sources of one distance band.  The target output
      changes by exactly a_pair * u_k * dv_j.  R^2 of the model's output change
      against the true change measures full-resolution retrieval at that
      distance.  A mechanism that reads a W-wide average around the source has
      an R^2 ceiling of ~1/W on this probe, because v is white noise.

  segment probe
      Resample u at one position near a segment start; every later position in
      the segment changes through the running-sum read.  Measures the triangle
      readout.

  local probe
      Resample v at a random position; the next position changes through the
      local product.  The easy column, reported so it can be seen NOT to
      differ between mechanisms.

  boundary leakage probe
      Resample every u in the segment *before* a boundary.  Ground truth after
      the boundary is exactly unchanged (asserted).  Any change in the model's
      prediction after the boundary is structure blur — content bleeding
      across a boundary that should be sharp.  Reported as
      rms(change after) / rms(change before); 0 is a perfectly sharp boundary.

All probes work on any sequence-to-sequence predictor (a closure from input
array to prediction array), so the same code evaluates trained models and the
closed-form oracle/blurred readouts used to validate the task itself.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models import Model, ModelConfig
from .structure import (
    attention_mass_in_structure,
    batch_union_mask,
    source_mass_at_pairs,
)
from .task import Batch, TaskConfig, generate_batch, CH_U, CH_V

EVAL_SEED = 123456
PAIR_GROUP_NAMES = ("near", "mid", "far")


def r2(dy_true: np.ndarray, dy_pred: np.ndarray) -> float:
    denom = float((dy_true**2).sum())
    if denom == 0:
        return float("nan")
    return 1.0 - float(((dy_pred - dy_true) ** 2).sum()) / denom


def probe_pair(predict, batch: Batch, rng, group: str):
    xv = batch.x.copy()
    rows = []
    for b, meta in enumerate(batch.metas):
        for j, k, name in meta["pairs"]:
            if name == group:
                xv[b, j, CH_V] = rng.standard_normal()
                rows.append((b, k))
    dy = batch.recompute_targets(xv) - batch.y
    dpred = predict(xv) - predict(batch.x)
    idx = tuple(np.array(rows).T)
    return r2(dy[idx], dpred[idx])


def probe_local(predict, batch: Batch, rng):
    N = batch.cfg.seq_len
    xv = batch.x.copy()
    rows = []
    for b, meta in enumerate(batch.metas):
        sources = {j for j, _k, _n in meta["pairs"]}
        while True:
            m = int(rng.integers(0, N - 1))
            if m not in sources:
                break
        xv[b, m, CH_V] = rng.standard_normal()
        rows.append((b, m + 1))
    dy = batch.recompute_targets(xv) - batch.y
    dpred = predict(xv) - predict(batch.x)
    idx = tuple(np.array(rows).T)
    return r2(dy[idx], dpred[idx])


def probe_segment(predict, batch: Batch, rng):
    N = batch.cfg.seq_len
    xv = batch.x.copy()
    rows = []
    for b, meta in enumerate(batch.metas):
        starts = meta["starts"]
        ends = np.append(starts[1:], N)
        seg = int(np.argmax(ends - starts))
        s, e = int(starts[seg]), int(ends[seg])
        m = s + 2
        xv[b, m, CH_U] = rng.standard_normal()
        rows.extend((b, i) for i in range(m + 1, e))
    dy = batch.recompute_targets(xv) - batch.y
    dpred = predict(xv) - predict(batch.x)
    idx = tuple(np.array(rows).T)
    return r2(dy[idx], dpred[idx])


def probe_leakage(predict, batch: Batch, rng, horizon: int = 8):
    """Returns rms(prediction change after boundary) / rms(before boundary)."""
    N = batch.cfg.seq_len
    xv = batch.x.copy()
    before_rows, after_rows = [], []
    for b, meta in enumerate(batch.metas):
        starts = meta["starts"]
        bi = int(np.argmin(np.abs(starts[1:] - N // 2))) + 1
        boundary, prev_start = int(starts[bi]), int(starts[bi - 1])
        xv[b, prev_start:boundary, CH_U] = rng.standard_normal(boundary - prev_start)
        before_rows.extend((b, i) for i in range(prev_start, boundary))
        after_rows.extend((b, i) for i in range(boundary, min(boundary + horizon, N)))
    dy = batch.recompute_targets(xv) - batch.y
    a_idx = tuple(np.array(after_rows).T)
    assert np.abs(dy[a_idx]).max() < 1e-4, "task bug: ground truth changed after boundary"
    dpred = predict(xv) - predict(batch.x)
    b_idx = tuple(np.array(before_rows).T)
    rms_after = float(np.sqrt((dpred[a_idx] ** 2).mean()))
    rms_before = float(np.sqrt((dpred[b_idx] ** 2).mean()))
    return rms_after / max(rms_before, 1e-9)


def run_probes(predict, batch: Batch, seed: int = 0) -> dict:
    """The full stratified suite for any predictor.  Fresh rng per probe so
    every predictor sees identical counterfactual edits."""
    out = {}
    for group in PAIR_GROUP_NAMES:
        out[f"pair_{group}_r2"] = probe_pair(predict, batch, np.random.default_rng(seed + 1), group)
    out["local_r2"] = probe_local(predict, batch, np.random.default_rng(seed + 2))
    out["segment_r2"] = probe_segment(predict, batch, np.random.default_rng(seed + 3))
    out["leakage_ratio"] = probe_leakage(predict, batch, np.random.default_rng(seed + 4))
    pred = predict(batch.x)
    var_y = float(batch.y.var())
    out["nmse_overall"] = float(((pred - batch.y) ** 2).mean()) / var_y
    out["noise_floor_nmse"] = batch.cfg.sigma**2 / var_y
    return out


def torch_predictor(model: Model, batch: Batch, **fwd_kwargs):
    """Wrap a model as a numpy predictor.  Oracle structure masks come from the
    batch metadata, which counterfactual edits never touch."""
    kwargs = dict(fwd_kwargs)
    if model.cfg.mixer == "oracle":
        kwargs["structure_mask"] = torch.from_numpy(
            batch_union_mask(batch.metas, batch.cfg.seq_len)
        )

    def predict(x_np):
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x_np), **kwargs)
        return pred.numpy()

    return predict


def structure_measurements(model: Model, batch: Batch) -> dict:
    """How much of the learned attention actually lives inside the ground-truth
    structure, and whether any head finds the far needle."""
    kwargs = {}
    if model.cfg.mixer == "oracle":
        kwargs["structure_mask"] = torch.from_numpy(
            batch_union_mask(batch.metas, batch.cfg.seq_len)
        )
    with torch.no_grad():
        _, extras = model(torch.from_numpy(batch.x), collect_attn=True, **kwargs)
    union = batch_union_mask(batch.metas, batch.cfg.seq_len)
    out = {"mass_in_structure": [], "far_source_mass": []}
    for attn in extras["attns"]:
        a = attn.numpy()
        out["mass_in_structure"].append(round(attention_mass_in_structure(a, union), 4))
        out["far_source_mass"].append(round(source_mass_at_pairs(a, batch.metas, "far"), 4))
    return out


def evaluate_checkpoint(ckpt_dir: Path, n_eval: int = 64, threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    cfg_all = json.loads((ckpt_dir / "config.json").read_text())
    model_cfg = ModelConfig.from_dict(cfg_all["model"])
    model = Model(model_cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", weights_only=True))
    model.eval()

    task_cfg = TaskConfig(seq_len=model_cfg.seq_len)
    batch = generate_batch(np.random.default_rng(EVAL_SEED), n_eval, task_cfg)

    metrics = {"mixer": model_cfg.mixer, "n_params": cfg_all["n_params"]}
    metrics.update(run_probes(torch_predictor(model, batch), batch))
    metrics["structure"] = structure_measurements(model, batch)

    if model_cfg.mixer == "full":
        # inference-time ablation: restrict the trained map to the ground-truth
        # structure with NO retraining.  If nothing degrades, the boundary
        # description captures everything the full map was doing.
        mask = torch.from_numpy(batch_union_mask(batch.metas, task_cfg.seq_len))
        ablated = run_probes(torch_predictor(model, batch, attn_mask=mask), batch)
        metrics["masked_to_structure"] = {
            k: ablated[k] for k in
            ("pair_near_r2", "pair_mid_r2", "pair_far_r2", "local_r2",
             "segment_r2", "leakage_ratio", "nmse_overall")
        }

    if model_cfg.mixer == "spanpred":
        # threshold-at-inference: replace the soft bias with a hard mask S>tau
        # and report how much survives, plus the budget actually spent.
        hard = run_probes(torch_predictor(model, batch, hard_structure=True), batch)
        with torch.no_grad():
            _, extras = model(torch.from_numpy(batch.x), hard_structure=True)
        density = extras["hard_mask"].float().sum().item() / (
            batch.x.shape[0] * task_cfg.seq_len
        )
        metrics["hard_threshold"] = {
            k: hard[k] for k in
            ("pair_near_r2", "pair_mid_r2", "pair_far_r2", "local_r2",
             "segment_r2", "leakage_ratio", "nmse_overall")
        }
        metrics["hard_threshold"]["mean_slots_per_row"] = round(density, 1)

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True, help="directory with model.pt + config.json")
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    ckpt_dir = Path(args.ckpt)
    metrics = evaluate_checkpoint(ckpt_dir, args.n_eval, args.threads)
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
