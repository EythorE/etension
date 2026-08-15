"""Validate the task itself, before any training.

Two questions must hold for the benchmark to mean anything:

1. The exact closed-form readout (perfect structural knowledge, perfect
   resolution) hits every probe ceiling: R^2 ~ 1 everywhere, zero boundary
   leakage, overall error at the noise floor.  This is "would perfect
   knowledge of the boundaries be enough" answered analytically — the
   structural prior is *sufficient* by construction.

2. Deliberately blurred readouts fail exactly the columns they should:
   - replacing the pair source v_j with a (2W+1)-window average collapses the
     far-pair column toward the ~1/(2W+1) ceiling while local and segment
     columns stay perfect.  This is the ceiling ANY mechanism that reaches
     distant positions only through W-wide averages inherits (the wavelet /
     pooled-pyramid objection, as a number);
   - snapping segment boundaries to a fixed grid (what a pooling grid does to
     a boundary it cannot align to) produces boundary leakage and degrades the
     segment column, while pairs are untouched.

If these hold, a mechanism's probe profile is attributable to the mechanism.
"""

import json
from pathlib import Path

import numpy as np

from .evaluate import EVAL_SEED, run_probes
from .task import TaskConfig, compute_targets, generate_batch


def closed_form(batch, pair_window=0, seg_grid=0):
    def predict(x_np):
        return np.stack([
            compute_targets(x_np[b], batch.metas[b], batch.cfg,
                            pair_window=pair_window, seg_grid=seg_grid)
            for b in range(x_np.shape[0])
        ])
    return predict


def main(out_path="results/task_validation.json", n_eval=64):
    cfg = TaskConfig()
    batch = generate_batch(np.random.default_rng(EVAL_SEED), n_eval, cfg)

    readouts = {
        "exact_oracle": closed_form(batch),
        "pair_blur_w8": closed_form(batch, pair_window=8),
        "pair_blur_w32": closed_form(batch, pair_window=32),
        "boundary_grid_16": closed_form(batch, seg_grid=16),
        "boundary_grid_64": closed_form(batch, seg_grid=64),
    }
    results = {name: run_probes(fn, batch) for name, fn in readouts.items()}

    ex = results["exact_oracle"]
    assert all(ex[k] > 0.999 for k in
               ("pair_near_r2", "pair_mid_r2", "pair_far_r2", "local_r2", "segment_r2"))
    assert ex["leakage_ratio"] < 1e-6
    # empirical noise power vs sigma^2: equal up to finite-sample fluctuation
    assert abs(ex["nmse_overall"] / ex["noise_floor_nmse"] - 1) < 0.05

    b8 = results["pair_blur_w8"]
    assert b8["pair_far_r2"] < 0.25 and b8["local_r2"] > 0.999 and b8["segment_r2"] > 0.999
    b32 = results["pair_blur_w32"]
    assert b32["pair_far_r2"] < 0.05

    g16 = results["boundary_grid_16"]
    assert g16["leakage_ratio"] > 0.1
    assert abs(g16["pair_far_r2"] - 1.0) < 1e-6  # pairs untouched by boundary blur

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    cols = ("pair_near_r2", "pair_mid_r2", "pair_far_r2", "segment_r2",
            "local_r2", "leakage_ratio", "nmse_overall")
    header = f"{'readout':<18}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    for name, r in results.items():
        print(f"{name:<18}" + "".join(f"{r[c]:>14.3f}" for c in cols))
    print(f"\nnoise floor nmse: {ex['noise_floor_nmse']:.3f}")
    print("task validation passed: exact oracle is perfect; blur fails exactly "
          "the intended columns.")
    return results


if __name__ == "__main__":
    main()
