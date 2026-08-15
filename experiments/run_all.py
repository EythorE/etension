"""Run the whole benchmark: validate the task, train every mechanism on the
identical data stream, evaluate with the stratified probe suite, render maps,
and write results/summary.md.

    python -m experiments.run_all --steps 3000
"""

import argparse
import json
from pathlib import Path

from etension import validate_task
from etension.evaluate import evaluate_checkpoint
from etension.train import build_parser as train_parser, train
from etension.visualize import render

DEFAULT_MIXERS = ["full", "local", "pooled", "lowrank", "oracle", "spanpred"]

COLS = [
    ("pair_near_r2", "pair near"),
    ("pair_mid_r2", "pair mid"),
    ("pair_far_r2", "pair FAR"),
    ("segment_r2", "segment"),
    ("local_r2", "local"),
    ("leakage_ratio", "leakage"),
    ("nmse_overall", "nmse"),
]


def summarize(results: dict, out_dir: Path, noise_floor: float):
    lines = [
        "# Results",
        "",
        "R^2 of predicted output change against exact ground-truth change under "
        "counterfactual input edits (1.0 = perfect tracking, 0 = blind). "
        "Columns are reported separately on purpose: local is easy and would "
        "dominate any average; the question lives in `pair FAR`.",
        "",
        f"Overall NMSE noise floor: {noise_floor:.3f} (irreducible).",
        "",
        "| mixer | " + " | ".join(label for _, label in COLS) + " |",
        "|" + "---|" * (len(COLS) + 1),
    ]
    for mixer, m in results.items():
        cells = [f"{m[k]:.3f}" if m.get(k) == m.get(k) else "—" for k, _ in COLS]
        lines.append(f"| {mixer} | " + " | ".join(cells) + " |")

    for mixer, m in results.items():
        for key, title in (
            ("masked_to_structure",
             "full attention, inference-time masked to ground-truth structure "
             "(no retraining) — does anything live outside the boundary description?"),
            ("hard_threshold",
             "spanpred with the soft bias replaced by a hard threshold at "
             "inference (YOLO-style)"),
        ):
            if key in m:
                lines += ["", f"## {mixer}: {title}", ""]
                lines.append("| " + " | ".join(k for k in m[key]) + " |")
                lines.append("|" + "---|" * len(m[key]))
                lines.append(
                    "| " + " | ".join(
                        f"{v:.3f}" if isinstance(v, float) else str(v)
                        for v in m[key].values()
                    ) + " |"
                )

    lines += ["", "## Attention mass inside the ground-truth structure (per layer)", ""]
    lines.append("| mixer | mass in structure | mass on far source (best head) |")
    lines.append("|---|---|---|")
    for mixer, m in results.items():
        s = m["structure"]
        lines.append(
            f"| {mixer} | {s['mass_in_structure']} | {s['far_source_mass']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixers", nargs="*", default=DEFAULT_MIXERS)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--n-eval", type=int, default=64)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_task.main(out_dir / "task_validation.json", n_eval=args.n_eval)
    noise_floor = validation["exact_oracle"]["noise_floor_nmse"]

    results = {}
    for mixer in args.mixers:
        targs = train_parser().parse_args(
            ["--mixer", mixer, "--steps", str(args.steps), "--out", str(out_dir)]
        )
        ckpt_dir = train(targs)
        metrics = evaluate_checkpoint(ckpt_dir, n_eval=args.n_eval)
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        render(ckpt_dir, out_dir / "figs" / f"{mixer}.png")
        results[mixer] = metrics

    summarize(results, out_dir, noise_floor)


if __name__ == "__main__":
    main()
