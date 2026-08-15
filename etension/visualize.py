"""Look at what the model decided mattered.

For each trained mechanism: the ground-truth structure (triangles anchored on
the diagonal + pair links), then every layer/head attention map on the same
held-out sequence.  For mechanisms without an explicit map (pooled, lowrank)
the *effective* per-position map is materialised.  For spanpred the predicted
soft structure S is shown next to the ground truth.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate import EVAL_SEED
from .models import Model, ModelConfig
from .structure import batch_union_mask, union_mask
from .task import TaskConfig, generate_batch


def _imshow(ax, m, title, cmap="magma", vmax=None):
    ax.imshow(m, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def gt_panel(ax, meta, N):
    m = union_mask(meta, N).astype(float) * 0.6
    for j, k, _name in meta["pairs"]:
        m[k, j] = 1.0
    _imshow(ax, m, "ground-truth structure", cmap="gray_r", vmax=1.0)


def render(ckpt_dir: Path, out_path: Path, seq_index: int = 0):
    cfg_all = json.loads((ckpt_dir / "config.json").read_text())
    model_cfg = ModelConfig.from_dict(cfg_all["model"])
    model = Model(model_cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", weights_only=True))
    model.eval()

    task_cfg = TaskConfig(seq_len=model_cfg.seq_len)
    batch = generate_batch(np.random.default_rng(EVAL_SEED), seq_index + 1, task_cfg)
    N = task_cfg.seq_len
    x = torch.from_numpy(batch.x[seq_index : seq_index + 1])
    meta = batch.metas[seq_index]

    kwargs = {}
    if model_cfg.mixer == "oracle":
        kwargs["structure_mask"] = torch.from_numpy(
            batch_union_mask([meta], N)
        )
    with torch.no_grad():
        _, extras = model(x, collect_attn=True, **kwargs)

    L, H = model_cfg.layers, model_cfg.heads
    ncols = H + 1
    fig, axes = plt.subplots(L, ncols, figsize=(2.2 * ncols, 2.2 * L))
    axes = np.atleast_2d(axes)

    gt_panel(axes[0, 0], meta, N)
    if model_cfg.mixer == "spanpred":
        S = extras["S"][0].numpy()
        _imshow(axes[1, 0], S, "predicted structure S", cmap="gray_r", vmax=1.0)
    else:
        axes[1, 0].axis("off") if L > 1 else None

    for layer in range(L):
        attn = extras["attns"][layer][0].numpy()  # (H, N, N)
        for h in range(H):
            a = attn[h]
            _imshow(axes[layer, h + 1], a, f"L{layer} H{h}",
                    vmax=max(float(np.quantile(a, 0.999)), 1e-3))

    fig.suptitle(f"{model_cfg.mixer} — attention maps vs ground-truth structure", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seq-index", type=int, default=0)
    args = p.parse_args()
    ckpt_dir = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt_dir.parent / "figs" / f"{ckpt_dir.name}.png"
    render(ckpt_dir, out, args.seq_index)


if __name__ == "__main__":
    main()
