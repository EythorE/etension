"""Train one mixing mechanism on the synthetic task.

Data is generated fresh every batch (an infinite stream), so there is no
overfitting axis — differences between mechanisms are representational, not
statistical.  All mixers see the identical data stream (same seed).

For mixer='spanpred' the ground-truth structure supervises the predictor heads
(boundary BCE + pair-link BCE) as an auxiliary loss when --supervision aux.
The attention itself is never hard-masked during training.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import Model, ModelConfig, MIXERS
from .structure import batch_union_mask, batch_pair_mask
from .task import TaskConfig, generate_batch, CH_B


def make_lr(step, steps, base_lr, warmup=100):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * t))


def train(args) -> Path:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed + 1)
    data_rng = np.random.default_rng(args.seed)

    task_cfg = TaskConfig(seq_len=args.seq_len)
    model_cfg = ModelConfig(
        mixer=args.mixer,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
    )
    model = Model(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    out_dir = Path(args.out) / args.mixer
    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = make_lr(step, args.steps, args.lr)

        batch = generate_batch(data_rng, args.batch, task_cfg)
        x = torch.from_numpy(batch.x)
        y = torch.from_numpy(batch.y)

        kwargs = {}
        if args.mixer == "oracle":
            kwargs["structure_mask"] = torch.from_numpy(
                batch_union_mask(batch.metas, args.seq_len)
            )
        pred, extras = model(x, **kwargs)
        mse = F.mse_loss(pred, y)
        # Pair-target rows are few (12 of 256) and their gradient has a
        # chicken-and-egg structure (routing the source payload only pays off
        # once the multiplicative readout exists, and vice versa), so the
        # retrieval circuit forms slowly under plain MSE.  Upweighting those
        # rows — identically for every mechanism — speeds circuit formation
        # without changing what any mechanism *can* represent, and the
        # counterfactual probes cannot be gamed by loss weighting: a mechanism
        # that cannot see v_j still cannot track it.
        if args.pair_row_weight != 1.0:
            w = torch.ones_like(y)
            for b, meta in enumerate(batch.metas):
                for _j, kk, _name in meta["pairs"]:
                    w[b, kk] = args.pair_row_weight
            loss = ((pred - y) ** 2 * w).sum() / w.sum()
        else:
            loss = mse
        aux_b = aux_l = None
        if args.mixer == "spanpred" and args.supervision == "aux":
            b_target = x[..., CH_B]
            n_b = b_target.sum().clamp(min=1)
            aux_b = F.binary_cross_entropy_with_logits(
                extras["boundary_logit"], b_target,
                pos_weight=torch.tensor((b_target.numel() - n_b.item()) / n_b),
            )
            link_target = torch.from_numpy(
                batch_pair_mask(batch.metas, args.seq_len).astype(np.float32)
            )
            causal = torch.tril(torch.ones(args.seq_len, args.seq_len, dtype=torch.bool))
            ll = extras["link_logit"][:, causal]
            lt = link_target[:, causal]
            aux_l = F.binary_cross_entropy_with_logits(
                ll, lt, pos_weight=torch.tensor(500.0)
            )
            loss = loss + args.aux_weight * (aux_b + aux_l)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == args.steps - 1:
            rec = {"step": step, "mse": mse.item(), "loss": loss.item(),
                   "elapsed": round(time.time() - t0, 1)}
            if aux_b is not None:
                rec["aux_boundary"] = aux_b.item()
                rec["aux_link"] = aux_l.item()
            log.append(rec)
            print(f"[{args.mixer}] step {step:5d}  mse {mse.item():.4f}  "
                  f"loss {loss.item():.4f}  ({rec['elapsed']}s)", flush=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "model": model_cfg.to_dict(),
        "train": vars(args),
        "n_params": model.n_params(),
    }, indent=2))
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"[{args.mixer}] saved to {out_dir}  ({model.n_params():,} params)")
    return out_dir


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixer", choices=MIXERS, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--supervision", choices=["aux", "none"], default="aux")
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--pair-row-weight", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=str, default="results")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
