"""One backbone, interchangeable token-mixing mechanisms.

All mechanisms share the same embedding, block structure, MLPs, and head; they
differ only in how position i is allowed to read from positions j <= i.  That
keeps the comparison about the mixing mechanism, not about capacity elsewhere.

  full      standard causal softmax attention (the O(N^2) reference)
  local     softmax attention restricted to a sliding window
  pooled    sliding window + attention over average-pooled chunk summaries at
            two scales.  The wavelet-pyramid stand-in: distant positions are
            reachable only through coarse averages — gist survives, sharpness
            does not.
  lowrank   causal linear attention (kernel feature maps + prefix sums).  The
            attention map is rank-bounded by the feature dimension: smooth
            structure survives, sharp selection does not.
  oracle    full attention with support restricted to the ground-truth
            structure (triangle + pairs + band).  Answers "would perfect
            knowledge of the boundaries be enough?" with the predictor
            question removed entirely.
  spanpred  full attention biased by a cheap learned structure predictor:
            per-position boundary scores turned into a differentiable soft
            triangle mask, plus a low-dimensional pair-link head.  Soft
            everywhere during training (YOLO-style), hard-thresholdable at
            inference.  Ground-truth structure supervises the predictor heads
            as an auxiliary loss — a teacher, not a cage: attention itself is
            never restricted to the supervised pattern during training.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import N_CHANNELS

MIXERS = ("full", "local", "pooled", "lowrank", "oracle", "spanpred")
NEG_INF = float("-inf")


@dataclass
class ModelConfig:
    mixer: str = "full"
    seq_len: int = 256
    dim: int = 64
    layers: int = 2
    heads: int = 4
    window: int = 16  # local / pooled sliding window
    pool_sizes: tuple = (16, 64)  # pooled: chunk sizes
    pred_hidden: int = 32  # spanpred: conv hidden width
    pred_rank: int = 8  # spanpred: pair-link head dimension
    band: int = 2  # structural local band width

    def to_dict(self):
        d = asdict(self)
        d["pool_sizes"] = list(self.pool_sizes)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["pool_sizes"] = tuple(d.get("pool_sizes", (16, 64)))
        return ModelConfig(**d)


def _causal(N, device):
    return torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))


class SoftmaxMixer(nn.Module):
    """full / local / oracle / spanpred: same QKV machinery, different masks."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        if cfg.mixer == "spanpred":
            # softplus(-1) ~ 0.31: the structure bias starts gentle and the
            # model learns how hard to lean on it.
            self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, h, ctx):
        cfg = self.cfg
        B, N, _ = h.shape
        H, dh = cfg.heads, cfg.dim // cfg.heads
        q, k, v = self.qkv(h).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-2, -1) / math.sqrt(dh)

        mask = _causal(N, h.device)[None, None]
        if cfg.mixer == "local":
            idx = torch.arange(N, device=h.device)
            band = (idx[:, None] - idx[None, :]) < cfg.window
            mask = mask & band[None, None]
        if cfg.mixer == "oracle":
            mask = mask & ctx["structure_mask"][:, None]
        if ctx.get("attn_mask") is not None:  # inference-time ablation
            mask = mask & ctx["attn_mask"][:, None]
        if cfg.mixer == "spanpred":
            if ctx.get("hard_structure"):
                mask = mask & ctx["hard_mask"][:, None]
            else:
                logits = logits + F.softplus(self.gate) * ctx["struct_log"][:, None]
        logits = logits.masked_fill(~mask, NEG_INF)

        attn = logits.softmax(-1)
        if ctx.get("collect_attn") is not None:
            ctx["collect_attn"].append(attn.detach())
        out = (attn @ v).transpose(1, 2).reshape(B, N, cfg.dim)
        return self.proj(out)


class PooledMixer(nn.Module):
    """Sliding window over positions + softmax over average-pooled chunk
    summaries at multiple scales.  Chunks are admitted only once they end
    before the local window, and only at chunk granularity — so anything
    outside the window is visible exclusively as an average."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, h, ctx):
        cfg = self.cfg
        B, N, _ = h.shape
        H, dh = cfg.heads, cfg.dim // cfg.heads
        q, k, v = self.qkv(h).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        scale = 1.0 / math.sqrt(dh)
        idx = torch.arange(N, device=h.device)

        local_logits = q @ k.transpose(-2, -1) * scale
        band = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < cfg.window)
        local_logits = local_logits.masked_fill(~band[None, None], NEG_INF)

        parts, values = [local_logits], [v]
        pool_meta = []
        for p in cfg.pool_sizes:
            C = N // p
            kp = k[:, :, : C * p].reshape(B, H, C, p, dh).mean(3)
            vp = v[:, :, : C * p].reshape(B, H, C, p, dh).mean(3)
            lg = q @ kp.transpose(-2, -1) * scale
            chunk_end = torch.arange(C, device=h.device) * p + (p - 1)
            allowed = chunk_end[None, :] <= idx[:, None] - cfg.window
            lg = lg.masked_fill(~allowed[None, None], NEG_INF)
            parts.append(lg)
            values.append(vp)
            pool_meta.append((C, p))

        attn = torch.cat(parts, dim=-1).softmax(-1)
        splits = [N] + [C for C, _ in pool_meta]
        chunks = attn.split(splits, dim=-1)
        out = chunks[0] @ values[0]
        for a, vp in zip(chunks[1:], values[1:]):
            out = out + a @ vp

        if ctx.get("collect_attn") is not None:
            # expand pooled columns back to positions: effective per-position map
            eff = chunks[0].detach().clone()
            for a, (C, p) in zip(chunks[1:], pool_meta):
                spread = a.detach()[..., None].expand(B, H, N, C, p).reshape(B, H, N, C * p) / p
                eff[:, :, :, : C * p] += spread
            ctx["collect_attn"].append(eff)

        out = out.transpose(1, 2).reshape(B, N, cfg.dim)
        return self.proj(out)


class LinearMixer(nn.Module):
    """Causal linear attention (Katharopoulos et al.): phi(q) . cumsum(phi(k) v^T).
    The effective attention map has rank <= head feature dim — the canonical
    smooth/low-rank compression of the map."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, h, ctx):
        cfg = self.cfg
        B, N, _ = h.shape
        H, dh = cfg.heads, cfg.dim // cfg.heads
        q, k, v = self.qkv(h).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        fq, fk = F.elu(q) + 1, F.elu(k) + 1
        kv = (fk.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2)  # (B,H,N,dh,dh)
        z = fk.cumsum(2)
        num = (fq.unsqueeze(-1) * kv).sum(-2)
        den = (fq * z).sum(-1, keepdim=True) + 1e-6
        if ctx.get("collect_attn") is not None:
            eff = (fq @ fk.transpose(-2, -1)).tril() / den
            ctx["collect_attn"].append(eff.detach())
        out = (num / den).transpose(1, 2).reshape(B, N, cfg.dim)
        return self.proj(out)


class StructurePredictor(nn.Module):
    """The cheap predictor: O(N) boundary scores from a small causal conv, plus
    a low-dimensional pair-link head over raw input channels.

    The boundary scores beta parameterise a differentiable soft triangle:
        T[i, j] = prod_{m in (j, i]} (1 - beta_m)
    i.e. "probability that no boundary separates j from i" — computed with a
    cumulative sum in log space, exact, and O(N) parameters to specify even
    though the mask it describes is full rank.  This is the discrete-choice
    workaround: scores are soft everywhere during training and only
    thresholded at inference.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hid, r = cfg.pred_hidden, cfg.pred_rank
        self.conv1 = nn.Conv1d(N_CHANNELS, hid, 5, padding=4)
        self.conv2 = nn.Conv1d(hid, hid, 5, padding=4)
        self.bhead = nn.Conv1d(hid, 1, 1)
        self.q_proj = nn.Linear(N_CHANNELS, r)
        self.k_proj = nn.Linear(N_CHANNELS, r)

    def forward(self, x_raw):
        B, N, _ = x_raw.shape
        h = x_raw.transpose(1, 2)
        h = F.gelu(self.conv1(h)[..., :N])
        h = F.gelu(self.conv2(h)[..., :N])
        blogit = self.bhead(h)[:, 0]  # (B, N)
        beta = torch.sigmoid(blogit).clamp(max=1 - 1e-4)

        cl = torch.cumsum(torch.log1p(-beta), dim=1)
        # exponent <= 0 wherever j <= i; clamp only kills the (masked) upper
        # triangle, where it would otherwise overflow to inf
        tri = torch.exp((cl[:, :, None] - cl[:, None, :]).clamp(max=0.0))
        causal = _causal(N, x_raw.device)[None]
        tri = tri * causal

        q = self.q_proj(x_raw)
        kk = self.k_proj(x_raw)
        link_logit = q @ kk.transpose(-2, -1) / math.sqrt(self.cfg.pred_rank)
        link = torch.sigmoid(link_logit) * causal

        idx = torch.arange(N, device=x_raw.device)
        band = ((idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] <= self.cfg.band))[None]

        S = (tri + link + band).clamp(max=1.0)
        struct_log = torch.log(S + 1e-6) * causal  # 0 where disallowed anyway after causal mask
        return struct_log, S, blogit, link_logit


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


def make_mixer(cfg: ModelConfig):
    if cfg.mixer == "pooled":
        return PooledMixer(cfg)
    if cfg.mixer == "lowrank":
        return LinearMixer(cfg)
    return SoftmaxMixer(cfg)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mixer = make_mixer(cfg)
        self.mlp = MLP(cfg.dim)

    def forward(self, h, ctx):
        h = h + self.mixer(self.ln1(h), ctx)
        h = h + self.mlp(self.ln2(h))
        return h


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.mixer in MIXERS, cfg.mixer
        self.cfg = cfg
        self.in_proj = nn.Linear(N_CHANNELS, cfg.dim)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, 1)
        self.predictor = StructurePredictor(cfg) if cfg.mixer == "spanpred" else None

    def forward(
        self,
        x,
        structure_mask=None,
        attn_mask=None,
        collect_attn=False,
        hard_structure=False,
        tau=0.5,
    ):
        """x: (B, N, N_CHANNELS) raw input channels.

        structure_mask : bool (B, N, N), required for mixer='oracle'
        attn_mask      : bool (B, N, N), optional inference-time support ablation
        hard_structure : spanpred only — replace the soft log-bias with a hard
                         mask S > tau (threshold-at-inference mode)
        """
        ctx = {}
        extras = {}
        if collect_attn:
            ctx["collect_attn"] = []
        if attn_mask is not None:
            ctx["attn_mask"] = attn_mask
        if self.cfg.mixer == "oracle":
            assert structure_mask is not None, "oracle mixer needs the ground-truth mask"
            ctx["structure_mask"] = structure_mask
        if self.predictor is not None:
            struct_log, S, blogit, link_logit = self.predictor(x)
            ctx["struct_log"] = struct_log
            extras["boundary_logit"] = blogit
            extras["link_logit"] = link_logit
            extras["S"] = S
            if hard_structure:
                eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None]
                ctx["hard_mask"] = (S > tau) | eye
                ctx["hard_structure"] = True
                extras["hard_mask"] = ctx["hard_mask"]

        h = self.in_proj(x) + self.pos[None, : x.shape[1]]
        for blk in self.blocks:
            h = blk(h, ctx)
        y = self.head(self.ln_f(h)).squeeze(-1)
        if collect_attn:
            extras["attns"] = ctx["collect_attn"]
        return y, extras

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
