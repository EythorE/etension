# etension — attention compression at full resolution

Attention costs N² because it builds an explicit map from every position to
every position. Every cheaper alternative replaces that map with a compressed
description of it. This repo is a controlled test of **which compression is the
right one** — specifically, whether the standard compressions throw away
exactly the thing that matters: **long-range connection at full resolution**.

## The claim under test

Hierarchical/pyramid schemes (wavelet-flavoured attention, pooled summaries)
reach distant positions only by climbing to a coarse scale, and climbing means
averaging — so a distant position is reachable only as a gist, never as a
sharp feature that stays sharp. Low-rank schemes keep smooth structure and
destroy edges by nature; they can only ever return a blurry reconstruction.

The structure actually visible in real attention maps is the opposite of
smooth: stripes, and **triangles** — contiguous regions anchored on the
diagonal, opening at a boundary and closing at the next one. A triangle is
full rank, so no low-rank scheme can represent it, yet it is fully described
by two numbers: its start and its end. That is the split this repo is built
around: structure that is **cheap to specify and impossible to approximate
smoothly**. The triangles are the bounding boxes; the objects to be detected
are the spans (the YOLO framing, taken literally).

A note on wavelets: the wavelet *transform* is invertible — fine detail at
distance exists in the pyramid as fine-scale coefficients. What blurs is the
*attention pattern over the pyramid*: touching the fine-scale coefficients of
a distant region costs as much as full attention, so pyramid-attention schemes
only touch the coarse ones. The failure is in which coefficients you can
afford to reach, not in the basis.

## The task

`etension/task.py` generates sequences where the ground-truth attention
structure is known exactly. The output is a sum of three multiplicative
interactions (the presence/magnitude of one feature modulates the contribution
of another — nothing is solvable by attending to either feature alone), plus
irreducible noise:

| component | definition | attention row needed | what it tests |
|---|---|---|---|
| local | `y_i += u_i · v_{i-1}` | previous position | the easy column |
| segment | `y_i += v_i · (Σ_{m=s_i..i} u_m)/√t` | the **triangle** `[s_i, i]` | boundary sharpness |
| pair | `y_k += 2 · u_k · v_j` for key-matched pairs `j → k` | one spike at `j` | full-resolution retrieval at distance |

Design decisions that make the test mean something:

- **`v` is white noise.** The average of any window around a pair source `j`
  carries almost none of `v_j`; a mechanism that reaches `j` only through a
  `(2W+1)`-wide average has a provable R² ceiling of ~`1/(2W+1)` on the pair
  probe. Sharpness is enforced by construction, not assumption.
- **Pair distances are sampled in three bands** (near ≤16, mid 24–64, far
  96–248) so retrieval is reported **per distance and never averaged** — local
  is easy, would dominate any aggregate, and the entire question lives in the
  far column.
- **Pairs are addressed by content, and addressing is made as learnable as
  possible on purpose**: the source advertises a unit-norm key in dedicated
  "address" channels, the target asks with the same key in separate "query"
  channels. Matched keys dot to exactly 1, unmatched to ~0, matching is linear
  in the input, and there is no self-match to gate away — so failures on the
  pair probe are attributable to the mechanism's resolution at distance, not
  to how hard the addressing circuit is to form.
- **Boundaries are visible in the input.** Discovering *what counts as* a
  boundary is not the axis under test; using one at range and resolution is.
- Targets are a deterministic function of the inputs given the metadata and a
  frozen noise draw, so counterfactual edits have **exact** ground-truth
  deltas.

## The mechanisms

One backbone (embedding → blocks → head); only the token mixer differs
(`etension/models.py`):

| mixer | what it is | what it stands in for |
|---|---|---|
| `full` | causal softmax attention | the O(N²) reference |
| `local` | sliding-window attention (w=16) | locality-only compression |
| `pooled` | window + softmax over average-pooled chunks at scales 16 and 64 | the wavelet pyramid: distance only via coarse averages |
| `lowrank` | causal linear attention (kernel feature maps, prefix sums) | low-rank maps: smooth survives, edges die |
| `oracle` | full attention with support restricted to ground-truth structure (triangle ∪ pairs ∪ band) | *is the structural prior sufficient?* |
| `spanpred` | full attention biased by a cheap learned structure predictor | *can a cheap predictor find the structure?* |

`spanpred` is the learned-transform idea made concrete. A small causal conv
predicts per-position boundary scores β; these parameterise a differentiable
soft triangle `T[i,j] = Π_{m∈(j,i]}(1−β_m)` ("no boundary separates j from i")
— O(N) numbers specifying a full-rank mask. A low-dimensional head scores pair
links. Both are **soft everywhere during training and thresholded only at
inference**, which is the YOLO answer to "discrete choices don't pass
gradients". The ground-truth structure supervises the predictor heads as an
auxiliary loss — **a teacher, not a cage**: attention is never architecturally
restricted to the supervised pattern; at inference the predictor is free to
place structure where supervision never showed it.

## The evaluation

`etension/evaluate.py` refuses aggregate error as a headline number. Each
capability is measured by a **counterfactual probe**: edit one input, recompute
the exact ground-truth change, and measure R² of the model's output change
against it.

- **Pair probe, per distance band** — resample `v` at pair sources; does the
  target's output track `u_k·Δv_j`? This is retrieval at full resolution.
- **Segment probe** — resample one `u` near a segment start; do all later
  positions in the segment track the change through the running sum?
- **Local probe** — the easy column, reported to show it does *not*
  discriminate.
- **Boundary leakage probe** — resample every `u` in the segment *before* a
  boundary. Ground truth after the boundary is exactly unchanged (asserted);
  any model-output change after the boundary is structure blur. Reported as
  rms(after)/rms(before).

Two separations are enforced by design:

1. **Prior correctness vs. predictor quality.** `oracle` isolates "would
   perfect knowledge of the boundaries be enough"; `spanpred` minus `oracle`
   is the predictor gap; `full` minus `oracle` is the prior's error. A good
   idea with a bad predictor can no longer look identical to a bad idea.
2. **Local vs. long-range.** Every table keeps the distance bands as separate
   columns. There is no aggregate that mixes them.

Two more measurements close the loop on "is the boundary structure a
sufficient description of what attention does":

- **Attention mass inside structure** — fraction of the trained full model's
  attention that falls inside the ground-truth mask, and mass on the far
  source at pair rows (best head).
- **Inference-time ablation** — take the *trained* full model, mask its
  attention to the ground-truth structure with no retraining, and re-run all
  probes. If nothing degrades, no important mass lives outside the boundary
  description.

## Task validity (before any training)

`etension/validate_task.py` runs closed-form readouts through the same probes
(`results/task_validation.json`):

| readout | pair near | pair mid | pair far | segment | leakage |
|---|---|---|---|---|---|
| exact oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| pair source blurred, W=8 | 0.126 | 0.127 | 0.127 | 1.000 | 0.000 |
| pair source blurred, W=32 | 0.040 | 0.038 | 0.037 | 1.000 | 0.000 |
| boundaries snapped to grid 16 | 1.000 | 1.000 | 1.000 | 0.956 | 0.459 |
| boundaries snapped to grid 64 | 1.000 | 1.000 | 1.000 | 0.825 | 0.497 |

Perfect structural knowledge is sufficient (exact oracle hits the noise floor,
NMSE 0.041 ≈ σ²/var(y)). Blur fails **exactly** the intended columns: window-
averaging the source kills only the pair columns, to the predicted ~1/(2W+1)
ceiling; snapping boundaries to a pooling grid produces only leakage and
segment degradation. A mechanism's probe profile is therefore attributable to
the mechanism.

## Results

_Filled in by `python -m experiments.run_all` — see `results/summary.md`._

## Running it

```
pip install -r requirements.txt
python -m etension.validate_task            # task validity, no training
python -m experiments.run_all --steps 4000  # everything: ~30-40 min on 4 CPU cores
# or individually:
python -m etension.train --mixer full --steps 4000
python -m etension.evaluate --ckpt results/full
python -m etension.visualize --ckpt results/full
```

## Scope, honestly stated

This benchmark measures **representational adequacy**, not wall-clock cost.
`spanpred` materialises its soft mask densely (O(N²) memory) because the
question here is whether boundary-parameterised structure *can carry the
information*; the O(N·K) sparse kernel that spends a budget of K coefficients
per row is the production goal, and `mean_slots_per_row` in the
threshold-at-inference results measures what K would need to be. Training
compute is also deliberately modest; the mechanisms are compared under an
identical budget, and conclusions are about their ordering, not their ceilings.
