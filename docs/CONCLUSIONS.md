# Conclusions: attention compression at full resolution

This document records what the benchmark in this repository established, and
describes the `spanpred` method — the learned boundary/span attention that the
experiments single out — in enough detail to reimplement it. Numbers refer to
the run in `results/` (six mechanisms, identical ~117k-param backbone,
identical data stream, 4000 steps; see `results/summary.md`). The task and
evaluation design are documented in the [README](../README.md).

## 1. The question

Attention costs $N^2$ because it builds an explicit map from every position to
every position. Every cheaper alternative replaces the map with a compressed
description. The question under test: **which compression preserves long-range
connection at full resolution** — the ability to read a sharp feature at
distance while it stays sharp — rather than only preserving the gist of
distant content?

The working hypothesis was that the structure worth keeping is the one visible
in real attention maps: stripes and **triangles** — contiguous spans anchored
on the diagonal, opening at a boundary and closing at the next. A triangle is
full rank (no low-rank factorization can represent it), yet it is fully
described by two numbers. Structure that is *cheap to specify and impossible
to approximate smoothly* is exactly what classical nonlinear approximation
theory says wavelets buy you for piecewise-smooth signals [15, 16, 17]; the
question is how to buy it for attention maps.

## 2. What the experiments established

**Finding 1 — the two standard compressions fail on orthogonal axes, both as
predicted.** The pooled pyramid (sliding window + attention over averaged
chunks at scales 16/64 — the attention-pattern equivalent of climbing a
wavelet pyramid [8, 9]) has global reach and it is worth nothing at
resolution: far-pair retrieval R² = 0.003, no head puts more than ~1% of mass
on the far source. Its *smooth* far read is fine (segment 0.821). Low-rank
attention (causal linear attention [5]) is the mirror image: best cheap
mechanism at the smooth read (segment 0.961), weak at sharp selection at
*every* distance — local retrieval one position away is 0.521, and pair
retrieval decays gradually (0.771 → 0.617 → 0.414) as prefix competition
washes out the kernel match. Pyramids fail by distance; low rank fails by
sharpness. The failure of each is quantitative, predicted in closed form
before training (`results/task_validation.json`), and attributable: a
mechanism that reaches a white-noise payload only through a $W$-wide average
has an R² ceiling of $\approx 1/W$ on the retrieval probe, and the pooled
mechanism performs at or below its ceiling.

**Finding 2 — the structural prior is sufficient.** Full attention trained
with its support restricted to the ground-truth structure (segment triangles ∪
pair links ∪ a width-2 local band) matches unconstrained full attention on
every retrieval column (far 0.993 vs 0.991) and *beats* it on boundary
sharpness (segment 0.985 vs 0.945; the mask performs the boundary exclusion
the unconstrained model must learn). Perfect knowledge of the boundaries is
enough. This isolates prior-correctness from predictor quality — the
separation the evaluation was designed to enforce.

**Finding 3 — a cheap predictor finds the structure.** `spanpred`
(Section 3) closes the oracle gap essentially to zero (far 0.992 vs oracle
0.993) with a structure predictor that is 8,753 parameters — 6.9% of the
model — reading raw input channels only.

**Finding 4 — thresholded at inference, retrieval survives on a budget of
~15 slots per row.** Replacing the soft structure bias with a hard mask
$S > 0.5$ keeps far retrieval at 0.969 while admitting a mean of 15.1
attention slots per row, versus ~128 for an average causal row. What degrades
is the smooth read (segment 0.654): thresholding clips the tails of long
triangles. The oracle row proves this is predictor *calibration*, not a
failure of the prior — with true boundaries the hard triangle costs nothing.

**Finding 5 — the boundary structure is a sufficient *specification*, not a
faithful *description*.** Masking the *trained, unconstrained* full model to
the ground-truth structure at inference destroys it (segment R² −8.6, NMSE
5.07): only 24% of its layer-0 mass lies inside the structure, even though its
best layer-0 head places 97% of its mass on the far source. The sharp,
needle-finding part of the computation lives inside the structural
description; the smooth machinery is broad, distributed, and load-bearing
outside it (compare the attention-sink and broad-mixing phenomena observed in
trained transformers [13]). Compression by boundaries must be imposed during
training, not read off a trained map afterwards.

**Finding 6 — the prior is an optimization aid, not just a cost model.**
Under plain MSE, full attention never formed the far-retrieval circuit
(far R² ≈ 0 after 3000 steps): routing the source payload has zero
first-order gradient until the multiplicative readout exists, and vice versa —
a saddle of the same character as the abrupt formation of induction heads
[12]. Escaping it required upweighting the pair rows in the loss (applied
identically to every mechanism; counterfactual probes cannot be gamed by loss
weighting). The structured mechanisms face the same saddle with a drastically
smaller search space.

| mixer | pair near | pair mid | pair **far** | segment | local | leakage ↓ |
|---|---|---|---|---|---|---|
| full | 0.993 | 0.991 | **0.991** | 0.945 | 0.990 | 0.024 |
| local (w=16) | 0.800 | 0.019 | **0.000** | 0.778 | 0.833 | 0.274 |
| pooled | 0.798 | 0.036 | **0.003** | 0.821 | 0.852 | 0.210 |
| lowrank | 0.771 | 0.617 | **0.414** | 0.961 | 0.521 | 0.048 |
| oracle | 0.994 | 0.989 | **0.993** | 0.985 | 0.896 | 0.055 |
| spanpred | 0.992 | 0.993 | **0.992** | 0.971 | 0.958 | 0.123 |

## 3. The spanpred method

### 3.1 Design requirements

The method is the YOLO framing [1] taken literally. YOLO solves discrete
object selection by predicting a *soft score everywhere* on a dense grid
during training and thresholding only at inference; ground-truth boxes enter
as supervision through a matching step, never as an architectural restriction
on where boxes may appear. Transposed to attention:

1. The objects to detect are **spans** (segment triangles) and **links**
   (pair edges). The attention map's support is parameterised by their
   boundary positions — $O(N)$ numbers describing a full-rank $N \times N$
   pattern.
2. Scores must be **soft everywhere during training** so gradients flow
   through the discrete where-does-a-coefficient-go choice (the alternative
   workarounds — straight-through estimators [11] or Gumbel-softmax
   relaxations [10] — inject bias or variance; the dense-soft/threshold-late
   route needs neither).
3. Ground truth, when available, supervises the *predictor heads* as an
   auxiliary loss — **a teacher, not a cage**. Attention is never restricted
   to the supervised pattern during training, and at inference the predictor
   is free to place structure where supervision never showed it. Because the
   prediction grid here *is* the sequence, YOLO's box-to-cell matching step is
   trivial (a boundary supervises $\beta$ at its position; a pair supervises
   one entry of the link matrix); it becomes a real matching problem only if
   spans are predicted as unanchored objects.

### 3.2 Structure parameterisation

The predictor (`StructurePredictor` in `etension/models.py`, 8,753 params)
reads only the raw input channels $x \in \mathbb{R}^{N \times C}$, not the
backbone's hidden states.

**Boundary head → soft triangles.** A two-layer causal convolution (kernel 5,
hidden width 32, GELU) produces per-position boundary logits, and
$\beta_m = \sigma(\text{logit}_m) \in [0,1]$ is the predicted probability that
a segment boundary sits at position $m$. These parameterise the soft triangle

$$T[i,j] \;=\; \prod_{m=j+1}^{i} (1-\beta_m), \qquad j \le i,\; T[i,i]=1 .$$

If boundaries were independent Bernoulli events, $T[i,j]$ is exactly the
probability that *no boundary separates $j$ from $i$* — i.e. that they share a
segment. The product telescopes into a cumulative sum in log space,

$$T[i,j] = \exp\big(c_i - c_j\big), \qquad c_i = \sum_{m \le i} \log(1-\beta_m),$$

so the whole triangle field costs one $O(N)$ scan to specify plus a
broadcasted subtraction to materialise; it is exact, differentiable, and the
mask it describes is full rank while being specified by $N$ numbers. (The
exponent is clamped at 0 before exponentiation: entries above the diagonal
would otherwise overflow, and valid entries always have non-positive
exponents.) This product form is the stick-breaking construction, which has
recently been used as a replacement for softmax itself [14]; here it
parameterises the *support*, and softmax survives inside it. One
step-function boundary per segment is what a sliding window or a pooling grid
cannot represent at arbitrary offsets — and boundary alignment is precisely
where pooled mechanisms bled (leakage 0.210 vs full's 0.024).

**Link head → soft pair edges.** Two linear projections of the raw input,
$q_k, \kappa_j \in \mathbb{R}^{8}$, score every causal position pair:

$$L[k,j] = \sigma\!\big(\langle q_k, \kappa_j\rangle / \sqrt{8}\big).$$

This is itself a miniature attention — but rank-8, over raw inputs, with no
value pathway; its only job is to say *where* an edge is, in the spirit of
content-based sparse addressing in Reformer and the Routing Transformer
[6, 7]. In the benchmark it is materialised densely ($O(N^2 \cdot 8)$); at
production scale the same head is what an LSH or top-$k$ candidate search
approximates in $O(N \log N)$.

**Union.** With a fixed causal band $B$ of width 2 (local interactions are
structure too, and cheap), the predicted structure is

$$S = \min\big(T + L + B,\; 1\big) \in [0,1]^{N \times N}.$$

### 3.3 Injection into attention: a log-prior bias, not a mask

Each attention layer adds the structure as a bias on its pre-softmax logits:

$$\text{logits}[i,j] \;=\; \frac{q_i^\top k_j}{\sqrt{d_h}} \;+\; \mathrm{softplus}(g)\,\log\big(S[i,j] + \varepsilon\big),$$

with $\varepsilon = 10^{-6}$ and a per-layer learned scalar gate $g$
(initialised so $\mathrm{softplus}(g) \approx 0.31$). Two properties matter:

- **Product-of-experts reading.** After softmax the attention weight is
  proportional to $e^{q^\top k/\sqrt{d_h}} \cdot S^{\mathrm{softplus}(g)}$ —
  content similarity multiplied by a structural prior raised to a learned
  temperature. With $g$ small the model ignores the predictor (early
  training, predictor still wrong); as the predictor sharpens, the model
  learns how hard to lean on it. $S \to 0$ drives the bias toward
  $\log\varepsilon \approx -13.8$, an effective exclusion — but a *soft* one,
  everywhere differentiable, all the way through training. This is the
  learned analogue of adaptive attention spans [4], with the span boundaries
  coming from a predictor rather than a per-head width parameter.
- **The backbone stays intact.** The mixer is ordinary multi-head attention;
  `spanpred` differs from `full` only by this added bias (and from `oracle`
  by bias-vs-hard-mask and predicted-vs-true structure). Comparisons isolate
  exactly one variable at each step of the chain
  full → oracle → spanpred-soft → spanpred-hard.

### 3.4 Training

Total loss:

$$\mathcal{L} = \underbrace{\text{weighted MSE}}_{\text{task}} \;+\; \lambda\,\big(\mathrm{BCE}(\beta,\, b^\*) + \mathrm{BCE}(L,\, E^\*)\big), \qquad \lambda = 1,$$

where $b^\*$ are true boundary indicators (positive-class weight ≈ 25, since
boundaries are ~1 in 26 positions) and $E^\*$ the true pair-edge matrix
(positive weight 500; positives are ~2·10⁻⁴ of causal entries). The auxiliary
terms touch only the predictor heads. With partially annotated data the
auxiliary term simply drops out on unannotated samples — the semi-supervised
regime discussed in §5. Nothing in the forward pass consumes ground truth, so
annotation coverage is a training-data property, not an architectural one.

### 3.5 Inference: threshold, and what the budget buys

At inference the soft bias can be replaced by the hard support
$\{S > \tau\} \cup \{\text{diagonal}\}$ (τ = 0.5 in the reported run). This
is YOLO's confidence threshold. Measured effect (`results/summary.json`):

| | soft bias | hard $S>0.5$ |
|---|---|---|
| pair far R² | 0.992 | 0.969 |
| segment R² | 0.971 | 0.654 |
| local R² | 0.958 | 0.955 |
| mean slots/row | ~128 (dense) | **15.1** |

The sharp capability — the one no baseline compression retains — survives
thresholding almost intact on a ~15-coefficient budget. The smooth read is
what suffers, because a triangle's soft membership $T[i,j]$ decays with
distance whenever the β's inside the span are not exactly zero, so a fixed
τ clips long-triangle tails. That is a calibration artifact of the predictor
(β systematically a few percent above zero inside segments), not a defect of
the span prior — the oracle's hard triangles cost nothing (segment 0.985).
Obvious remedies, untested: calibrate β (temperature or bias on the boundary
head), per-row-adaptive τ, or keeping the triangle term soft while
thresholding only links.

### 3.6 Complexity, honestly

The benchmark materialises $S$ densely, because the question here is
representational. The method is *specified* in $O(N)$ + $O(\text{links})$:
boundary scan, then triangles as intervals per row (a row's triangle is
`[last_boundary(i), i]`, one interval), plus a candidate set of links per row
from top-$k$ on the 8-dim link space. A production kernel would gather
$K \approx 15$ keys per query ($O(NK)$), which is the regime hardware-aligned
trainable-sparse designs like Native Sparse Attention target [18] — NSA's
compressed-branch + selected-branch decomposition is the same
smooth-plus-sharp split this benchmark motivates, with the difference that
here the *selected* support is span-parameterised and supervisable.

### 3.7 What spanpred does not yet show

- Leakage 0.123 vs full's 0.024: the soft boundary is not yet as sharp as a
  learned hard exclusion. Same calibration lever as §3.5.
- Supervision was always-on in the reported run. `--supervision none` exists
  and is untested; the partial-coverage sweep (§5) is the experiment that
  matters for practice.
- One scale of structure (segments). Real maps plausibly need nested spans;
  the stick-breaking parameterisation composes (a β-field per level, products
  of products), but this is unimplemented.

## 4. Relation to prior work

Fixed sparse patterns — strided/dilated [2], sliding window + global tokens
[3], and their randomised variants — are the "cage": they bound cost by
deciding *in advance* where attention may go, and a boundary that falls
between grid lines is invisible to them (the pooled row of the results table
is the honest version of this family, and its leakage number is the cost).
Low-rank and kernel factorizations [5, 19, 20] compress the map's *values*
smoothly and inherit the blur this repo measures directly. Content-based
sparse methods [6, 7] and learned spans/expiry [4, 21] move the support into
the model, as here, but parameterise it per-query or per-head rather than as
explicit named structure (boundaries, links) that supervision can touch and a
human can read off a plot. Hierarchical sequence models discover boundaries
for *representation* (HM-RNN's boundary detectors [22], dynamic token pooling
[23], hierarchical transformers [24]); spanpred discovers them for the
*attention support* while keeping token resolution everywhere. The
stick-breaking product appears in [14] as a softmax replacement; here it
builds the support prior instead. On the signal-processing side, the premise
that discontinuities are cheap in location-based descriptions and expensive
in smooth bases is classical [15, 16, 17] — the contribution of this repo's
experiments is measuring which side of that dichotomy each attention
compression lands on, with the interaction map's ground truth known exactly.

## 5. Open questions, ranked by information per unit work

1. **Partial supervision sweep** — supervise the predictor on a fraction
   $p \in \{1, 0.1, 0.01, 0\}$ of samples; plot far-pair R² vs $p$. The
   architecture needs no change. This answers the practical question: how
   many annotated interaction maps buy the oracle's performance?
2. **Calibrated thresholding** — close the segment gap at $K \approx 15$
   (§3.5 remedies). If it closes, the budget story is complete.
3. **Unsupervised structure discovery** — `--supervision none`: does the
   task loss alone shape β into boundaries? Finding 6 predicts it will be
   slow or stuck; measuring *how* stuck quantifies the teacher's value.
4. **A literal wavelet mixer with coefficient selection** — keys/values =
   Haar coefficients, a selector spends budget across scales. Tests whether
   basis-domain selection can match position-domain selection (§3.6's NSA
   comparison, in a controlled setting).
5. **O(NK) kernel + scale** — take the measured $K$, implement the gather,
   and test at $N$ where the quadratic reference actually hurts.

## References

[1] J. Redmon, S. Divvala, R. Girshick, A. Farhadi. *You Only Look Once:
Unified, Real-Time Object Detection.* CVPR 2016.

[2] R. Child, S. Gray, A. Radford, I. Sutskever. *Generating Long Sequences
with Sparse Transformers.* arXiv:1904.10509, 2019.

[3] I. Beltagy, M. E. Peters, A. Cohan. *Longformer: The Long-Document
Transformer.* arXiv:2004.05150, 2020.

[4] S. Sukhbaatar, E. Grave, P. Bojanowski, A. Joulin. *Adaptive Attention
Span in Transformers.* ACL 2019.

[5] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. *Transformers are RNNs:
Fast Autoregressive Transformers with Linear Attention.* ICML 2020.

[6] N. Kitaev, Ł. Kaiser, A. Levskaya. *Reformer: The Efficient Transformer.*
ICLR 2020.

[7] A. Roy, M. Saffar, A. Vaswani, D. Grangier. *Efficient Content-Based
Sparse Attention with Routing Transformers.* TACL 2021.

[8] Z. Zhu, R. Soricut. *H-Transformer-1D: Fast One-Dimensional Hierarchical
Attention for Sequences.* ACL 2021.

[9] M. Zaheer, G. Guruganesh, A. Dubey, et al. *Big Bird: Transformers for
Longer Sequences.* NeurIPS 2020.

[10] E. Jang, S. Gu, B. Poole. *Categorical Reparameterization with
Gumbel-Softmax.* ICLR 2017; C. J. Maddison, A. Mnih, Y. W. Teh. *The Concrete
Distribution.* ICLR 2017.

[11] Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating
Gradients Through Stochastic Neurons for Conditional Computation.*
arXiv:1308.3432, 2013.

[12] C. Olsson, N. Elhage, N. Nanda, et al. *In-context Learning and
Induction Heads.* Transformer Circuits Thread, 2022.

[13] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming
Language Models with Attention Sinks.* ICLR 2024.

[14] S. Tan, Y. Shen, S. Yang, A. Courville, R. Panda. *Stick-Breaking
Attention.* arXiv:2410.17980, 2024.

[15] S. Mallat. *A Wavelet Tour of Signal Processing.* Academic Press, 3rd
ed., 2008.

[16] D. L. Donoho, I. M. Johnstone. *Ideal Spatial Adaptation by Wavelet
Shrinkage.* Biometrika 81(3), 1994.

[17] R. A. DeVore. *Nonlinear Approximation.* Acta Numerica 7, 1998.

[18] J. Yuan, H. Gao, D. Dai, et al. *Native Sparse Attention:
Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089,
2025.

[19] S. Wang, B. Z. Li, M. Khabsa, H. Fang, H. Ma. *Linformer: Self-Attention
with Linear Complexity.* arXiv:2006.04768, 2020.

[20] K. Choromanski, V. Likhosherstov, D. Dohan, et al. *Rethinking Attention
with Performers.* ICLR 2021.

[21] S. Sukhbaatar, D. Ju, S. Poff, et al. *Not All Memories are Created
Equal: Learning to Forget by Expiring.* ICML 2021.

[22] J. Chung, S. Ahn, Y. Bengio. *Hierarchical Multiscale Recurrent Neural
Networks.* ICLR 2017.

[23] P. Nawrot, J. Chorowski, A. Łańcucki, E. M. Ponti. *Efficient
Transformers with Dynamic Token Pooling.* ACL 2023.

[24] P. Nawrot, S. Tworkowski, M. Tyrolski, et al. *Hierarchical Transformers
Are More Efficient Language Models.* Findings of NAACL 2022.
