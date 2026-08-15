# Results

R^2 of predicted output change against exact ground-truth change under counterfactual input edits (1.0 = perfect tracking, 0 = blind). Columns are reported separately on purpose: local is easy and would dominate any average; the question lives in `pair FAR`.

Overall NMSE noise floor: 0.039 (irreducible).

| mixer | pair near | pair mid | pair FAR | segment | local | leakage | nmse |
|---|---|---|---|---|---|---|---|
| full | 0.993 | 0.991 | 0.991 | 0.945 | 0.990 | 0.024 | 0.083 |
| local | 0.800 | 0.019 | -0.000 | 0.778 | 0.833 | 0.274 | 0.292 |
| pooled | 0.798 | 0.036 | 0.003 | 0.821 | 0.852 | 0.210 | 0.256 |
| lowrank | 0.771 | 0.617 | 0.414 | 0.961 | 0.521 | 0.048 | 0.364 |
| oracle | 0.994 | 0.989 | 0.993 | 0.985 | 0.896 | 0.055 | 0.078 |
| spanpred | 0.992 | 0.993 | 0.992 | 0.971 | 0.958 | 0.123 | 0.094 |

## full: full attention, inference-time masked to ground-truth structure (no retraining) — does anything live outside the boundary description?

| pair_near_r2 | pair_mid_r2 | pair_far_r2 | local_r2 | segment_r2 | leakage_ratio | nmse_overall |
|---|---|---|---|---|---|---|
| 0.621 | 0.637 | 0.498 | 0.224 | -8.552 | 0.768 | 5.072 |

## spanpred: spanpred with the soft bias replaced by a hard threshold at inference (YOLO-style)

| pair_near_r2 | pair_mid_r2 | pair_far_r2 | local_r2 | segment_r2 | leakage_ratio | nmse_overall | mean_slots_per_row |
|---|---|---|---|---|---|---|---|
| 0.977 | 0.989 | 0.969 | 0.955 | 0.654 | 0.247 | 0.264 | 15.100 |

## Attention mass inside the ground-truth structure (per layer)

| mixer | mass in structure | mass on far source (best head) |
|---|---|---|
| full | [0.2398, 0.996] | [0.9727, 0.0] |
| local | [0.7504, 0.9366] | [0.0, 0.0] |
| pooled | [0.6123, 0.8974] | [0.0086, 0.0001] |
| lowrank | [0.2612, 0.8275] | [0.0805, 0.0001] |
| oracle | [1.0, 1.0] | [0.9796, 0.0908] |
| spanpred | [0.6958, 0.9935] | [0.9788, 0.0005] |
