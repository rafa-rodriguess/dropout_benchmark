# S1 — Hyperparameter Search Grids

This supplementary file is **auto-generated** by `dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py`.
It documents the complete candidate search grids for all tuned models in the benchmark.
For Family B, grids are read from JSON metadata files produced at pipeline run-time.
For Family A, grids are extracted from the pipeline source constants via AST parsing.

*Generated: 2026-06-19 23:25 UTC*

---

## Family B — Static Early-Window (Comparable Arm)

All Family B models use enrollment-level early-window features.
Validation uses a 20% enrollment-level hold-out (GroupShuffleSplit) stratified by event,
except Neural-MTLR and DeepHit which use a 10% internal fraction.

### Cox Comparable

**Candidates:** 24  |  **Selection criterion:** highest validation C-index

| Candidate | penalizer | l1_ratio |
| --- | --- | --- |
| 1 | 0.001 | 0.0 |
| 2 | 0.001 | 0.25 |
| 3 | 0.001 | 0.5 |
| 4 | 0.001 | 0.75 |
| 5 | 0.01 | 0.0 |
| 6 | 0.01 | 0.25 |
| 7 | 0.01 | 0.5 |
| 8 | 0.01 | 0.75 |
| 9 | 0.05 | 0.0 |
| 10 | 0.05 | 0.25 |
| 11 | 0.05 | 0.5 |
| 12 | 0.05 | 0.75 |
| 13 | 0.1 | 0.0 |
| 14 | 0.1 | 0.25 |
| 15 | 0.1 | 0.5 |
| 16 | 0.1 | 0.75 |
| 17 | 0.2 | 0.0 |
| 18 | 0.2 | 0.25 |
| 19 | 0.2 | 0.5 |
| 20 | 0.2 | 0.75 |
| 21 | 0.5 | 0.0 |
| 22 | 0.5 | 0.25 |
| 23 | 0.5 | 0.5 |
| 24 | 0.5 | 0.75 |

### DeepSurv

**Candidates:** 24  |  **Selection criterion:** lowest validation loss (early stopping, patience=10)
*Architecture grid: [32,16], [64,32], [128,64]. Early stopping on val loss.*

| Candidate | hidden_dims | dropout | learning_rate | weight_decay |
| --- | --- | --- | --- | --- |
| 1 | [32, 16] | 0.1 | 5e-04 | 1e-05 |
| 2 | [32, 16] | 0.1 | 5e-04 | 1e-04 |
| 3 | [32, 16] | 0.1 | 0.001 | 1e-05 |
| 4 | [32, 16] | 0.1 | 0.001 | 1e-04 |
| 5 | [32, 16] | 0.3 | 5e-04 | 1e-05 |
| 6 | [32, 16] | 0.3 | 5e-04 | 1e-04 |
| 7 | [32, 16] | 0.3 | 0.001 | 1e-05 |
| 8 | [32, 16] | 0.3 | 0.001 | 1e-04 |
| 9 | [64, 32] | 0.1 | 5e-04 | 1e-05 |
| 10 | [64, 32] | 0.1 | 5e-04 | 1e-04 |
| 11 | [64, 32] | 0.1 | 0.001 | 1e-05 |
| 12 | [64, 32] | 0.1 | 0.001 | 1e-04 |
| 13 | [64, 32] | 0.3 | 5e-04 | 1e-05 |
| 14 | [64, 32] | 0.3 | 5e-04 | 1e-04 |
| 15 | [64, 32] | 0.3 | 0.001 | 1e-05 |
| 16 | [64, 32] | 0.3 | 0.001 | 1e-04 |
| 17 | [128, 64] | 0.1 | 5e-04 | 1e-05 |
| 18 | [128, 64] | 0.1 | 5e-04 | 1e-04 |
| 19 | [128, 64] | 0.1 | 0.001 | 1e-05 |
| 20 | [128, 64] | 0.1 | 0.001 | 1e-04 |
| 21 | [128, 64] | 0.3 | 5e-04 | 1e-05 |
| 22 | [128, 64] | 0.3 | 5e-04 | 1e-04 |
| 23 | [128, 64] | 0.3 | 0.001 | 1e-05 |
| 24 | [128, 64] | 0.3 | 0.001 | 1e-04 |

### Random Survival Forest

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS

| Candidate | n_estimators | min_samples_leaf | max_depth | max_features |
| --- | --- | --- | --- | --- |
| 1 | 200 | 3 | None | sqrt |
| 2 | 300 | 5 | None | sqrt |
| 3 | 400 | 3 | None | sqrt |
| 4 | 200 | 10 | None | sqrt |
| 5 | 300 | 20 | None | sqrt |
| 6 | 400 | 5 | None | 0.8 |
| 7 | 200 | 3 | None | 0.7 |
| 8 | 300 | 10 | None | 0.8 |
| 9 | 200 | 3 | 8 | sqrt |
| 10 | 300 | 5 | 8 | sqrt |
| 11 | 400 | 3 | 8 | sqrt |
| 12 | 200 | 10 | 8 | 0.8 |
| 13 | 300 | 20 | 8 | 0.8 |
| 14 | 400 | 5 | 8 | 0.7 |
| 15 | 200 | 3 | 8 | 0.7 |
| 16 | 300 | 10 | 8 | 0.7 |
| 17 | 200 | 3 | 15 | sqrt |
| 18 | 300 | 5 | 15 | sqrt |
| 19 | 400 | 3 | 15 | sqrt |
| 20 | 200 | 10 | 15 | 0.8 |
| 21 | 300 | 20 | 15 | 0.8 |
| 22 | 400 | 5 | 15 | 0.7 |
| 23 | 200 | 3 | 15 | 0.7 |
| 24 | 300 | 10 | 15 | 0.7 |

### Neural-MTLR

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS (early stopping, patience=8)
*num_durations controls survival-time discretization. Early stopping on val IBS.*

| Candidate | num_durations | hidden_dims | dropout | learning_rate | weight_decay | batch_size |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | [64, 32] | 0.1 | 0.005 | 1e-05 | 512 |
| 2 | 20 | [64, 32] | 0.1 | 0.01 | 1e-05 | 512 |
| 3 | 20 | [64, 32] | 0.2 | 0.005 | 1e-04 | 512 |
| 4 | 20 | [64, 32] | 0.2 | 0.01 | 1e-04 | 512 |
| 5 | 20 | [128, 64] | 0.1 | 0.005 | 1e-05 | 512 |
| 6 | 20 | [128, 64] | 0.1 | 0.01 | 1e-05 | 512 |
| 7 | 20 | [128, 64] | 0.2 | 0.005 | 1e-04 | 512 |
| 8 | 20 | [128, 64] | 0.2 | 0.01 | 1e-04 | 512 |
| 9 | 20 | [256, 128] | 0.1 | 0.005 | 1e-05 | 512 |
| 10 | 20 | [256, 128] | 0.2 | 0.005 | 1e-04 | 512 |
| 11 | 20 | [256, 128] | 0.2 | 0.01 | 1e-04 | 1024 |
| 12 | 20 | [256, 128] | 0.3 | 0.001 | 1e-04 | 1024 |
| 13 | 40 | [64, 32] | 0.1 | 0.005 | 1e-05 | 512 |
| 14 | 40 | [64, 32] | 0.2 | 0.005 | 1e-04 | 512 |
| 15 | 40 | [64, 32] | 0.2 | 0.01 | 1e-04 | 512 |
| 16 | 40 | [64, 32] | 0.3 | 0.001 | 1e-04 | 512 |
| 17 | 40 | [128, 64] | 0.1 | 0.005 | 1e-05 | 512 |
| 18 | 40 | [128, 64] | 0.1 | 0.01 | 1e-05 | 512 |
| 19 | 40 | [128, 64] | 0.2 | 0.005 | 1e-04 | 512 |
| 20 | 40 | [128, 64] | 0.2 | 0.001 | 1e-04 | 1024 |
| 21 | 40 | [256, 128] | 0.1 | 0.005 | 1e-05 | 512 |
| 22 | 40 | [256, 128] | 0.2 | 0.005 | 1e-04 | 512 |
| 23 | 40 | [256, 128] | 0.2 | 0.01 | 1e-04 | 1024 |
| 24 | 40 | [256, 128] | 0.3 | 0.001 | 1e-04 | 1024 |

### DeepHit

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS (early stopping, patience=8)
*alpha/sigma are DeepHit-specific ranking loss weights.*

| Candidate | num_durations | hidden_dims | dropout | alpha | sigma | learning_rate | weight_decay | batch_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | [64, 32] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 2 | 20 | [64, 32] | 0.1 | 0.2 | 0.1 | 0.01 | 1e-05 | 512 |
| 3 | 20 | [64, 32] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 4 | 20 | [64, 32] | 0.2 | 0.3 | 0.2 | 0.01 | 1e-04 | 512 |
| 5 | 20 | [128, 64] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 6 | 20 | [128, 64] | 0.1 | 0.2 | 0.2 | 0.01 | 1e-05 | 512 |
| 7 | 20 | [128, 64] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 8 | 20 | [128, 64] | 0.2 | 0.3 | 0.2 | 0.001 | 1e-04 | 1024 |
| 9 | 20 | [256, 128] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 10 | 20 | [256, 128] | 0.2 | 0.3 | 0.2 | 0.001 | 1e-04 | 1024 |
| 11 | 20 | [256, 128] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 12 | 20 | [256, 128] | 0.3 | 0.2 | 0.1 | 0.001 | 1e-04 | 1024 |
| 13 | 40 | [64, 32] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 14 | 40 | [64, 32] | 0.1 | 0.2 | 0.2 | 0.01 | 1e-05 | 512 |
| 15 | 40 | [64, 32] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 16 | 40 | [64, 32] | 0.2 | 0.3 | 0.2 | 0.01 | 1e-04 | 512 |
| 17 | 40 | [128, 64] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 18 | 40 | [128, 64] | 0.1 | 0.2 | 0.2 | 0.01 | 1e-05 | 512 |
| 19 | 40 | [128, 64] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 20 | 40 | [128, 64] | 0.2 | 0.3 | 0.2 | 0.001 | 1e-04 | 1024 |
| 21 | 40 | [256, 128] | 0.1 | 0.2 | 0.1 | 0.005 | 1e-05 | 512 |
| 22 | 40 | [256, 128] | 0.2 | 0.3 | 0.1 | 0.005 | 1e-04 | 512 |
| 23 | 40 | [256, 128] | 0.2 | 0.3 | 0.2 | 0.01 | 1e-04 | 1024 |
| 24 | 40 | [256, 128] | 0.3 | 0.2 | 0.1 | 0.001 | 1e-04 | 1024 |

### GB Cox

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS

| Candidate | loss | learning_rate | n_estimators | max_depth | subsample |
| --- | --- | --- | --- | --- | --- |
| 1 | coxph | 0.02 | 100 | 1 | 1.0 |
| 2 | coxph | 0.03 | 150 | 1 | 1.0 |
| 3 | coxph | 0.05 | 150 | 1 | 1.0 |
| 4 | coxph | 0.05 | 250 | 1 | 0.8 |
| 5 | coxph | 0.08 | 150 | 1 | 0.8 |
| 6 | coxph | 0.1 | 100 | 1 | 1.0 |
| 7 | coxph | 0.1 | 150 | 1 | 0.8 |
| 8 | coxph | 0.1 | 300 | 1 | 0.8 |
| 9 | coxph | 0.02 | 200 | 2 | 1.0 |
| 10 | coxph | 0.03 | 200 | 2 | 1.0 |
| 11 | coxph | 0.05 | 200 | 2 | 1.0 |
| 12 | coxph | 0.05 | 250 | 2 | 0.8 |
| 13 | coxph | 0.08 | 200 | 2 | 0.8 |
| 14 | coxph | 0.08 | 250 | 2 | 0.8 |
| 15 | coxph | 0.1 | 150 | 2 | 1.0 |
| 16 | coxph | 0.1 | 300 | 2 | 0.8 |
| 17 | coxph | 0.02 | 300 | 3 | 1.0 |
| 18 | coxph | 0.03 | 250 | 3 | 1.0 |
| 19 | coxph | 0.05 | 200 | 3 | 0.8 |
| 20 | coxph | 0.05 | 300 | 3 | 1.0 |
| 21 | coxph | 0.08 | 150 | 3 | 0.8 |
| 22 | coxph | 0.08 | 300 | 3 | 0.8 |
| 23 | coxph | 0.1 | 100 | 3 | 0.8 |
| 24 | coxph | 0.1 | 200 | 3 | 0.8 |

### Royston-Parmar

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS

| Candidate | n_baseline_knots | penalizer | l1_ratio |
| --- | --- | --- | --- |
| 1 | 2 | 0.01 | 0.0 |
| 2 | 2 | 0.025 | 0.0 |
| 3 | 2 | 0.05 | 0.0 |
| 4 | 2 | 0.1 | 0.0 |
| 5 | 2 | 0.15 | 0.0 |
| 6 | 2 | 0.2 | 0.0 |
| 7 | 2 | 0.3 | 0.0 |
| 8 | 2 | 0.5 | 0.0 |
| 9 | 2 | 0.7 | 0.0 |
| 10 | 2 | 1.0 | 0.0 |
| 11 | 2 | 2.0 | 0.0 |
| 12 | 2 | 5.0 | 0.0 |
| 13 | 3 | 0.01 | 0.0 |
| 14 | 3 | 0.025 | 0.0 |
| 15 | 3 | 0.05 | 0.0 |
| 16 | 3 | 0.1 | 0.0 |
| 17 | 3 | 0.15 | 0.0 |
| 18 | 3 | 0.2 | 0.0 |
| 19 | 3 | 0.3 | 0.0 |
| 20 | 3 | 0.5 | 0.0 |
| 21 | 3 | 0.7 | 0.0 |
| 22 | 3 | 1.0 | 0.0 |
| 23 | 3 | 2.0 | 0.0 |
| 24 | 3 | 5.0 | 0.0 |

### Weibull AFT

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS

| Candidate | penalizer | l1_ratio |
| --- | --- | --- |
| 1 | 1e-04 | 0.0 |
| 2 | 2e-04 | 0.0 |
| 3 | 5e-04 | 0.0 |
| 4 | 0.001 | 0.0 |
| 5 | 0.002 | 0.0 |
| 6 | 0.005 | 0.0 |
| 7 | 0.01 | 0.0 |
| 8 | 0.02 | 0.0 |
| 9 | 0.05 | 0.0 |
| 10 | 0.1 | 0.0 |
| 11 | 0.15 | 0.0 |
| 12 | 0.2 | 0.0 |
| 13 | 0.3 | 0.0 |
| 14 | 0.4 | 0.0 |
| 15 | 0.5 | 0.0 |
| 16 | 0.6 | 0.0 |
| 17 | 0.7 | 0.0 |
| 18 | 0.8 | 0.0 |
| 19 | 1.0 | 0.0 |
| 20 | 1.5 | 0.0 |
| 21 | 2.0 | 0.0 |
| 22 | 3.0 | 0.0 |
| 23 | 4.0 | 0.0 |
| 24 | 5.0 | 0.0 |

### XGBoost AFT

**Candidates:** 24  |  **Selection criterion:** lowest validation IBS

| Candidate | aft_loss_distribution | aft_loss_distribution_scale | learning_rate | num_boost_round | max_depth | min_child_weight | subsample | colsample_bytree | reg_lambda | reg_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | logistic | 1.0 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 2 | logistic | 1.0 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 3 | logistic | 1.0 | 0.03 | 400 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 4 | logistic | 1.0 | 0.05 | 400 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 5 | logistic | 1.5 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 6 | logistic | 1.5 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 7 | logistic | 1.5 | 0.03 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 8 | logistic | 1.5 | 0.05 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 9 | logistic | 2.0 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 10 | logistic | 2.0 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 11 | logistic | 2.0 | 0.03 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 12 | logistic | 2.0 | 0.05 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 13 | normal | 1.0 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 14 | normal | 1.0 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 15 | normal | 1.0 | 0.03 | 400 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 16 | normal | 1.0 | 0.05 | 400 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 17 | normal | 1.5 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 18 | normal | 1.5 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 19 | normal | 1.5 | 0.03 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 20 | normal | 1.5 | 0.05 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 21 | normal | 2.0 | 0.03 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 22 | normal | 2.0 | 0.05 | 200 | 2 | 1.0 | 0.8 | 0.8 | 1.0 | 0.0 |
| 23 | normal | 2.0 | 0.03 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |
| 24 | normal | 2.0 | 0.05 | 400 | 2 | 2.0 | 0.8 | 0.8 | 5.0 | 0.0 |

---

## Family A — Dynamic Weekly (Person-Period Arm)

All Family A models use person-period rows (one row per enrollment per week).
Validation uses a 10–20% enrollment-level GroupShuffleSplit.

### Linear Discrete-Time Hazard

**Candidates:** 24  |  **Selection criterion:** highest row-level validation C-index
*Penalty type (L1/L2) × C regularization strength.*

| Candidate | penalty | C |
| --- | --- | --- |
| 1 | l1 | 0.001 |
| 2 | l1 | 0.005 |
| 3 | l1 | 0.01 |
| 4 | l1 | 0.05 |
| 5 | l1 | 0.1 |
| 6 | l1 | 0.5 |
| 7 | l1 | 1.0 |
| 8 | l1 | 5.0 |
| 9 | l1 | 10.0 |
| 10 | l1 | 50.0 |
| 11 | l1 | 100.0 |
| 12 | l1 | 500.0 |
| 13 | l2 | 0.001 |
| 14 | l2 | 0.005 |
| 15 | l2 | 0.01 |
| 16 | l2 | 0.05 |
| 17 | l2 | 0.1 |
| 18 | l2 | 0.5 |
| 19 | l2 | 1.0 |
| 20 | l2 | 5.0 |
| 21 | l2 | 10.0 |
| 22 | l2 | 50.0 |
| 23 | l2 | 100.0 |
| 24 | l2 | 500.0 |

### Poisson Piecewise-Exponential

**Candidates:** 24  |  **Selection criterion:** lowest validation log-loss on discrete hazard
*1-D grid over L2 regularization strength (alpha). Single-param sweep via statsmodels GLM.*

| Candidate | alpha |
| --- | --- |
| 1 | 1e-05 |
| 2 | 2e-05 |
| 3 | 5e-05 |
| 4 | 1e-04 |
| 5 | 2e-04 |
| 6 | 5e-04 |
| 7 | 0.001 |
| 8 | 0.002 |
| 9 | 0.005 |
| 10 | 0.01 |
| 11 | 0.02 |
| 12 | 0.05 |
| 13 | 0.1 |
| 14 | 0.2 |
| 15 | 0.3 |
| 16 | 0.5 |
| 17 | 0.7 |
| 18 | 1.0 |
| 19 | 2.0 |
| 20 | 5.0 |
| 21 | 10.0 |
| 22 | 20.0 |
| 23 | 50.0 |
| 24 | 100.0 |

### GB Weekly Hazard

**Candidates:** 24  |  **Selection criterion:** lowest validation log-loss (early stopping)
*HistGradientBoostingClassifier with early stopping. Row budget capped by feature count.*

| Candidate | max_depth | learning_rate | max_iter | min_samples_leaf | l2_regularization |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0.02 | 80 | 300 | 5.0 |
| 2 | 2 | 0.02 | 150 | 200 | 5.0 |
| 3 | 2 | 0.03 | 120 | 300 | 5.0 |
| 4 | 2 | 0.05 | 120 | 300 | 1.0 |
| 5 | 2 | 0.05 | 200 | 160 | 5.0 |
| 6 | 2 | 0.08 | 150 | 200 | 5.0 |
| 7 | 2 | 0.1 | 100 | 250 | 5.0 |
| 8 | 2 | 0.1 | 200 | 160 | 10.0 |
| 9 | 3 | 0.02 | 150 | 300 | 10.0 |
| 10 | 3 | 0.03 | 120 | 200 | 5.0 |
| 11 | 3 | 0.03 | 200 | 160 | 5.0 |
| 12 | 3 | 0.05 | 120 | 160 | 5.0 |
| 13 | 3 | 0.05 | 200 | 200 | 10.0 |
| 14 | 3 | 0.08 | 100 | 300 | 5.0 |
| 15 | 3 | 0.08 | 200 | 160 | 10.0 |
| 16 | 3 | 0.1 | 80 | 300 | 10.0 |
| 17 | 4 | 0.02 | 200 | 300 | 10.0 |
| 18 | 4 | 0.03 | 150 | 250 | 10.0 |
| 19 | 4 | 0.05 | 100 | 300 | 10.0 |
| 20 | 4 | 0.05 | 200 | 200 | 10.0 |
| 21 | 4 | 0.08 | 80 | 300 | 5.0 |
| 22 | 4 | 0.1 | 80 | 200 | 10.0 |
| 23 | 4 | 0.1 | 150 | 160 | 10.0 |
| 24 | 4 | 0.1 | 250 | 160 | 10.0 |

### Neural Discrete-Time

**Candidates:** 24  |  **Selection criterion:** lowest validation loss (early stopping, patience=5)
*Grid product: 3 arch × 2 dropout × 2 lr × 2 wd = 24 candidates. Override: candidate_id=9 uses dropout=0.05.*

| Candidate | hidden_dims | dropout | learning_rate | weight_decay |
| --- | --- | --- | --- | --- |
| 1 | [32, 16] | 0.1 | 0.001 | 1e-05 |
| 2 | [32, 16] | 0.1 | 0.001 | 1e-04 |
| 3 | [32, 16] | 0.1 | 5e-04 | 1e-05 |
| 4 | [32, 16] | 0.1 | 5e-04 | 1e-04 |
| 5 | [32, 16] | 0.3 | 0.001 | 1e-05 |
| 6 | [32, 16] | 0.3 | 0.001 | 1e-04 |
| 7 | [32, 16] | 0.3 | 5e-04 | 1e-05 |
| 8 | [32, 16] | 0.3 | 5e-04 | 1e-04 |
| 9 | [64, 32] | 0.05 | 0.001 | 1e-05 |
| 10 | [64, 32] | 0.1 | 0.001 | 1e-04 |
| 11 | [64, 32] | 0.1 | 5e-04 | 1e-05 |
| 12 | [64, 32] | 0.1 | 5e-04 | 1e-04 |
| 13 | [64, 32] | 0.3 | 0.001 | 1e-05 |
| 14 | [64, 32] | 0.3 | 0.001 | 1e-04 |
| 15 | [64, 32] | 0.3 | 5e-04 | 1e-05 |
| 16 | [64, 32] | 0.3 | 5e-04 | 1e-04 |
| 17 | [128, 64] | 0.1 | 0.001 | 1e-05 |
| 18 | [128, 64] | 0.1 | 0.001 | 1e-04 |
| 19 | [128, 64] | 0.1 | 5e-04 | 1e-05 |
| 20 | [128, 64] | 0.1 | 5e-04 | 1e-04 |
| 21 | [128, 64] | 0.3 | 0.001 | 1e-05 |
| 22 | [128, 64] | 0.3 | 0.001 | 1e-04 |
| 23 | [128, 64] | 0.3 | 5e-04 | 1e-05 |
| 24 | [128, 64] | 0.3 | 5e-04 | 1e-04 |

---

## Notes on tuning scope

Tuning was deliberately controlled rather than exhaustive: the goal was
comparable benchmark representatives, not unbounded per-model optimization.
All candidates were evaluated under the same preprocessing pipeline
(median imputation, constant-missing categoricals, one-hot encoding,
standard scaling fitted on training rows only).

All models were subjected to an equivalent tuning budget of 24 candidate
evaluations, ensuring that no model benefited from a disproportionately
larger search effort. The budget of 24 candidates was set by the most
complex model in the benchmark (DeepSurv), whose architecture and
regularisation space required the largest grid; all remaining models were
expanded to the same ceiling to ensure symmetric search effort.

Full machine-readable candidate records are available in
`outputs_benchmark_survival/metadata/` as JSON files.
