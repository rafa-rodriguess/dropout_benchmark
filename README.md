# Student Dropout Survival Benchmark

A reproducible, survival-oriented benchmark for temporal student dropout risk modelling in Learning Analytics, built on the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

The pipeline compares **14 tuned model families** organised into two methodologically distinct arms under a harmonised evaluation protocol that integrates predictive performance, ablation, explainability, and calibration.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Families](#model-families)
- [Evaluation Protocol](#evaluation-protocol)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Utility Scripts](#utility-scripts)
- [Citation](#citation)

---

## Overview

Standard dropout prediction studies frequently:
- compare models under heterogeneous protocols,
- prioritise discrimination (AUC) over temporal interpretability and calibration, and
- lack ablation or explainability layers to diagnose *why* a model performs as it does.

This benchmark addresses those gaps by:

1. **Harmonising two representational arms** — a *dynamic weekly arm* (person-period hazard models) and a *comparable continuous-time arm* (early-window survival models) — evaluated under the same survival-oriented metric set.
2. **Measuring four analytical layers**: predictive performance (IBS, TD concordance, Brier scores), ablation (static vs. temporal-behavioural signal removal), SHAP-based explainability, and horizon-wise calibration.
3. **Enforcing a bounded tuning discipline** and a full evidence-freeze contract so every number in the final results is traced back to a deterministic, auditable artifact.

---

## Dataset

**OULAD** — Open University Learning Analytics Dataset  
Source: <https://analyse.kmi.open.ac.uk/open_dataset>

Required raw files (place inside `content/`):

| File | Description |
|---|---|
| `studentInfo.csv` | Demographic and registration data |
| `studentRegistration.csv` | Enrolment and unregistration dates |
| `studentVle.csv` | Weekly VLE click interactions |
| `courses.csv` | Module and presentation metadata |
| `vle.csv` | VLE activity-type catalogue |
| `studentAssessment.csv` | Assessment submission records |
| `assessments.csv` | Assessment design metadata |

Alternatively, extract `anonymisedData.zip` (included in the repository) into `content/`.

The **event definition** used throughout the benchmark is: *Withdrawn with a valid `date_unregistration`*.  
The **unit of analysis** is the enrolment (student × module × presentation).

---

## Pipeline Architecture

The pipeline is divided into seven sequential stages, each implemented as a standalone Python script:

```
Stage A  →  Stage B  →  Stage C  →  Stage D (D01–D16)  →  Stage E  →  Stage F  →  Stage G
Foundation  Features    Split       Modelling             Post-hoc    Ablation    Explainability
                                                          Audits      & Stability & Paper Export
```

| Stage | Script(s) | Role |
|---|---|---|
| **A** | `dropout_bench_v3_A_1_foundation.py`, `A_2_runtime_config.py` | Load raw OULAD sources into DuckDB; build the enrolment backbone and canonical survival-ready table; write `benchmark_shared_config.toml` |
| **B** | `dropout_bench_v3_B_feature_engineering_refatorado_v5.py` | Construct weekly person-period features and enrolment-level early-window features; write `benchmark_modeling_contract.toml` |
| **C** | `dropout_bench_v3_C_split_and_audit_from_scratch_v2_minimal_changes.py` | Build the canonical 70/30 enrolment-level train/test split; materialise split-propagated DuckDB tables |
| **D01** | `dropout_bench_v3_D_01_contract_runtime_materialization.py` | Materialise the runtime contract for downstream D stages |
| **D02–D08** | `D_02_A … D_08_B` | **Dynamic arm**: Linear Discrete-Time Hazard, Neural Discrete-Time Survival, Poisson Piecewise-Exponential, GB Weekly Hazard, CatBoost Weekly Hazard (each with a sensitivity/weighted variant) |
| **D09–D15** | `D_09 … D_15` | **Comparable arm**: Random Survival Forest, Gradient-Boosted Cox, Weibull AFT, Royston-Parmar, XGBoost AFT, Neural-MTLR, DeepHit |
| **D16** | `dropout_bench_v3_D_16_benchmark_consolidation.py` | Consolidate all per-family metrics into the unified benchmark leaderboard |
| **E** | `dropout_bench_v3_E_posthoc_audits_refatorado_v21.py` | Post-hoc audit stack: calibration, proportional-hazards, sensitivity, bootstrap uncertainty |
| **F** | `dropout_bench_v3_F_ablation_stability_refatorado_v30.py` | Ablation analysis (static vs. temporal-behavioural block removal) on the manuscript-facing representative subset |
| **G** | `dropout_bench_v3_G_explainability_paper_refatorado_v7.py` | SHAP explainability, calibration figures, and full paper-facing evidence freeze (tables, figures, metadata) |

All stages read from and write to a single **DuckDB** database (`outputs_benchmark_survival/benchmark_survival.duckdb`), which acts as the canonical analytical store.

---

## Model Families

### Dynamic Weekly Arm (person-period representation)

| # | Family | Type |
|---|---|---|
| D02 | Linear Discrete-Time Hazard | Linear |
| D03 | Neural Discrete-Time Survival | Neural |
| D06 | Poisson Piecewise-Exponential | Linear |
| D07 | Gradient-Boosted Weekly Hazard | Tree |
| D08 | CatBoost Weekly Hazard | Tree |

### Comparable Continuous-Time Arm (early 4-week enrolment window)

| # | Family | Type |
|---|---|---|
| D09 | Random Survival Forest (RSF) | Tree |
| D10 | Gradient-Boosted Cox | Tree |
| D11 | Weibull AFT | Parametric |
| D12 | Royston-Parmar | Parametric |
| D13 | XGBoost AFT | Tree |
| D14 | Neural-MTLR | Neural |
| D15 | DeepHit | Neural |
| D04 | Cox (Early Window) | Linear |
| D05 | DeepSurv | Neural |

All 14 families are trained with a **bounded hyperparameter search** (not exhaustive grid search) and evaluated under a shared survival-oriented protocol.

---

## Evaluation Protocol

| Metric | Description |
|---|---|
| **IBS** | Integrated Brier Score — mean squared survival error across the time axis |
| **TD Concordance** | Time-dependent C-index — ordinal discriminative ability |
| **Brier@10/20/30** | Horizon-specific Brier scores at weeks 10, 20, and 30 |
| **Calibration** | Horizon-wise reliability diagrams and calibration gap summaries |
| **Bootstrap CI** | 200 no-refit enrolment-level resamples (comparable arm) |
| **Ablation** | Feature-block removal (static vs. temporal-behavioural) on 8 representative families |
| **Explainability** | SHAP block-level dominance on the same 8 families |

Results are reported **within each arm separately**; a cross-arm ranking is not warranted because the two arms use fundamentally different risk formulations.

---

## Repository Structure

```
.
├── content/                          # Raw OULAD CSV files (not tracked)
├── exe/                              # Shell helper scripts
│   ├── run_a1_to_d15_sequential_resume.sh   # Full pipeline runner (A → D15)
│   ├── status_a1_to_d15_sequential.sh       # Status checker for above
│   ├── run_python_sequence_resume.sh        # Generic resumable sequence runner
│   ├── status_python_sequence.sh            # Status for above
│   ├── executar_git_tracking.sh             # Stage + commit + push (≤50 MiB)
│   └── matar_duckdb_zumbis.sh               # Kill stale DuckDB processes
├── graphwiz/                         # Graphviz source (.txt) and rendered PNGs for pipeline diagrams
├── notebooks/                        # Jupyter notebooks (exploratory / development)
├── outputs_benchmark_survival/       # All pipeline outputs (generated)
│   ├── benchmark_survival.duckdb     # Analytical DuckDB store
│   ├── data/                         # Intermediate datasets
│   ├── figures/                      # Stage-level diagnostic figures
│   ├── logs/                         # Execution logs
│   ├── metadata/                     # JSON metadata, configs, run records
│   ├── models/                       # Serialised model objects
│   ├── tables/                       # Stage-level CSV tables
│   ├── paper_main/                   # Frozen paper-facing artifacts (figures + tables)
│   └── paper_appendix/               # Frozen appendix artifacts
├── paper/                            # Manuscript source and bibliography
├── dropout_bench_v3_A_1_foundation.py
├── dropout_bench_v3_A_2_runtime_config.py
├── dropout_bench_v3_B_feature_engineering_refatorado_v5.py
├── dropout_bench_v3_C_split_and_audit_from_scratch_v2_minimal_changes.py
├── dropout_bench_v3_D_00_common.py   # Shared utilities for all D stages
├── dropout_bench_v3_D_01_contract_runtime_materialization.py
├── dropout_bench_v3_D_02_A … D_15.py  # Individual model families
├── dropout_bench_v3_D_16_benchmark_consolidation.py
├── dropout_bench_v3_E_posthoc_audits_refatorado_v21.py
├── dropout_bench_v3_F_ablation_stability_refatorado_v30.py
├── dropout_bench_v3_G_explainability_paper_refatorado_v7.py
├── benchmark_shared_config.toml      # Shared runtime configuration
├── benchmark_modeling_contract.toml  # Modeling feature contract (written by Stage B)
├── util.py                           # Shared low-level utilities
└── requirements.txt                  # Python dependencies
```

---

## Requirements

- **Python 3.9.x** (tested with 3.9.13)
- **OS**: macOS or Linux

Core Python dependencies (`requirements.txt`):

```
duckdb
numpy
pandas
matplotlib
catboost
xgboost
scikit-learn
scikit-survival
lifelines
torch
torchtuples
pycox
requests
tomli            # Python < 3.11 only; stdlib tomllib used on 3.11+
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/rafa-rodriguess/dropout_benchmark.git
cd dropout_benchmark

# 2. Create and activate a virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw OULAD files in content/
#    Either copy the CSVs manually or extract the included zip:
unzip anonymisedData.zip -d content/
```

---

## Running the Pipeline

### Full sequential run (A → D15), with resume support

```bash
bash exe/run_a1_to_d15_sequential_resume.sh
```

This script:
- runs all stages from A1 through D15 in order,
- writes a checkpoint file so it can **resume from the last successful stage** after a failure,
- logs all output to `run_logs/a1_to_d15_resume/`.

Check status at any time:

```bash
bash exe/status_a1_to_d15_sequential.sh
```

### Run individual stages

```bash
python dropout_bench_v3_A_1_foundation.py
python dropout_bench_v3_B_feature_engineering_refatorado_v5.py
python dropout_bench_v3_C_split_and_audit_from_scratch_v2_minimal_changes.py
python dropout_bench_v3_D_01_contract_runtime_materialization.py
# ... continue through D02–D16 in order
python dropout_bench_v3_E_posthoc_audits_refatorado_v21.py
python dropout_bench_v3_F_ablation_stability_refatorado_v30.py
python dropout_bench_v3_G_explainability_paper_refatorado_v7.py
```

### Run a specific model subset (D09 → D15)

The `.vscode/tasks.json` defines several pre-configured task sequences, including:

```bash
# D9 through D15 (comparable arm)
# run-d5.9-to-d5.15

# D4, D5, D9 with configured early window
# rerun-d5.4-d5.5-d5.9-full-windows
```

These can be invoked from the VS Code **Run Task** menu or adapted for direct shell execution.

### Override the Python interpreter

```bash
PYTHON_BIN=/path/to/python bash exe/run_a1_to_d15_sequential_resume.sh
```

---

## Outputs

After a complete run, the key artifacts are:

| Path | Content |
|---|---|
| `outputs_benchmark_survival/benchmark_survival.duckdb` | Full analytical store — all modeling tables, metrics, calibration data |
| `outputs_benchmark_survival/tables/` | Per-stage CSV exports of primary metrics |
| `outputs_benchmark_survival/figures/` | Diagnostic figures (calibration, ablation, explainability) |
| `outputs_benchmark_survival/models/` | Serialised tuned model objects |
| `outputs_benchmark_survival/paper_main/` | Frozen paper-facing figures and tables |
| `outputs_benchmark_survival/paper_appendix/` | Frozen appendix figures and tables |
| `outputs_benchmark_survival/metadata/` | JSON metadata: run records, configs, benchmark contracts |
| `outputs_benchmark_survival/logs/` | Execution logs with timestamps |

The **paper-main artifacts** are the canonical quantitative contract for the manuscript — they are written once by Stage G and treated as immutable by the paper integration layer.

---

## Configuration

### `benchmark_shared_config.toml`

Written by Stage A and consumed by all downstream stages.

```toml
[benchmark]
seed = 42
test_size = 0.3
early_window_weeks = 4
benchmark_horizons = [10, 20, 30]
event_definition = "Withdrawn with valid date_unregistration"

[runtime]
cpu_cores = 8
tuning_parallel_backend = "processes"
```

### `benchmark_modeling_contract.toml`

Written by Stage B. Defines the stable feature sets and modeling parameters.  
**Paper-aligned values are locked** — changing them breaks comparability with published results.

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `exe/executar_git_tracking.sh` | Stage all changed files ≤ 50 MiB, commit, and push to `origin` |
| `exe/matar_duckdb_zumbis.sh` | Kill stale DuckDB lock processes (useful after interrupted runs) |
| `exe/run_python_sequence_resume.sh` | Generic resumable Python script sequence runner |
| `exe/status_python_sequence.sh` | Status for the generic sequence runner |

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{RafaelGitHubBenchmark2026,
  author       = {da Silva, Rafael and Eicher, Jeff and Longo, Gregory},
  title        = {dropout\_benchmark},
  year         = {2026},
  howpublished = {\url{https://github.com/rafa-rodriguess/dropout_benchmark}},
  note         = {GitHub repository; pipeline scripts, figures, and frozen benchmark artifacts}
}
```

---

## License

This repository contains research code. The OULAD dataset is distributed under its own licence — see the [OULAD data page](https://analyse.kmi.open.ac.uk/open_dataset) for terms of use.
