# Student Dropout Survival Benchmark

A reproducible, survival-oriented benchmark for temporal student dropout risk modelling in Learning Analytics, built on the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

The pipeline compares **14 model families** organised into two methodologically distinct arms under a harmonised evaluation protocol that integrates predictive performance, ablation, explainability, and calibration. All 13 tuned families (excluding CatBoost, which serves as a robustness probe) are evaluated under a **uniform 24-candidate hyperparameter search budget** (Experiment AA), ensuring no model benefits from a disproportionately larger tuning effort.

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
- [Paper Submissions](#paper-submissions)
- [Citation](#citation)

---

## Overview

Standard dropout prediction studies frequently:
- compare models under heterogeneous protocols,
- prioritise discrimination (AUC) over temporal interpretability and calibration, and
- lack ablation or explainability layers to diagnose *why* a model performs as it does.

This benchmark addresses those gaps by:

1. **Harmonising two representational arms** — a *dynamic weekly arm* (Family A, person-period hazard models) and a *comparable continuous-time arm* (Family B, early-window survival models) — evaluated under the same survival-oriented metric set.
2. **Measuring four analytical layers**: predictive performance (IBS, TD concordance, Brier scores), ablation (static vs. temporal-behavioural signal removal), SHAP-based explainability, and horizon-wise calibration.
3. **Enforcing a uniform tuning discipline**: all 13 tuned model families are evaluated over an identical 24-candidate bounded search, ensuring methodological symmetry across the arms.
4. **Enforcing a full evidence-freeze contract** so every number in the final results is traced back to a deterministic, auditable artifact.

### Key results (post Experiment AA)

| Metric | Family B leader | Value |
|--------|----------------|-------|
| Integrated Brier Score | Random Survival Forest | 0.1118 |
| TD Concordance | Random Survival Forest | 0.6735 |
| IBS bootstrap rank-1 share | RSF / Neural-MTLR | 64% / 27% |
| C-index bootstrap rank-1 share | RSF | 90% |

Within Family A, all five models cluster within a 0.0011 IBS band (0.1399–0.1410), with Poisson Piecewise-Exponential narrowly leading.

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
| **D02–D08** | `D_02_A … D_08_B` | **Family A (dynamic arm)**: Linear Discrete-Time Hazard, Neural Discrete-Time Survival, Poisson Piecewise-Exponential, GB Weekly Hazard, CatBoost Weekly Hazard (each with a sensitivity/weighted variant) |
| **D09–D15** | `D_09 … D_15` | **Family B (comparable arm)**: Random Survival Forest, Gradient-Boosted Cox, Weibull AFT, Royston-Parmar, XGBoost AFT, Neural-MTLR, DeepHit |
| **D04, D05** | `D_04`, `D_05` | **Family B**: Cox (early window), DeepSurv |
| **D16** | `dropout_bench_v3_D_16_benchmark_consolidation.py` | Consolidate all per-family metrics into the unified benchmark leaderboard |
| **E** | `dropout_bench_v3_E_posthoc_audits_refatorado_v21.py` | Post-hoc audit stack: calibration, proportional-hazards, sensitivity, bootstrap uncertainty (200 resamples) |
| **F** | `dropout_bench_v3_F_ablation_stability_refatorado_v30.py` | Ablation analysis (static vs. temporal-behavioural block removal) on the manuscript-facing representative subset |
| **G** | `dropout_bench_v3_G_explainability_paper_refatorado_v7.py` | SHAP explainability, calibration figures, and full paper-facing evidence freeze (tables, figures, metadata) |
| **G13** | `dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py` | Auto-generates `S1_hyperparameter_grids.md` — the complete candidate search grids for all tuned models, sourced from pipeline metadata JSON files |

All stages read from and write to a single **DuckDB** database (`outputs_benchmark_survival/benchmark_survival.duckdb`), which acts as the canonical analytical store.

---

## Model Families

### Family A — Dynamic Weekly Arm (person-period representation)

| # | Family | Type | Tuning candidates |
|---|---|---|---|
| D02 | Linear Discrete-Time Hazard | Linear | 24 |
| D03 | Neural Discrete-Time Survival | Neural | 24 |
| D06 | Poisson Piecewise-Exponential | Linear | 24 |
| D07 | Gradient-Boosted Weekly Hazard | Tree | 24 |
| D08 | CatBoost Weekly Hazard | Tree | robustness probe |

### Family B — Comparable Continuous-Time Arm (early 4-week enrolment window)

| # | Family | Type | Tuning candidates |
|---|---|---|---|
| D04 | Cox (Early Window) | Linear | 24 |
| D05 | DeepSurv | Neural | 24 |
| D09 | Random Survival Forest (RSF) | Tree | 24 |
| D10 | Gradient-Boosted Cox | Tree | 24 |
| D11 | Weibull AFT | Parametric | 24 |
| D12 | Royston-Parmar | Parametric | 24 |
| D13 | XGBoost AFT | Tree | 24 |
| D14 | Neural-MTLR | Neural | 24 |
| D15 | DeepHit | Neural | 24 |

All 13 tuned families are evaluated under an **equivalent 24-candidate bounded search** (Experiment AA). The budget of 24 was set to match the pre-existing ceiling of the most heavily searched model (DeepSurv) and expanded uniformly upward; no model was penalised by a downward revision.

---

## Evaluation Protocol

| Metric | Description |
|---|---|
| **IBS** | Integrated Brier Score — mean squared survival error across the time axis |
| **TD Concordance** | Time-dependent C-index — ordinal discriminative ability |
| **Brier@10/20/30** | Horizon-specific Brier scores at weeks 10, 20, and 30 |
| **Calibration** | Horizon-wise reliability diagrams and calibration gap summaries |
| **Bootstrap CI** | 200 no-refit enrolment-level resamples (Family B only) |
| **Ablation** | Feature-block removal (static vs. temporal-behavioural) on 8 representative families |
| **Explainability** | SHAP block-level dominance on the same 8 families |

Results are reported **within each arm separately**; a cross-arm ranking is not warranted because the two arms use fundamentally different risk formulations.

---

## Repository Structure

```
.
├── content/                              # Raw OULAD CSV files (not tracked)
├── exe/                                  # Shell and Python helper scripts
│   ├── run_a1_to_d15_sequential_resume.sh   # Full pipeline runner (A → D15)
│   ├── status_a1_to_d15_sequential.sh       # Status checker for above
│   ├── run_python_sequence_resume.sh        # Generic resumable sequence runner
│   ├── status_python_sequence.sh            # Status for above
│   ├── run_exp_aa.sh                        # Experiment AA re-execution sequence
│   ├── generate_figures_pdf.py              # Convert paper figures PNG → PDF
│   ├── executar_git_tracking.sh             # Stage + commit + push (≤50 MiB)
│   └── matar_duckdb_zumbis.sh               # Kill stale DuckDB processes
├── graphwiz/                             # Graphviz source (.txt) and rendered PNGs for pipeline diagrams
├── notebooks/                            # Jupyter notebooks (exploratory / development)
├── paper2/                               # Manuscript sources and submission packages
│   ├── IJAIED_submission/                # Springer Nature IJAIED submission (v1, v2)
│   ├── ceai_submission/                  # Elsevier CEAI submission
│   ├── old_jedm/                         # Previous JEDM submission drafts (archived)
│   └── refs/                             # Bibliography support files
├── papers_compare/                       # Reference papers used in the comparative analysis
├── outputs_benchmark_survival/           # All pipeline outputs (generated)
│   ├── benchmark_survival.duckdb         # Analytical DuckDB store
│   ├── data/                             # Intermediate datasets
│   ├── figures/                          # Stage-level diagnostic figures
│   ├── logs/                             # Execution logs
│   ├── metadata/                         # JSON metadata, configs, run records
│   ├── models/                           # Serialised model objects
│   ├── tables/                           # Stage-level CSV tables
│   ├── paper_main/                       # Frozen paper-facing artifacts (figures + tables)
│   └── paper_appendix/                   # Frozen appendix artifacts
├── exp-AA.md                             # Experiment AA design doc + post-execution log
├── S1_hyperparameter_grids.md            # Auto-generated supplementary: full tuning grids
├── dropout_bench_v3_A_1_foundation.py
├── dropout_bench_v3_A_2_runtime_config.py
├── dropout_bench_v3_B_feature_engineering_refatorado_v5.py
├── dropout_bench_v3_C_split_and_audit_from_scratch_v2_minimal_changes.py
├── dropout_bench_v3_D_00_common.py       # Shared utilities for all D stages
├── dropout_bench_v3_D_01_contract_runtime_materialization.py
├── dropout_bench_v3_D_02_A … D_15.py    # Individual model families
├── dropout_bench_v3_D_16_benchmark_consolidation.py
├── dropout_bench_v3_E_posthoc_audits_refatorado_v21.py
├── dropout_bench_v3_F_ablation_stability_refatorado_v30.py
├── dropout_bench_v3_G_explainability_paper_refatorado_v7.py
├── dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py
├── benchmark_shared_config.toml          # Shared runtime configuration (written by Stage A)
├── benchmark_modeling_contract.toml      # Feature contract (written by Stage B)
├── util.py                               # Shared low-level utilities
└── requirements.txt                      # Python dependencies
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
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_A_1_foundation.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_B_feature_engineering_refatorado_v5.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_C_split_and_audit_from_scratch_v2_minimal_changes.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_D_01_contract_runtime_materialization.py
# ... continue through D02–D16 in order
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_E_posthoc_audits_refatorado_v21.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_F_ablation_stability_refatorado_v30.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_G_explainability_paper_refatorado_v7.py
/Users/rafars/.pyenv/versions/3.9.13/bin/python dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py
```

### Re-run only the Experiment AA sequence (D stages onward)

For re-tuning after grid changes without re-running A/B/C:

```bash
bash exe/run_exp_aa.sh
```

This covers: D02 → D03 → D04 → D06 → D07 → D09 → D10 → D11 → D12 → D13 → D14 → D15 → D16 → E → F → G → G13.  
Estimated total runtime: **5–8 hours**.

### Run a specific model subset

The `.vscode/tasks.json` defines several pre-configured task sequences, including:

- `run-d5.9-to-d5.15` — comparable arm only (D09–D15)
- `rerun-d5.4-d5.5-d5.9-full-windows` — D04, D05, D09 with early window

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
| `S1_hyperparameter_grids.md` | Auto-generated supplementary file with full tuning grids for all 13 models |

The **paper-main artifacts** are the canonical quantitative contract for the manuscript — they are written once by Stage G and treated as immutable by the paper integration layer. `S1_hyperparameter_grids.md` is re-generated by G13 and should be updated whenever tuning grids change.

---

## Configuration

### `benchmark_shared_config.toml`

Written by Stage A and consumed by all downstream stages.

```toml
[paths]
data_dir = "content"
output_dir = "outputs_benchmark_survival"
duckdb_filename = "benchmark_survival.duckdb"
# ... additional path keys

[benchmark]
seed = 42
test_size = 0.3
early_window_weeks = 4
main_enrollment_window_weeks = 4
temporal_buckets_q = 4
unit_of_analysis = "enrollment"
time_granularity = "weekly"
event_definition = "Withdrawn with valid date_unregistration"

[keys]
enrollment_key = ["id_student", "code_module", "code_presentation"]

[runtime]
cpu_cores = 8
tuning_parallel_backend = "processes"
```

### `benchmark_modeling_contract.toml`

Written by Stage B. Defines the stable feature sets and modeling parameters.  
**Paper-aligned values are locked** — changing them breaks comparability with published results.

### `exp-AA.md`

Design document and post-execution log for Experiment AA (uniform 24-candidate tuning). Contains the full table of grid changes per model, the re-execution sequence, risk scenarios, and the step-by-step guide for updating the manuscript `.tex` after re-execution. Status: **CONCLUÍDO** (executed 2026-06-19/20).

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `exe/run_exp_aa.sh` | Re-execute all D stages + downstream in the Experiment AA order |
| `exe/generate_figures_pdf.py` | Convert paper figures from PNG to PDF for LaTeX inclusion |
| `exe/executar_git_tracking.sh` | Stage all changed files ≤ 50 MiB, commit, and push to `origin` |
| `exe/matar_duckdb_zumbis.sh` | Kill stale DuckDB lock processes (useful after interrupted runs) |
| `exe/run_python_sequence_resume.sh` | Generic resumable Python script sequence runner |
| `exe/status_python_sequence.sh` | Status for the generic sequence runner |

---

## Paper Submissions

The manuscript is currently under review at two venues (distinct versions):

| Venue | Directory | Status |
|---|---|---|
| [IJAIED](https://www.springer.com/journal/40593) — Int. J. of AI in Education (Springer Nature) | `paper2/IJAIED_submission/` | v2 in preparation |
| [CEAI](https://www.sciencedirect.com/journal/computers-and-education-artificial-intelligence) — Computers and Education: AI (Elsevier) | `paper2/ceai_submission/` | v1 submitted |

Both submission packages include the frozen paper figures and the `S1_hyperparameter_grids.md` supplementary file generated by G13.

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
