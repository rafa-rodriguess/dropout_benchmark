from __future__ import annotations

"""
Production not-weighted neural discrete-time hazard benchmark module for the official D5.3 arm.

What this file does:
- prepares the neural discrete-time hazard treatment for the person-period benchmark arm
- tunes and fits the neural survival model under the official benchmark protocol
- evaluates the tuned model with survival, calibration, row-level, and hazard diagnostics
- persists the trained model state, fitted preprocessor, DuckDB audit tables, and JSON metadata artifacts
- can optionally rerun the dynamic arm under an explicit information limit up to each window `w` from the centralized modeling contract

Main processing purpose:
- materialize the full neural benchmark arm deterministically from DuckDB-ready tables without notebook-specific runtime state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_p18_neural_not_weighted_treatment_summary, table_p18_neural_not_weighted_feature_manifest,
  table_p18_neural_not_weighted_canonical_alignment, table_p18_neural_not_weighted_output_feature_manifest,
  table_neural_not_weighted_tuning_results, table_neural_not_weighted_tuned_training_history,
  table_neural_not_weighted_tuned_test_predictions, table_neural_not_weighted_tuned_primary_metrics,
  table_neural_not_weighted_tuned_brier_by_horizon, table_neural_not_weighted_tuned_secondary_metrics,
  table_neural_not_weighted_tuned_td_auc_support_audit, table_neural_not_weighted_tuned_row_diagnostics,
  table_neural_not_weighted_tuned_support_by_horizon, table_neural_not_weighted_tuned_calibration_summary,
  table_neural_not_weighted_tuned_calibration_bins_by_horizon,
  table_neural_not_weighted_tuned_predicted_vs_observed_survival,
  table_neural_not_weighted_tuned_hazard_audit_summary, table_neural_not_weighted_tuned_hazard_by_week
- outputs_benchmark_survival/metadata/metadata_p18_neural_not_weighted_treatment.json
- outputs_benchmark_survival/metadata/neural_not_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/neural_discrete_time_survival_not_weighted_tuned.pt
- outputs_benchmark_survival/models/neural_discrete_time_not_weighted_preprocessor.joblib

Main DuckDB tables used as inputs:
- pp_neural_hazard_ready_train
- pp_neural_hazard_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_p18_neural_not_weighted_treatment_summary
- table_p18_neural_not_weighted_feature_manifest
- table_p18_neural_not_weighted_canonical_alignment
- table_p18_neural_not_weighted_output_feature_manifest
- table_neural_not_weighted_tuning_results
- table_neural_not_weighted_tuned_training_history
- table_neural_not_weighted_tuned_test_predictions
- table_neural_not_weighted_tuned_primary_metrics
- table_neural_not_weighted_tuned_brier_by_horizon
- table_neural_not_weighted_tuned_secondary_metrics
- table_neural_not_weighted_tuned_td_auc_support_audit
- table_neural_not_weighted_tuned_row_diagnostics
- table_neural_not_weighted_tuned_support_by_horizon
- table_neural_not_weighted_tuned_calibration_summary
- table_neural_not_weighted_tuned_calibration_bins_by_horizon
- table_neural_not_weighted_tuned_predicted_vs_observed_survival
- table_neural_not_weighted_tuned_hazard_audit_summary
- table_neural_not_weighted_tuned_hazard_by_week
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/metadata_p18_neural_not_weighted_treatment.json
- outputs_benchmark_survival/metadata/neural_not_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/neural_discrete_time_survival_not_weighted_tuned.pt
- outputs_benchmark_survival/models/neural_discrete_time_not_weighted_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, or CSV-based workflows are permitted

Execution modes:
- default execution iterates over the full contract-driven window grid from `benchmark.early_window_sensitivity_weeks`
"""

import itertools
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.evaluation import EvalSurv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv, check_y_survival
from torch.utils.data import DataLoader, TensorDataset

from dropout_bench_v3_D_00_common import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_OBSERVED_RISK_METHOD,
    append_suffix_before_extension,
    apply_name_suffix,
    build_ipcw_calibration_artifacts,
    resolve_benchmark_horizons,
    resolve_calibration_bins,
    resolve_early_window_sensitivity_weeks,
    summarize_calibration_by_horizon,
)


if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    import tomli as toml_reader


STAGE_PREFIX = "5.3"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_NAME = "dropout_bench_v3_D_5_3_not_weighted.ipynb"
PREVIEW_ROWS = 20
with open(PROJECT_ROOT / "benchmark_modeling_contract.toml", "rb") as _contract_file_obj:
    _MODULE_MODELING_CONTRACT = toml_reader.load(_contract_file_obj)
BENCHMARK_HORIZONS = tuple(resolve_benchmark_horizons(_MODULE_MODELING_CONTRACT["benchmark"]))
CALIBRATION_BINS = resolve_calibration_bins(_MODULE_MODELING_CONTRACT["benchmark"])
REQUIRED_SHARED_PATH_KEYS = [
    "output_dir",
    "tables_subdir",
    "metadata_subdir",
    "models_subdir",
    "data_output_subdir",
    "duckdb_filename",
]
REQUIRED_MODELING_KEYS = ["benchmark", "modeling", "feature_contract"]
REQUIRED_INPUT_TABLES = ["pp_neural_hazard_ready_train", "pp_neural_hazard_ready_test"]
REQUIRED_PERSON_PERIOD_COLUMNS = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "week",
    "event_t",
    "event_observed",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_clicks_week",
    "active_this_week",
    "n_vle_rows_week",
    "n_distinct_sites_week",
    "cum_clicks_until_t",
    "recency",
    "streak",
]
CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]
NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "week",
    "total_clicks_week",
    "active_this_week",
    "n_vle_rows_week",
    "n_distinct_sites_week",
    "cum_clicks_until_t",
    "recency",
    "streak",
]
FEATURE_ALIAS_MAP = {"total_clicks": "total_clicks_week"}
MODEL_NAME = "neural_discrete_time_survival_not_weighted_tuned"
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 1024
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
NEURAL_GRID_OVERRIDES = {
    9: {"dropout": 0.05},
}
NEURAL_GRID = tuple(
    {
        "candidate_id": candidate_id,
        "hidden_dims": list(hidden_dims),
        "dropout": float(NEURAL_GRID_OVERRIDES.get(candidate_id, {}).get("dropout", dropout)),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    for candidate_id, (hidden_dims, dropout, learning_rate, weight_decay) in enumerate(
        itertools.product(
            ([64, 32], [128, 64]),
            (0.10, 0.30),
            (1e-3, 5e-4),
            (1e-5, 1e-4),
        ),
        start=1,
    )
)
MAX_DATA_LOADER_WORKERS = 8


@dataclass
class PipelineContext:
    project_root: Path
    script_name: str
    notebook_name: str
    config_toml_path: Path
    modeling_contract_toml_path: Path
    run_metadata_path: Path
    output_dir: Path
    tables_dir: Path
    metadata_dir: Path
    models_dir: Path
    data_output_dir: Path
    duckdb_path: Path
    run_id: str
    random_seed: int
    test_size: float
    early_window_weeks: int
    main_enrollment_window_weeks: int
    early_window_sensitivity_weeks: list[int]
    cpu_cores: int
    shared_config: dict[str, Any]
    shared_modeling_contract: dict[str, Any]
    run_metadata: dict[str, Any]
    con: Any


@dataclass(frozen=True)
class WindowRunSpec:
    active_window_weeks: int | None
    output_suffix: str
    is_window_truncated_run: bool
    run_label: str
    train_table_name: str
    test_table_name: str


@dataclass
class NeuralTreatmentState:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    feature_columns: list[str]
    expected_features_raw: list[str]
    expected_features_resolved: list[str]
    preprocessor: ColumnTransformer
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    feature_names_out: list[str]


class TunedHazardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.network(tensor)


def log_stage_start(block_number: str, title: str) -> None:
    print(f"[START] {STAGE_PREFIX} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# ==============================================================")
    print(f"# {block_number} - {title}")
    print("# ==============================================================")


def log_stage_end(block_number: str) -> None:
    print(f"[END] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_artifact(label: str, location: str) -> None:
    print(f"ARTIFACT | {label} | {location}")


def require_mapping_keys(mapping: dict[str, Any], required_keys: list[str], mapping_name: str) -> None:
    missing_keys = [key for key in required_keys if key not in mapping]
    if missing_keys:
        raise KeyError(f"{mapping_name} is missing required keys: {', '.join(missing_keys)}")


def require_list_of_strings(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")
    return list(value)


def print_table_audit(con: Any, table_name: str, label: str, preview_rows: int = PREVIEW_ROWS) -> None:
    row_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    preview_df = con.execute(f"SELECT * FROM {table_name} LIMIT {preview_rows}").fetchdf()
    column_count = int(len(preview_df.columns))
    print(f"[{label}]")
    print(f"table_name={table_name}")
    print(f"rows={row_count}, cols={column_count}")
    if preview_df.empty:
        print("[empty table]")
    else:
        print(preview_df.to_string(index=False))


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_dynamic_window_suffix(active_window_weeks: int | None) -> str:
    return "" if active_window_weeks is None else f"_w{int(active_window_weeks)}"


def resolve_dynamic_window_execution_plan(ctx: PipelineContext) -> list[WindowRunSpec]:
    return [
        WindowRunSpec(
            active_window_weeks=int(window_weeks),
            output_suffix=_build_dynamic_window_suffix(int(window_weeks)),
            is_window_truncated_run=True,
            run_label=f"dynamic_window_w{int(window_weeks)}",
            train_table_name="pp_neural_hazard_ready_train",
            test_table_name="pp_neural_hazard_ready_test",
        )
        for window_weeks in ctx.early_window_sensitivity_weeks
    ]


def materialize_dataframe_table(
    ctx: PipelineContext,
    df: pd.DataFrame,
    table_name: str,
    block_number: str,
    label: str,
) -> None:
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_d_5_3_materialize_df__"
    try:
        ctx.con.unregister(temp_view_name)
    except Exception:
        pass
    ctx.con.execute(f"DROP TABLE IF EXISTS {table_name}")
    ctx.con.register(temp_view_name, df)
    try:
        ctx.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_view_name}")
    finally:
        ctx.con.unregister(temp_view_name)
    register_duckdb_table(
        con=ctx.con,
        table_name=table_name,
        notebook_name=ctx.notebook_name,
        cell_name=block_number,
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_table_audit(ctx.con, table_name, label=label)
    print_artifact(label, f"duckdb://{table_name}")


def available_tables(con: Any) -> set[str]:
    return set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())


def require_tables(con: Any, required_tables: list[str], block_number: str) -> None:
    missing_tables = [table_name for table_name in required_tables if table_name not in available_tables(con)]
    if missing_tables:
        raise FileNotFoundError(
            f"{block_number}: missing required DuckDB table(s): {', '.join(missing_tables)}"
        )


def get_table_columns(con: Any, table_name: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]


def require_columns(con: Any, table_name: str, required_columns: list[str]) -> None:
    actual_columns = set(get_table_columns(con, table_name))
    missing_columns = sorted(set(required_columns) - actual_columns)
    if missing_columns:
        raise KeyError(f"{table_name} is missing required columns: {', '.join(missing_columns)}")


def load_required_table(ctx: PipelineContext, table_name: str, required_columns: list[str], block_number: str) -> pd.DataFrame:
    require_tables(ctx.con, [table_name], block_number=block_number)
    require_columns(ctx.con, table_name, required_columns)
    df = ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()
    if df.empty:
        raise ValueError(f"{table_name} is empty.")
    return df


def ensure_binary_target(series: pd.Series, series_name: str) -> pd.Series:
    target = pd.to_numeric(series, errors="raise").astype(int)
    unique_values = sorted(target.dropna().unique().tolist())
    if unique_values != [0, 1]:
        raise ValueError(f"{series_name} must be binary with values [0, 1]. Found: {unique_values}")
    return target


def build_truth_by_enrollment(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_columns = ["enrollment_id", "event_observed", "t_event_week", "t_final_week"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{dataset_name} is missing required columns: {', '.join(missing_columns)}")
    truth_df = (
        df.groupby("enrollment_id", as_index=False)
        .agg(
            event=("event_observed", "max"),
            t_event_week=("t_event_week", "first"),
            t_final_week=("t_final_week", "first"),
        )
        .sort_values("enrollment_id")
        .reset_index(drop=True)
    )
    truth_df["event"] = ensure_binary_target(truth_df["event"], f"{dataset_name}.event")
    # FIX: use study-level enrollment duration, not the training-window end week.
    # t_event_week: actual withdrawal week (for dropouts); t_final_week: last observed week (for non-dropouts).
    truth_df["duration"] = truth_df.apply(
        lambda r: int(r["t_event_week"]) if pd.notna(r["t_event_week"]) else int(r["t_final_week"]),
        axis=1,
    ).astype(int)
    if truth_df["duration"].lt(0).any():
        raise ValueError(f"{dataset_name} contains negative enrollment durations.")
    return truth_df[["enrollment_id", "event", "duration"]]


def get_prediction_at_horizon(predictions_df: pd.DataFrame, horizon_week: int, value_column: str) -> pd.DataFrame:
    eligible_df = predictions_df.loc[predictions_df["week"] <= horizon_week, ["enrollment_id", "week", value_column]].copy()
    if eligible_df.empty:
        raise ValueError(f"No prediction rows are available at or before horizon {horizon_week}.")
    return (
        eligible_df.sort_values(["enrollment_id", "week"])
        .groupby("enrollment_id", as_index=False)
        .tail(1)[["enrollment_id", value_column]]
        .reset_index(drop=True)
    )


def compute_ipcw_time_dependent_auc(
    survival_train: Any,
    survival_test: Any,
    risk_scores: np.ndarray,
    horizon_week: float,
    tied_tol: float = 1e-8,
) -> float:
    test_event, test_time = check_y_survival(survival_test)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if risk_scores.ndim != 1:
        raise ValueError("risk_scores must be one-dimensional for single-horizon IPCW AUC computation.")
    if risk_scores.shape[0] != test_time.shape[0]:
        raise ValueError("risk_scores length must match the number of survival_test rows.")
    if not np.isfinite(risk_scores).all():
        raise ValueError("risk_scores contains non-finite values.")

    censoring_estimator = CensoringDistributionEstimator()
    censoring_estimator.fit(survival_train)
    ipcw = censoring_estimator.predict_ipcw(survival_test)

    order = np.argsort(-risk_scores)
    sorted_time = test_time[order]
    sorted_event = test_event[order]
    sorted_scores = risk_scores[order]
    sorted_ipcw = ipcw[order]

    is_case = (sorted_time <= horizon_week) & sorted_event
    is_control = sorted_time > horizon_week
    n_controls = int(is_control.sum())
    if n_controls <= 0:
        raise ValueError(f"No dynamic controls are available at horizon {horizon_week} for IPCW AUC.")
    case_weight_total = float((is_case * sorted_ipcw).sum())
    if case_weight_total <= 0.0:
        raise ValueError(f"No weighted cumulative cases are available at horizon {horizon_week} for IPCW AUC.")

    estimate_diff = np.concatenate(([np.inf], sorted_scores))
    is_tied = np.absolute(np.diff(estimate_diff)) <= tied_tol

    true_pos = np.cumsum(is_case * sorted_ipcw) / case_weight_total
    false_pos = np.cumsum(is_control) / n_controls

    tied_indices = np.flatnonzero(is_tied) - 1
    true_pos_no_ties = np.delete(true_pos, tied_indices)
    false_pos_no_ties = np.delete(false_pos, tied_indices)
    true_pos_no_ties = np.r_[0.0, true_pos_no_ties]
    false_pos_no_ties = np.r_[0.0, false_pos_no_ties]
    return float(np.trapezoid(true_pos_no_ties, false_pos_no_ties))


def set_deterministic_state(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def initialize_context() -> PipelineContext:
    from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
    from util import ensure_pipeline_catalog, open_duckdb_connection

    # Inputs:
    # - benchmark_shared_config.toml
    # - benchmark_modeling_contract.toml
    # - outputs_benchmark_survival/metadata/run_metadata.json
    # Outputs:
    # - validated runtime context in memory
    # - active DuckDB connection
    # - ensured pipeline catalog objects
    log_stage_start("5.3.1", "Lightweight runtime bootstrap")

    config_toml_path = PROJECT_ROOT / "benchmark_shared_config.toml"
    modeling_contract_toml_path = PROJECT_ROOT / "benchmark_modeling_contract.toml"
    run_metadata_path = PROJECT_ROOT / "outputs_benchmark_survival" / "metadata" / "run_metadata.json"

    if not config_toml_path.exists():
        raise FileNotFoundError(f"Missing shared benchmark config TOML: {config_toml_path}")
    if not modeling_contract_toml_path.exists():
        raise FileNotFoundError(f"Missing modeling contract TOML: {modeling_contract_toml_path}")
    if not run_metadata_path.exists():
        raise FileNotFoundError(f"Missing execution metadata JSON: {run_metadata_path}")

    with open(config_toml_path, "rb") as file_obj:
        shared_config = toml_reader.load(file_obj)
    with open(modeling_contract_toml_path, "rb") as file_obj:
        shared_modeling_contract = toml_reader.load(file_obj)
    with open(run_metadata_path, "r", encoding="utf-8") as file_obj:
        run_metadata = json.load(file_obj)

    require_mapping_keys(shared_config, ["paths"], "benchmark_shared_config.toml")
    require_mapping_keys(shared_modeling_contract, REQUIRED_MODELING_KEYS, "benchmark_modeling_contract.toml")
    require_mapping_keys(run_metadata, ["run_id"], "run_metadata.json")

    paths_config = shared_config["paths"]
    require_mapping_keys(paths_config, REQUIRED_SHARED_PATH_KEYS, "benchmark_shared_config.toml [paths]")
    benchmark_config = shared_modeling_contract["benchmark"]
    require_mapping_keys(
        benchmark_config,
        ["seed", "test_size", "early_window_weeks", "main_enrollment_window_weeks"],
        "benchmark_modeling_contract.toml [benchmark]",
    )

    output_dir = PROJECT_ROOT / str(paths_config["output_dir"])
    tables_dir = output_dir / str(paths_config["tables_subdir"])
    metadata_dir = output_dir / str(paths_config["metadata_subdir"])
    models_dir = output_dir / str(paths_config["models_subdir"])
    data_output_dir = output_dir / str(paths_config["data_output_subdir"])
    duckdb_path = output_dir / str(paths_config["duckdb_filename"])

    for directory_path in [output_dir, tables_dir, metadata_dir, models_dir, data_output_dir]:
        directory_path.mkdir(parents=True, exist_ok=True)

    run_id = str(run_metadata["run_id"]).strip()
    if not run_id:
        raise ValueError("run_metadata.json contains an empty run_id.")

    cpu_cores = configure_runtime_cpu_cores(shared_config)
    set_deterministic_state(int(benchmark_config["seed"]))
    con = open_duckdb_connection(duckdb_path)
    ensure_pipeline_catalog(con)

    ctx = PipelineContext(
        project_root=PROJECT_ROOT,
        script_name=SCRIPT_NAME,
        notebook_name=NOTEBOOK_NAME,
        config_toml_path=config_toml_path,
        modeling_contract_toml_path=modeling_contract_toml_path,
        run_metadata_path=run_metadata_path,
        output_dir=output_dir,
        tables_dir=tables_dir,
        metadata_dir=metadata_dir,
        models_dir=models_dir,
        data_output_dir=data_output_dir,
        duckdb_path=duckdb_path,
        run_id=run_id,
        random_seed=int(benchmark_config["seed"]),
        test_size=float(benchmark_config["test_size"]),
        early_window_weeks=int(benchmark_config["early_window_weeks"]),
        main_enrollment_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
        early_window_sensitivity_weeks=resolve_early_window_sensitivity_weeks(benchmark_config),
        cpu_cores=cpu_cores,
        shared_config=shared_config,
        shared_modeling_contract=shared_modeling_contract,
        run_metadata=run_metadata,
        con=con,
    )

    print(f"- SCRIPT_NAME: {ctx.script_name}")
    print(f"- RUN_ID: {ctx.run_id}")
    print(f"- CPU_CORES: {ctx.cpu_cores}")
    print(f"- DUCKDB_PATH: {ctx.duckdb_path}")
    print(f"- BENCHMARK_HORIZONS: {list(BENCHMARK_HORIZONS)}")
    print(f"- CALIBRATION_BINS: {CALIBRATION_BINS}")
    print(f"- EARLY_WINDOW_SENSITIVITY_WEEKS: {ctx.early_window_sensitivity_weeks}")
    print_artifact("shared_config", str(ctx.config_toml_path))
    print_artifact("modeling_contract", str(ctx.modeling_contract_toml_path))
    print_artifact("run_metadata", str(ctx.run_metadata_path))

    log_stage_end("5.3.1")
    return ctx


def build_neural_treatment(ctx: PipelineContext, run_spec: WindowRunSpec) -> NeuralTreatmentState:
    # Inputs:
    # - pp_neural_hazard_ready_train and pp_neural_hazard_ready_test DuckDB tables
    # - feature_contract.static_features and feature_contract.temporal_features_discrete from benchmark_modeling_contract.toml
    # Outputs:
    # - in-memory neural treatment matrices and labels
    # - DuckDB tables table_p18_neural_not_weighted_treatment_summary, table_p18_neural_not_weighted_feature_manifest,
    #   table_p18_neural_not_weighted_canonical_alignment, table_p18_neural_not_weighted_output_feature_manifest
    # - metadata artifact metadata_p18_neural_not_weighted_treatment.json
    log_stage_start("5.3.2", "Prepare the neural discrete-time treatment")

    train_df = load_required_table(
        ctx,
        table_name=run_spec.train_table_name,
        required_columns=REQUIRED_PERSON_PERIOD_COLUMNS,
        block_number="5.3.2",
    )
    test_df = load_required_table(
        ctx,
        table_name=run_spec.test_table_name,
        required_columns=REQUIRED_PERSON_PERIOD_COLUMNS,
        block_number="5.3.2",
    )
    if run_spec.is_window_truncated_run:
        train_df = train_df.loc[train_df["week"].astype(int) <= int(run_spec.active_window_weeks)].copy()
        test_df = test_df.loc[test_df["week"].astype(int) <= int(run_spec.active_window_weeks)].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Window-truncated neural treatment produced empty train/test tables for w={int(run_spec.active_window_weeks)}."
            )

    feature_contract = ctx.shared_modeling_contract["feature_contract"]
    require_mapping_keys(
        feature_contract,
        ["static_features", "temporal_features_discrete"],
        "benchmark_modeling_contract.toml [feature_contract]",
    )
    expected_features_raw = require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
    expected_features_raw.extend(
        require_list_of_strings(
            feature_contract["temporal_features_discrete"],
            "feature_contract.temporal_features_discrete",
        )
    )
    expected_features_resolved = [FEATURE_ALIAS_MAP.get(column, column) for column in expected_features_raw]

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    missing_from_defined = [column for column in expected_features_resolved if column not in feature_columns]
    missing_in_train = [column for column in feature_columns if column not in train_df.columns]
    missing_in_test = [column for column in feature_columns if column not in test_df.columns]
    if missing_from_defined:
        raise ValueError(
            "Configured canonical features are not covered by the operational neural treatment spec after alias resolution: "
            f"{missing_from_defined}. Raw expected features were: {expected_features_raw}"
        )
    if missing_in_train or missing_in_test:
        raise ValueError(
            "Operational neural feature columns are missing from the materialized train/test tables. "
            f"Missing in train: {missing_in_train}. Missing in test: {missing_in_test}"
        )

    target_col = "event_t"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' is missing from the neural ready tables.")

    X_train_raw = train_df[feature_columns].copy()
    X_test_raw = test_df[feature_columns].copy()
    y_train = ensure_binary_target(train_df[target_col], "pp_neural_hazard_ready_train.event_t")
    y_test = ensure_binary_target(test_df[target_col], "pp_neural_hazard_ready_test.event_t")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)
    if not np.isfinite(X_train).all():
        raise ValueError("Neural training design matrix contains non-finite values after preprocessing.")
    if not np.isfinite(X_test).all():
        raise ValueError("Neural test design matrix contains non-finite values after preprocessing.")

    feature_names_out = preprocessor.get_feature_names_out().tolist()
    treatment_summary_df = pd.DataFrame(
        [
            {
                "model_family": "neural_discrete_time_survival",
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "n_input_features_raw": int(len(feature_columns)),
                "n_numeric_features": int(len(NUMERIC_FEATURES)),
                "n_categorical_features": int(len(CATEGORICAL_FEATURES)),
                "n_output_features_after_preprocessing": int(len(feature_names_out)),
                "train_event_rate": float(y_train.mean()),
                "test_event_rate": float(y_test.mean()),
            }
        ]
    )
    feature_manifest_df = pd.DataFrame(
        {
            "feature_name_operational": feature_columns,
            "feature_role": ["categorical" if column in CATEGORICAL_FEATURES else "numeric" for column in feature_columns],
            "present_in_train": [column in train_df.columns for column in feature_columns],
            "present_in_test": [column in test_df.columns for column in feature_columns],
            "covered_by_canonical_config_after_alias": [column in expected_features_resolved for column in feature_columns],
            "canonical_source_name": [
                next(
                    (
                        raw_name
                        for raw_name, resolved_name in zip(expected_features_raw, expected_features_resolved)
                        if resolved_name == column
                    ),
                    "",
                )
                for column in feature_columns
            ],
        }
    )
    canonical_alignment_df = pd.DataFrame(
        {
            "canonical_feature_raw": expected_features_raw,
            "canonical_feature_resolved": expected_features_resolved,
            "covered_by_operational_treatment_spec": [
                column in feature_columns for column in expected_features_resolved
            ],
        }
    )
    output_feature_manifest_df = pd.DataFrame({"preprocessed_feature_name": feature_names_out})

    materialize_dataframe_table(
        ctx,
        df=treatment_summary_df,
        table_name=apply_name_suffix("table_p18_neural_not_weighted_treatment_summary", run_spec.output_suffix),
        block_number="5.3.2",
        label=f"Stage 5.3.2 {apply_name_suffix('table_p18_neural_not_weighted_treatment_summary', run_spec.output_suffix)} — Neural treatment summary",
    )
    materialize_dataframe_table(
        ctx,
        df=feature_manifest_df,
        table_name=apply_name_suffix("table_p18_neural_not_weighted_feature_manifest", run_spec.output_suffix),
        block_number="5.3.2",
        label=f"Stage 5.3.2 {apply_name_suffix('table_p18_neural_not_weighted_feature_manifest', run_spec.output_suffix)} — Neural feature manifest",
    )
    materialize_dataframe_table(
        ctx,
        df=canonical_alignment_df,
        table_name=apply_name_suffix("table_p18_neural_not_weighted_canonical_alignment", run_spec.output_suffix),
        block_number="5.3.2",
        label=f"Stage 5.3.2 {apply_name_suffix('table_p18_neural_not_weighted_canonical_alignment', run_spec.output_suffix)} — Canonical feature alignment",
    )
    materialize_dataframe_table(
        ctx,
        df=output_feature_manifest_df,
        table_name=apply_name_suffix("table_p18_neural_not_weighted_output_feature_manifest", run_spec.output_suffix),
        block_number="5.3.2",
        label=f"Stage 5.3.2 {apply_name_suffix('table_p18_neural_not_weighted_output_feature_manifest', run_spec.output_suffix)} — Preprocessed feature manifest",
    )

    treatment_metadata_path = ctx.metadata_dir / append_suffix_before_extension(
        "metadata_p18_neural_not_weighted_treatment.json",
        run_spec.output_suffix,
    )
    treatment_metadata = {
        "step": "P18",
        "model_family": "neural_discrete_time_survival",
        "run_label": run_spec.run_label,
        "is_window_truncated_run": bool(run_spec.is_window_truncated_run),
        "active_window_weeks": None if run_spec.active_window_weeks is None else int(run_spec.active_window_weeks),
        "output_suffix": run_spec.output_suffix,
        "train_table": run_spec.train_table_name,
        "test_table": run_spec.test_table_name,
        "target_column": target_col,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "operational_feature_columns": feature_columns,
        "canonical_expected_features_raw": expected_features_raw,
        "canonical_expected_features_resolved": expected_features_resolved,
        "feature_alias_map": FEATURE_ALIAS_MAP,
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_output_features_after_preprocessing": int(len(feature_names_out)),
        "train_event_rate": float(y_train.mean()),
        "test_event_rate": float(y_test.mean()),
        "methodological_note": "All learned preprocessing operations were fit on training data only and then applied unchanged to test data.",
        "design_note": "Enrollment truth for neural evaluation is derived directly from the neural person-period tables to preserve identifier consistency with the person-period prediction surface.",
    }
    if run_spec.is_window_truncated_run:
        treatment_metadata["cross_arm_parity_note"] = (
            "This run truncates weekly person-period information at or before the active window and is intended for cross-arm information-parity analysis."
        )
    save_json(treatment_metadata, treatment_metadata_path)
    print_artifact("metadata_p18_neural_treatment", str(treatment_metadata_path))
    print(treatment_metadata_path.read_text(encoding="utf-8"))

    log_stage_end("5.3.2")
    return NeuralTreatmentState(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        feature_columns=feature_columns,
        expected_features_raw=expected_features_raw,
        expected_features_resolved=expected_features_resolved,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names_out=feature_names_out,
    )


def fit_neural_candidate(
    X_train_internal: np.ndarray,
    y_train_internal: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    input_dim: int,
    candidate: dict[str, Any],
    random_seed: int,
    device: torch.device,
    data_loader_workers: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_deterministic_state(random_seed)
    model = TunedHazardMLP(
        input_dim=input_dim,
        hidden_dims=list(candidate["hidden_dims"]),
        dropout=float(candidate["dropout"]),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )

    train_tensor_dataset = TensorDataset(
        torch.from_numpy(X_train_internal.astype(np.float32)),
        torch.from_numpy(y_train_internal.astype(np.float32)).view(-1, 1),
    )
    validation_tensor_dataset = TensorDataset(
        torch.from_numpy(X_validation.astype(np.float32)),
        torch.from_numpy(y_validation.astype(np.float32)).view(-1, 1),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        num_workers=int(data_loader_workers),
        persistent_workers=bool(data_loader_workers > 0),
    )
    validation_loader = DataLoader(
        validation_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=int(data_loader_workers),
        persistent_workers=bool(data_loader_workers > 0),
    )

    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                logits = model(batch_features)
                validation_losses.append(float(criterion(logits, batch_targets).item()))

        epoch_validation_loss = float(np.mean(validation_losses))
        if epoch_validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    if best_state_dict is None or best_epoch is None:
        raise RuntimeError(f"Neural tuning candidate {candidate['candidate_id']} did not produce a valid state dict.")

    candidate_result = {
        "candidate_id": int(candidate["candidate_id"]),
        "hidden_dims": str(list(candidate["hidden_dims"])),
        "dropout": float(candidate["dropout"]),
        "learning_rate": float(candidate["learning_rate"]),
        "weight_decay": float(candidate["weight_decay"]),
        "best_val_loss": best_validation_loss,
        "best_epoch": int(best_epoch),
    }
    return candidate_result, best_state_dict


def predict_probabilities_in_batches(
    model: TunedHazardMLP,
    X_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_matrix.shape[0], batch_size):
            stop = min(start + batch_size, X_matrix.shape[0])
            batch_tensor = torch.from_numpy(X_matrix[start:stop].astype(np.float32)).to(device)
            probabilities.append(torch.sigmoid(model(batch_tensor)).cpu().numpy().reshape(-1))
    predictions = np.concatenate(probabilities, axis=0)
    if predictions.shape[0] != X_matrix.shape[0]:
        raise ValueError("Neural batch prediction returned a length inconsistent with the design matrix.")
    return predictions


def tune_and_evaluate_neural_model(ctx: PipelineContext, treatment: NeuralTreatmentState, run_spec: WindowRunSpec) -> None:
    # Inputs:
    # - in-memory neural treatment matrices and labels from block 5.3.2
    # - benchmark tuning configuration and deterministic random seed
    # Outputs:
    # - DuckDB tuning, prediction, metric, calibration, hazard, and audit tables
    # - neural_discrete_time_survival_not_weighted_tuned.pt
    # - neural_discrete_time_not_weighted_preprocessor.joblib
    # - neural_not_weighted_tuned_model_config.json
    log_stage_start("5.3.3", "Tune and evaluate the neural discrete-time hazard model")

    enrollment_groups = treatment.train_df["enrollment_id"].astype(str).to_numpy()
    if len(np.unique(enrollment_groups)) < 2:
        raise ValueError("The neural training table must contain at least two enrollment groups for validation splitting.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=ctx.random_seed)
    subtrain_index, validation_index = next(
        splitter.split(treatment.X_train, treatment.y_train.to_numpy(), groups=enrollment_groups)
    )
    X_subtrain = treatment.X_train[subtrain_index]
    X_validation = treatment.X_train[validation_index]
    y_subtrain = treatment.y_train.iloc[subtrain_index].to_numpy(dtype=np.float32)
    y_validation = treatment.y_train.iloc[validation_index].to_numpy(dtype=np.float32)
    ensure_binary_target(pd.Series(y_validation), "neural validation target")
    data_loader_workers = max(0, min(int(ctx.cpu_cores) - 1, MAX_DATA_LOADER_WORKERS))
    print(f"- DATA_LOADER_WORKERS: {data_loader_workers}")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    candidate_rows: list[dict[str, Any]] = []
    best_candidate_config: dict[str, Any] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_loss: float | None = None

    for candidate in NEURAL_GRID:
        candidate_row, candidate_state_dict = fit_neural_candidate(
            X_train_internal=X_subtrain,
            y_train_internal=y_subtrain,
            X_validation=X_validation,
            y_validation=y_validation,
            input_dim=int(treatment.X_train.shape[1]),
            candidate=candidate,
            random_seed=ctx.random_seed,
            device=device,
            data_loader_workers=data_loader_workers,
        )
        candidate_rows.append(candidate_row)
        if best_validation_loss is None or float(candidate_row["best_val_loss"]) < best_validation_loss:
            best_validation_loss = float(candidate_row["best_val_loss"])
            best_candidate_config = dict(candidate)
            best_candidate_config["best_val_loss"] = float(candidate_row["best_val_loss"])
            best_candidate_config["best_epoch"] = int(candidate_row["best_epoch"])
            best_state_dict = candidate_state_dict

    if best_candidate_config is None or best_state_dict is None:
        raise RuntimeError("Neural tuning failed to produce a valid best candidate.")

    tuning_results_df = pd.DataFrame(candidate_rows).sort_values(
        ["best_val_loss", "candidate_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_model = TunedHazardMLP(
        input_dim=int(treatment.X_train.shape[1]),
        hidden_dims=list(best_candidate_config["hidden_dims"]),
        dropout=float(best_candidate_config["dropout"]),
    ).to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    test_hazard = np.clip(
        predict_probabilities_in_batches(best_model, treatment.X_test, batch_size=4096, device=device),
        1e-8,
        1.0 - 1e-8,
    )
    if not np.isfinite(test_hazard).all():
        raise ValueError("The tuned neural model produced non-finite hazard predictions.")

    test_predictions_df = treatment.test_df.copy().sort_values(["enrollment_id", "week"]).reset_index(drop=True)
    test_predictions_df["pred_hazard"] = test_hazard
    test_predictions_df["pred_survival"] = test_predictions_df.groupby("enrollment_id")["pred_hazard"].transform(
        lambda series: (1.0 - series).cumprod()
    )
    test_predictions_df["pred_risk"] = 1.0 - test_predictions_df["pred_survival"]
    if not np.isfinite(test_predictions_df[["pred_hazard", "pred_survival", "pred_risk"]].to_numpy()).all():
        raise ValueError("The tuned neural model produced non-finite survival or risk predictions.")

    truth_train_df = build_truth_by_enrollment(treatment.train_df, run_spec.train_table_name)
    truth_test_df = build_truth_by_enrollment(treatment.test_df, run_spec.test_table_name)
    durations_test = truth_test_df["duration"].astype(int).to_numpy()
    events_test = truth_test_df["event"].astype(int).to_numpy()
    y_train_surv = Surv.from_arrays(
        event=truth_train_df["event"].astype(bool).to_numpy(),
        time=truth_train_df["duration"].astype(float).to_numpy(),
    )
    y_test_surv = Surv.from_arrays(
        event=truth_test_df["event"].astype(bool).to_numpy(),
        time=truth_test_df["duration"].astype(float).to_numpy(),
    )

    survival_wide_df = (
        test_predictions_df[["enrollment_id", "week", "pred_survival"]]
        .drop_duplicates(subset=["enrollment_id", "week"])
        .pivot(index="week", columns="enrollment_id", values="pred_survival")
        .sort_index()
    )
    max_evaluation_week = max(int(test_predictions_df["week"].max()), int(max(BENCHMARK_HORIZONS)))
    full_week_index = pd.Index(np.arange(0, max_evaluation_week + 1, dtype=int), name="week")
    survival_wide_df = survival_wide_df.reindex(full_week_index).ffill().fillna(1.0)
    ordered_test_enrollment_ids = truth_test_df["enrollment_id"].tolist()
    survival_wide_df = survival_wide_df.reindex(columns=ordered_test_enrollment_ids)
    if survival_wide_df.isna().any().any():
        raise ValueError("The neural survival surface contains missing values after enrollment alignment.")

    eval_surv = EvalSurv(
        surv=survival_wide_df,
        durations=durations_test,
        events=events_test,
        censor_surv="km",
    )
    max_supported_brier_week = int(np.max(durations_test)) - 1
    if max_supported_brier_week < 1:
        raise ValueError("No admissible positive Brier evaluation times are available for the truncated test follow-up.")
    brier_time_grid = np.arange(1, min(max(BENCHMARK_HORIZONS), max_supported_brier_week) + 1, dtype=int)
    survival_estimate_matrix = survival_wide_df.loc[brier_time_grid, ordered_test_enrollment_ids].to_numpy(dtype=float).T
    brier_times, brier_scores = sksurv_brier_score(
        y_train_surv,
        y_test_surv,
        survival_estimate_matrix,
        brier_time_grid.astype(float),
    )
    if len(brier_times) < 2:
        integrated_brier = float(brier_scores.astype(float)[0])
        ibs_notes = "sksurv_brier_score_single_admissible_time"
    else:
        integrated_brier = float(
            np.trapezoid(brier_scores.astype(float), brier_times.astype(float))
            / (float(brier_times[-1]) - float(brier_times[0]))
        )
        ibs_notes = "sksurv_integrated_brier_score"

    primary_metrics_df = pd.DataFrame(
        [
            {
                "metric_name": "ibs",
                "metric_category": "primary",
                "metric_value": integrated_brier,
                "notes": ibs_notes,
            },
            {
                "metric_name": "c_index",
                "metric_category": "co_primary",
                "metric_value": float(eval_surv.concordance_td()),
                "notes": "pycox_concordance_td",
            },
        ]
    )
    brier_by_horizon_df = pd.DataFrame(
        {
            "horizon_week": brier_times.astype(int),
            "metric_name": ["brier_at_horizon"] * len(brier_times),
            "metric_category": ["primary"] * len(brier_times),
            "metric_value": brier_scores.astype(float),
            "notes": ["sksurv_brier_score"] * len(brier_times),
        }
    ).loc[lambda df: df["horizon_week"].isin(BENCHMARK_HORIZONS)].reset_index(drop=True)

    support_rows: list[dict[str, Any]] = []
    risk_auc_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    predicted_vs_observed_rows: list[dict[str, Any]] = []
    for horizon_week in BENCHMARK_HORIZONS:
        pred_survival_h = get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_survival").rename(
            columns={"pred_survival": "pred_survival_h"}
        )
        pred_risk_h = get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_risk").rename(
            columns={"pred_risk": "pred_risk_h"}
        )
        evaluable_df = truth_test_df.merge(pred_survival_h, on="enrollment_id", how="left").merge(
            pred_risk_h,
            on="enrollment_id",
            how="left",
        )
        evaluable_df["is_evaluable_at_h"] = (
            ((evaluable_df["event"] == 1) & (evaluable_df["duration"] <= horizon_week))
            | (evaluable_df["duration"] >= horizon_week)
        ).astype(int)
        evaluable_df = evaluable_df.loc[evaluable_df["is_evaluable_at_h"] == 1].copy()
        if evaluable_df.empty:
            raise ValueError(f"No evaluable enrollments are available at horizon {horizon_week}.")
        evaluable_df["observed_event_by_h"] = (
            (evaluable_df["event"] == 1) & (evaluable_df["duration"] <= horizon_week)
        ).astype(int)
        evaluable_df["observed_survival_by_h"] = 1 - evaluable_df["observed_event_by_h"]
        observed_event_nunique = int(evaluable_df["observed_event_by_h"].nunique())

        support_rows.append(
            {
                "horizon_week": int(horizon_week),
                "n_evaluable_enrollments": int(evaluable_df.shape[0]),
                "n_events_by_horizon": int(evaluable_df["observed_event_by_h"].sum()),
                "event_rate_by_horizon": float(evaluable_df["observed_event_by_h"].mean()),
            }
        )
        risk_auc_rows.append(
            {
                "horizon_week": int(horizon_week),
                "metric_name": "risk_auc_at_horizon",
                "metric_category": "secondary",
                "metric_value": float(roc_auc_score(evaluable_df["observed_event_by_h"], evaluable_df["pred_risk_h"])) if observed_event_nunique >= 2 else np.nan,
                "notes": "roc_auc_on_evaluable_subset" if observed_event_nunique >= 2 else "roc_auc_undefined_single_observed_class",
            }
        )
        calibration_table, predicted_vs_observed_row = build_ipcw_calibration_artifacts(
            y_train_surv,
            evaluable_df[["enrollment_id", "event", "duration", "pred_risk_h"]],
            horizon_week,
            CALIBRATION_BINS,
        )
        predicted_vs_observed_rows.append(predicted_vs_observed_row)
        calibration_rows.extend(calibration_table.to_dict(orient="records"))

    support_by_horizon_df = pd.DataFrame(support_rows)
    risk_auc_by_horizon_df = pd.DataFrame(risk_auc_rows)
    predicted_vs_observed_survival_df = pd.DataFrame(predicted_vs_observed_rows)
    calibration_bins_df = pd.DataFrame(calibration_rows)
    calibration_summary_df = summarize_calibration_by_horizon(calibration_bins_df)

    max_ipcw_time = float(truth_train_df.loc[truth_train_df["event"] == 0, "duration"].max())
    if not np.isfinite(max_ipcw_time):
        raise ValueError("Could not determine a valid censoring-support horizon for the neural model.")

    td_auc_rows: list[dict[str, Any]] = []
    td_auc_audit_rows: list[dict[str, Any]] = []
    for horizon_week in BENCHMARK_HORIZONS:
        pred_risk_h = get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_risk")
        supported_df = truth_test_df.merge(pred_risk_h, on="enrollment_id", how="left")
        supported_df = supported_df.loc[
            (supported_df["duration"] < max_ipcw_time) & supported_df["pred_risk"].notna()
        ].copy()
        if supported_df.empty:
            horizon_auc = np.nan
            td_auc_notes = "ipcw_auc_undefined_no_supported_rows"
        elif supported_df["event"].nunique() < 2:
            horizon_auc = np.nan
            td_auc_notes = "ipcw_auc_undefined_single_observed_class"
        else:
            supported_test_surv = Surv.from_arrays(
                event=supported_df["event"].astype(bool).to_numpy(),
                time=supported_df["duration"].astype(float).to_numpy(),
            )
            horizon_auc = compute_ipcw_time_dependent_auc(
                y_train_surv,
                supported_test_surv,
                supported_df["pred_risk"].to_numpy(),
                float(horizon_week),
            )
            td_auc_notes = (
                f"ipcw_supported_subset(duration < {max_ipcw_time:.0f}); "
                f"retained={supported_df.shape[0]}/{truth_test_df.shape[0]}; "
                f"auc={horizon_auc:.6f}; local_ipcw_auc"
            )
        td_auc_rows.append(
            {
                "horizon_week": int(horizon_week),
                "metric_name": "time_dependent_auc",
                "metric_category": "secondary",
                "metric_value": horizon_auc,
                "notes": td_auc_notes,
            }
        )
        td_auc_audit_rows.append(
            {
                "horizon_week": int(horizon_week),
                "max_ipcw_time": max_ipcw_time,
                "n_total_test_enrollments": int(truth_test_df.shape[0]),
                "n_supported_test_enrollments": int(supported_df.shape[0]),
                "mean_auc": horizon_auc,
                "notes": td_auc_notes,
            }
        )

    secondary_metrics_df = pd.concat(
        [risk_auc_by_horizon_df, pd.DataFrame(td_auc_rows)],
        ignore_index=True,
    ).sort_values(["horizon_week", "metric_name"]).reset_index(drop=True)
    td_auc_support_audit_df = pd.DataFrame(td_auc_audit_rows)

    row_diagnostics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "row_level_roc_auc": float(roc_auc_score(treatment.y_test, test_hazard)),
                "row_level_pr_auc": float(average_precision_score(treatment.y_test, test_hazard)),
                "row_level_log_loss": float(log_loss(treatment.y_test, test_hazard, labels=[0, 1])),
                "row_level_brier": float(brier_score_loss(treatment.y_test, test_hazard)),
            }
        ]
    )
    hazard_audit_summary_df = pd.DataFrame(
        [
            {
                "mean_pred_hazard_test": float(np.mean(test_hazard)),
                "median_pred_hazard_test": float(np.median(test_hazard)),
                "p90_pred_hazard_test": float(np.quantile(test_hazard, 0.90)),
                "p99_pred_hazard_test": float(np.quantile(test_hazard, 0.99)),
                "min_pred_hazard_test": float(np.min(test_hazard)),
                "max_pred_hazard_test": float(np.max(test_hazard)),
            }
        ]
    )
    hazard_by_week_df = (
        test_predictions_df.groupby("week", as_index=False)
        .agg(
            mean_pred_hazard=("pred_hazard", "mean"),
            median_pred_hazard=("pred_hazard", "median"),
            mean_pred_survival=("pred_survival", "mean"),
            mean_pred_risk=("pred_risk", "mean"),
            n_rows=("enrollment_id", "count"),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )

    history_df = pd.DataFrame(
        [
            {
                "candidate_id": int(best_candidate_config["candidate_id"]),
                "hidden_dims": str(list(best_candidate_config["hidden_dims"])),
                "dropout": float(best_candidate_config["dropout"]),
                "learning_rate": float(best_candidate_config["learning_rate"]),
                "weight_decay": float(best_candidate_config["weight_decay"]),
                "best_val_loss": float(best_candidate_config["best_val_loss"]),
                "best_epoch": int(best_candidate_config["best_epoch"]),
            }
        ]
    )

    model_path = ctx.models_dir / append_suffix_before_extension(
        "neural_discrete_time_survival_not_weighted_tuned.pt",
        run_spec.output_suffix,
    )
    preprocessor_path = ctx.models_dir / append_suffix_before_extension(
        "neural_discrete_time_not_weighted_preprocessor.joblib",
        run_spec.output_suffix,
    )
    config_path = ctx.metadata_dir / append_suffix_before_extension(
        "neural_not_weighted_tuned_model_config.json",
        run_spec.output_suffix,
    )
    torch.save(best_model.state_dict(), model_path)
    joblib.dump(treatment.preprocessor, preprocessor_path)

    materialize_dataframe_table(
        ctx,
        df=tuning_results_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuning_results", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuning_results', run_spec.output_suffix)} — Neural tuning results",
    )
    materialize_dataframe_table(
        ctx,
        df=history_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_training_history", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_training_history', run_spec.output_suffix)} — Neural best-candidate training summary",
    )
    materialize_dataframe_table(
        ctx,
        df=test_predictions_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_test_predictions", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_test_predictions', run_spec.output_suffix)} — Neural test predictions",
    )
    materialize_dataframe_table(
        ctx,
        df=primary_metrics_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_primary_metrics", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_primary_metrics', run_spec.output_suffix)} — Neural primary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=brier_by_horizon_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_brier_by_horizon", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_brier_by_horizon', run_spec.output_suffix)} — Neural Brier scores by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=secondary_metrics_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_secondary_metrics", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_secondary_metrics', run_spec.output_suffix)} — Neural secondary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=td_auc_support_audit_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_td_auc_support_audit", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_td_auc_support_audit', run_spec.output_suffix)} — Neural IPCW support audit",
    )
    materialize_dataframe_table(
        ctx,
        df=row_diagnostics_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_row_diagnostics", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_row_diagnostics', run_spec.output_suffix)} — Neural row-level diagnostics",
    )
    materialize_dataframe_table(
        ctx,
        df=support_by_horizon_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_support_by_horizon", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_support_by_horizon', run_spec.output_suffix)} — Neural support by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_summary_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_calibration_summary", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_calibration_summary', run_spec.output_suffix)} — Neural calibration summary",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_bins_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_calibration_bins_by_horizon", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_calibration_bins_by_horizon', run_spec.output_suffix)} — Neural calibration bins",
    )
    materialize_dataframe_table(
        ctx,
        df=predicted_vs_observed_survival_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_predicted_vs_observed_survival", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_predicted_vs_observed_survival', run_spec.output_suffix)} — Neural predicted versus observed survival",
    )
    materialize_dataframe_table(
        ctx,
        df=hazard_audit_summary_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_hazard_audit_summary", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_hazard_audit_summary', run_spec.output_suffix)} — Neural hazard audit summary",
    )
    materialize_dataframe_table(
        ctx,
        df=hazard_by_week_df,
        table_name=apply_name_suffix("table_neural_not_weighted_tuned_hazard_by_week", run_spec.output_suffix),
        block_number="5.3.3",
        label=f"Stage 5.3.3 {apply_name_suffix('table_neural_not_weighted_tuned_hazard_by_week', run_spec.output_suffix)} — Neural hazard by week",
    )

    model_config = {
        "model_name": MODEL_NAME,
        "search_strategy": "controlled_grid_search",
        "selection_criterion": "lowest_validation_loss",
        "validation_split": {
            "method": "GroupShuffleSplit",
            "unit": "enrollment_id",
            "test_size": VALIDATION_FRACTION,
            "random_seed": ctx.random_seed,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss()",
        "random_state": ctx.random_seed,
        "run_label": run_spec.run_label,
        "is_window_truncated_run": bool(run_spec.is_window_truncated_run),
        "active_window_weeks": None if run_spec.active_window_weeks is None else int(run_spec.active_window_weeks),
        "output_suffix": run_spec.output_suffix,
        "device": device_name,
        "benchmark_horizons": list(BENCHMARK_HORIZONS),
        "calibration_bins": CALIBRATION_BINS,
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
        "calibration_observed_risk_method": CALIBRATION_OBSERVED_RISK_METHOD,
        "best_candidate": {
            "candidate_id": int(best_candidate_config["candidate_id"]),
            "hidden_dims": list(best_candidate_config["hidden_dims"]),
            "dropout": float(best_candidate_config["dropout"]),
            "learning_rate": float(best_candidate_config["learning_rate"]),
            "weight_decay": float(best_candidate_config["weight_decay"]),
            "best_val_loss": float(best_candidate_config["best_val_loss"]),
            "best_epoch": int(best_candidate_config["best_epoch"]),
        },
        "preprocessing": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_encoder": "OneHotEncoder(handle_unknown=ignore, sparse_output=False)",
            "numeric_scaler": "StandardScaler",
        },
        "weighting_strategy": "none; this official D5.3 variant uses the same grouped validation protocol and evaluation contract without positive-class loss weighting.",
        "design_note": "Enrollment truth and evaluation alignment are derived directly from the neural person-period tables so the Item 6 comparison changes only the fitting loss in the weighted sensitivity counterpart, not the evaluation contract.",
    }
    save_json(model_config, config_path)
    print_artifact("neural_model", str(model_path))
    print_artifact("neural_preprocessor", str(preprocessor_path))
    print_artifact("neural_tuned_model_config", str(config_path))
    print(config_path.read_text(encoding="utf-8"))
    print(history_df.to_string(index=False))

    log_stage_end("5.3.3")


def close_context(ctx: PipelineContext) -> None:
    from util import close_duckdb_connection

    # Inputs:
    # - active DuckDB connection
    # Outputs:
    # - clean DuckDB shutdown
    log_stage_start("5.3.4", "Close the DuckDB runtime cleanly")
    ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
    log_stage_end("5.3.4")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    ctx = initialize_context()
    try:
        run_specs = resolve_dynamic_window_execution_plan(ctx)
        plan_text = ", ".join(
            spec.run_label if spec.active_window_weeks is None else f"{spec.run_label}(suffix={spec.output_suffix})"
            for spec in run_specs
        )
        print(f"[RUN_PLAN] neural_dynamic_execution={plan_text}")
        for run_spec in run_specs:
            if run_spec.is_window_truncated_run:
                print(f"[RUN] dynamic_window_weeks={int(run_spec.active_window_weeks)} output_suffix={run_spec.output_suffix}")
            else:
                print("[RUN] official_full_information output_suffix=''")
            treatment_state = build_neural_treatment(ctx, run_spec)
            tune_and_evaluate_neural_model(ctx, treatment_state, run_spec)
    finally:
        close_context(ctx)


if __name__ == "__main__":
    main()