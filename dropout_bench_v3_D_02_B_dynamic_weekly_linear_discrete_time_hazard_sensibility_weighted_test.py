from __future__ import annotations

"""
Production weighted linear discrete-time hazard benchmark module for Item 6 sensitivity.

What this file does:
- prepares the linear discrete-time hazard treatment for the person-period benchmark arm
- tunes and fits the linear survival model under the official benchmark protocol
- evaluates the tuned model with survival, calibration, and row-level diagnostics
- persists the trained model, fitted preprocessor, DuckDB audit tables, and JSON metadata artifacts
- can optionally rerun the dynamic arm under an explicit information limit up to each window `w` from the centralized modeling contract

Main processing purpose:
- materialize the full linear benchmark arm deterministically from DuckDB-ready tables without notebook-specific runtime state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_p16_linear_weighted_treatment_audit, table_p16_linear_weighted_feature_manifest,
  table_p16_linear_weighted_canonical_alignment, table_p16_linear_weighted_output_feature_manifest,
  table_linear_weighted_tuning_results, table_linear_weighted_tuned_test_predictions,
  table_linear_weighted_tuned_primary_metrics, table_linear_weighted_tuned_brier_by_horizon,
  table_linear_weighted_tuned_secondary_metrics, table_linear_weighted_tuned_td_auc_support_audit,
  table_linear_weighted_tuned_row_diagnostics, table_linear_weighted_tuned_support_by_horizon,
  table_linear_weighted_tuned_calibration_summary, table_linear_weighted_tuned_calibration_bins_by_horizon,
  table_linear_weighted_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/metadata_p16_linear_weighted_treatment.json
- outputs_benchmark_survival/metadata/linear_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/linear_discrete_time_hazard_weighted_tuned.joblib
- outputs_benchmark_survival/models/linear_discrete_time_weighted_preprocessor.joblib

Main DuckDB tables used as inputs:
- pp_linear_hazard_ready_train
- pp_linear_hazard_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_p16_linear_weighted_treatment_audit
- table_p16_linear_weighted_feature_manifest
- table_p16_linear_weighted_canonical_alignment
- table_p16_linear_weighted_output_feature_manifest
- table_linear_weighted_tuning_results
- table_linear_weighted_tuned_test_predictions
- table_linear_weighted_tuned_primary_metrics
- table_linear_weighted_tuned_brier_by_horizon
- table_linear_weighted_tuned_secondary_metrics
- table_linear_weighted_tuned_td_auc_support_audit
- table_linear_weighted_tuned_row_diagnostics
- table_linear_weighted_tuned_support_by_horizon
- table_linear_weighted_tuned_calibration_summary
- table_linear_weighted_tuned_calibration_bins_by_horizon
- table_linear_weighted_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/metadata_p16_linear_weighted_treatment.json
- outputs_benchmark_survival/metadata/linear_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/linear_discrete_time_hazard_weighted_tuned.joblib
- outputs_benchmark_survival/models/linear_discrete_time_weighted_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, or CSV-based workflows are permitted

Execution modes:
- default execution iterates over the full contract-driven window grid from `benchmark.early_window_sensitivity_weeks`
"""

import json
import inspect
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv, check_y_survival

from dropout_bench_v3_D_00_common import (
    append_suffix_before_extension,
    apply_name_suffix,
    resolve_early_window_sensitivity_weeks,
)


if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    import tomli as toml_reader


STAGE_PREFIX = "5.2"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_NAME = "dropout_bench_v3_D_5_2_weighted.ipynb"
PREVIEW_ROWS = 20
BENCHMARK_HORIZONS = (10, 20, 30)
CALIBRATION_BINS = 10
REQUIRED_SHARED_PATH_KEYS = [
    "output_dir",
    "tables_subdir",
    "metadata_subdir",
    "models_subdir",
    "data_output_subdir",
    "duckdb_filename",
]
REQUIRED_MODELING_KEYS = ["benchmark", "modeling", "feature_contract"]
REQUIRED_INPUT_TABLES = ["pp_linear_hazard_ready_train", "pp_linear_hazard_ready_test"]
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
MODEL_NAME = "linear_discrete_time_hazard_weighted_tuned"
VALIDATION_TEST_SIZE = 0.20
LINEAR_TUNING_GRID = (
    {"candidate_id": 1, "penalty": "l1", "C": 0.01},
    {"candidate_id": 2, "penalty": "l1", "C": 0.1},
    {"candidate_id": 3, "penalty": "l1", "C": 1.0},
    {"candidate_id": 4, "penalty": "l1", "C": 10.0},
    {"candidate_id": 5, "penalty": "l2", "C": 0.01},
    {"candidate_id": 6, "penalty": "l2", "C": 0.1},
    {"candidate_id": 7, "penalty": "l2", "C": 1.0},
    {"candidate_id": 8, "penalty": "l2", "C": 10.0},
)


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
    cpu_cores: int
    random_seed: int
    test_size: float
    early_window_weeks: int
    main_enrollment_window_weeks: int
    early_window_sensitivity_weeks: list[int]
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
class LinearTreatmentState:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    feature_columns: list[str]
    expected_features_raw: list[str]
    expected_features_resolved: list[str]
    preprocessor: ColumnTransformer
    X_train: Any
    X_test: Any
    y_train: pd.Series
    y_test: pd.Series
    feature_names_out: list[str]


def build_logistic_regression_candidate(candidate: dict[str, Any], random_seed: int) -> LogisticRegression:
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    common_kwargs = {
        "C": float(candidate["C"]),
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": random_seed,
    }
    if penalty_default == "deprecated":
        common_kwargs["l1_ratio"] = 1.0 if str(candidate["penalty"]) == "l1" else 0.0
    else:
        common_kwargs["penalty"] = str(candidate["penalty"])
    return LogisticRegression(**common_kwargs)


def evaluate_linear_candidate(
    candidate: dict[str, Any],
    random_seed: int,
    X_subtrain: Any,
    y_subtrain: np.ndarray,
    X_validation: Any,
    y_validation: np.ndarray,
) -> dict[str, Any]:
    model = build_logistic_regression_candidate(candidate, random_seed=random_seed)
    model.fit(X_subtrain, y_subtrain)
    validation_pred = np.clip(model.predict_proba(X_validation)[:, 1], 1e-8, 1.0 - 1e-8)
    validation_log_loss = float(log_loss(y_validation, validation_pred, labels=[0, 1]))
    validation_brier = float(brier_score_loss(y_validation, validation_pred))
    validation_roc_auc = float(roc_auc_score(y_validation, validation_pred))
    validation_pr_auc = float(average_precision_score(y_validation, validation_pred))
    return {
        "candidate_id": int(candidate["candidate_id"]),
        "penalty": str(candidate["penalty"]),
        "l1_ratio": 1.0 if str(candidate["penalty"]) == "l1" else 0.0,
        "C": float(candidate["C"]),
        "val_log_loss": validation_log_loss,
        "val_brier": validation_brier,
        "val_roc_auc": validation_roc_auc,
        "val_pr_auc": validation_pr_auc,
    }


def log_stage_start(block_number: str, title: str) -> None:
    print(f"[START] {STAGE_PREFIX} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("# ==============================================================", flush=True)
    print(f"# {block_number} - {title}", flush=True)
    print("# ==============================================================", flush=True)


def log_stage_end(block_number: str) -> None:
    print(f"[END] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


def log_progress(block_number: str, message: str) -> None:
    print(f"[PROGRESS] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}", flush=True)


def print_artifact(label: str, location: str) -> None:
    print(f"ARTIFACT | {label} | {location}", flush=True)


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
    print(f"[{label}]", flush=True)
    print(f"table_name={table_name}", flush=True)
    print(f"rows={row_count}, cols={column_count}", flush=True)
    if preview_df.empty:
        print("[empty table]", flush=True)
    else:
        print(preview_df.to_string(index=False), flush=True)


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
            train_table_name="pp_linear_hazard_ready_train",
            test_table_name="pp_linear_hazard_ready_test",
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

    temp_view_name = "__stage_d_5_2_materialize_df__"
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


def load_required_table(ctx: PipelineContext, table_name: str, required_columns: list[str]) -> pd.DataFrame:
    require_tables(ctx.con, [table_name], block_number="5.2.2")
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


def initialize_context() -> PipelineContext:
    from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores, resolve_runtime_tuning_parallel_backend
    from util import ensure_pipeline_catalog, open_duckdb_connection

    # Inputs:
    # - benchmark_shared_config.toml
    # - benchmark_modeling_contract.toml
    # - outputs_benchmark_survival/metadata/run_metadata.json
    # Outputs:
    # - validated runtime context in memory
    # - active DuckDB connection
    # - ensured pipeline catalog objects
    log_stage_start("5.2.1", "Lightweight runtime bootstrap")
    log_progress("5.2.1", "Resolving config, contract, and metadata paths")

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
    log_progress("5.2.1", "Configuration artifacts loaded into memory")

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
    tuning_parallel_backend = resolve_runtime_tuning_parallel_backend(shared_config)
    log_progress("5.2.1", f"Runtime CPU configuration resolved to {cpu_cores} cores")
    log_progress("5.2.1", f"Runtime tuning backend resolved to '{tuning_parallel_backend}'")

    con = open_duckdb_connection(duckdb_path)
    ensure_pipeline_catalog(con)
    log_progress("5.2.1", "DuckDB connection opened and pipeline catalog ensured")

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
        cpu_cores=cpu_cores,
        random_seed=int(benchmark_config["seed"]),
        test_size=float(benchmark_config["test_size"]),
        early_window_weeks=int(benchmark_config["early_window_weeks"]),
        main_enrollment_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
        early_window_sensitivity_weeks=resolve_early_window_sensitivity_weeks(benchmark_config),
        shared_config=shared_config,
        shared_modeling_contract=shared_modeling_contract,
        run_metadata=run_metadata,
        con=con,
    )

    print(f"- SCRIPT_NAME: {ctx.script_name}", flush=True)
    print(f"- RUN_ID: {ctx.run_id}", flush=True)
    print(f"- CPU_CORES: {ctx.cpu_cores}", flush=True)
    print(f"- DUCKDB_PATH: {ctx.duckdb_path}", flush=True)
    print(f"- BENCHMARK_HORIZONS: {list(BENCHMARK_HORIZONS)}", flush=True)
    print(f"- CALIBRATION_BINS: {CALIBRATION_BINS}", flush=True)
    print(f"- EARLY_WINDOW_SENSITIVITY_WEEKS: {ctx.early_window_sensitivity_weeks}", flush=True)
    print_artifact("shared_config", str(ctx.config_toml_path))
    print_artifact("modeling_contract", str(ctx.modeling_contract_toml_path))
    print_artifact("run_metadata", str(ctx.run_metadata_path))
    log_stage_end("5.2.1")
    return ctx


def build_linear_treatment(ctx: PipelineContext, run_spec: WindowRunSpec) -> LinearTreatmentState:
    # Inputs:
    # - benchmark_modeling_contract.toml feature contract
    # - DuckDB tables pp_linear_hazard_ready_train and pp_linear_hazard_ready_test
    # Outputs:
    # - validated train and test dataframes in memory
    # - sparse fitted preprocessing object in memory
    # - transformed train and test matrices in memory
    # - DuckDB audit tables and metadata_p16_linear_weighted_treatment.json
    log_stage_start("5.2.2", "Prepare the linear discrete-time treatment")
    log_progress("5.2.2", "Resolving feature contract and expected canonical features")

    feature_contract = ctx.shared_modeling_contract["feature_contract"]
    static_features = require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
    temporal_features_discrete = require_list_of_strings(
        feature_contract["temporal_features_discrete"],
        "feature_contract.temporal_features_discrete",
    )
    expected_features_raw = static_features + temporal_features_discrete
    expected_features_resolved = [FEATURE_ALIAS_MAP.get(feature_name, feature_name) for feature_name in expected_features_raw]

    train_df = load_required_table(ctx, run_spec.train_table_name, REQUIRED_PERSON_PERIOD_COLUMNS)
    test_df = load_required_table(ctx, run_spec.test_table_name, REQUIRED_PERSON_PERIOD_COLUMNS)
    if run_spec.is_window_truncated_run:
        train_df = train_df.loc[train_df["week"].astype(int) <= int(run_spec.active_window_weeks)].copy()
        test_df = test_df.loc[test_df["week"].astype(int) <= int(run_spec.active_window_weeks)].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Window-truncated linear treatment produced empty train/test tables for w={int(run_spec.active_window_weeks)}."
            )
    log_progress("5.2.2", f"Loaded training and test tables with shapes train={train_df.shape}, test={test_df.shape}")

    missing_from_treatment_spec = [
        feature_name for feature_name in expected_features_resolved if feature_name not in CATEGORICAL_FEATURES + NUMERIC_FEATURES
    ]
    if missing_from_treatment_spec:
        raise ValueError(
            "Configured canonical features are not covered by the operational linear treatment spec after alias resolution: "
            + ", ".join(missing_from_treatment_spec)
        )

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    missing_in_train = [column for column in feature_columns if column not in train_df.columns]
    missing_in_test = [column for column in feature_columns if column not in test_df.columns]
    if missing_in_train or missing_in_test:
        raise KeyError(
            "Operational linear feature columns are missing from the materialized train/test tables. "
            f"Missing in train: {missing_in_train}. Missing in test: {missing_in_test}"
        )

    target_col = "event_t"
    y_train = ensure_binary_target(train_df[target_col], "pp_linear_hazard_ready_train.event_t")
    y_test = ensure_binary_target(test_df[target_col], "pp_linear_hazard_ready_test.event_t")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MaxAbsScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    log_progress("5.2.2", "Fitting sparse preprocessor on training data")
    X_train = preprocessor.fit_transform(train_df[feature_columns].copy())
    log_progress("5.2.2", "Transforming test data with fitted preprocessor")
    X_test = preprocessor.transform(test_df[feature_columns].copy())
    feature_names_out = preprocessor.get_feature_names_out().tolist()
    if len(feature_names_out) == 0:
        raise ValueError("Linear preprocessing produced zero output features.")

    preproc_audit_df = pd.DataFrame(
        {
            "component": [
                "target_column",
                "n_train_rows",
                "n_test_rows",
                "n_operational_features",
                "n_numeric_features",
                "n_categorical_features",
                "n_output_features_after_preprocessing",
                "train_event_rate",
                "test_event_rate",
            ],
            "value": [
                target_col,
                int(len(train_df)),
                int(len(test_df)),
                int(len(feature_columns)),
                int(len(NUMERIC_FEATURES)),
                int(len(CATEGORICAL_FEATURES)),
                int(len(feature_names_out)),
                float(y_train.mean()),
                float(y_test.mean()),
            ],
        }
    )
    feature_manifest_df = pd.DataFrame(
        {
            "feature_name_operational": feature_columns,
            "feature_role": ["categorical" if feature_name in CATEGORICAL_FEATURES else "numeric" for feature_name in feature_columns],
            "present_in_train": [feature_name in train_df.columns for feature_name in feature_columns],
            "present_in_test": [feature_name in test_df.columns for feature_name in feature_columns],
            "covered_by_canonical_config_after_alias": [feature_name in expected_features_resolved for feature_name in feature_columns],
            "canonical_source_name": [
                next(
                    (
                        raw_feature
                        for raw_feature, resolved_feature in zip(expected_features_raw, expected_features_resolved)
                        if resolved_feature == feature_name
                    ),
                    "",
                )
                for feature_name in feature_columns
            ],
        }
    )
    canonical_alignment_df = pd.DataFrame(
        {
            "canonical_feature_raw": expected_features_raw,
            "canonical_feature_resolved": expected_features_resolved,
            "covered_by_operational_treatment_spec": [
                feature_name in feature_columns for feature_name in expected_features_resolved
            ],
        }
    )
    output_feature_manifest_df = pd.DataFrame({"preprocessed_feature_name": feature_names_out})
    log_progress("5.2.2", f"Preprocessing finished with {len(feature_names_out)} output features")

    log_progress("5.2.2", "Materializing treatment audit tables")
    materialize_dataframe_table(
        ctx,
        df=preproc_audit_df,
        table_name=apply_name_suffix("table_p16_linear_weighted_treatment_audit", run_spec.output_suffix),
        block_number="5.2.2",
        label=f"Stage 5.2.2 {apply_name_suffix('table_p16_linear_weighted_treatment_audit', run_spec.output_suffix)} — Linear treatment audit",
    )
    materialize_dataframe_table(
        ctx,
        df=feature_manifest_df,
        table_name=apply_name_suffix("table_p16_linear_weighted_feature_manifest", run_spec.output_suffix),
        block_number="5.2.2",
        label=f"Stage 5.2.2 {apply_name_suffix('table_p16_linear_weighted_feature_manifest', run_spec.output_suffix)} — Linear feature manifest",
    )
    materialize_dataframe_table(
        ctx,
        df=canonical_alignment_df,
        table_name=apply_name_suffix("table_p16_linear_weighted_canonical_alignment", run_spec.output_suffix),
        block_number="5.2.2",
        label=f"Stage 5.2.2 {apply_name_suffix('table_p16_linear_weighted_canonical_alignment', run_spec.output_suffix)} — Canonical feature alignment",
    )
    materialize_dataframe_table(
        ctx,
        df=output_feature_manifest_df,
        table_name=apply_name_suffix("table_p16_linear_weighted_output_feature_manifest", run_spec.output_suffix),
        block_number="5.2.2",
        label=f"Stage 5.2.2 {apply_name_suffix('table_p16_linear_weighted_output_feature_manifest', run_spec.output_suffix)} — Preprocessed feature manifest",
    )

    treatment_metadata = {
        "step": "P16",
        "model_family": MODEL_NAME,
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
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_output_features_after_preprocessing": int(len(feature_names_out)),
        "train_event_rate": float(y_train.mean()),
        "test_event_rate": float(y_test.mean()),
        "design_note": "Sparse linear preprocessing is used to eliminate dense-matrix runtime warnings while preserving the benchmark feature contract.",
        "methodological_note": "All learned preprocessing operations were fit on training data only and then applied unchanged to test data.",
    }
    if run_spec.is_window_truncated_run:
        treatment_metadata["cross_arm_parity_note"] = (
            "This run truncates weekly person-period information at or before the active window and is intended for cross-arm information-parity analysis."
        )
    treatment_metadata_path = ctx.metadata_dir / append_suffix_before_extension(
        "metadata_p16_linear_weighted_treatment.json",
        run_spec.output_suffix,
    )
    save_json(treatment_metadata, treatment_metadata_path)
    print_artifact("metadata_p16_linear_treatment", str(treatment_metadata_path))
    print(treatment_metadata_path.read_text(encoding="utf-8"), flush=True)
    log_progress("5.2.2", "Treatment metadata persisted")

    log_stage_end("5.2.2")
    return LinearTreatmentState(
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


def tune_and_evaluate_linear_model(ctx: PipelineContext, treatment: LinearTreatmentState, run_spec: WindowRunSpec) -> None:
    from dropout_bench_v3_A_2_runtime_config import resolve_runtime_tuning_parallel_backend

    # Inputs:
    # - transformed sparse train and test matrices from block 5.2.2
    # - DuckDB-ready linear train and test tables
    # - benchmark horizons and calibration configuration
    # Outputs:
    # - trained model and fitted preprocessor artifacts
    # - DuckDB tuning, prediction, metric, calibration, and audit tables
    # - linear_weighted_tuned_model_config.json
    log_stage_start("5.2.3", "Tune and evaluate the linear discrete-time hazard model")
    log_progress("5.2.3", "Preparing grouped validation split")

    enrollment_groups = treatment.train_df["enrollment_id"].astype(str).to_numpy()
    if len(np.unique(enrollment_groups)) < 2:
        raise ValueError("The linear training table must contain at least two enrollment groups for validation splitting.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_TEST_SIZE, random_state=ctx.random_seed)
    subtrain_index, validation_index = next(
        splitter.split(treatment.X_train, treatment.y_train.to_numpy(), groups=enrollment_groups)
    )
    X_subtrain = treatment.X_train[subtrain_index]
    X_validation = treatment.X_train[validation_index]
    y_subtrain = treatment.y_train.iloc[subtrain_index].to_numpy()
    y_validation = treatment.y_train.iloc[validation_index].to_numpy()
    ensure_binary_target(pd.Series(y_validation), "linear validation target")
    log_progress(
        "5.2.3",
        f"Validation split ready with subtrain_rows={X_subtrain.shape[0]}, validation_rows={X_validation.shape[0]}",
    )

    tuning_workers = max(1, min(int(ctx.cpu_cores), len(LINEAR_TUNING_GRID)))
    tuning_parallel_backend = resolve_runtime_tuning_parallel_backend(ctx.shared_config)
    print(f"- TUNING_WORKERS: {tuning_workers}", flush=True)
    print(f"- TUNING_BACKEND: {tuning_parallel_backend}", flush=True)
    log_progress("5.2.3", f"Starting tuning across {len(LINEAR_TUNING_GRID)} candidates")

    candidate_rows = joblib.Parallel(n_jobs=tuning_workers, prefer=tuning_parallel_backend)(
        joblib.delayed(evaluate_linear_candidate)(
            candidate=candidate,
            random_seed=ctx.random_seed,
            X_subtrain=X_subtrain,
            y_subtrain=y_subtrain,
            X_validation=X_validation,
            y_validation=y_validation,
        )
        for candidate in LINEAR_TUNING_GRID
    )

    if not candidate_rows:
        raise RuntimeError("Linear tuning failed to produce a valid candidate model.")

    log_progress("5.2.3", f"Tuning completed with {len(candidate_rows)} evaluated candidates")
    tuning_results_df = pd.DataFrame(candidate_rows).sort_values(
        by=["val_log_loss", "val_brier", "candidate_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best_candidate = tuning_results_df.iloc[0].to_dict()
    best_model = build_logistic_regression_candidate(best_candidate, random_seed=ctx.random_seed)
    log_progress("5.2.3", f"Best candidate selected: id={int(best_candidate['candidate_id'])}, penalty={best_candidate['penalty']}, C={float(best_candidate['C'])}")
    log_progress("5.2.3", "Fitting best model on full training data")
    best_model.fit(treatment.X_train, treatment.y_train.to_numpy())
    log_progress("5.2.3", "Generating row-level predictions on test data")
    test_pred_row = np.clip(best_model.predict_proba(treatment.X_test)[:, 1], 1e-8, 1.0 - 1e-8)

    test_predictions_df = treatment.test_df.copy().sort_values(["enrollment_id", "week"]).reset_index(drop=True)
    test_predictions_df["pred_hazard"] = test_pred_row
    test_predictions_df["pred_survival"] = test_predictions_df.groupby("enrollment_id")["pred_hazard"].transform(
        lambda series: (1.0 - series).cumprod()
    )
    test_predictions_df["pred_risk"] = 1.0 - test_predictions_df["pred_survival"]
    if not np.isfinite(test_predictions_df[["pred_hazard", "pred_survival", "pred_risk"]].to_numpy()).all():
        raise ValueError("The tuned linear model produced non-finite predictions.")

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
        raise ValueError("The linear survival surface contains missing values after enrollment alignment.")
    log_progress("5.2.3", "Survival surface assembled; computing benchmark metrics")
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
        predicted_vs_observed_rows.append(
            {
                "horizon_week": int(horizon_week),
                "n_evaluable_enrollments": int(evaluable_df.shape[0]),
                "mean_predicted_survival": float(evaluable_df["pred_survival_h"].mean()),
                "mean_observed_survival": float(evaluable_df["observed_survival_by_h"].mean()),
                "abs_gap": float(abs(evaluable_df["pred_survival_h"].mean() - evaluable_df["observed_survival_by_h"].mean())),
            }
        )

        ranked_scores = evaluable_df["pred_risk_h"].rank(method="first")
        n_bins = int(min(CALIBRATION_BINS, evaluable_df.shape[0]))
        evaluable_df["calibration_bin"] = pd.qcut(ranked_scores, q=n_bins, labels=False)
        calibration_table = (
            evaluable_df.groupby("calibration_bin", as_index=False)
            .agg(
                n=("enrollment_id", "count"),
                mean_predicted_risk=("pred_risk_h", "mean"),
                observed_event_rate=("observed_event_by_h", "mean"),
            )
            .sort_values("calibration_bin")
            .reset_index(drop=True)
        )
        calibration_table["horizon_week"] = int(horizon_week)
        calibration_table["abs_calibration_gap"] = (
            calibration_table["mean_predicted_risk"] - calibration_table["observed_event_rate"]
        ).abs()
        calibration_rows.extend(calibration_table.to_dict(orient="records"))

    support_by_horizon_df = pd.DataFrame(support_rows)
    risk_auc_by_horizon_df = pd.DataFrame(risk_auc_rows)
    predicted_vs_observed_survival_df = pd.DataFrame(predicted_vs_observed_rows)
    calibration_bins_df = pd.DataFrame(calibration_rows)

    calibration_summary_df = (
        calibration_bins_df.groupby("horizon_week", as_index=False)
        .apply(
            lambda group_df: pd.Series(
                {
                    "metric_name": "calibration_at_horizon",
                    "metric_category": "primary",
                    "metric_value": float(np.average(group_df["abs_calibration_gap"], weights=group_df["n"])),
                    "notes": "Weighted absolute calibration gap across bins",
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    log_progress("5.2.3", "Calibration summaries computed; starting IPCW-dependent metrics")

    max_ipcw_time = float(truth_train_df.loc[truth_train_df["event"] == 0, "duration"].max())
    if not np.isfinite(max_ipcw_time):
        raise ValueError("Could not determine a valid censoring-support horizon for the linear model.")
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
            y_test_surv = Surv.from_arrays(
                event=supported_df["event"].astype(bool).to_numpy(),
                time=supported_df["duration"].astype(float).to_numpy(),
            )
            horizon_auc = compute_ipcw_time_dependent_auc(
                y_train_surv,
                y_test_surv,
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

    time_dependent_auc_df = pd.DataFrame(td_auc_rows)
    time_dependent_auc_audit_df = pd.DataFrame(td_auc_audit_rows)
    secondary_metrics_df = pd.concat([risk_auc_by_horizon_df, time_dependent_auc_df], ignore_index=True).sort_values(
        ["horizon_week", "metric_name"]
    ).reset_index(drop=True)

    ensure_binary_target(treatment.y_test, "pp_linear_hazard_ready_test.event_t")
    row_diagnostics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "row_level_roc_auc": float(roc_auc_score(treatment.y_test, test_pred_row)),
                "row_level_pr_auc": float(average_precision_score(treatment.y_test, test_pred_row)),
                "row_level_log_loss": float(log_loss(treatment.y_test, test_pred_row, labels=[0, 1])),
                "row_level_brier": float(brier_score_loss(treatment.y_test, test_pred_row)),
            }
        ]
    )
    log_progress("5.2.3", "All evaluation metrics computed; persisting artifacts and DuckDB tables")

    model_path = ctx.models_dir / append_suffix_before_extension(
        "linear_discrete_time_hazard_weighted_tuned.joblib",
        run_spec.output_suffix,
    )
    preprocessor_path = ctx.models_dir / append_suffix_before_extension(
        "linear_discrete_time_weighted_preprocessor.joblib",
        run_spec.output_suffix,
    )
    config_path = ctx.metadata_dir / append_suffix_before_extension(
        "linear_weighted_tuned_model_config.json",
        run_spec.output_suffix,
    )

    joblib.dump(best_model, model_path)
    joblib.dump(treatment.preprocessor, preprocessor_path)

    materialize_dataframe_table(
        ctx,
        df=tuning_results_df,
        table_name=apply_name_suffix("table_linear_weighted_tuning_results", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuning_results', run_spec.output_suffix)} — Linear tuning results",
    )
    materialize_dataframe_table(
        ctx,
        df=test_predictions_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_test_predictions", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_test_predictions', run_spec.output_suffix)} — Linear test predictions",
    )
    materialize_dataframe_table(
        ctx,
        df=primary_metrics_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_primary_metrics", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_primary_metrics', run_spec.output_suffix)} — Linear primary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=brier_by_horizon_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_brier_by_horizon", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_brier_by_horizon', run_spec.output_suffix)} — Linear Brier scores by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=secondary_metrics_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_secondary_metrics", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_secondary_metrics', run_spec.output_suffix)} — Linear secondary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=time_dependent_auc_audit_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_td_auc_support_audit", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_td_auc_support_audit', run_spec.output_suffix)} — Linear IPCW support audit",
    )
    materialize_dataframe_table(
        ctx,
        df=row_diagnostics_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_row_diagnostics", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_row_diagnostics', run_spec.output_suffix)} — Linear row-level diagnostics",
    )
    materialize_dataframe_table(
        ctx,
        df=support_by_horizon_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_support_by_horizon", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_support_by_horizon', run_spec.output_suffix)} — Linear support by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_summary_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_calibration_summary", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_calibration_summary', run_spec.output_suffix)} — Linear calibration summary",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_bins_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_calibration_bins_by_horizon", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_calibration_bins_by_horizon', run_spec.output_suffix)} — Linear calibration bins",
    )
    materialize_dataframe_table(
        ctx,
        df=predicted_vs_observed_survival_df,
        table_name=apply_name_suffix("table_linear_weighted_tuned_predicted_vs_observed_survival", run_spec.output_suffix),
        block_number="5.2.3",
        label=f"Stage 5.2.3 {apply_name_suffix('table_linear_weighted_tuned_predicted_vs_observed_survival', run_spec.output_suffix)} — Linear predicted versus observed survival",
    )

    config_payload = {
        "model_name": MODEL_NAME,
        "search_space": {
            "penalty": [candidate["penalty"] for candidate in LINEAR_TUNING_GRID[:4]] + [candidate["penalty"] for candidate in LINEAR_TUNING_GRID[4:8]],
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 2000,
        "tuning_parallel_workers": tuning_workers,
        "tuning_parallel_backend": tuning_parallel_backend,
        "run_label": run_spec.run_label,
        "is_window_truncated_run": bool(run_spec.is_window_truncated_run),
        "active_window_weeks": None if run_spec.active_window_weeks is None else int(run_spec.active_window_weeks),
        "output_suffix": run_spec.output_suffix,
        "selection_metric": "val_log_loss",
        "validation_split": {
            "method": "GroupShuffleSplit",
            "test_size": VALIDATION_TEST_SIZE,
            "group_column": "enrollment_id",
            "random_seed": ctx.random_seed,
        },
        "benchmark_horizons": list(BENCHMARK_HORIZONS),
        "calibration_bins": CALIBRATION_BINS,
        "best_candidate": {
            "candidate_id": int(best_candidate["candidate_id"]),
            "penalty": str(best_candidate["penalty"]),
            "l1_ratio": float(best_candidate["l1_ratio"]),
            "C": float(best_candidate["C"]),
            "val_log_loss": float(best_candidate["val_log_loss"]),
            "val_brier": float(best_candidate["val_brier"]),
            "val_roc_auc": float(best_candidate["val_roc_auc"]),
            "val_pr_auc": float(best_candidate["val_pr_auc"]),
        },
        "preprocessing": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_encoder": "OneHotEncoder(handle_unknown=ignore, sparse_output=True)",
            "numeric_scaler": "MaxAbsScaler",
        },
        "weighting_strategy": "liblinear class_weight='balanced' on the same weekly dynamic feature contract as the official D5.2 baseline.",
        "design_note": "The repaired IPCW-supported time-dependent AUC path is integrated directly into this weighted D5.2 variant so the Item 6 comparison changes only the fitting weights, not the evaluation stack.",
    }
    save_json(config_payload, config_path)
    print_artifact("linear_model", str(model_path))
    print_artifact("linear_preprocessor", str(preprocessor_path))
    print_artifact("linear_tuned_model_config", str(config_path))
    print(config_path.read_text(encoding="utf-8"), flush=True)
    print(pd.DataFrame([best_candidate]).to_string(index=False), flush=True)
    log_progress("5.2.3", "Model artifacts, config, and benchmark tables persisted successfully")

    log_stage_end("5.2.3")


def close_context(ctx: PipelineContext) -> None:
    from util import close_duckdb_connection

    # Inputs:
    # - active DuckDB connection
    # Outputs:
    # - clean DuckDB shutdown
    log_stage_start("5.2.4", "Close the DuckDB runtime cleanly")
    ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
    log_stage_end("5.2.4")


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
        print(f"[RUN_PLAN] linear_dynamic_execution={plan_text}", flush=True)
        for run_spec in run_specs:
            if run_spec.is_window_truncated_run:
                print(
                    f"[RUN] dynamic_window_weeks={int(run_spec.active_window_weeks)} output_suffix={run_spec.output_suffix}",
                    flush=True,
                )
            else:
                print("[RUN] official_full_information output_suffix=''", flush=True)
            treatment_state = build_linear_treatment(ctx, run_spec)
            tune_and_evaluate_linear_model(ctx, treatment_state, run_spec)
    finally:
        close_context(ctx)


if __name__ == "__main__":
    main()