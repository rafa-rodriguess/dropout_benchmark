from __future__ import annotations

"""
Production comparable-window DeepSurv benchmark module for stage D5.5.

What this file does:
- prepares the comparable-window DeepSurv treatment directly from DuckDB-ready enrollment tables
- tunes and fits the continuous-time neural survival model under the official benchmark protocol
- evaluates the tuned model with survival, calibration, row-level, and support diagnostics
- persists the trained model state, fitted preprocessing pipeline, DuckDB audit tables, and JSON metadata artifacts

Main processing purpose:
- materialize the full comparable DeepSurv benchmark arm deterministically from DuckDB-ready tables without notebook-specific runtime state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_deepsurv_preprocessing_summary, table_deepsurv_raw_feature_manifest,
  table_deepsurv_feature_names_out, table_deepsurv_tuning_results,
  table_deepsurv_tuned_training_history, table_deepsurv_tuned_test_predictions,
  table_deepsurv_tuned_primary_metrics, table_deepsurv_tuned_brier_by_horizon,
  table_deepsurv_tuned_secondary_metrics, table_deepsurv_tuned_td_auc_support_audit,
  table_deepsurv_tuned_row_diagnostics, table_deepsurv_tuned_support_by_horizon,
  table_deepsurv_tuned_calibration_summary, table_deepsurv_tuned_calibration_bins_by_horizon,
  table_deepsurv_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/deepsurv_preprocessing_config.json
- outputs_benchmark_survival/metadata/deepsurv_tuned_model_config.json
- outputs_benchmark_survival/models/deepsurv_tuned.pt
- outputs_benchmark_survival/models/deepsurv_preprocessor.joblib

Main DuckDB tables used as inputs:
- enrollment_deepsurv_ready_train
- enrollment_deepsurv_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_deepsurv_preprocessing_summary
- table_deepsurv_raw_feature_manifest
- table_deepsurv_feature_names_out
- table_deepsurv_tuning_results
- table_deepsurv_tuned_training_history
- table_deepsurv_tuned_test_predictions
- table_deepsurv_tuned_primary_metrics
- table_deepsurv_tuned_brier_by_horizon
- table_deepsurv_tuned_secondary_metrics
- table_deepsurv_tuned_td_auc_support_audit
- table_deepsurv_tuned_row_diagnostics
- table_deepsurv_tuned_support_by_horizon
- table_deepsurv_tuned_calibration_summary
- table_deepsurv_tuned_calibration_bins_by_horizon
- table_deepsurv_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/deepsurv_preprocessing_config.json
- outputs_benchmark_survival/metadata/deepsurv_tuned_model_config.json
- outputs_benchmark_survival/models/deepsurv_tuned.pt
- outputs_benchmark_survival/models/deepsurv_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, notebook globals, CSV-based workflows, or permissive optional imports are permitted
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
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv, check_y_survival

from dropout_bench_v3_D_00_common import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_OBSERVED_RISK_METHOD,
    append_suffix_before_extension,
    apply_name_suffix,
    build_ipcw_calibration_artifacts,
    build_window_suffix,
    comparable_numeric_features,
    comparable_required_columns,
    comparable_window_feature_names,
    resolve_benchmark_horizons,
    resolve_calibration_bins,
    resolve_comparable_window_execution_plan,
    resolve_variant_table_name,
    summarize_calibration_by_horizon,
)


if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    import tomli as toml_reader


STAGE_PREFIX = "5.5"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_IMPORT_NAME = SCRIPT_PATH.stem
if __name__ == "__main__":
    sys.modules.setdefault(MODULE_IMPORT_NAME, sys.modules[__name__])

NOTEBOOK_NAME = "dropout_bench_v3_D_5_5.ipynb"
PREVIEW_ROWS = 20
with open(PROJECT_ROOT / "benchmark_modeling_contract.toml", "rb") as _contract_file_obj:
    _MODULE_MODELING_CONTRACT = toml_reader.load(_contract_file_obj)
BENCHMARK_HORIZONS = tuple(resolve_benchmark_horizons(_MODULE_MODELING_CONTRACT["benchmark"]))
CALIBRATION_BINS = resolve_calibration_bins(_MODULE_MODELING_CONTRACT["benchmark"])
MODEL_NAME = "deepsurv_tuned"
VALIDATION_TEST_SIZE = 0.20
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
REQUIRED_SHARED_PATH_KEYS = [
    "output_dir",
    "tables_subdir",
    "metadata_subdir",
    "models_subdir",
    "data_output_subdir",
    "duckdb_filename",
]
REQUIRED_MODELING_KEYS = ["benchmark", "modeling", "feature_contract"]
REQUIRED_FEATURE_CONTRACT_KEYS = [
    "static_features",
    "main_enrollment_window_features",
    "optional_comparable_window_features",
]
REQUIRED_INPUT_TABLES = ["enrollment_deepsurv_ready_train", "enrollment_deepsurv_ready_test"]
BASE_REQUIRED_DEEPSURV_COLUMNS = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "event",
    "duration",
    "duration_raw",
    "used_zero_week_fallback_for_censoring",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
]
CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]
STATIC_NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
]
DEEPSURV_GRID = tuple(
    {
        "candidate_id": candidate_id,
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    for candidate_id, (hidden_dims, dropout, learning_rate, weight_decay) in enumerate(
        itertools.product(
            ([32, 16], [64, 32], [128, 64]),
            (0.10, 0.30),
            (5e-4, 1e-3),
            (1e-5, 1e-4),
        ),
        start=1,
    )
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
    random_seed: int
    test_size: float
    early_window_weeks: int
    main_enrollment_window_weeks: int
    effective_window_weeks: int
    cpu_cores: int
    input_train_table: str
    input_test_table: str
    artifact_suffix: str
    shared_config: dict[str, Any]
    shared_modeling_contract: dict[str, Any]
    run_metadata: dict[str, Any]
    con: Any


@dataclass
class DeepSurvTreatmentState:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    expected_features_raw: list[str]
    expected_features_resolved: list[str]
    numeric_fill_values: dict[str, float]
    preprocessor: Pipeline
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names_out: list[str]


class NumericPrefillTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns: list[str], numeric_columns: list[str], fill_values: dict[str, float]) -> None:
        self.feature_columns = list(feature_columns)
        self.numeric_columns = list(numeric_columns)
        self.fill_values = {str(key): float(value) for key, value in fill_values.items()}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NumericPrefillTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NumericPrefillTransformer requires a pandas DataFrame input.")
        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            raise KeyError(f"NumericPrefillTransformer is missing required columns: {', '.join(missing_columns)}")
        missing_fill_values = [column for column in self.numeric_columns if column not in self.fill_values]
        if missing_fill_values:
            raise KeyError(f"NumericPrefillTransformer is missing fill values for: {', '.join(missing_fill_values)}")
        self.feature_names_in_ = np.asarray(self.feature_columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NumericPrefillTransformer requires a pandas DataFrame input.")
        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            raise KeyError(f"NumericPrefillTransformer is missing required columns: {', '.join(missing_columns)}")
        transformed_df = X.loc[:, self.feature_columns].copy()
        for column in self.numeric_columns:
            transformed_df[column] = pd.to_numeric(transformed_df[column], errors="raise").fillna(self.fill_values[column])
            transformed_df[column] = transformed_df[column].astype(float)
            if transformed_df[column].isna().any():
                raise ValueError(f"NumericPrefillTransformer left missing values in numeric column '{column}'.")
            if not np.isfinite(transformed_df[column].to_numpy(dtype=float)).all():
                raise ValueError(f"NumericPrefillTransformer produced non-finite values in numeric column '{column}'.")
        return transformed_df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(self.feature_columns, dtype=object)
        return np.asarray(input_features, dtype=object)


NumericPrefillTransformer.__module__ = MODULE_IMPORT_NAME


def log_stage_start(block_number: str, title: str) -> None:
    print(f"[START] {STAGE_PREFIX} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# ==============================================================")
    print(f"# {block_number} - {title}")
    print("# ==============================================================")


def log_stage_end(block_number: str) -> None:
    print(f"[END] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def artifact_name(ctx: PipelineContext, base_name: str) -> str:
    return apply_name_suffix(base_name, ctx.artifact_suffix)


def artifact_filename(ctx: PipelineContext, base_filename: str) -> str:
    return append_suffix_before_extension(base_filename, ctx.artifact_suffix)


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


def materialize_dataframe_table(
    ctx: PipelineContext,
    df: pd.DataFrame,
    table_name: str,
    block_number: str,
    label: str,
) -> None:
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_d_5_5_materialize_df__"
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


def ensure_unique_enrollment_rows(df: pd.DataFrame, dataset_name: str) -> None:
    enrollment_ids = df["enrollment_id"].astype(str)
    duplicated_ids = enrollment_ids[enrollment_ids.duplicated()].unique().tolist()
    if duplicated_ids:
        preview = ", ".join(duplicated_ids[:10])
        raise ValueError(f"{dataset_name} contains duplicated enrollment_id values: {preview}")


def build_truth_by_enrollment(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_columns = ["enrollment_id", "event", "duration"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{dataset_name} is missing required columns: {', '.join(missing_columns)}")
    ensure_unique_enrollment_rows(df, dataset_name)
    truth_df = df.loc[:, required_columns].copy()
    truth_df["enrollment_id"] = truth_df["enrollment_id"].astype(str)
    truth_df["event"] = ensure_binary_target(truth_df["event"], f"{dataset_name}.event")
    truth_df["duration"] = pd.to_numeric(truth_df["duration"], errors="raise").astype(int)
    if truth_df["duration"].lt(0).any():
        raise ValueError(f"{dataset_name} contains negative enrollment durations.")
    return truth_df.sort_values("enrollment_id").reset_index(drop=True)


def build_matrix_in_enrollment_order(
    df: pd.DataFrame,
    matrix: np.ndarray,
    ordered_enrollment_ids: list[str],
    dataset_name: str,
) -> np.ndarray:
    enrollment_ids = df["enrollment_id"].astype(str).tolist()
    if len(enrollment_ids) != matrix.shape[0]:
        raise ValueError(
            f"{dataset_name} matrix row count does not match the number of enrollment rows: "
            f"{matrix.shape[0]} versus {len(enrollment_ids)}."
        )
    if len(set(enrollment_ids)) != len(enrollment_ids):
        raise ValueError(f"{dataset_name} contains duplicated enrollment_id values and cannot be reindexed safely.")
    row_index_by_enrollment = {enrollment_id: row_index for row_index, enrollment_id in enumerate(enrollment_ids)}
    missing_ids = [enrollment_id for enrollment_id in ordered_enrollment_ids if enrollment_id not in row_index_by_enrollment]
    if missing_ids:
        raise KeyError(f"{dataset_name} is missing enrollment_ids required for canonical ordering: {missing_ids[:10]}")
    ordered_positions = [row_index_by_enrollment[enrollment_id] for enrollment_id in ordered_enrollment_ids]
    ordered_matrix = np.asarray(matrix[ordered_positions], dtype=np.float32)
    if not np.isfinite(ordered_matrix).all():
        raise ValueError(f"{dataset_name} contains non-finite values after canonical enrollment ordering.")
    return ordered_matrix


def align_survival_surface_to_week_grid(raw_survival_df: pd.DataFrame, max_week: int) -> pd.DataFrame:
    if raw_survival_df.empty:
        raise ValueError("The DeepSurv survival surface is empty.")
    source_times = np.asarray(raw_survival_df.index, dtype=float)
    if source_times.ndim != 1 or source_times.size == 0:
        raise ValueError("The DeepSurv survival surface index is invalid.")
    if not np.all(np.diff(source_times) >= 0.0):
        raise ValueError("The DeepSurv survival surface index must be sorted in ascending order.")

    aligned_rows: list[np.ndarray] = []
    for week in range(max_week + 1):
        position = int(np.searchsorted(source_times, float(week), side="right") - 1)
        if position < 0:
            row_values = np.ones(raw_survival_df.shape[1], dtype=float)
        else:
            row_values = raw_survival_df.iloc[position].to_numpy(dtype=float)
        aligned_rows.append(np.clip(row_values, 1e-8, 1.0))
    aligned_df = pd.DataFrame(
        aligned_rows,
        index=pd.Index(np.arange(0, max_week + 1, dtype=int), name="week"),
        columns=raw_survival_df.columns,
    )
    if aligned_df.isna().any().any():
        raise ValueError("The aligned DeepSurv survival surface contains missing values.")
    return aligned_df


def get_survival_at_horizon(survival_wide_df: pd.DataFrame, horizon_week: int) -> pd.Series:
    if horizon_week not in survival_wide_df.index:
        raise KeyError(f"The survival surface is missing the required horizon {horizon_week}.")
    return survival_wide_df.loc[int(horizon_week)].astype(float)


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


def patch_torchtuples_adamw_compatibility() -> None:
    def before_step(self: Any) -> bool:
        if self.normalized:
            weight_decay = self._normalized_weight_decay()
        else:
            weight_decay = self.weight_decay
        for group in self.model.optimizer.param_groups:
            lr = group["lr"]
            alpha = group.get("initial_lr", lr)
            eta = lr / alpha
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.data = parameter.data.add(parameter.data, alpha=-weight_decay * eta)
        return False

    tt.callbacks.DecoupledWeightDecay.before_step = before_step


def initialize_context(window_weeks: int) -> PipelineContext:
    from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
    from util import ensure_pipeline_catalog, open_duckdb_connection

    log_stage_start("5.5.1", "Lightweight runtime bootstrap")

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
    cpu_cores = configure_runtime_cpu_cores(shared_config)

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

    effective_window_weeks = int(window_weeks)
    artifact_suffix = build_window_suffix(
        canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
        active_window_weeks=effective_window_weeks,
    )
    input_train_table = resolve_variant_table_name(
        "enrollment_deepsurv_ready_train",
        canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
        active_window_weeks=effective_window_weeks,
    )
    input_test_table = resolve_variant_table_name(
        "enrollment_deepsurv_ready_test",
        canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
        active_window_weeks=effective_window_weeks,
    )

    set_deterministic_state(int(benchmark_config["seed"]))
    patch_torchtuples_adamw_compatibility()
    con = open_duckdb_connection(duckdb_path)
    ensure_pipeline_catalog(con)
    require_tables(con, [input_train_table, input_test_table], block_number="5.5.1")

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
        effective_window_weeks=effective_window_weeks,
        cpu_cores=cpu_cores,
        input_train_table=input_train_table,
        input_test_table=input_test_table,
        artifact_suffix=artifact_suffix,
        shared_config=shared_config,
        shared_modeling_contract=shared_modeling_contract,
        run_metadata=run_metadata,
        con=con,
    )

    print_artifact("benchmark_shared_config", str(config_toml_path))
    print_artifact("benchmark_modeling_contract", str(modeling_contract_toml_path))
    print_artifact("run_metadata", str(run_metadata_path))
    print_artifact("duckdb_path", str(duckdb_path))
    print(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "random_seed": ctx.random_seed,
                "test_size": ctx.test_size,
                "validation_test_size": VALIDATION_TEST_SIZE,
                "cpu_cores": ctx.cpu_cores,
                "early_window_weeks": ctx.early_window_weeks,
                "main_enrollment_window_weeks": ctx.main_enrollment_window_weeks,
                "effective_window_weeks": ctx.effective_window_weeks,
                "input_train_table": ctx.input_train_table,
                "input_test_table": ctx.input_test_table,
                "benchmark_horizons": list(BENCHMARK_HORIZONS),
                "calibration_bins": CALIBRATION_BINS,
            },
            indent=2,
        )
    )

    log_stage_end("5.5.1")
    return ctx


def build_deepsurv_treatment(ctx: PipelineContext) -> DeepSurvTreatmentState:
    log_stage_start("5.5.2", "Build comparable-window DeepSurv treatment")

    feature_contract = ctx.shared_modeling_contract["feature_contract"]
    require_mapping_keys(feature_contract, REQUIRED_FEATURE_CONTRACT_KEYS, "benchmark_modeling_contract.toml [feature_contract]")

    static_features = require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
    canonical_main_window_features = require_list_of_strings(
        feature_contract["main_enrollment_window_features"],
        "feature_contract.main_enrollment_window_features",
    )
    canonical_optional_comparable_features = require_list_of_strings(
        feature_contract["optional_comparable_window_features"],
        "feature_contract.optional_comparable_window_features",
    )

    if int(ctx.effective_window_weeks) == int(ctx.main_enrollment_window_weeks):
        main_window_features = canonical_main_window_features
        optional_comparable_features = canonical_optional_comparable_features
    else:
        active_feature_names = comparable_window_feature_names(ctx.effective_window_weeks)
        main_window_features = list(STATIC_NUMERIC_FEATURES) + [active_feature_names["main_clicks_feature"]]
        optional_comparable_features = [
            active_feature_names["main_active_feature"],
            active_feature_names["main_mean_clicks_feature"],
        ]

    expected_features_raw = static_features + main_window_features + optional_comparable_features
    expected_features_resolved = list(dict.fromkeys(expected_features_raw))
    numeric_features = comparable_numeric_features(
        ctx.effective_window_weeks,
        static_numeric_features=STATIC_NUMERIC_FEATURES,
    )
    operational_feature_columns = CATEGORICAL_FEATURES + numeric_features
    expected_comparable_numeric_features = list(dict.fromkeys(main_window_features + optional_comparable_features))
    if sorted(expected_comparable_numeric_features) != sorted(numeric_features):
        raise ValueError(
            "The comparable DeepSurv numeric feature specification does not match the canonical modeling contract. "
            f"Contract numeric features: {expected_comparable_numeric_features}. Operational numeric features: {numeric_features}."
        )

    required_columns = comparable_required_columns(BASE_REQUIRED_DEEPSURV_COLUMNS, ctx.effective_window_weeks)

    train_df = load_required_table(ctx, ctx.input_train_table, required_columns, block_number="5.5.2")
    test_df = load_required_table(ctx, ctx.input_test_table, required_columns, block_number="5.5.2")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["enrollment_id"] = train_df["enrollment_id"].astype(str)
    test_df["enrollment_id"] = test_df["enrollment_id"].astype(str)
    train_df["event"] = ensure_binary_target(train_df["event"], f"{ctx.input_train_table}.event")
    test_df["event"] = ensure_binary_target(test_df["event"], f"{ctx.input_test_table}.event")
    train_df["duration"] = pd.to_numeric(train_df["duration"], errors="raise").astype(int)
    test_df["duration"] = pd.to_numeric(test_df["duration"], errors="raise").astype(int)
    ensure_unique_enrollment_rows(train_df, ctx.input_train_table)
    ensure_unique_enrollment_rows(test_df, ctx.input_test_table)

    numeric_fill_values: dict[str, float] = {}
    for column in numeric_features:
        train_numeric = pd.to_numeric(train_df[column], errors="raise")
        observed_values = train_numeric.dropna()
        numeric_fill_values[column] = float(observed_values.median()) if not observed_values.empty else 0.0

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    preprocessor = Pipeline(
        steps=[
            (
                "numeric_prefill",
                NumericPrefillTransformer(
                    feature_columns=operational_feature_columns,
                    numeric_columns=numeric_features,
                    fill_values=numeric_fill_values,
                ),
            ),
            ("column_transformer", column_transformer),
        ]
    )

    X_train = np.asarray(preprocessor.fit_transform(train_df[operational_feature_columns]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_df[operational_feature_columns]), dtype=np.float32)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"The comparable DeepSurv transformed matrices must be 2D. Got train={X_train.shape}, test={X_test.shape}.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("The comparable DeepSurv transformed matrices contain non-finite values.")

    feature_names_out = preprocessor.get_feature_names_out().astype(str).tolist()
    preprocessing_summary_df = pd.DataFrame(
        [
            {
                "model_family": "deepsurv",
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "n_input_features_raw": int(len(operational_feature_columns)),
                "n_numeric_features": int(len(numeric_features)),
                "n_categorical_features": int(len(CATEGORICAL_FEATURES)),
                "n_features_after_transform": int(len(feature_names_out)),
                "n_events_train": int(train_df["event"].sum()),
                "n_events_test": int(test_df["event"].sum()),
                "mean_duration_train": float(train_df["duration"].mean()),
                "mean_duration_test": float(test_df["duration"].mean()),
                "numeric_prefill_policy": "deterministic_train_median_or_zero_when_all_missing",
                "categorical_imputation": "constant_missing",
                "categorical_encoding": "one_hot_handle_unknown_ignore",
                "numeric_scaling": "standard_scaler_fit_on_train_only",
                "output_dtype": "float32",
                "comparability_note": "same conceptual input design as the comparable Cox benchmark",
                "window_weeks": int(ctx.effective_window_weeks),
            }
        ]
    )
    raw_feature_manifest_df = pd.DataFrame(
        {
            "feature_name_operational": operational_feature_columns,
            "feature_role": ["categorical" if column in CATEGORICAL_FEATURES else "numeric" for column in operational_feature_columns],
            "feature_scope": [
                "fixed_demographic_metadata" if column in CATEGORICAL_FEATURES else "canonical_comparable_numeric"
                for column in operational_feature_columns
            ],
            "present_in_train": [column in train_df.columns for column in operational_feature_columns],
            "present_in_test": [column in test_df.columns for column in operational_feature_columns],
            "covered_by_canonical_config": [column in expected_features_resolved for column in operational_feature_columns],
            "numeric_prefill_value": [numeric_fill_values.get(column, np.nan) for column in operational_feature_columns],
        }
    )
    output_feature_manifest_df = pd.DataFrame({"feature_name_out": feature_names_out})

    materialize_dataframe_table(
        ctx,
        df=preprocessing_summary_df,
        table_name=artifact_name(ctx, "table_deepsurv_preprocessing_summary"),
        block_number="5.5.2",
        label="Stage 5.5.2 table_deepsurv_preprocessing_summary - DeepSurv preprocessing summary",
    )
    materialize_dataframe_table(
        ctx,
        df=raw_feature_manifest_df,
        table_name=artifact_name(ctx, "table_deepsurv_raw_feature_manifest"),
        block_number="5.5.2",
        label="Stage 5.5.2 table_deepsurv_raw_feature_manifest - DeepSurv raw feature manifest",
    )
    materialize_dataframe_table(
        ctx,
        df=output_feature_manifest_df,
        table_name=artifact_name(ctx, "table_deepsurv_feature_names_out"),
        block_number="5.5.2",
        label="Stage 5.5.2 table_deepsurv_feature_names_out - DeepSurv output feature manifest",
    )

    preprocessing_config = {
        "model_family": "deepsurv",
        "duration_column": "duration",
        "event_column": "event",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": numeric_features,
        "raw_feature_columns": operational_feature_columns,
        "expected_features_raw": expected_features_raw,
        "expected_features_resolved": expected_features_resolved,
        "numeric_prefill_policy": "deterministic_train_median_or_zero_when_all_missing",
        "numeric_fill_values": numeric_fill_values,
        "categorical_imputation": "constant_missing",
        "categorical_encoding": "one_hot_handle_unknown_ignore",
        "numeric_scaling": "standard_scaler_fit_on_train_only",
        "feature_names_out": feature_names_out,
        "comparability_note": "same conceptual input design as the comparable Cox benchmark",
        "window_weeks": int(ctx.effective_window_weeks),
    }
    preprocessing_config_path = ctx.metadata_dir / artifact_filename(ctx, "deepsurv_preprocessing_config.json")
    save_json(preprocessing_config, preprocessing_config_path)
    print_artifact("deepsurv_preprocessing_config", str(preprocessing_config_path))
    print(preprocessing_config_path.read_text(encoding="utf-8"))

    log_stage_end("5.5.2")
    return DeepSurvTreatmentState(
        train_df=train_df,
        test_df=test_df,
        feature_columns=operational_feature_columns,
        expected_features_raw=expected_features_raw,
        expected_features_resolved=expected_features_resolved,
        numeric_fill_values=numeric_fill_values,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        feature_names_out=feature_names_out,
    )


def fit_deepsurv_candidate(
    X_train_internal: np.ndarray,
    durations_train_internal: np.ndarray,
    events_train_internal: np.ndarray,
    X_validation: np.ndarray,
    durations_validation: np.ndarray,
    events_validation: np.ndarray,
    input_dim: int,
    candidate: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    set_deterministic_state(random_seed)

    network = tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=list(candidate["hidden_dims"]),
        out_features=1,
        batch_norm=True,
        dropout=float(candidate["dropout"]),
        output_bias=False,
    )
    optimizer = tt.optim.AdamW(
        lr=float(candidate["learning_rate"]),
        decoupled_weight_decay=float(candidate["weight_decay"]),
    )
    model = CoxPH(network, optimizer)
    model.optimizer.set_lr(float(candidate["learning_rate"]))
    callbacks = [tt.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE)]
    train_target = (durations_train_internal.astype(np.float32), events_train_internal.astype(np.float32))
    validation_target = (durations_validation.astype(np.float32), events_validation.astype(np.float32))
    log = model.fit(
        X_train_internal.astype(np.float32),
        train_target,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=False,
        val_data=(X_validation.astype(np.float32), validation_target),
    )
    history_df = log.to_pandas().reset_index().rename(columns={"index": "epoch"})
    if "train_loss" not in history_df.columns and "loss" in history_df.columns:
        history_df = history_df.rename(columns={"loss": "train_loss"})
    if "val_loss" not in history_df.columns:
        raise KeyError(f"DeepSurv candidate {candidate['candidate_id']} did not return validation losses.")
    history_df["epoch"] = history_df["epoch"].astype(int) + 1
    history_df["candidate_id"] = int(candidate["candidate_id"])

    best_row_index = int(history_df["val_loss"].astype(float).idxmin())
    best_val_loss = float(history_df.loc[best_row_index, "val_loss"])
    best_epoch = int(history_df.loc[best_row_index, "epoch"])
    candidate_result = {
        "candidate_id": int(candidate["candidate_id"]),
        "hidden_dims": str(list(candidate["hidden_dims"])),
        "dropout": float(candidate["dropout"]),
        "learning_rate": float(candidate["learning_rate"]),
        "weight_decay": float(candidate["weight_decay"]),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    return candidate_result, history_df


def tune_and_evaluate_deepsurv_model(ctx: PipelineContext, treatment: DeepSurvTreatmentState) -> None:
    log_stage_start("5.5.3", "Tune and evaluate the comparable-window DeepSurv model")

    truth_train_df = build_truth_by_enrollment(treatment.train_df, ctx.input_train_table)
    truth_test_df = build_truth_by_enrollment(treatment.test_df, ctx.input_test_table)
    ordered_train_enrollment_ids = truth_train_df["enrollment_id"].astype(str).tolist()
    ordered_test_enrollment_ids = truth_test_df["enrollment_id"].astype(str).tolist()

    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_TEST_SIZE,
        random_state=ctx.random_seed,
    )
    train_internal_idx, val_internal_idx = next(
        group_splitter.split(
            treatment.X_train,
            treatment.train_df["event"].to_numpy(),
            groups=treatment.train_df["enrollment_id"].astype(str).to_numpy(),
        )
    )

    X_train_internal = np.asarray(treatment.X_train[train_internal_idx], dtype=np.float32)
    X_val_internal = np.asarray(treatment.X_train[val_internal_idx], dtype=np.float32)
    durations_train_internal = treatment.train_df.iloc[train_internal_idx]["duration"].to_numpy(dtype=np.float32)
    durations_val_internal = treatment.train_df.iloc[val_internal_idx]["duration"].to_numpy(dtype=np.float32)
    events_train_internal = treatment.train_df.iloc[train_internal_idx]["event"].to_numpy(dtype=np.int32)
    events_val_internal = treatment.train_df.iloc[val_internal_idx]["event"].to_numpy(dtype=np.int32)

    candidate_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_candidate_history_df: pd.DataFrame | None = None
    best_val_loss = float("inf")

    for candidate in DEEPSURV_GRID:
        candidate_result, history_df = fit_deepsurv_candidate(
            X_train_internal=X_train_internal,
            durations_train_internal=durations_train_internal,
            events_train_internal=events_train_internal,
            X_validation=X_val_internal,
            durations_validation=durations_val_internal,
            events_validation=events_val_internal,
            input_dim=int(treatment.X_train.shape[1]),
            candidate=candidate,
            random_seed=ctx.random_seed + int(candidate["candidate_id"]),
        )
        candidate_rows.append(candidate_result)
        if float(candidate_result["best_val_loss"]) < best_val_loss:
            best_val_loss = float(candidate_result["best_val_loss"])
            best_candidate = {
                "candidate_id": int(candidate["candidate_id"]),
                "hidden_dims": list(candidate["hidden_dims"]),
                "dropout": float(candidate["dropout"]),
                "learning_rate": float(candidate["learning_rate"]),
                "weight_decay": float(candidate["weight_decay"]),
                "best_val_loss": float(candidate_result["best_val_loss"]),
                "best_epoch": int(candidate_result["best_epoch"]),
            }
            best_candidate_history_df = history_df.copy()

    if best_candidate is None or best_candidate_history_df is None:
        raise RuntimeError("DeepSurv tuning failed to produce a valid best candidate.")

    tuning_results_df = pd.DataFrame(candidate_rows).sort_values(
        ["best_val_loss", "candidate_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    set_deterministic_state(ctx.random_seed)
    final_network = tt.practical.MLPVanilla(
        in_features=int(treatment.X_train.shape[1]),
        num_nodes=list(best_candidate["hidden_dims"]),
        out_features=1,
        batch_norm=True,
        dropout=float(best_candidate["dropout"]),
        output_bias=False,
    )
    final_optimizer = tt.optim.AdamW(
        lr=float(best_candidate["learning_rate"]),
        decoupled_weight_decay=float(best_candidate["weight_decay"]),
    )
    final_model = CoxPH(final_network, final_optimizer)
    final_model.optimizer.set_lr(float(best_candidate["learning_rate"]))
    final_train_target = (
        treatment.train_df["duration"].to_numpy(dtype=np.float32),
        treatment.train_df["event"].to_numpy(dtype=np.float32),
    )
    _ = final_model.fit(
        np.asarray(treatment.X_train, dtype=np.float32),
        final_train_target,
        batch_size=BATCH_SIZE,
        epochs=int(best_candidate["best_epoch"]),
        verbose=False,
    )
    final_model.compute_baseline_hazards()

    ordered_X_test = build_matrix_in_enrollment_order(
        treatment.test_df,
        treatment.X_test,
        ordered_test_enrollment_ids,
        "enrollment_deepsurv_ready_test",
    )
    raw_survival_df = final_model.predict_surv_df(ordered_X_test)
    raw_survival_df.columns = ordered_test_enrollment_ids
    raw_survival_df.columns.name = "enrollment_id"
    raw_survival_df.index = raw_survival_df.index.astype(float)
    max_duration = int(max(truth_train_df["duration"].max(), truth_test_df["duration"].max(), max(BENCHMARK_HORIZONS)))
    survival_wide_df = align_survival_surface_to_week_grid(raw_survival_df, max_week=max_duration)
    survival_wide_df = survival_wide_df.reindex(columns=ordered_test_enrollment_ids)
    if survival_wide_df.isna().any().any():
        raise ValueError("The tuned DeepSurv survival surface contains missing values after enrollment alignment.")

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

    eval_surv = EvalSurv(
        surv=survival_wide_df,
        durations=durations_test,
        events=events_test,
        censor_surv="km",
    )
    brier_time_grid = np.arange(1, max(BENCHMARK_HORIZONS) + 1, dtype=int)
    survival_estimate_matrix = survival_wide_df.loc[brier_time_grid, ordered_test_enrollment_ids].to_numpy(dtype=float).T
    brier_times, brier_scores = sksurv_brier_score(
        y_train_surv,
        y_test_surv,
        survival_estimate_matrix,
        brier_time_grid.astype(float),
    )
    if len(brier_times) < 2:
        raise ValueError("At least two time points are required to compute the integrated Brier score.")
    integrated_brier = float(
        np.trapezoid(brier_scores.astype(float), brier_times.astype(float))
        / (float(brier_times[-1]) - float(brier_times[0]))
    )

    primary_metrics_df = pd.DataFrame(
        [
            {
                "metric_name": "ibs",
                "metric_category": "primary",
                "metric_value": integrated_brier,
                "notes": "sksurv_integrated_brier_score",
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

    prediction_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    risk_auc_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    predicted_vs_observed_rows: list[dict[str, Any]] = []
    for horizon_week in BENCHMARK_HORIZONS:
        pred_survival_h = get_survival_at_horizon(survival_wide_df, horizon_week)
        pred_risk_h = 1.0 - pred_survival_h

        horizon_predictions_df = truth_test_df.copy()
        horizon_predictions_df["horizon_week"] = int(horizon_week)
        horizon_predictions_df["pred_survival_h"] = pred_survival_h.reindex(truth_test_df["enrollment_id"]).to_numpy(dtype=float)
        horizon_predictions_df["pred_risk_h"] = pred_risk_h.reindex(truth_test_df["enrollment_id"]).to_numpy(dtype=float)
        if horizon_predictions_df[["pred_survival_h", "pred_risk_h"]].isna().any().any():
            raise ValueError(f"The tuned DeepSurv predictions contain missing values at horizon {horizon_week}.")
        prediction_rows.extend(horizon_predictions_df.to_dict(orient="records"))

        evaluable_df = horizon_predictions_df.copy()
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
        ensure_binary_target(evaluable_df["observed_event_by_h"], f"deepsurv observed_event_by_h@{horizon_week}")
        if evaluable_df["observed_event_by_h"].nunique() < 2:
            raise ValueError(
                f"risk_auc_at_horizon is undefined at horizon {horizon_week}: only one observed class is present."
            )

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
                "metric_value": float(roc_auc_score(evaluable_df["observed_event_by_h"], evaluable_df["pred_risk_h"])),
                "notes": "roc_auc_on_evaluable_subset",
            }
        )
        calibration_table, predicted_vs_observed_row = build_ipcw_calibration_artifacts(
            y_train_surv,
            horizon_predictions_df,
            horizon_week,
            CALIBRATION_BINS,
        )
        predicted_vs_observed_rows.append(predicted_vs_observed_row)
        calibration_rows.extend(calibration_table.to_dict(orient="records"))

    test_predictions_df = pd.DataFrame(prediction_rows).sort_values(
        ["horizon_week", "enrollment_id"]
    ).reset_index(drop=True)
    support_by_horizon_df = pd.DataFrame(support_rows)
    risk_auc_by_horizon_df = pd.DataFrame(risk_auc_rows)
    predicted_vs_observed_survival_df = pd.DataFrame(predicted_vs_observed_rows)
    calibration_bins_df = pd.DataFrame(calibration_rows)

    calibration_summary_df = summarize_calibration_by_horizon(calibration_bins_df)

    max_ipcw_time = float(truth_train_df.loc[truth_train_df["event"] == 0, "duration"].max())
    if not np.isfinite(max_ipcw_time):
        raise ValueError("Could not determine a valid censoring-support horizon for the DeepSurv model.")

    td_auc_rows: list[dict[str, Any]] = []
    td_auc_audit_rows: list[dict[str, Any]] = []
    for horizon_week in BENCHMARK_HORIZONS:
        horizon_prediction_df = test_predictions_df.loc[
            test_predictions_df["horizon_week"] == int(horizon_week),
            ["enrollment_id", "pred_risk_h"],
        ].copy()
        supported_df = truth_test_df.merge(horizon_prediction_df, on="enrollment_id", how="left")
        supported_df = supported_df.loc[
            (supported_df["duration"] < max_ipcw_time) & supported_df["pred_risk_h"].notna()
        ].copy()
        if supported_df.empty:
            raise ValueError(f"No IPCW-supported evaluation rows are available at horizon {horizon_week}.")
        if supported_df["event"].nunique() < 2:
            raise ValueError(
                f"The IPCW-supported evaluation subset at horizon {horizon_week} does not contain both event classes."
            )

        supported_test_surv = Surv.from_arrays(
            event=supported_df["event"].astype(bool).to_numpy(),
            time=supported_df["duration"].astype(float).to_numpy(),
        )
        horizon_auc = compute_ipcw_time_dependent_auc(
            y_train_surv,
            supported_test_surv,
            supported_df["pred_risk_h"].to_numpy(dtype=float),
            float(horizon_week),
        )
        td_auc_rows.append(
            {
                "horizon_week": int(horizon_week),
                "metric_name": "time_dependent_auc",
                "metric_category": "secondary",
                "metric_value": horizon_auc,
                "notes": (
                    f"ipcw_supported_subset(duration < {max_ipcw_time:.0f}); "
                    f"retained={supported_df.shape[0]}/{truth_test_df.shape[0]}; "
                    f"auc={horizon_auc:.6f}; local_ipcw_auc"
                ),
            }
        )
        td_auc_audit_rows.append(
            {
                "horizon_week": int(horizon_week),
                "max_ipcw_time": max_ipcw_time,
                "n_total_test_enrollments": int(truth_test_df.shape[0]),
                "n_supported_test_enrollments": int(supported_df.shape[0]),
                "mean_auc": horizon_auc,
                "notes": "IPCW-supported subset used for DeepSurv time-dependent AUC",
            }
        )

    time_dependent_auc_df = pd.DataFrame(td_auc_rows)
    td_auc_support_audit_df = pd.DataFrame(td_auc_audit_rows)
    secondary_metrics_df = pd.concat(
        [risk_auc_by_horizon_df, time_dependent_auc_df],
        ignore_index=True,
    ).sort_values(["horizon_week", "metric_name"]).reset_index(drop=True)

    row_pred_survival = np.asarray(
        [
            float(survival_wide_df.loc[int(duration), enrollment_id])
            for enrollment_id, duration in zip(truth_test_df["enrollment_id"], truth_test_df["duration"])
        ],
        dtype=float,
    )
    row_pred_risk = 1.0 - row_pred_survival
    if not np.isfinite(row_pred_risk).all():
        raise ValueError("The tuned DeepSurv event-time risk diagnostics contain non-finite values.")

    row_roc_auc = float(roc_auc_score(truth_test_df["event"], row_pred_risk))
    row_pr_auc = float(average_precision_score(truth_test_df["event"], row_pred_risk))
    row_log_loss = float(log_loss(truth_test_df["event"], np.clip(row_pred_risk, 1e-8, 1.0 - 1e-8), labels=[0, 1]))
    row_brier = float(np.mean((row_pred_risk - truth_test_df["event"].to_numpy(dtype=float)) ** 2))
    row_diagnostics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "row_level_roc_auc": row_roc_auc,
                "row_level_pr_auc": row_pr_auc,
                "row_level_log_loss": row_log_loss,
                "row_level_brier": row_brier,
                "diagnostic_note": "auxiliary enrollment-level proxy only; not a primary cross-family benchmark metric",
            }
        ]
    )

    model_path = ctx.models_dir / artifact_filename(ctx, "deepsurv_tuned.pt")
    preprocessor_path = ctx.models_dir / artifact_filename(ctx, "deepsurv_preprocessor.joblib")
    config_path = ctx.metadata_dir / artifact_filename(ctx, "deepsurv_tuned_model_config.json")
    torch.save(final_model.net.state_dict(), model_path)
    joblib.dump(treatment.preprocessor, preprocessor_path)

    materialize_dataframe_table(
        ctx,
        df=tuning_results_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuning_results"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuning_results - DeepSurv tuning results",
    )
    materialize_dataframe_table(
        ctx,
        df=best_candidate_history_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_training_history"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_training_history - DeepSurv best-candidate training history",
    )
    materialize_dataframe_table(
        ctx,
        df=test_predictions_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_test_predictions"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_test_predictions - DeepSurv test predictions",
    )
    materialize_dataframe_table(
        ctx,
        df=primary_metrics_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_primary_metrics"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_primary_metrics - DeepSurv primary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=brier_by_horizon_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_brier_by_horizon"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_brier_by_horizon - DeepSurv Brier scores by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=secondary_metrics_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_secondary_metrics"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_secondary_metrics - DeepSurv secondary metrics",
    )
    materialize_dataframe_table(
        ctx,
        df=td_auc_support_audit_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_td_auc_support_audit"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_td_auc_support_audit - DeepSurv IPCW support audit",
    )
    materialize_dataframe_table(
        ctx,
        df=row_diagnostics_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_row_diagnostics"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_row_diagnostics - DeepSurv row-level diagnostics",
    )
    materialize_dataframe_table(
        ctx,
        df=support_by_horizon_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_support_by_horizon"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_support_by_horizon - DeepSurv support by horizon",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_summary_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_calibration_summary"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_calibration_summary - DeepSurv calibration summary",
    )
    materialize_dataframe_table(
        ctx,
        df=calibration_bins_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_calibration_bins_by_horizon"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_calibration_bins_by_horizon - DeepSurv calibration bins",
    )
    materialize_dataframe_table(
        ctx,
        df=predicted_vs_observed_survival_df,
        table_name=artifact_name(ctx, "table_deepsurv_tuned_predicted_vs_observed_survival"),
        block_number="5.5.3",
        label="Stage 5.5.3 table_deepsurv_tuned_predicted_vs_observed_survival - DeepSurv predicted versus observed survival",
    )

    config_payload = {
        "model_name": MODEL_NAME,
        "selection_metric": "lowest_validation_loss",
        "validation_split": {
            "method": "GroupShuffleSplit",
            "test_size": float(VALIDATION_TEST_SIZE),
            "group_column": "enrollment_id",
            "random_seed": int(ctx.random_seed),
        },
        "effective_window_weeks": int(ctx.effective_window_weeks),
        "input_train_table": ctx.input_train_table,
        "input_test_table": ctx.input_test_table,
        "search_space": [
            {
                "candidate_id": int(candidate["candidate_id"]),
                "hidden_dims": list(candidate["hidden_dims"]),
                "dropout": float(candidate["dropout"]),
                "learning_rate": float(candidate["learning_rate"]),
                "weight_decay": float(candidate["weight_decay"]),
            }
            for candidate in DEEPSURV_GRID
        ],
        "batch_norm": True,
        "output_bias": False,
        "batch_size": int(BATCH_SIZE),
        "epochs": int(MAX_EPOCHS),
        "patience": int(EARLY_STOPPING_PATIENCE),
        "benchmark_horizons": list(BENCHMARK_HORIZONS),
        "calibration_bins": int(CALIBRATION_BINS),
        "calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
        "calibration_observed_risk_method": CALIBRATION_OBSERVED_RISK_METHOD,
        "best_candidate": best_candidate,
        "preprocessing": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": treatment.feature_columns[len(CATEGORICAL_FEATURES):],
            "numeric_prefill_values": treatment.numeric_fill_values,
            "feature_names_out": treatment.feature_names_out,
        },
        "design_note": (
            "The comparable DeepSurv arm integrates deterministic numeric prefill, canonical enrollment ordering, "
            "and a local IPCW-supported time-dependent AUC path directly into D5.5 so no downstream repair stage is required."
        ),
    }
    save_json(config_payload, config_path)
    print_artifact("deepsurv_model", str(model_path))
    print_artifact("deepsurv_preprocessor", str(preprocessor_path))
    print_artifact("deepsurv_tuned_model_config", str(config_path))
    print(config_path.read_text(encoding="utf-8"))
    print(pd.DataFrame([best_candidate]).to_string(index=False))

    log_stage_end("5.5.3")


def close_context(ctx: PipelineContext) -> None:
    from util import close_duckdb_connection

    log_stage_start("5.5.4", "Close runtime resources")
    close_duckdb_connection(ctx.con)
    print_artifact("duckdb_connection", f"closed::{ctx.duckdb_path}")
    log_stage_end("5.5.4")


def run_single_window(window_weeks: int) -> None:
    ctx: PipelineContext | None = None
    try:
        ctx = initialize_context(window_weeks)
        treatment = build_deepsurv_treatment(ctx)
        tune_and_evaluate_deepsurv_model(ctx, treatment)
    finally:
        if ctx is not None:
            close_context(ctx)


def resolve_execution_windows() -> list[int]:
    modeling_contract_toml_path = PROJECT_ROOT / "benchmark_modeling_contract.toml"
    if not modeling_contract_toml_path.exists():
        raise FileNotFoundError(f"Missing modeling contract TOML: {modeling_contract_toml_path}")
    with open(modeling_contract_toml_path, "rb") as file_obj:
        shared_modeling_contract = toml_reader.load(file_obj)
    benchmark_config = shared_modeling_contract.get("benchmark")
    if not isinstance(benchmark_config, dict):
        raise TypeError("benchmark_modeling_contract.toml is missing the [benchmark] section.")
    return resolve_comparable_window_execution_plan(benchmark_config)


def main() -> None:
    execution_windows = resolve_execution_windows()
    for window_weeks in execution_windows:
        print(f"[RUN_PLAN] comparable_window_weeks={int(window_weeks)} script={SCRIPT_NAME}")
        run_single_window(int(window_weeks))


if __name__ == "__main__":
    main()