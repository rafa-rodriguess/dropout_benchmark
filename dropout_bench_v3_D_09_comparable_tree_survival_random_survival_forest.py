from __future__ import annotations

"""
Production comparable-window Random Survival Forest benchmark module for stage D5.9.

What this file does:
- prepares the comparable-window RSF treatment directly from DuckDB-ready enrollment tables
- tunes and fits a compact random survival forest under the official benchmark protocol
- evaluates the tuned model with survival, calibration, support, and auxiliary row-level diagnostics
- persists the trained model, fitted preprocessing pipeline, DuckDB audit tables, and JSON metadata artifacts

Main processing purpose:
- materialize the full comparable nonlinear tree-ensemble benchmark arm deterministically from DuckDB-ready tables without notebook runtime state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_rsf_preprocessing_summary, table_rsf_raw_feature_manifest,
  table_rsf_feature_names_out, table_rsf_tuning_results,
  table_rsf_tuned_test_predictions, table_rsf_tuned_primary_metrics,
  table_rsf_tuned_brier_by_horizon, table_rsf_tuned_secondary_metrics,
  table_rsf_tuned_td_auc_support_audit, table_rsf_tuned_row_diagnostics,
  table_rsf_tuned_support_by_horizon, table_rsf_tuned_calibration_summary,
  table_rsf_tuned_calibration_bins_by_horizon,
  table_rsf_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/rsf_preprocessing_config.json
- outputs_benchmark_survival/metadata/rsf_tuned_model_config.json
- outputs_benchmark_survival/models/rsf_tuned.joblib
- outputs_benchmark_survival/models/rsf_preprocessor.joblib

Main DuckDB tables used as inputs:
- enrollment_cox_ready_train
- enrollment_cox_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_rsf_preprocessing_summary
- table_rsf_raw_feature_manifest
- table_rsf_feature_names_out
- table_rsf_tuning_results
- table_rsf_tuned_test_predictions
- table_rsf_tuned_primary_metrics
- table_rsf_tuned_brier_by_horizon
- table_rsf_tuned_secondary_metrics
- table_rsf_tuned_td_auc_support_audit
- table_rsf_tuned_row_diagnostics
- table_rsf_tuned_support_by_horizon
- table_rsf_tuned_calibration_summary
- table_rsf_tuned_calibration_bins_by_horizon
- table_rsf_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/rsf_preprocessing_config.json
- outputs_benchmark_survival/metadata/rsf_tuned_model_config.json
- outputs_benchmark_survival/models/rsf_tuned.joblib
- outputs_benchmark_survival/models/rsf_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, notebook globals, or CSV-based workflows are permitted
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
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


STAGE_PREFIX = "5.9"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

MODULE_IMPORT_NAME = SCRIPT_PATH.stem
if __name__ == "__main__":
	sys.modules.setdefault(MODULE_IMPORT_NAME, sys.modules[__name__])

NOTEBOOK_NAME = "dropout_bench_D_v5_9_random_survival_forest.ipynb"
PREVIEW_ROWS = 20
with open(PROJECT_ROOT / "benchmark_modeling_contract.toml", "rb") as _contract_file_obj:
	_MODULE_MODELING_CONTRACT = toml_reader.load(_contract_file_obj)
BENCHMARK_HORIZONS = tuple(resolve_benchmark_horizons(_MODULE_MODELING_CONTRACT["benchmark"]))
CALIBRATION_BINS = resolve_calibration_bins(_MODULE_MODELING_CONTRACT["benchmark"])
VALIDATION_TEST_SIZE = 0.10
LOW_VARIANCE_THRESHOLD = 1e-12
MODEL_NAME = "rsf_tuned"
RSF_N_JOBS = 4
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
REQUIRED_INPUT_TABLES = ["enrollment_cox_ready_train", "enrollment_cox_ready_test"]
BASE_REQUIRED_CONTINUOUS_COLUMNS = [
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
RSF_TUNING_GRID = (
	{
		"candidate_id": 1,
		"n_estimators": 200,
		"min_samples_leaf": 3,
		"max_depth": None,
		"max_features": "sqrt",
	},
	{
		"candidate_id": 2,
		"n_estimators": 300,
		"min_samples_leaf": 5,
		"max_depth": None,
		"max_features": "sqrt",
	},
	{
		"candidate_id": 3,
		"n_estimators": 200,
		"min_samples_leaf": 3,
		"max_depth": 8,
		"max_features": 0.8,
	},
	{
		"candidate_id": 4,
		"n_estimators": 300,
		"min_samples_leaf": 10,
		"max_depth": 8,
		"max_features": 0.7,
	},
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
class RSFTreatmentState:
	train_df: pd.DataFrame
	test_df: pd.DataFrame
	feature_columns: list[str]
	expected_features_raw: list[str]
	expected_features_resolved: list[str]
	numeric_fill_values: dict[str, float]
	preprocessor: Pipeline
	X_train: np.ndarray
	X_test: np.ndarray
	feature_names_before_filter: list[str]
	feature_names_out: list[str]
	dropped_feature_names: list[str]
	transformed_variances: list[float]


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


class LowVarianceFilter(BaseEstimator, TransformerMixin):
	def __init__(self, threshold: float = LOW_VARIANCE_THRESHOLD) -> None:
		self.threshold = float(threshold)

	def fit(self, X: Any, y: Any = None) -> "LowVarianceFilter":
		array = np.asarray(X, dtype=float)
		if array.ndim != 2:
			raise ValueError(f"LowVarianceFilter expects a 2D array. Got shape={array.shape}.")
		self.variances_ = np.var(array, axis=0).astype(float)
		self.support_mask_ = self.variances_ > self.threshold
		if not bool(np.any(self.support_mask_)):
			raise ValueError("All transformed RSF features were removed by the low-variance filter.")
		return self

	def transform(self, X: Any) -> np.ndarray:
		array = np.asarray(X, dtype=float)
		if array.ndim != 2:
			raise ValueError(f"LowVarianceFilter expects a 2D array. Got shape={array.shape}.")
		return array[:, self.support_mask_]

	def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
		if input_features is None:
			raise ValueError("LowVarianceFilter requires input_features to resolve output feature names.")
		input_feature_array = np.asarray(input_features, dtype=object)
		return input_feature_array[self.support_mask_]

	def get_dropped_feature_names(self, input_features: list[str]) -> list[str]:
		input_feature_array = np.asarray(input_features, dtype=object)
		return input_feature_array[~self.support_mask_].astype(str).tolist()


NumericPrefillTransformer.__module__ = MODULE_IMPORT_NAME
LowVarianceFilter.__module__ = MODULE_IMPORT_NAME


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

	temp_view_name = "__stage_d_5_9_materialize_df__"
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
	required_columns = ["enrollment_id", "event", "duration"]
	missing_columns = [column for column in required_columns if column not in df.columns]
	if missing_columns:
		raise KeyError(f"{dataset_name} is missing required columns: {', '.join(missing_columns)}")

	grouped_df = (
		df.groupby("enrollment_id", as_index=False)
		.agg(
			event_min=("event", "min"),
			event_max=("event", "max"),
			duration_min=("duration", "min"),
			duration_max=("duration", "max"),
		)
		.sort_values("enrollment_id")
		.reset_index(drop=True)
	)
	inconsistent_event_ids = grouped_df.loc[grouped_df["event_min"] != grouped_df["event_max"], "enrollment_id"]
	inconsistent_duration_ids = grouped_df.loc[
		grouped_df["duration_min"] != grouped_df["duration_max"], "enrollment_id"
	]
	if not inconsistent_event_ids.empty:
		raise ValueError(
			f"{dataset_name} contains enrollment_id values with inconsistent event labels: "
			f"{', '.join(inconsistent_event_ids.astype(str).tolist()[:10])}"
		)
	if not inconsistent_duration_ids.empty:
		raise ValueError(
			f"{dataset_name} contains enrollment_id values with inconsistent durations: "
			f"{', '.join(inconsistent_duration_ids.astype(str).tolist()[:10])}"
		)

	truth_df = grouped_df.loc[:, ["enrollment_id", "event_max", "duration_max"]].rename(
		columns={"event_max": "event", "duration_max": "duration"}
	)
	truth_df["enrollment_id"] = truth_df["enrollment_id"].astype(str)
	truth_df["event"] = ensure_binary_target(truth_df["event"], f"{dataset_name}.event")
	truth_df["duration"] = pd.to_numeric(truth_df["duration"], errors="raise").astype(int)
	if truth_df["duration"].lt(0).any():
		raise ValueError(f"{dataset_name} contains negative enrollment durations.")
	return truth_df


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


def initialize_context(window_weeks: int) -> PipelineContext:
	from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
	from util import ensure_pipeline_catalog, open_duckdb_connection

	log_stage_start("5.9.1", "Lightweight runtime bootstrap")

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
		"enrollment_cox_ready_train",
		canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
		active_window_weeks=effective_window_weeks,
	)
	input_test_table = resolve_variant_table_name(
		"enrollment_cox_ready_test",
		canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
		active_window_weeks=effective_window_weeks,
	)

	con = open_duckdb_connection(duckdb_path)
	ensure_pipeline_catalog(con)
	require_tables(con, [input_train_table, input_test_table], block_number="5.9.1")

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
				"early_window_weeks": ctx.early_window_weeks,
				"main_enrollment_window_weeks": ctx.main_enrollment_window_weeks,
				"effective_window_weeks": ctx.effective_window_weeks,
				"input_train_table": ctx.input_train_table,
				"input_test_table": ctx.input_test_table,
			},
			indent=2,
		)
	)

	log_stage_end("5.9.1")
	return ctx


def build_rsf_treatment(ctx: PipelineContext) -> RSFTreatmentState:
	log_stage_start("5.9.2", "Build comparable-window RSF treatment")

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
			"The comparable RSF numeric feature specification does not match the canonical modeling contract. "
			f"Contract numeric features: {expected_comparable_numeric_features}. Operational numeric features: {numeric_features}."
		)

	required_columns = comparable_required_columns(BASE_REQUIRED_CONTINUOUS_COLUMNS, ctx.effective_window_weeks)

	train_df = load_required_table(ctx, ctx.input_train_table, required_columns, block_number="5.9.2")
	test_df = load_required_table(ctx, ctx.input_test_table, required_columns, block_number="5.9.2")
	train_df = train_df.copy()
	test_df = test_df.copy()
	train_df["enrollment_id"] = train_df["enrollment_id"].astype(str)
	test_df["enrollment_id"] = test_df["enrollment_id"].astype(str)
	train_df["event"] = ensure_binary_target(train_df["event"], f"{ctx.input_train_table}.event")
	test_df["event"] = ensure_binary_target(test_df["event"], f"{ctx.input_test_table}.event")
	train_df["duration"] = pd.to_numeric(train_df["duration"], errors="raise").astype(int)
	test_df["duration"] = pd.to_numeric(test_df["duration"], errors="raise").astype(int)

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
			("low_variance_filter", LowVarianceFilter(threshold=LOW_VARIANCE_THRESHOLD)),
		]
	)

	X_train = np.asarray(preprocessor.fit_transform(train_df[operational_feature_columns]), dtype=np.float32)
	X_test = np.asarray(preprocessor.transform(test_df[operational_feature_columns]), dtype=np.float32)
	if X_train.ndim != 2 or X_test.ndim != 2:
		raise ValueError(f"The comparable RSF transformed matrices must be 2D. Got train={X_train.shape}, test={X_test.shape}.")
	if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
		raise ValueError("The comparable RSF transformed matrices contain non-finite values.")

	fitted_column_transformer: ColumnTransformer = preprocessor.named_steps["column_transformer"]
	fitted_low_variance_filter: LowVarianceFilter = preprocessor.named_steps["low_variance_filter"]
	feature_names_before_filter = fitted_column_transformer.get_feature_names_out().astype(str).tolist()
	feature_names_out = preprocessor.get_feature_names_out().astype(str).tolist()
	dropped_feature_names = fitted_low_variance_filter.get_dropped_feature_names(feature_names_before_filter)
	retained_variances = fitted_low_variance_filter.variances_[fitted_low_variance_filter.support_mask_].astype(float).tolist()

	preprocessing_summary_df = pd.DataFrame(
		[
			{
				"model_family": "random_survival_forest",
				"train_rows": int(X_train.shape[0]),
				"test_rows": int(X_test.shape[0]),
				"n_input_features_raw": int(len(operational_feature_columns)),
				"n_numeric_features": int(len(numeric_features)),
				"n_categorical_features": int(len(CATEGORICAL_FEATURES)),
				"n_features_after_encoding_before_filter": int(len(feature_names_before_filter)),
				"n_features_after_variance_filter": int(len(feature_names_out)),
				"n_features_dropped_by_variance_filter": int(len(dropped_feature_names)),
				"n_events_train": int(train_df["event"].sum()),
				"n_events_test": int(test_df["event"].sum()),
				"mean_duration_train": float(train_df["duration"].mean()),
				"mean_duration_test": float(test_df["duration"].mean()),
				"numeric_prefill_policy": "deterministic_train_median_or_zero_when_all_missing",
				"categorical_imputation": "constant_missing",
				"categorical_encoding": "one_hot_handle_unknown_ignore",
				"numeric_scaling": "standard_scaler_fit_on_train_only",
				"variance_filter_threshold": float(LOW_VARIANCE_THRESHOLD),
				"output_dtype": "float32",
				"rsf_positioning_note": "early-window comparable random survival forest benchmark",
				"window_weeks": int(ctx.effective_window_weeks),
			}
		]
	)
	raw_feature_manifest_df = pd.DataFrame(
		{
			"feature_name_operational": operational_feature_columns,
			"feature_role": ["categorical" if column in CATEGORICAL_FEATURES else "numeric" for column in operational_feature_columns],
			"present_in_train": [column in train_df.columns for column in operational_feature_columns],
			"present_in_test": [column in test_df.columns for column in operational_feature_columns],
			"covered_by_canonical_config": [column in expected_features_resolved for column in operational_feature_columns],
			"numeric_prefill_value": [numeric_fill_values.get(column, np.nan) for column in operational_feature_columns],
		}
	)
	output_feature_manifest_df = pd.DataFrame(
		{
			"feature_name_out": feature_names_out,
			"transformed_variance_train": retained_variances,
		}
	)

	materialize_dataframe_table(
		ctx,
		df=preprocessing_summary_df,
		table_name=artifact_name(ctx, "table_rsf_preprocessing_summary"),
		block_number="5.9.2",
		label="Stage 5.9.2 table_rsf_preprocessing_summary - RSF preprocessing summary",
	)
	materialize_dataframe_table(
		ctx,
		df=raw_feature_manifest_df,
		table_name=artifact_name(ctx, "table_rsf_raw_feature_manifest"),
		block_number="5.9.2",
		label="Stage 5.9.2 table_rsf_raw_feature_manifest - RSF raw feature manifest",
	)
	materialize_dataframe_table(
		ctx,
		df=output_feature_manifest_df,
		table_name=artifact_name(ctx, "table_rsf_feature_names_out"),
		block_number="5.9.2",
		label="Stage 5.9.2 table_rsf_feature_names_out - RSF output feature manifest",
	)

	preprocessing_config = {
		"model_family": "random_survival_forest",
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
		"variance_filter_threshold": float(LOW_VARIANCE_THRESHOLD),
		"feature_names_before_filter": feature_names_before_filter,
		"feature_names_out": feature_names_out,
		"dropped_low_variance_features": dropped_feature_names,
		"rsf_positioning_note": "early-window comparable random survival forest benchmark",
		"window_weeks": int(ctx.effective_window_weeks),
	}
	preprocessing_config_path = ctx.metadata_dir / artifact_filename(ctx, "rsf_preprocessing_config.json")
	save_json(preprocessing_config, preprocessing_config_path)
	print_artifact("rsf_preprocessing_config", str(preprocessing_config_path))
	print(preprocessing_config_path.read_text(encoding="utf-8"))

	log_stage_end("5.9.2")
	return RSFTreatmentState(
		train_df=train_df,
		test_df=test_df,
		feature_columns=operational_feature_columns,
		expected_features_raw=expected_features_raw,
		expected_features_resolved=expected_features_resolved,
		numeric_fill_values=numeric_fill_values,
		preprocessor=preprocessor,
		X_train=X_train,
		X_test=X_test,
		feature_names_before_filter=feature_names_before_filter,
		feature_names_out=feature_names_out,
		dropped_feature_names=dropped_feature_names,
		transformed_variances=retained_variances,
	)


def align_feature_matrix(
	matrix: np.ndarray,
	feature_names: list[str],
	raw_enrollment_ids: list[str],
	ordered_enrollment_ids: list[str],
	matrix_name: str,
) -> np.ndarray:
	if len(raw_enrollment_ids) != len(set(raw_enrollment_ids)):
		raise ValueError(f"{matrix_name} contains duplicate enrollment_id values and cannot be aligned deterministically.")
	aligned_df = pd.DataFrame(matrix, index=raw_enrollment_ids, columns=feature_names).reindex(ordered_enrollment_ids)
	if aligned_df.isna().any().any():
		raise ValueError(f"{matrix_name} contains missing values after enrollment alignment.")
	return aligned_df.to_numpy(dtype=np.float32)


def evaluate_survival_functions_on_grid(functions: list[Any], time_grid: np.ndarray) -> np.ndarray:
	survival_columns: list[np.ndarray] = []
	for function in functions:
		event_times = np.asarray(function.x, dtype=float)
		event_survival = np.asarray(function.y, dtype=float)
		if event_times.ndim != 1 or event_survival.ndim != 1 or event_times.shape[0] != event_survival.shape[0]:
			raise ValueError("Random Survival Forest returned an invalid survival step function.")
		insertion_points = np.searchsorted(event_times, time_grid, side="right") - 1
		survival_values = np.ones(time_grid.shape[0], dtype=float)
		valid_mask = insertion_points >= 0
		survival_values[valid_mask] = event_survival[insertion_points[valid_mask]]
		if not np.isfinite(survival_values).all():
			raise ValueError("Random Survival Forest produced non-finite survival values on the evaluation grid.")
		survival_columns.append(np.clip(survival_values, 0.0, 1.0))
	return np.column_stack(survival_columns)


def surv_functions_to_frame(functions: list[Any], enrollment_ids: list[str], time_grid: np.ndarray) -> pd.DataFrame:
	if time_grid.ndim != 1:
		raise ValueError("time_grid must be one-dimensional.")
	if len(enrollment_ids) != len(functions):
		raise ValueError("The number of enrollment IDs must match the number of survival functions.")
	survival_matrix = evaluate_survival_functions_on_grid(functions, time_grid)
	survival_wide_df = pd.DataFrame(survival_matrix, index=pd.Index(time_grid.astype(int), name="week"), columns=enrollment_ids)
	if survival_wide_df.isna().any().any():
		raise ValueError("The survival surface contains missing values after tabulation.")
	return survival_wide_df


def score_continuous_candidate(
	survival_wide_df: pd.DataFrame,
	truth_train_df: pd.DataFrame,
	truth_test_df: pd.DataFrame,
) -> tuple[float, float]:
	durations_train = truth_train_df["duration"].astype(float).to_numpy()
	events_train = truth_train_df["event"].astype(bool).to_numpy()
	y_train_surv = Surv.from_arrays(event=events_train, time=durations_train)
	y_test_surv = Surv.from_arrays(
		event=truth_test_df["event"].astype(bool).to_numpy(),
		time=truth_test_df["duration"].astype(float).to_numpy(),
	)
	durations_test = truth_test_df["duration"].astype(int).to_numpy()
	events_test = truth_test_df["event"].astype(int).to_numpy()

	eval_surv = EvalSurv(
		surv=survival_wide_df,
		durations=durations_test,
		events=events_test,
		censor_surv="km",
	)
	brier_time_grid = np.arange(1, max(BENCHMARK_HORIZONS) + 1, dtype=int)
	survival_estimate_matrix = survival_wide_df.loc[brier_time_grid, truth_test_df["enrollment_id"]].to_numpy(dtype=float).T
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
	return integrated_brier, float(eval_surv.concordance_td())


def tune_and_evaluate_rsf_model(ctx: PipelineContext, treatment: RSFTreatmentState) -> None:
	log_stage_start("5.9.3", "Tune and evaluate the comparable-window Random Survival Forest")

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

	subtrain_raw_ids = treatment.train_df.iloc[train_internal_idx]["enrollment_id"].astype(str).tolist()
	val_raw_ids = treatment.train_df.iloc[val_internal_idx]["enrollment_id"].astype(str).tolist()
	truth_subtrain_df = build_truth_by_enrollment(treatment.train_df.iloc[train_internal_idx], "rsf_validation_subtrain")
	truth_val_df = build_truth_by_enrollment(treatment.train_df.iloc[val_internal_idx], "rsf_validation_holdout")
	X_subtrain = align_feature_matrix(
		treatment.X_train[train_internal_idx],
		treatment.feature_names_out,
		subtrain_raw_ids,
		truth_subtrain_df["enrollment_id"].astype(str).tolist(),
		matrix_name="RSF subtrain matrix",
	)
	X_val = align_feature_matrix(
		treatment.X_train[val_internal_idx],
		treatment.feature_names_out,
		val_raw_ids,
		truth_val_df["enrollment_id"].astype(str).tolist(),
		matrix_name="RSF validation matrix",
	)
	y_subtrain = Surv.from_arrays(
		event=truth_subtrain_df["event"].astype(bool).to_numpy(),
		time=truth_subtrain_df["duration"].astype(float).to_numpy(),
	)

	tuning_rows: list[dict[str, Any]] = []
	best_candidate: dict[str, Any] | None = None
	for candidate in RSF_TUNING_GRID:
		candidate_model = RandomSurvivalForest(
			n_estimators=int(candidate["n_estimators"]),
			min_samples_leaf=int(candidate["min_samples_leaf"]),
			max_depth=candidate["max_depth"],
			max_features=candidate["max_features"],
			n_jobs=ctx.cpu_cores,
			random_state=ctx.random_seed,
		)
		candidate_model.fit(X_subtrain, y_subtrain)
		val_time_grid = np.arange(0, int(max(truth_val_df["duration"].max(), max(BENCHMARK_HORIZONS))) + 1, dtype=float)
		val_survival_df = surv_functions_to_frame(
			list(candidate_model.predict_survival_function(X_val)),
			truth_val_df["enrollment_id"].astype(str).tolist(),
			val_time_grid,
		)
		val_ibs, val_c_index = score_continuous_candidate(val_survival_df, truth_subtrain_df, truth_val_df)
		candidate_row = {
			"candidate_id": int(candidate["candidate_id"]),
			"n_estimators": int(candidate["n_estimators"]),
			"min_samples_leaf": int(candidate["min_samples_leaf"]),
			"max_depth": -1 if candidate["max_depth"] is None else int(candidate["max_depth"]),
			"max_features": str(candidate["max_features"]),
			"val_ibs": float(val_ibs),
			"val_c_index": float(val_c_index),
		}
		tuning_rows.append(candidate_row)
		if best_candidate is None or float(val_ibs) < float(best_candidate["val_ibs"]):
			best_candidate = {
				"candidate_id": int(candidate["candidate_id"]),
				"params": {
					"n_estimators": int(candidate["n_estimators"]),
					"min_samples_leaf": int(candidate["min_samples_leaf"]),
					"max_depth": candidate["max_depth"],
					"max_features": candidate["max_features"],
				},
				"val_ibs": float(val_ibs),
				"val_c_index": float(val_c_index),
			}

	if best_candidate is None:
		raise RuntimeError("The Random Survival Forest tuning grid did not produce a best candidate.")

	tuning_results_df = pd.DataFrame(tuning_rows).sort_values(
		["val_ibs", "candidate_id"],
		ascending=[True, True],
		kind="mergesort",
	).reset_index(drop=True)

	X_train_full = align_feature_matrix(
		treatment.X_train,
		treatment.feature_names_out,
		treatment.train_df["enrollment_id"].astype(str).tolist(),
		ordered_train_enrollment_ids,
		matrix_name="RSF full-train matrix",
	)
	X_test_aligned = align_feature_matrix(
		treatment.X_test,
		treatment.feature_names_out,
		treatment.test_df["enrollment_id"].astype(str).tolist(),
		ordered_test_enrollment_ids,
		matrix_name="RSF test matrix",
	)
	y_train_full = Surv.from_arrays(
		event=truth_train_df["event"].astype(bool).to_numpy(),
		time=truth_train_df["duration"].astype(float).to_numpy(),
	)
	y_test_surv = Surv.from_arrays(
		event=truth_test_df["event"].astype(bool).to_numpy(),
		time=truth_test_df["duration"].astype(float).to_numpy(),
	)

	best_model = RandomSurvivalForest(
		n_jobs=ctx.cpu_cores,
		random_state=ctx.random_seed,
		**best_candidate["params"],
	)
	best_model.fit(X_train_full, y_train_full)

	rsf_time_grid = np.arange(0, int(max(truth_test_df["duration"].max(), max(BENCHMARK_HORIZONS))) + 1, dtype=float)
	survival_wide_df = surv_functions_to_frame(
		list(best_model.predict_survival_function(X_test_aligned)),
		ordered_test_enrollment_ids,
		rsf_time_grid,
	)
	if survival_wide_df.isna().any().any():
		raise ValueError("The tuned RSF survival surface contains missing values after enrollment alignment.")

	durations_test = truth_test_df["duration"].astype(int).to_numpy()
	events_test = truth_test_df["event"].astype(int).to_numpy()
	eval_surv = EvalSurv(
		surv=survival_wide_df,
		durations=durations_test,
		events=events_test,
		censor_surv="km",
	)
	brier_time_grid = np.arange(1, max(BENCHMARK_HORIZONS) + 1, dtype=int)
	survival_estimate_matrix = survival_wide_df.loc[brier_time_grid, ordered_test_enrollment_ids].to_numpy(dtype=float).T
	brier_times, brier_scores = sksurv_brier_score(
		y_train_full,
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
			raise ValueError(f"The tuned RSF predictions contain missing values at horizon {horizon_week}.")
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
		ensure_binary_target(evaluable_df["observed_event_by_h"], f"observed_event_by_h@{horizon_week}")
		if evaluable_df["observed_event_by_h"].nunique() < 2:
			raise ValueError(
				f"The evaluable subset at horizon {horizon_week} does not contain both event classes for risk AUC."
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
			y_train_full,
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
		raise ValueError("Could not determine a valid censoring-support horizon for the comparable RSF model.")
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
		y_test_supported_surv = Surv.from_arrays(
			event=supported_df["event"].astype(bool).to_numpy(),
			time=supported_df["duration"].astype(float).to_numpy(),
		)
		horizon_auc = compute_ipcw_time_dependent_auc(
			y_train_full,
			y_test_supported_surv,
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
				"notes": "IPCW-supported subset used for RSF time-dependent AUC",
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
		raise ValueError("The tuned RSF event-time risk diagnostics contain non-finite values.")
	if truth_test_df["event"].nunique() < 2:
		raise ValueError("The RSF test truth does not contain both event classes for row-level diagnostics.")

	row_diagnostics_df = pd.DataFrame(
		[
			{
				"model_name": MODEL_NAME,
				"row_level_roc_auc": float(roc_auc_score(truth_test_df["event"], row_pred_risk)),
				"row_level_pr_auc": float(average_precision_score(truth_test_df["event"], row_pred_risk)),
				"row_level_log_loss": float(log_loss(truth_test_df["event"], np.clip(row_pred_risk, 1e-8, 1.0 - 1e-8), labels=[0, 1])),
				"row_level_brier": float(np.mean((row_pred_risk - truth_test_df["event"].to_numpy(dtype=float)) ** 2)),
				"diagnostic_note": "auxiliary enrollment-level proxy only; not a primary cross-family benchmark metric",
			}
		]
	)

	model_path = ctx.models_dir / artifact_filename(ctx, "rsf_tuned.joblib")
	preprocessor_path = ctx.models_dir / artifact_filename(ctx, "rsf_preprocessor.joblib")
	config_path = ctx.metadata_dir / artifact_filename(ctx, "rsf_tuned_model_config.json")
	joblib.dump(best_model, model_path)
	joblib.dump(treatment.preprocessor, preprocessor_path)

	materialize_dataframe_table(
		ctx,
		df=tuning_results_df,
		table_name=artifact_name(ctx, "table_rsf_tuning_results"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuning_results - RSF tuning results",
	)
	materialize_dataframe_table(
		ctx,
		df=test_predictions_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_test_predictions"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_test_predictions - RSF test predictions",
	)
	materialize_dataframe_table(
		ctx,
		df=primary_metrics_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_primary_metrics"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_primary_metrics - RSF primary metrics",
	)
	materialize_dataframe_table(
		ctx,
		df=brier_by_horizon_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_brier_by_horizon"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_brier_by_horizon - RSF Brier scores by horizon",
	)
	materialize_dataframe_table(
		ctx,
		df=secondary_metrics_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_secondary_metrics"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_secondary_metrics - RSF secondary metrics",
	)
	materialize_dataframe_table(
		ctx,
		df=td_auc_support_audit_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_td_auc_support_audit"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_td_auc_support_audit - RSF IPCW support audit",
	)
	materialize_dataframe_table(
		ctx,
		df=row_diagnostics_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_row_diagnostics"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_row_diagnostics - RSF row-level diagnostics",
	)
	materialize_dataframe_table(
		ctx,
		df=support_by_horizon_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_support_by_horizon"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_support_by_horizon - RSF support by horizon",
	)
	materialize_dataframe_table(
		ctx,
		df=calibration_summary_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_calibration_summary"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_calibration_summary - RSF calibration summary",
	)
	materialize_dataframe_table(
		ctx,
		df=calibration_bins_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_calibration_bins_by_horizon"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_calibration_bins_by_horizon - RSF calibration bins",
	)
	materialize_dataframe_table(
		ctx,
		df=predicted_vs_observed_survival_df,
		table_name=artifact_name(ctx, "table_rsf_tuned_predicted_vs_observed_survival"),
		block_number="5.9.3",
		label="Stage 5.9.3 table_rsf_tuned_predicted_vs_observed_survival - RSF predicted versus observed survival",
	)

	config_payload = {
		"model_name": MODEL_NAME,
		"selection_metric": "lowest_validation_ibs",
		"validation_split": {
			"method": "GroupShuffleSplit",
			"test_size": float(VALIDATION_TEST_SIZE),
			"group_column": "enrollment_id",
			"random_seed": int(ctx.random_seed),
		},
		"search_space": [
			{
				"candidate_id": int(candidate["candidate_id"]),
				"n_estimators": int(candidate["n_estimators"]),
				"min_samples_leaf": int(candidate["min_samples_leaf"]),
				"max_depth": candidate["max_depth"],
				"max_features": candidate["max_features"],
			}
			for candidate in RSF_TUNING_GRID
		],
		"benchmark_horizons": list(BENCHMARK_HORIZONS),
		"calibration_bins": int(CALIBRATION_BINS),
		"calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": CALIBRATION_OBSERVED_RISK_METHOD,
		"variance_filter_threshold": float(LOW_VARIANCE_THRESHOLD),
		"n_jobs": int(ctx.cpu_cores),
		"effective_window_weeks": int(ctx.effective_window_weeks),
		"input_train_table": ctx.input_train_table,
		"input_test_table": ctx.input_test_table,
		"best_candidate": best_candidate,
		"preprocessing": {
			"categorical_features": CATEGORICAL_FEATURES,
			"numeric_features": treatment.feature_columns[len(CATEGORICAL_FEATURES):],
			"numeric_prefill_values": treatment.numeric_fill_values,
			"feature_names_before_filter": treatment.feature_names_before_filter,
			"feature_names_out": treatment.feature_names_out,
			"dropped_low_variance_features": treatment.dropped_feature_names,
		},
		"design_note": (
			"The comparable Random Survival Forest arm preserves the canonical early-window contract, "
			"uses the same deterministic preprocessing as the linear continuous-time baselines, "
			"and tunes only a compact notebook-traceable forest grid under validation IBS."
		),
	}
	save_json(config_payload, config_path)
	print_artifact("rsf_model", str(model_path))
	print_artifact("rsf_preprocessor", str(preprocessor_path))
	print_artifact("rsf_tuned_model_config", str(config_path))
	print(config_path.read_text(encoding="utf-8"))
	print(pd.DataFrame([best_candidate]).to_string(index=False))

	log_stage_end("5.9.3")


def close_context(ctx: PipelineContext) -> None:
	from util import close_duckdb_connection

	log_stage_start("5.9.4", "Close runtime resources")
	close_duckdb_connection(ctx.con)
	print_artifact("duckdb_connection", f"closed::{ctx.duckdb_path}")
	log_stage_end("5.9.4")


def resolve_execution_windows() -> list[int]:
	contract_path = PROJECT_ROOT / "benchmark_modeling_contract.toml"
	with contract_path.open("rb") as handle:
		contract = toml_reader.load(handle)
	benchmark_config = contract.get("benchmark", {})
	return resolve_comparable_window_execution_plan(benchmark_config)


def run_single_window(window_weeks: int) -> None:
	ctx: PipelineContext | None = None
	try:
		ctx = initialize_context(window_weeks)
		treatment = build_rsf_treatment(ctx)
		tune_and_evaluate_rsf_model(ctx, treatment)
	finally:
		if ctx is not None:
			close_context(ctx)


def main() -> None:
	execution_windows = resolve_execution_windows()
	for window_weeks in execution_windows:
		print(f"[RUN_PLAN] comparable_window_weeks={int(window_weeks)}")
		run_single_window(int(window_weeks))


if __name__ == "__main__":
	main()
