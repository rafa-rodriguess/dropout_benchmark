from __future__ import annotations

"""
Production XGBoost AFT benchmark module for stage D5.13.

What this file does:
- prepares the comparable-window continuous-time treatment directly from DuckDB-ready enrollment tables
- applies deterministic preprocessing and feature validation for the XGBoost AFT model
- tunes and fits the XGBoost AFT benchmark under the official grouped validation protocol
- evaluates the tuned model with survival, calibration, support, time-dependent AUC, and auxiliary row-level diagnostics
- persists the trained model, fitted preprocessing pipeline, DuckDB audit tables, and JSON metadata artifacts

Main processing purpose:
- materialize the full comparable XGBoost AFT benchmark arm deterministically from DuckDB-ready tables without notebook runtime state, CSV workflows, or hidden execution state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_xgb_aft_preprocessing_summary, table_xgb_aft_raw_feature_manifest,
	table_xgb_aft_feature_names_out, table_xgb_aft_guardrails, table_xgb_aft_tuning_results,
  table_xgb_aft_tuned_test_predictions, table_xgb_aft_tuned_primary_metrics,
  table_xgb_aft_tuned_brier_by_horizon, table_xgb_aft_tuned_secondary_metrics,
  table_xgb_aft_tuned_td_auc_support_audit, table_xgb_aft_tuned_row_diagnostics,
  table_xgb_aft_tuned_support_by_horizon, table_xgb_aft_tuned_calibration_summary,
  table_xgb_aft_tuned_calibration_bins_by_horizon,
  table_xgb_aft_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/xgb_aft_preprocessing_config.json
- outputs_benchmark_survival/metadata/xgb_aft_tuned_model_config.json
- outputs_benchmark_survival/models/xgb_aft_tuned.joblib
- outputs_benchmark_survival/models/xgb_aft_preprocessor.joblib

Main DuckDB tables used as inputs:
- enrollment_cox_ready_train
- enrollment_cox_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_xgb_aft_preprocessing_summary
- table_xgb_aft_raw_feature_manifest
- table_xgb_aft_feature_names_out
- table_xgb_aft_guardrails
- table_xgb_aft_tuning_results
- table_xgb_aft_tuned_test_predictions
- table_xgb_aft_tuned_primary_metrics
- table_xgb_aft_tuned_brier_by_horizon
- table_xgb_aft_tuned_secondary_metrics
- table_xgb_aft_tuned_td_auc_support_audit
- table_xgb_aft_tuned_row_diagnostics
- table_xgb_aft_tuned_support_by_horizon
- table_xgb_aft_tuned_calibration_summary
- table_xgb_aft_tuned_calibration_bins_by_horizon
- table_xgb_aft_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/xgb_aft_preprocessing_config.json
- outputs_benchmark_survival/metadata/xgb_aft_tuned_model_config.json
- outputs_benchmark_survival/models/xgb_aft_tuned.joblib
- outputs_benchmark_survival/models/xgb_aft_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, notebook globals, or CSV-based workflows are permitted
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tomli as toml_reader
from pycox.evaluation import EvalSurv
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.util import Surv
import xgboost as xgb

import dropout_bench_v3_D_09_comparable_tree_survival_random_survival_forest as base


STAGE_PREFIX = "5.13"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_D_5_13.ipynb"
MODEL_NAME = "xgb_aft_tuned"
MODEL_IMPLEMENTATION = "xgboost_survival_aft"
PREVIEW_ROWS = 20
BENCHMARK_HORIZONS = base.BENCHMARK_HORIZONS
CALIBRATION_BINS = base.CALIBRATION_BINS
VALIDATION_TEST_SIZE = 0.10
LOW_VARIANCE_THRESHOLD = 1e-12
XGB_AFT_DURATION_FLOOR = 1.0
XGB_AFT_SEARCH_ROW_CAP = 50000
XGB_AFT_FINAL_ROW_CAP = 50000
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
XGB_AFT_TUNING_GRID = (
	{
		"candidate_id": 1,
		"aft_loss_distribution": "logistic",
		"aft_loss_distribution_scale": 1.0,
		"learning_rate": 0.05,
		"num_boost_round": 200,
		"max_depth": 2,
		"min_child_weight": 1.0,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_lambda": 1.0,
		"reg_alpha": 0.0,
	},
	{
		"candidate_id": 2,
		"aft_loss_distribution": "normal",
		"aft_loss_distribution_scale": 1.0,
		"learning_rate": 0.05,
		"num_boost_round": 200,
		"max_depth": 2,
		"min_child_weight": 1.0,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_lambda": 1.0,
		"reg_alpha": 0.0,
	},
	{
		"candidate_id": 3,
		"aft_loss_distribution": "logistic",
		"aft_loss_distribution_scale": 1.5,
		"learning_rate": 0.05,
		"num_boost_round": 300,
		"max_depth": 2,
		"min_child_weight": 1.0,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_lambda": 1.0,
		"reg_alpha": 0.0,
	},
	{
		"candidate_id": 4,
		"aft_loss_distribution": "normal",
		"aft_loss_distribution_scale": 1.5,
		"learning_rate": 0.03,
		"num_boost_round": 400,
		"max_depth": 2,
		"min_child_weight": 2.0,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_lambda": 5.0,
		"reg_alpha": 0.0,
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
class XGBAFTTreatmentState:
	train_df: pd.DataFrame
	test_df: pd.DataFrame
	feature_columns: list[str]
	expected_features_raw: list[str]
	expected_features_resolved: list[str]
	numeric_fill_values: dict[str, float]
	preprocessor: Any
	X_train: np.ndarray
	X_test: np.ndarray
	feature_names_before_filter: list[str]
	feature_names_out: list[str]
	dropped_feature_names: list[str]
	transformed_variances: list[float]
	truth_train_df: pd.DataFrame
	truth_test_df: pd.DataFrame


@dataclass
class XGBAFTGuardrailState:
	X_subtrain: np.ndarray
	X_val: np.ndarray
	X_train_full: np.ndarray
	X_test_aligned: np.ndarray
	truth_subtrain_df: pd.DataFrame
	truth_val_df: pd.DataFrame
	subtrain_sample_idx: np.ndarray
	full_sample_idx: np.ndarray
	summary: dict[str, Any]


def log_stage_start(block_number: str, title: str) -> None:
	print(f"[START] {block_number} - {base.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("# ==============================================================")
	print(f"# {block_number} - {title}")
	print("# ==============================================================")


def log_stage_end(block_number: str) -> None:
	print(f"[END] {block_number} - {base.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==============================================================
# 5.13.1 - Lightweight runtime bootstrap
# ==============================================================
# What this block does:
# - Initializes shared paths, configuration objects, run metadata, and the DuckDB connection for the D5.13 stage.
# Inputs:
# - files read: benchmark_shared_config.toml, benchmark_modeling_contract.toml, outputs_benchmark_survival/metadata/run_metadata.json
# - DuckDB tables read: enrollment_cox_ready_train, enrollment_cox_ready_test
# - configuration values used: paths.*, benchmark.seed, benchmark.test_size, benchmark.early_window_weeks, benchmark.main_enrollment_window_weeks
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: pipeline_table_catalog, vw_pipeline_table_catalog_schema
# - objects returned in memory: PipelineContext
def initialize_context(window_weeks: int) -> PipelineContext:
	from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
	from util import ensure_pipeline_catalog, open_duckdb_connection

	log_stage_start("5.13.1", "Lightweight runtime bootstrap")

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

	base.require_mapping_keys(shared_config, ["paths"], "benchmark_shared_config.toml")
	base.require_mapping_keys(shared_modeling_contract, REQUIRED_MODELING_KEYS, "benchmark_modeling_contract.toml")
	base.require_mapping_keys(run_metadata, ["run_id"], "run_metadata.json")

	paths_config = shared_config["paths"]
	base.require_mapping_keys(paths_config, REQUIRED_SHARED_PATH_KEYS, "benchmark_shared_config.toml [paths]")
	benchmark_config = shared_modeling_contract["benchmark"]
	base.require_mapping_keys(
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
	artifact_suffix = base.build_window_suffix(
		canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
		active_window_weeks=effective_window_weeks,
	)
	input_train_table = base.resolve_variant_table_name(
		"enrollment_cox_ready_train",
		canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
		active_window_weeks=effective_window_weeks,
	)
	input_test_table = base.resolve_variant_table_name(
		"enrollment_cox_ready_test",
		canonical_window_weeks=int(benchmark_config["main_enrollment_window_weeks"]),
		active_window_weeks=effective_window_weeks,
	)

	con = open_duckdb_connection(duckdb_path)
	ensure_pipeline_catalog(con)
	base.require_tables(con, [input_train_table, input_test_table], block_number="5.13.1")

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

	base.print_artifact("benchmark_shared_config", str(config_toml_path))
	base.print_artifact("benchmark_modeling_contract", str(modeling_contract_toml_path))
	base.print_artifact("run_metadata", str(run_metadata_path))
	base.print_artifact("duckdb_path", str(duckdb_path))
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

	log_stage_end("5.13.1")
	return ctx


# ==============================================================
# 5.13.2 - Build the comparable XGBoost AFT treatment
# ==============================================================
# What this block does:
# - Loads the comparable-arm continuous-time tables from DuckDB, validates the canonical feature contract, applies deterministic preprocessing, and records preprocessing audit outputs.
# Inputs:
# - files read: benchmark_modeling_contract.toml
# - DuckDB tables read: enrollment_cox_ready_train, enrollment_cox_ready_test
# - configuration values used: feature_contract.static_features, feature_contract.main_enrollment_window_features, feature_contract.optional_comparable_window_features, benchmark.main_enrollment_window_weeks
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/xgb_aft_preprocessing_config.json, outputs_benchmark_survival/models/xgb_aft_preprocessor.joblib
# - DuckDB tables created or updated: table_xgb_aft_preprocessing_summary, table_xgb_aft_raw_feature_manifest, table_xgb_aft_feature_names_out
# - objects returned in memory: XGBAFTTreatmentState
def build_xgb_aft_treatment(ctx: PipelineContext) -> XGBAFTTreatmentState:
	log_stage_start("5.13.2", "Build comparable XGBoost AFT treatment")

	feature_contract = ctx.shared_modeling_contract["feature_contract"]
	base.require_mapping_keys(feature_contract, REQUIRED_FEATURE_CONTRACT_KEYS, "benchmark_modeling_contract.toml [feature_contract]")

	static_features = base.require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
	canonical_main_window_features = base.require_list_of_strings(
		feature_contract["main_enrollment_window_features"],
		"feature_contract.main_enrollment_window_features",
	)
	canonical_optional_comparable_features = base.require_list_of_strings(
		feature_contract["optional_comparable_window_features"],
		"feature_contract.optional_comparable_window_features",
	)

	if int(ctx.effective_window_weeks) == int(ctx.main_enrollment_window_weeks):
		main_window_features = canonical_main_window_features
		optional_comparable_features = canonical_optional_comparable_features
	else:
		active_feature_names = base.comparable_window_feature_names(ctx.effective_window_weeks)
		main_window_features = list(STATIC_NUMERIC_FEATURES) + [active_feature_names["main_clicks_feature"]]
		optional_comparable_features = [
			active_feature_names["main_active_feature"],
			active_feature_names["main_mean_clicks_feature"],
		]

	expected_features_raw = static_features + main_window_features + optional_comparable_features
	expected_features_resolved = list(dict.fromkeys(expected_features_raw))
	numeric_features = base.comparable_numeric_features(
		ctx.effective_window_weeks,
		static_numeric_features=STATIC_NUMERIC_FEATURES,
	)
	feature_columns = CATEGORICAL_FEATURES + numeric_features
	expected_numeric_features = list(dict.fromkeys(main_window_features + optional_comparable_features))
	if sorted(expected_numeric_features) != sorted(numeric_features):
		raise ValueError(
			"The comparable XGBoost AFT numeric feature specification does not match the canonical modeling contract. "
			f"Contract numeric features: {expected_numeric_features}. Operational numeric features: {numeric_features}."
		)

	required_columns = base.comparable_required_columns(BASE_REQUIRED_CONTINUOUS_COLUMNS, ctx.effective_window_weeks)

	train_df = base.load_required_table(ctx, ctx.input_train_table, required_columns, block_number="5.13.2")
	test_df = base.load_required_table(ctx, ctx.input_test_table, required_columns, block_number="5.13.2")
	train_df = train_df.copy()
	test_df = test_df.copy()
	train_df["enrollment_id"] = train_df["enrollment_id"].astype(str)
	test_df["enrollment_id"] = test_df["enrollment_id"].astype(str)
	train_df["event"] = base.ensure_binary_target(train_df["event"], f"{ctx.input_train_table}.event")
	test_df["event"] = base.ensure_binary_target(test_df["event"], f"{ctx.input_test_table}.event")
	train_df["duration"] = pd.to_numeric(train_df["duration"], errors="raise").astype(int)
	test_df["duration"] = pd.to_numeric(test_df["duration"], errors="raise").astype(int)

	numeric_fill_values: dict[str, float] = {}
	for column_name in numeric_features:
		train_numeric = pd.to_numeric(train_df[column_name], errors="raise")
		observed_values = train_numeric.dropna()
		numeric_fill_values[column_name] = float(observed_values.median()) if not observed_values.empty else 0.0

	preprocessor = base.Pipeline(
		steps=[
			(
				"numeric_prefill",
				base.NumericPrefillTransformer(
					feature_columns=feature_columns,
					numeric_columns=numeric_features,
					fill_values=numeric_fill_values,
				),
			),
			(
				"column_transformer",
				base.ColumnTransformer(
					transformers=[
						(
							"num",
							base.Pipeline(
								steps=[("scaler", base.StandardScaler())]
							),
							numeric_features,
						),
						(
							"cat",
							base.Pipeline(
								steps=[
									("imputer", base.SimpleImputer(strategy="constant", fill_value="missing")),
									("onehot", base.OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
								]
							),
							CATEGORICAL_FEATURES,
						),
					],
					remainder="drop",
					verbose_feature_names_out=True,
				),
			),
			("low_variance_filter", base.LowVarianceFilter(threshold=LOW_VARIANCE_THRESHOLD)),
		]
	)

	X_train = np.asarray(preprocessor.fit_transform(train_df[feature_columns]), dtype=np.float32)
	X_test = np.asarray(preprocessor.transform(test_df[feature_columns]), dtype=np.float32)
	if X_train.ndim != 2 or X_test.ndim != 2:
		raise ValueError(f"The XGBoost AFT transformed matrices must be 2D. Got train={X_train.shape}, test={X_test.shape}.")
	if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
		raise ValueError("The XGBoost AFT transformed matrices contain non-finite values.")

	fitted_column_transformer = preprocessor.named_steps["column_transformer"]
	fitted_low_variance_filter = preprocessor.named_steps["low_variance_filter"]
	feature_names_before_filter = fitted_column_transformer.get_feature_names_out().astype(str).tolist()
	feature_names_out = preprocessor.get_feature_names_out().astype(str).tolist()
	dropped_feature_names = fitted_low_variance_filter.get_dropped_feature_names(feature_names_before_filter)
	retained_variances = fitted_low_variance_filter.variances_[fitted_low_variance_filter.support_mask_].astype(float).tolist()

	truth_train_df = base.build_truth_by_enrollment(train_df, ctx.input_train_table)
	truth_test_df = base.build_truth_by_enrollment(test_df, ctx.input_test_table)

	preprocessing_summary_df = pd.DataFrame(
		[
			{
				"model_family": "xgboost_aft",
				"train_rows": int(X_train.shape[0]),
				"test_rows": int(X_test.shape[0]),
				"n_input_features_raw": int(len(feature_columns)),
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
				"model_positioning_note": "early-window comparable XGBoost AFT benchmark",
				"window_weeks": int(ctx.effective_window_weeks),
			}
		]
	)
	raw_feature_manifest_df = pd.DataFrame(
		{
			"feature_name_operational": feature_columns,
			"feature_role": ["categorical" if column_name in CATEGORICAL_FEATURES else "numeric" for column_name in feature_columns],
			"present_in_train": [column_name in train_df.columns for column_name in feature_columns],
			"present_in_test": [column_name in test_df.columns for column_name in feature_columns],
			"covered_by_canonical_config": [column_name in expected_features_resolved for column_name in feature_columns],
			"numeric_prefill_value": [numeric_fill_values.get(column_name, np.nan) for column_name in feature_columns],
		}
	)
	output_feature_manifest_df = pd.DataFrame(
		{
			"feature_name_out": feature_names_out,
			"transformed_variance_train": retained_variances,
		}
	)

	base.materialize_dataframe_table(
		ctx,
		df=preprocessing_summary_df,
		table_name=base.artifact_name(ctx, "table_xgb_aft_preprocessing_summary"),
		block_number="5.13.2",
		label="Stage 5.13.2 table_xgb_aft_preprocessing_summary - XGBoost AFT preprocessing summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=raw_feature_manifest_df,
		table_name=base.artifact_name(ctx, "table_xgb_aft_raw_feature_manifest"),
		block_number="5.13.2",
		label="Stage 5.13.2 table_xgb_aft_raw_feature_manifest - XGBoost AFT raw feature manifest",
	)
	base.materialize_dataframe_table(
		ctx,
		df=output_feature_manifest_df,
		table_name=base.artifact_name(ctx, "table_xgb_aft_feature_names_out"),
		block_number="5.13.2",
		label="Stage 5.13.2 table_xgb_aft_feature_names_out - XGBoost AFT output feature manifest",
	)

	preprocessing_config = {
		"model_family": "xgboost_aft",
		"duration_column": "duration",
		"event_column": "event",
		"categorical_features": CATEGORICAL_FEATURES,
		"numeric_features": numeric_features,
		"raw_feature_columns": feature_columns,
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
		"model_positioning_note": "early-window comparable XGBoost AFT benchmark",
		"window_weeks": int(ctx.effective_window_weeks),
	}
	preprocessing_config_path = ctx.metadata_dir / base.artifact_filename(ctx, "xgb_aft_preprocessing_config.json")
	base.save_json(preprocessing_config, preprocessing_config_path)
	base.print_artifact("xgb_aft_preprocessing_config", str(preprocessing_config_path))
	print(preprocessing_config_path.read_text(encoding="utf-8"))

	log_stage_end("5.13.2")
	return XGBAFTTreatmentState(
		train_df=train_df,
		test_df=test_df,
		feature_columns=feature_columns,
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
		truth_train_df=truth_train_df,
		truth_test_df=truth_test_df,
	)


def evaluate_continuous_survival(
	model_name: str,
	survival_wide_df: pd.DataFrame,
	truth_train_df: pd.DataFrame,
	truth_test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
	durations_train = truth_train_df["duration"].astype(float).to_numpy()
	events_train = truth_train_df["event"].astype(bool).to_numpy()
	y_train_surv = Surv.from_arrays(event=events_train, time=durations_train)
	y_test_surv = Surv.from_arrays(
		event=truth_test_df["event"].astype(bool).to_numpy(),
		time=truth_test_df["duration"].astype(float).to_numpy(),
	)
	durations_test = truth_test_df["duration"].astype(int).to_numpy()
	events_test = truth_test_df["event"].astype(int).to_numpy()

	eval_surv = EvalSurv(surv=survival_wide_df, durations=durations_test, events=events_test, censor_surv="km")
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

	primary_metrics_df = pd.DataFrame(
		[
			{"metric_name": "ibs", "metric_category": "primary", "metric_value": integrated_brier, "notes": "sksurv_integrated_brier_score"},
			{"metric_name": "c_index", "metric_category": "co_primary", "metric_value": float(eval_surv.concordance_td()), "notes": "pycox_concordance_td"},
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
		pred_survival_h = base.get_survival_at_horizon(survival_wide_df, horizon_week)
		pred_risk_h = 1.0 - pred_survival_h

		horizon_predictions_df = truth_test_df.copy()
		horizon_predictions_df["horizon_week"] = int(horizon_week)
		horizon_predictions_df["pred_survival_h"] = pred_survival_h.reindex(truth_test_df["enrollment_id"]).to_numpy(dtype=float)
		horizon_predictions_df["pred_risk_h"] = pred_risk_h.reindex(truth_test_df["enrollment_id"]).to_numpy(dtype=float)
		if horizon_predictions_df[["pred_survival_h", "pred_risk_h"]].isna().any().any():
			raise ValueError(f"The tuned XGBoost AFT predictions contain missing values at horizon {horizon_week}.")
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
		base.ensure_binary_target(evaluable_df["observed_event_by_h"], f"observed_event_by_h@{horizon_week}")
		if evaluable_df["observed_event_by_h"].nunique() < 2:
			raise ValueError(f"The evaluable subset at horizon {horizon_week} does not contain both event classes for risk AUC.")

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
		calibration_table, predicted_vs_observed_row = base.build_ipcw_calibration_artifacts(
			y_train_surv,
			horizon_predictions_df,
			horizon_week,
			CALIBRATION_BINS,
		)
		predicted_vs_observed_rows.append(predicted_vs_observed_row)
		calibration_rows.extend(calibration_table.to_dict(orient="records"))

	test_predictions_df = pd.DataFrame(prediction_rows).sort_values(["horizon_week", "enrollment_id"]).reset_index(drop=True)
	support_by_horizon_df = pd.DataFrame(support_rows)
	risk_auc_by_horizon_df = pd.DataFrame(risk_auc_rows)
	predicted_vs_observed_survival_df = pd.DataFrame(predicted_vs_observed_rows)
	calibration_bins_df = pd.DataFrame(calibration_rows)

	calibration_summary_df = base.summarize_calibration_by_horizon(calibration_bins_df)

	max_ipcw_time = float(truth_train_df.loc[truth_train_df["event"] == 0, "duration"].max())
	if not np.isfinite(max_ipcw_time):
		raise ValueError("Could not determine a valid censoring-support horizon for the XGBoost AFT model.")
	td_auc_rows: list[dict[str, Any]] = []
	td_auc_audit_rows: list[dict[str, Any]] = []
	for horizon_week in BENCHMARK_HORIZONS:
		horizon_prediction_df = test_predictions_df.loc[test_predictions_df["horizon_week"] == int(horizon_week), ["enrollment_id", "pred_risk_h"]].copy()
		supported_df = truth_test_df.merge(horizon_prediction_df, on="enrollment_id", how="left")
		supported_df = supported_df.loc[(supported_df["duration"] < max_ipcw_time) & supported_df["pred_risk_h"].notna()].copy()
		if supported_df.empty:
			raise ValueError(f"No IPCW-supported evaluation rows are available at horizon {horizon_week}.")
		if supported_df["event"].nunique() < 2:
			raise ValueError(f"The IPCW-supported evaluation subset at horizon {horizon_week} does not contain both event classes.")
		y_test_supported_surv = Surv.from_arrays(
			event=supported_df["event"].astype(bool).to_numpy(),
			time=supported_df["duration"].astype(float).to_numpy(),
		)
		horizon_auc = base.compute_ipcw_time_dependent_auc(
			y_train_surv,
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
					f"ipcw_supported_subset(duration < {max_ipcw_time:.0f}); retained={supported_df.shape[0]}/{truth_test_df.shape[0]}; auc={horizon_auc:.6f}; local_ipcw_auc"
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
				"notes": "IPCW-supported subset used for XGBoost AFT time-dependent AUC",
			}
		)

	time_dependent_auc_df = pd.DataFrame(td_auc_rows)
	td_auc_support_audit_df = pd.DataFrame(td_auc_audit_rows)
	secondary_metrics_df = pd.concat([risk_auc_by_horizon_df, time_dependent_auc_df], ignore_index=True).sort_values(["horizon_week", "metric_name"]).reset_index(drop=True)

	row_pred_survival = np.asarray(
		[
			float(survival_wide_df.loc[int(max(float(duration), XGB_AFT_DURATION_FLOOR)), enrollment_id])
			for enrollment_id, duration in zip(truth_test_df["enrollment_id"], truth_test_df["duration"])
		],
		dtype=float,
	)
	row_pred_risk = 1.0 - row_pred_survival
	if not np.isfinite(row_pred_risk).all():
		raise ValueError("The tuned XGBoost AFT event-time risk diagnostics contain non-finite values.")
	if truth_test_df["event"].nunique() < 2:
		raise ValueError("The XGBoost AFT test truth does not contain both event classes for row-level diagnostics.")

	row_diagnostics_df = pd.DataFrame(
		[
			{
				"model_name": model_name,
				"row_level_roc_auc": float(roc_auc_score(truth_test_df["event"], row_pred_risk)),
				"row_level_pr_auc": float(average_precision_score(truth_test_df["event"], row_pred_risk)),
				"row_level_log_loss": float(log_loss(truth_test_df["event"], np.clip(row_pred_risk, 1e-8, 1.0 - 1e-8), labels=[0, 1])),
				"row_level_brier": float(np.mean((row_pred_risk - truth_test_df["event"].to_numpy(dtype=float)) ** 2)),
				"diagnostic_note": "auxiliary enrollment-level proxy only; not a primary cross-family benchmark metric",
			}
		]
	)

	return {
		"test_predictions_df": test_predictions_df,
		"primary_metrics_df": primary_metrics_df,
		"brier_by_horizon_df": brier_by_horizon_df,
		"secondary_metrics_df": secondary_metrics_df,
		"td_auc_support_audit_df": td_auc_support_audit_df,
		"row_diagnostics_df": row_diagnostics_df,
		"support_by_horizon_df": support_by_horizon_df,
		"calibration_summary_df": calibration_summary_df,
		"calibration_bins_df": calibration_bins_df,
		"predicted_vs_observed_survival_df": predicted_vs_observed_survival_df,
	}


# ==============================================================
# 5.13.3 - Compute runtime guardrails for gradient boosting
# ==============================================================
# What this block does:
# - Creates the grouped validation split, applies deterministic enrollment-level sampling for the boosting fit, and persists the guardrail audit table.
# Inputs:
# - files read: none
# - DuckDB tables read: enrollment_cox_ready_train
# - configuration values used: benchmark.seed, validation split policy, search row cap, final row cap
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: table_xgb_aft_guardrails
# - objects returned in memory: XGBAFTGuardrailState
def safe_xgb_aft_row_budget(n_rows: int, n_features: int, requested_cap: int) -> int:
	feature_adjusted_cap = max(12000, int(600000 / max(1, n_features)))
	return int(min(n_rows, requested_cap, feature_adjusted_cap))


def build_aft_label_bounds(truth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	duration = np.maximum(truth_df["duration"].astype(float).to_numpy(), XGB_AFT_DURATION_FLOOR)
	event = truth_df["event"].astype(int).to_numpy()
	lower_bound = duration.astype(np.float32)
	upper_bound = np.where(event == 1, duration, np.inf).astype(np.float32)
	return lower_bound, upper_bound


def build_aft_dmatrix(matrix: np.ndarray, truth_df: pd.DataFrame) -> xgb.DMatrix:
	dmatrix = xgb.DMatrix(np.asarray(matrix, dtype=np.float32))
	lower_bound, upper_bound = build_aft_label_bounds(truth_df)
	dmatrix.set_float_info("label_lower_bound", lower_bound)
	dmatrix.set_float_info("label_upper_bound", upper_bound)
	return dmatrix


def build_xgb_aft_params(candidate: dict[str, Any], random_seed: int, cpu_cores: int) -> dict[str, Any]:
	return {
		"objective": "survival:aft",
		"eval_metric": "aft-nloglik",
		"aft_loss_distribution": str(candidate["aft_loss_distribution"]),
		"aft_loss_distribution_scale": float(candidate["aft_loss_distribution_scale"]),
		"tree_method": "hist",
		"learning_rate": float(candidate["learning_rate"]),
		"max_depth": int(candidate["max_depth"]),
		"min_child_weight": float(candidate["min_child_weight"]),
		"subsample": float(candidate["subsample"]),
		"colsample_bytree": float(candidate["colsample_bytree"]),
		"lambda": float(candidate["reg_lambda"]),
		"alpha": float(candidate["reg_alpha"]),
		"nthread": int(cpu_cores),
		"verbosity": 0,
		"seed": int(random_seed),
	}


def aft_survival_probabilities(
	pred_location: np.ndarray,
	time_grid: np.ndarray,
	distribution: str,
	scale: float,
) -> np.ndarray:
	safe_time_grid = np.maximum(np.asarray(time_grid, dtype=float), XGB_AFT_DURATION_FLOOR)
	pred_location = np.asarray(pred_location, dtype=float)
	if pred_location.ndim != 1:
		raise ValueError("AFT location predictions must be one-dimensional.")
	z = (np.log(safe_time_grid)[:, None] - pred_location[None, :]) / float(scale)
	if distribution == "logistic":
		survival_matrix = np.exp(-np.logaddexp(0.0, z))
	elif distribution == "normal":
		erf_values = np.frompyfunc(math.erf, 1, 1)(z / np.sqrt(2.0)).astype(float)
		survival_matrix = 0.5 * (1.0 - erf_values)
	elif distribution == "extreme":
		survival_matrix = np.exp(-np.exp(z))
	else:
		raise ValueError(f"Unsupported AFT distribution: {distribution}")
	if not np.isfinite(survival_matrix).all():
		raise ValueError("The XGBoost AFT survival reconstruction produced non-finite values.")
	return np.clip(survival_matrix.astype(float), 0.0, 1.0)


def xgb_aft_survival_frame(
	booster: xgb.Booster,
	matrix: np.ndarray,
	enrollment_ids: list[str],
	time_grid: np.ndarray,
	distribution: str,
	scale: float,
) -> pd.DataFrame:
	dtest = xgb.DMatrix(np.asarray(matrix, dtype=np.float32))
	pred_location = booster.predict(dtest)
	survival_matrix = aft_survival_probabilities(pred_location, time_grid, distribution, scale)
	survival_wide_df = pd.DataFrame(survival_matrix, index=time_grid.astype(int), columns=enrollment_ids)
	survival_wide_df.columns.name = "enrollment_id"
	survival_wide_df.index.name = "week"
	if survival_wide_df.isna().any().any():
		raise ValueError("The XGBoost AFT survival surface contains missing values after reconstruction.")
	return survival_wide_df


def sample_continuous_rows(frame: pd.DataFrame, row_cap: int, seed: int) -> np.ndarray:
	if frame.shape[0] <= row_cap:
		return np.arange(frame.shape[0], dtype=np.int32)
	if "event" not in frame.columns or "duration" not in frame.columns:
		raise KeyError("Continuous row sampling requires 'event' and 'duration' columns.")

	rng = np.random.default_rng(seed)
	work = frame.loc[:, ["event", "duration"]].copy()
	work["row_idx"] = np.arange(frame.shape[0], dtype=np.int32)
	work["event"] = pd.to_numeric(work["event"], errors="raise").astype(int)
	work["duration"] = pd.to_numeric(work["duration"], errors="raise").astype(float)
	work["duration_bucket"] = pd.cut(
		work["duration"],
		bins=[-np.inf, 10, 20, 30, np.inf],
		labels=["t01_10", "t11_20", "t21_30", "t31_plus"],
		include_lowest=True,
	).astype(str)
	work["stratum"] = work["event"].astype(str) + "__" + work["duration_bucket"]

	stratum_counts = work["stratum"].value_counts().sort_index()
	raw_alloc = (stratum_counts / stratum_counts.sum()) * row_cap
	alloc = np.floor(raw_alloc).astype(int)
	residual = int(row_cap - alloc.sum())
	if residual > 0:
		residual_order = (raw_alloc - alloc).sort_values(ascending=False)
		for stratum_name in residual_order.index[:residual]:
			alloc.loc[stratum_name] += 1

	sampled_chunks: list[np.ndarray] = []
	for stratum_name, n_take in alloc.items():
		if n_take <= 0:
			continue
		stratum_rows = work.loc[work["stratum"] == stratum_name, "row_idx"].to_numpy(dtype=np.int32)
		n_take = min(int(n_take), stratum_rows.shape[0])
		if n_take == stratum_rows.shape[0]:
			sampled_chunks.append(stratum_rows)
		else:
			sampled_chunks.append(rng.choice(stratum_rows, size=n_take, replace=False).astype(np.int32))

	sampled = np.concatenate(sampled_chunks) if sampled_chunks else np.empty(0, dtype=np.int32)
	if sampled.shape[0] == 0:
		raise ValueError("Continuous row sampling produced zero rows.")
	sampled = np.unique(sampled).astype(np.int32)
	sampled.sort()
	return sampled


def build_xgb_aft_guardrails(ctx: PipelineContext, treatment: XGBAFTTreatmentState) -> XGBAFTGuardrailState:
	log_stage_start("5.13.3", "Compute runtime guardrails for gradient boosting")

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
	truth_subtrain_df = base.build_truth_by_enrollment(treatment.train_df.iloc[train_internal_idx], "xgb_aft_validation_subtrain")
	truth_val_df = base.build_truth_by_enrollment(treatment.train_df.iloc[val_internal_idx], "xgb_aft_validation_holdout")
	X_subtrain_full = base.align_feature_matrix(
		treatment.X_train[train_internal_idx],
		treatment.feature_names_out,
		subtrain_raw_ids,
		truth_subtrain_df["enrollment_id"].astype(str).tolist(),
		matrix_name="XGBoost AFT subtrain matrix",
	)
	X_val = base.align_feature_matrix(
		treatment.X_train[val_internal_idx],
		treatment.feature_names_out,
		val_raw_ids,
		truth_val_df["enrollment_id"].astype(str).tolist(),
		matrix_name="XGBoost AFT validation matrix",
	)
	X_train_full_all = base.align_feature_matrix(
		treatment.X_train,
		treatment.feature_names_out,
		treatment.train_df["enrollment_id"].astype(str).tolist(),
		treatment.truth_train_df["enrollment_id"].astype(str).tolist(),
		matrix_name="XGBoost AFT full-train matrix",
	)
	X_test_aligned = base.align_feature_matrix(
		treatment.X_test,
		treatment.feature_names_out,
		treatment.test_df["enrollment_id"].astype(str).tolist(),
		treatment.truth_test_df["enrollment_id"].astype(str).tolist(),
		matrix_name="XGBoost AFT test matrix",
	)

	search_row_cap = safe_xgb_aft_row_budget(X_subtrain_full.shape[0], X_subtrain_full.shape[1], XGB_AFT_SEARCH_ROW_CAP)
	final_row_cap = safe_xgb_aft_row_budget(X_train_full_all.shape[0], X_train_full_all.shape[1], XGB_AFT_FINAL_ROW_CAP)
	subtrain_sample_idx = sample_continuous_rows(truth_subtrain_df, search_row_cap, ctx.random_seed + 100)
	full_sample_idx = sample_continuous_rows(treatment.truth_train_df, final_row_cap, ctx.random_seed + 101)

	summary = {
		"validation_test_size": float(VALIDATION_TEST_SIZE),
		"search_row_cap": int(search_row_cap),
		"final_row_cap": int(final_row_cap),
		"search_rows_selected": int(subtrain_sample_idx.shape[0]),
		"final_rows_selected": int(full_sample_idx.shape[0]),
		"subtrain_rows": int(X_subtrain_full.shape[0]),
		"validation_rows": int(X_val.shape[0]),
		"full_train_rows": int(X_train_full_all.shape[0]),
		"subtrain_enrollments": int(truth_subtrain_df.shape[0]),
		"validation_enrollments": int(truth_val_df.shape[0]),
		"full_train_enrollments": int(treatment.truth_train_df.shape[0]),
		"n_features": int(X_train_full_all.shape[1]),
		"subtrain_event_rate": float(truth_subtrain_df["event"].mean()),
		"validation_event_rate": float(truth_val_df["event"].mean()),
		"full_train_event_rate": float(treatment.truth_train_df["event"].mean()),
		"sampled_subtrain_event_rate": float(truth_subtrain_df.iloc[subtrain_sample_idx]["event"].mean()),
		"sampled_full_train_event_rate": float(treatment.truth_train_df.iloc[full_sample_idx]["event"].mean()),
		"random_seed": int(ctx.random_seed),
	}

	base.materialize_dataframe_table(
		ctx,
		df=pd.DataFrame([summary]),
		table_name=base.artifact_name(ctx, "table_xgb_aft_guardrails"),
		block_number="5.13.3",
		label="Stage 5.13.3 table_xgb_aft_guardrails - XGBoost AFT runtime guardrails",
	)
	log_stage_end("5.13.3")
	return XGBAFTGuardrailState(
		X_subtrain=X_subtrain_full,
		X_val=X_val,
		X_train_full=X_train_full_all,
		X_test_aligned=X_test_aligned,
		truth_subtrain_df=truth_subtrain_df,
		truth_val_df=truth_val_df,
		subtrain_sample_idx=subtrain_sample_idx,
		full_sample_idx=full_sample_idx,
		summary=summary,
	)


# ==============================================================
# 5.13.4 - Tune, fit, evaluate, and persist the XGBoost AFT model
# ==============================================================
# What this block does:
# - Tunes the compact XGBoost AFT grid on the grouped validation split, fits the best model on the full training treatment, evaluates the resulting survival functions, and persists all outputs.
# Inputs:
# - files read: none
# - DuckDB tables read: enrollment_cox_ready_train, enrollment_cox_ready_test
# - configuration values used: benchmark.seed, validation split policy, tuning grid, benchmark horizons, calibration bins
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/xgb_aft_tuned_model_config.json, outputs_benchmark_survival/models/xgb_aft_tuned.joblib, outputs_benchmark_survival/models/xgb_aft_preprocessor.joblib
# - DuckDB tables created or updated: table_xgb_aft_tuning_results, table_xgb_aft_tuned_test_predictions, table_xgb_aft_tuned_primary_metrics, table_xgb_aft_tuned_brier_by_horizon, table_xgb_aft_tuned_secondary_metrics, table_xgb_aft_tuned_td_auc_support_audit, table_xgb_aft_tuned_row_diagnostics, table_xgb_aft_tuned_support_by_horizon, table_xgb_aft_tuned_calibration_summary, table_xgb_aft_tuned_calibration_bins_by_horizon, table_xgb_aft_tuned_predicted_vs_observed_survival
# - objects returned in memory: none
def tune_and_evaluate_xgb_aft_model(
	ctx: PipelineContext,
	treatment: XGBAFTTreatmentState,
	guardrails: XGBAFTGuardrailState,
) -> None:
	log_stage_start("5.13.4", "Tune and evaluate the XGBoost AFT model")

	subtrain_truth_sampled_df = guardrails.truth_subtrain_df.iloc[guardrails.subtrain_sample_idx].reset_index(drop=True)
	dsubtrain = build_aft_dmatrix(guardrails.X_subtrain[guardrails.subtrain_sample_idx], subtrain_truth_sampled_df)

	tuning_rows: list[dict[str, Any]] = []
	best_candidate: dict[str, Any] | None = None
	for candidate in XGB_AFT_TUNING_GRID:
		candidate_params = build_xgb_aft_params(candidate, ctx.random_seed, ctx.cpu_cores)
		candidate_model = xgb.train(
			params=candidate_params,
			dtrain=dsubtrain,
			num_boost_round=int(candidate["num_boost_round"]),
			verbose_eval=False,
		)
		val_time_grid = np.arange(1, int(max(guardrails.truth_val_df["duration"].max(), max(BENCHMARK_HORIZONS))) + 1, dtype=float)
		val_survival_df = xgb_aft_survival_frame(
			candidate_model,
			guardrails.X_val,
			guardrails.truth_val_df["enrollment_id"].astype(str).tolist(),
			val_time_grid,
			distribution=str(candidate["aft_loss_distribution"]),
			scale=float(candidate["aft_loss_distribution_scale"]),
		)
		val_ibs, val_c_index = base.score_continuous_candidate(val_survival_df, guardrails.truth_subtrain_df, guardrails.truth_val_df)
		candidate_row = {
			"candidate_id": int(candidate["candidate_id"]),
			"aft_loss_distribution": str(candidate["aft_loss_distribution"]),
			"aft_loss_distribution_scale": float(candidate["aft_loss_distribution_scale"]),
			"learning_rate": float(candidate["learning_rate"]),
			"num_boost_round": int(candidate["num_boost_round"]),
			"max_depth": int(candidate["max_depth"]),
			"min_child_weight": float(candidate["min_child_weight"]),
			"subsample": float(candidate["subsample"]),
			"colsample_bytree": float(candidate["colsample_bytree"]),
			"reg_lambda": float(candidate["reg_lambda"]),
			"reg_alpha": float(candidate["reg_alpha"]),
			"val_ibs": float(val_ibs),
			"val_c_index": float(val_c_index),
		}
		tuning_rows.append(candidate_row)
		if best_candidate is None or float(val_ibs) < float(best_candidate["val_ibs"]):
			best_candidate = {
				"candidate_id": int(candidate["candidate_id"]),
				"params": {
					"aft_loss_distribution": str(candidate["aft_loss_distribution"]),
					"aft_loss_distribution_scale": float(candidate["aft_loss_distribution_scale"]),
					"learning_rate": float(candidate["learning_rate"]),
					"num_boost_round": int(candidate["num_boost_round"]),
					"max_depth": int(candidate["max_depth"]),
					"min_child_weight": float(candidate["min_child_weight"]),
					"subsample": float(candidate["subsample"]),
					"colsample_bytree": float(candidate["colsample_bytree"]),
					"reg_lambda": float(candidate["reg_lambda"]),
					"reg_alpha": float(candidate["reg_alpha"]),
				},
				"val_ibs": float(val_ibs),
				"val_c_index": float(val_c_index),
				"implementation": MODEL_IMPLEMENTATION,
			}

	if best_candidate is None:
		raise RuntimeError("The XGBoost AFT tuning grid did not produce a best candidate.")

	tuning_results_df = pd.DataFrame(tuning_rows).sort_values(
		["val_ibs", "candidate_id"],
		ascending=[True, True],
		kind="mergesort",
	).reset_index(drop=True)

	train_full_truth_sampled_df = treatment.truth_train_df.iloc[guardrails.full_sample_idx].reset_index(drop=True)
	dtrain_full = build_aft_dmatrix(guardrails.X_train_full[guardrails.full_sample_idx], train_full_truth_sampled_df)

	best_params = build_xgb_aft_params(best_candidate["params"], ctx.random_seed, ctx.cpu_cores)
	best_model = xgb.train(
		params=best_params,
		dtrain=dtrain_full,
		num_boost_round=int(best_candidate["params"]["num_boost_round"]),
		verbose_eval=False,
	)

	xgb_aft_time_grid = np.arange(1, int(max(treatment.truth_test_df["duration"].max(), max(BENCHMARK_HORIZONS))) + 1, dtype=float)
	survival_wide_df = xgb_aft_survival_frame(
		best_model,
		guardrails.X_test_aligned,
		treatment.truth_test_df["enrollment_id"].astype(str).tolist(),
		xgb_aft_time_grid,
		distribution=str(best_candidate["params"]["aft_loss_distribution"]),
		scale=float(best_candidate["params"]["aft_loss_distribution_scale"]),
	)
	if survival_wide_df.isna().any().any():
		raise ValueError("The tuned XGBoost AFT survival surface contains missing values after enrollment alignment.")

	artifacts = evaluate_continuous_survival(
		model_name=MODEL_NAME,
		survival_wide_df=survival_wide_df,
		truth_train_df=treatment.truth_train_df,
		truth_test_df=treatment.truth_test_df,
	)

	model_path = ctx.models_dir / base.artifact_filename(ctx, "xgb_aft_tuned.joblib")
	preprocessor_path = ctx.models_dir / base.artifact_filename(ctx, "xgb_aft_preprocessor.joblib")
	config_path = ctx.metadata_dir / base.artifact_filename(ctx, "xgb_aft_tuned_model_config.json")
	joblib.dump(best_model, model_path)
	joblib.dump(treatment.preprocessor, preprocessor_path)

	base.materialize_dataframe_table(
		ctx,
		df=tuning_results_df,
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuning_results"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuning_results - XGBoost AFT tuning results",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["test_predictions_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_test_predictions"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_test_predictions - XGBoost AFT test predictions",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["primary_metrics_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_primary_metrics"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_primary_metrics - XGBoost AFT primary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["brier_by_horizon_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_brier_by_horizon"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_brier_by_horizon - XGBoost AFT Brier scores by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["secondary_metrics_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_secondary_metrics"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_secondary_metrics - XGBoost AFT secondary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["td_auc_support_audit_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_td_auc_support_audit"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_td_auc_support_audit - XGBoost AFT IPCW support audit",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["row_diagnostics_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_row_diagnostics"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_row_diagnostics - XGBoost AFT row-level diagnostics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["support_by_horizon_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_support_by_horizon"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_support_by_horizon - XGBoost AFT support by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_summary_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_calibration_summary"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_calibration_summary - XGBoost AFT calibration summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_bins_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_calibration_bins_by_horizon"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_calibration_bins_by_horizon - XGBoost AFT calibration bins",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["predicted_vs_observed_survival_df"],
		table_name=base.artifact_name(ctx, "table_xgb_aft_tuned_predicted_vs_observed_survival"),
		block_number="5.13.4",
		label="Stage 5.13.4 table_xgb_aft_tuned_predicted_vs_observed_survival - XGBoost AFT predicted versus observed survival",
	)

	config_payload = {
		"model_name": MODEL_NAME,
		"selection_metric": "lowest_validation_ibs",
		"implementation": MODEL_IMPLEMENTATION,
		"validation_split": {
			"method": "GroupShuffleSplit",
			"test_size": float(VALIDATION_TEST_SIZE),
			"group_column": "enrollment_id",
			"random_seed": int(ctx.random_seed),
		},
		"search_space": [
			{
				"candidate_id": int(candidate["candidate_id"]),
				"aft_loss_distribution": str(candidate["aft_loss_distribution"]),
				"aft_loss_distribution_scale": float(candidate["aft_loss_distribution_scale"]),
				"learning_rate": float(candidate["learning_rate"]),
				"num_boost_round": int(candidate["num_boost_round"]),
				"max_depth": int(candidate["max_depth"]),
				"min_child_weight": float(candidate["min_child_weight"]),
				"subsample": float(candidate["subsample"]),
				"colsample_bytree": float(candidate["colsample_bytree"]),
				"reg_lambda": float(candidate["reg_lambda"]),
				"reg_alpha": float(candidate["reg_alpha"]),
			}
			for candidate in XGB_AFT_TUNING_GRID
		],
		"benchmark_horizons": list(BENCHMARK_HORIZONS),
		"calibration_bins": int(CALIBRATION_BINS),
		"calibration_contract_version": base.CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": base.CALIBRATION_OBSERVED_RISK_METHOD,
		"variance_filter_threshold": float(LOW_VARIANCE_THRESHOLD),
		"duration_floor": float(XGB_AFT_DURATION_FLOOR),
		"effective_window_weeks": int(ctx.effective_window_weeks),
		"input_train_table": ctx.input_train_table,
		"input_test_table": ctx.input_test_table,
		"best_candidate": best_candidate,
		"guardrails": guardrails.summary,
		"preprocessing": {
			"categorical_features": CATEGORICAL_FEATURES,
			"numeric_features": treatment.feature_columns[len(CATEGORICAL_FEATURES):],
			"numeric_prefill_values": treatment.numeric_fill_values,
			"feature_names_before_filter": treatment.feature_names_before_filter,
			"feature_names_out": treatment.feature_names_out,
			"dropped_low_variance_features": treatment.dropped_feature_names,
		},
		"design_note": (
			"The comparable XGBoost AFT arm preserves the canonical early-window contract, "
			"uses deterministic train-only preprocessing, fits ranged censoring labels through xgboost.DMatrix, "
			"and reconstructs survival curves from the tuned AFT location predictions under validation IBS."
		),
	}
	base.save_json(config_payload, config_path)
	base.print_artifact("xgb_aft_model", str(model_path))
	base.print_artifact("xgb_aft_preprocessor", str(preprocessor_path))
	base.print_artifact("xgb_aft_tuned_model_config", str(config_path))
	print(config_path.read_text(encoding="utf-8"))
	print(pd.DataFrame([best_candidate]).to_string(index=False))

	log_stage_end("5.13.4")


# ==============================================================
# 5.13.5 - Close runtime resources
# ==============================================================
# What this block does:
# - Closes the DuckDB connection after all XGBoost AFT artifacts have been materialized.
# Inputs:
# - files read: none
# - DuckDB tables read: none
# - configuration values used: none
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: none
# - objects returned in memory: none
def close_context(ctx: PipelineContext) -> None:
	from util import close_duckdb_connection

	log_stage_start("5.13.5", "Close runtime resources")
	close_duckdb_connection(ctx.con)
	base.print_artifact("duckdb_connection", f"closed::{ctx.duckdb_path}")
	log_stage_end("5.13.5")


def run_single_window(window_weeks: int) -> None:
	ctx: PipelineContext | None = None
	try:
		ctx = initialize_context(window_weeks)
		treatment = build_xgb_aft_treatment(ctx)
		guardrails = build_xgb_aft_guardrails(ctx, treatment)
		tune_and_evaluate_xgb_aft_model(ctx, treatment, guardrails)
	finally:
		if ctx is not None:
			close_context(ctx)


def main() -> None:
	execution_windows = base.resolve_execution_windows()
	for window_weeks in execution_windows:
		print(f"[RUN_PLAN] comparable_window_weeks={int(window_weeks)}")
		run_single_window(int(window_weeks))


if __name__ == "__main__":
	main()
