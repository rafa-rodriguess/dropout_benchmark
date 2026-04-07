from __future__ import annotations

"""
Production Neural-MTLR benchmark module for stage D5.14.

What this file does:
- prepares the comparable-window continuous-time treatment directly from DuckDB-ready enrollment tables
- applies deterministic preprocessing and feature validation for the Neural-MTLR model
- tunes and fits the Neural-MTLR benchmark under the official grouped validation protocol
- evaluates the tuned model with survival, calibration, support, time-dependent AUC, and auxiliary row-level diagnostics
- persists the trained model, fitted preprocessing pipeline, DuckDB audit tables, and JSON metadata artifacts

Main processing purpose:
- materialize the full comparable Neural-MTLR benchmark arm deterministically from DuckDB-ready tables without notebook runtime state, CSV workflows, or hidden execution state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_mtlr_preprocessing_summary, table_mtlr_raw_feature_manifest,
	table_mtlr_feature_names_out, table_mtlr_guardrails, table_mtlr_tuning_results,
  table_mtlr_tuned_test_predictions, table_mtlr_tuned_primary_metrics,
  table_mtlr_tuned_brier_by_horizon, table_mtlr_tuned_secondary_metrics,
  table_mtlr_tuned_td_auc_support_audit, table_mtlr_tuned_row_diagnostics,
  table_mtlr_tuned_support_by_horizon, table_mtlr_tuned_calibration_summary,
  table_mtlr_tuned_calibration_bins_by_horizon,
  table_mtlr_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/mtlr_preprocessing_config.json
- outputs_benchmark_survival/metadata/mtlr_tuned_model_config.json
- outputs_benchmark_survival/models/mtlr_tuned.pt
- outputs_benchmark_survival/models/mtlr_preprocessor.joblib

Main DuckDB tables used as inputs:
- enrollment_cox_ready_train
- enrollment_cox_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_mtlr_preprocessing_summary
- table_mtlr_raw_feature_manifest
- table_mtlr_feature_names_out
- table_mtlr_guardrails
- table_mtlr_tuning_results
- table_mtlr_tuned_test_predictions
- table_mtlr_tuned_primary_metrics
- table_mtlr_tuned_brier_by_horizon
- table_mtlr_tuned_secondary_metrics
- table_mtlr_tuned_td_auc_support_audit
- table_mtlr_tuned_row_diagnostics
- table_mtlr_tuned_support_by_horizon
- table_mtlr_tuned_calibration_summary
- table_mtlr_tuned_calibration_bins_by_horizon
- table_mtlr_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/mtlr_preprocessing_config.json
- outputs_benchmark_survival/metadata/mtlr_tuned_model_config.json
- outputs_benchmark_survival/models/mtlr_tuned.pt
- outputs_benchmark_survival/models/mtlr_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, missing dependencies, or invalid contracts raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, notebook globals, or CSV-based workflows are permitted
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tomli as toml_reader
from pycox.evaluation import EvalSurv
from pycox.models import MTLR
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.util import Surv
import torch
import torchtuples as tt

import dropout_bench_v3_D_09_comparable_tree_survival_random_survival_forest as base


STAGE_PREFIX = "5.14"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_D_5_14.ipynb"
MODEL_NAME = "mtlr_tuned"
MODEL_IMPLEMENTATION = "pycox_mtlr"
PREVIEW_ROWS = 20
BENCHMARK_HORIZONS = base.BENCHMARK_HORIZONS
CALIBRATION_BINS = base.CALIBRATION_BINS
VALIDATION_TEST_SIZE = 0.10
LOW_VARIANCE_THRESHOLD = 1e-12
MTLR_MIN_DURATION = 1.0
MTLR_SEARCH_ROW_CAP = 50000
MTLR_FINAL_ROW_CAP = 50000
MTLR_MAX_EPOCHS = 80
MTLR_EARLY_STOPPING_PATIENCE = 8
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
MTLR_TUNING_GRID = (
	{
		"candidate_id": 1,
		"num_durations": 20,
		"hidden_dims": [64, 32],
		"dropout": 0.10,
		"learning_rate": 1e-2,
		"weight_decay": 1e-5,
		"batch_size": 512,
	},
	{
		"candidate_id": 2,
		"num_durations": 20,
		"hidden_dims": [128, 64],
		"dropout": 0.10,
		"learning_rate": 5e-3,
		"weight_decay": 1e-5,
		"batch_size": 512,
	},
	{
		"candidate_id": 3,
		"num_durations": 40,
		"hidden_dims": [64, 32],
		"dropout": 0.20,
		"learning_rate": 5e-3,
		"weight_decay": 1e-4,
		"batch_size": 512,
	},
	{
		"candidate_id": 4,
		"num_durations": 30,
		"hidden_dims": [128, 64],
		"dropout": 0.20,
		"learning_rate": 1e-3,
		"weight_decay": 1e-4,
		"batch_size": 1024,
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
class MTLRTreatmentState:
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
class MTLRGuardrailState:
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
# 5.14.1 - Lightweight runtime bootstrap
# ==============================================================
# What this block does:
# - Initializes shared paths, configuration objects, run metadata, and the DuckDB connection for the D5.14 stage.
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

	log_stage_start("5.14.1", "Lightweight runtime bootstrap")

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
	base.require_tables(con, [input_train_table, input_test_table], block_number="5.14.1")

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

	log_stage_end("5.14.1")
	return ctx


# ==============================================================
# 5.14.2 - Build the comparable Neural-MTLR treatment
# ==============================================================
# What this block does:
# - Loads the comparable-arm continuous-time tables from DuckDB, validates the canonical feature contract, applies deterministic preprocessing, and records preprocessing audit outputs.
# Inputs:
# - files read: benchmark_modeling_contract.toml
# - DuckDB tables read: enrollment_cox_ready_train, enrollment_cox_ready_test
# - configuration values used: feature_contract.static_features, feature_contract.main_enrollment_window_features, feature_contract.optional_comparable_window_features, benchmark.main_enrollment_window_weeks
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/mtlr_preprocessing_config.json, outputs_benchmark_survival/models/mtlr_preprocessor.joblib
# - DuckDB tables created or updated: table_mtlr_preprocessing_summary, table_mtlr_raw_feature_manifest, table_mtlr_feature_names_out
# - objects returned in memory: MTLRTreatmentState
def build_mtlr_treatment(ctx: PipelineContext) -> MTLRTreatmentState:
	log_stage_start("5.14.2", "Build comparable Neural-MTLR treatment")

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
			"The comparable Neural-MTLR numeric feature specification does not match the canonical modeling contract. "
			f"Contract numeric features: {expected_numeric_features}. Operational numeric features: {numeric_features}."
		)

	required_columns = base.comparable_required_columns(BASE_REQUIRED_CONTINUOUS_COLUMNS, ctx.effective_window_weeks)

	train_df = base.load_required_table(ctx, ctx.input_train_table, required_columns, block_number="5.14.2")
	test_df = base.load_required_table(ctx, ctx.input_test_table, required_columns, block_number="5.14.2")
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
		raise ValueError(f"The Neural-MTLR transformed matrices must be 2D. Got train={X_train.shape}, test={X_test.shape}.")
	if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
		raise ValueError("The Neural-MTLR transformed matrices contain non-finite values.")

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
				"model_family": "neural_mtlr",
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
				"model_positioning_note": "early-window comparable Neural-MTLR benchmark",
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
		table_name=base.artifact_name(ctx, "table_mtlr_preprocessing_summary"),
		block_number="5.14.2",
		label="Stage 5.14.2 table_mtlr_preprocessing_summary - Neural-MTLR preprocessing summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=raw_feature_manifest_df,
		table_name=base.artifact_name(ctx, "table_mtlr_raw_feature_manifest"),
		block_number="5.14.2",
		label="Stage 5.14.2 table_mtlr_raw_feature_manifest - Neural-MTLR raw feature manifest",
	)
	base.materialize_dataframe_table(
		ctx,
		df=output_feature_manifest_df,
		table_name=base.artifact_name(ctx, "table_mtlr_feature_names_out"),
		block_number="5.14.2",
		label="Stage 5.14.2 table_mtlr_feature_names_out - Neural-MTLR output feature manifest",
	)

	preprocessing_config = {
		"model_family": "neural_mtlr",
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
		"model_positioning_note": "early-window comparable Neural-MTLR benchmark",
		"window_weeks": int(ctx.effective_window_weeks),
	}
	preprocessing_config_path = ctx.metadata_dir / base.artifact_filename(ctx, "mtlr_preprocessing_config.json")
	base.save_json(preprocessing_config, preprocessing_config_path)
	base.print_artifact("mtlr_preprocessing_config", str(preprocessing_config_path))
	print(preprocessing_config_path.read_text(encoding="utf-8"))

	log_stage_end("5.14.2")
	return MTLRTreatmentState(
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
			raise ValueError(f"The tuned Neural-MTLR predictions contain missing values at horizon {horizon_week}.")
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
		raise ValueError("Could not determine a valid censoring-support horizon for the Neural-MTLR model.")
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
				"notes": "IPCW-supported subset used for Neural-MTLR time-dependent AUC",
			}
		)

	time_dependent_auc_df = pd.DataFrame(td_auc_rows)
	td_auc_support_audit_df = pd.DataFrame(td_auc_audit_rows)
	secondary_metrics_df = pd.concat([risk_auc_by_horizon_df, time_dependent_auc_df], ignore_index=True).sort_values(["horizon_week", "metric_name"]).reset_index(drop=True)

	row_pred_survival = np.asarray(
		[
			float(survival_wide_df.loc[int(max(float(duration), MTLR_MIN_DURATION)), enrollment_id])
			for enrollment_id, duration in zip(truth_test_df["enrollment_id"], truth_test_df["duration"])
		],
		dtype=float,
	)
	row_pred_risk = 1.0 - row_pred_survival
	if not np.isfinite(row_pred_risk).all():
		raise ValueError("The tuned Neural-MTLR event-time risk diagnostics contain non-finite values.")
	if truth_test_df["event"].nunique() < 2:
		raise ValueError("The Neural-MTLR test truth does not contain both event classes for row-level diagnostics.")

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
# 5.14.3 - Compute runtime guardrails for Neural-MTLR
# ==============================================================
# What this block does:
# - Creates the grouped validation split, applies deterministic enrollment-level sampling for the boosting fit, and persists the guardrail audit table.
# Inputs:
# - files read: none
# - DuckDB tables read: enrollment_cox_ready_train
# - configuration values used: benchmark.seed, validation split policy, search row cap, final row cap
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: table_mtlr_guardrails
# - objects returned in memory: MTLRGuardrailState
def safe_mtlr_row_budget(n_rows: int, n_features: int, requested_cap: int) -> int:
	_ = n_features
	return int(min(n_rows, requested_cap))


def set_deterministic_state(random_seed: int) -> None:
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(random_seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def build_mtlr_target_arrays(truth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	durations = np.maximum(truth_df["duration"].astype(float).to_numpy(), MTLR_MIN_DURATION)
	events = truth_df["event"].astype(int).to_numpy()
	return durations.astype(np.float32), events.astype(np.int64)


def build_mtlr_network(input_dim: int, candidate: dict[str, Any], out_features: int) -> tt.practical.MLPVanilla:
	return tt.practical.MLPVanilla(
		in_features=input_dim,
		num_nodes=list(candidate["hidden_dims"]),
		out_features=out_features,
		batch_norm=False,
		dropout=float(candidate["dropout"]),
	)


def normalize_mtlr_survival_frame(
	raw_survival_df: pd.DataFrame,
	enrollment_ids: list[str],
	max_week: int,
) -> pd.DataFrame:
	raw = raw_survival_df.copy()
	raw.columns = list(enrollment_ids)
	raw.columns.name = "enrollment_id"
	raw.index = pd.Index(pd.to_numeric(raw.index, errors="raise").astype(float), name="raw_time")
	raw = raw.sort_index()
	raw_values = raw.to_numpy(dtype=float)
	if raw_values.ndim != 2 or raw_values.shape[1] != len(enrollment_ids):
		raise ValueError("The Neural-MTLR survival surface has an invalid shape.")
	week_grid = np.arange(1, int(max_week) + 1, dtype=float)
	cut_times = raw.index.to_numpy(dtype=float)
	positions = np.searchsorted(cut_times, week_grid, side="right") - 1
	survival_matrix = np.ones((week_grid.shape[0], raw_values.shape[1]), dtype=float)
	valid_mask = positions >= 0
	if valid_mask.any():
		survival_matrix[valid_mask] = raw_values[positions[valid_mask]]
	survival_wide_df = pd.DataFrame(survival_matrix, index=week_grid.astype(int), columns=enrollment_ids)
	survival_wide_df.index.name = "week"
	if survival_wide_df.isna().any().any():
		raise ValueError("The Neural-MTLR survival surface contains missing values after normalization.")
	if not np.isfinite(survival_wide_df.to_numpy(dtype=float)).all():
		raise ValueError("The Neural-MTLR survival surface contains non-finite values after normalization.")
	return survival_wide_df.clip(lower=0.0, upper=1.0)


def fit_mtlr_candidate(
	X_subtrain: np.ndarray,
	truth_subtrain_df: pd.DataFrame,
	X_val: np.ndarray,
	truth_val_df: pd.DataFrame,
	candidate: dict[str, Any],
	input_dim: int,
	random_seed: int,
	device: str,
) -> tuple[dict[str, Any], pd.DataFrame, MTLR]:
	set_deterministic_state(random_seed)
	subtrain_durations, subtrain_events = build_mtlr_target_arrays(truth_subtrain_df)
	val_durations, val_events = build_mtlr_target_arrays(truth_val_df)
	labtrans = MTLR.label_transform(int(candidate["num_durations"]))
	y_subtrain = labtrans.fit_transform(subtrain_durations, subtrain_events)
	y_val = labtrans.transform(val_durations, val_events)
	net = build_mtlr_network(input_dim=input_dim, candidate=candidate, out_features=int(labtrans.out_features))
	model = MTLR(
		net,
		tt.optim.Adam(lr=float(candidate["learning_rate"]), weight_decay=float(candidate["weight_decay"])),
		duration_index=labtrans.cuts,
		device=device,
	)
	training_log = model.fit(
		np.asarray(X_subtrain, dtype=np.float32),
		y_subtrain,
		batch_size=int(candidate["batch_size"]),
		epochs=int(MTLR_MAX_EPOCHS),
		verbose=False,
		val_data=(np.asarray(X_val, dtype=np.float32), y_val),
		callbacks=[tt.callbacks.EarlyStopping(patience=int(MTLR_EARLY_STOPPING_PATIENCE))],
	)
	history_df = training_log.to_pandas().reset_index(drop=True)
	if history_df.empty or "val_loss" not in history_df.columns:
		raise RuntimeError(f"MTLR tuning candidate {candidate['candidate_id']} did not produce a usable validation history.")
	history_df["epoch"] = np.arange(1, history_df.shape[0] + 1, dtype=int)
	history_df["candidate_id"] = int(candidate["candidate_id"])
	best_row = history_df.loc[history_df["val_loss"].astype(float).idxmin()]
	max_week = int(max(float(truth_val_df["duration"].max()), float(max(BENCHMARK_HORIZONS))))
	val_survival_df = normalize_mtlr_survival_frame(
		model.predict_surv_df(np.asarray(X_val, dtype=np.float32)),
		truth_val_df["enrollment_id"].astype(str).tolist(),
		max_week=max_week,
	)
	val_ibs, val_c_index = base.score_continuous_candidate(val_survival_df, truth_subtrain_df, truth_val_df)
	candidate_result = {
		"candidate_id": int(candidate["candidate_id"]),
		"num_durations": int(candidate["num_durations"]),
		"hidden_dims": str(list(candidate["hidden_dims"])),
		"dropout": float(candidate["dropout"]),
		"learning_rate": float(candidate["learning_rate"]),
		"weight_decay": float(candidate["weight_decay"]),
		"batch_size": int(candidate["batch_size"]),
		"best_epoch": int(best_row["epoch"]),
		"best_val_loss": float(best_row["val_loss"]),
		"val_ibs": float(val_ibs),
		"val_c_index": float(val_c_index),
	}
	return candidate_result, history_df, model


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


def build_mtlr_guardrails(ctx: PipelineContext, treatment: MTLRTreatmentState) -> MTLRGuardrailState:
	log_stage_start("5.14.3", "Compute runtime guardrails for Neural-MTLR")

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
	truth_subtrain_df = base.build_truth_by_enrollment(treatment.train_df.iloc[train_internal_idx], "mtlr_validation_subtrain")
	truth_val_df = base.build_truth_by_enrollment(treatment.train_df.iloc[val_internal_idx], "mtlr_validation_holdout")
	X_subtrain_full = base.align_feature_matrix(
		treatment.X_train[train_internal_idx],
		treatment.feature_names_out,
		subtrain_raw_ids,
		truth_subtrain_df["enrollment_id"].astype(str).tolist(),
		matrix_name="Neural-MTLR subtrain matrix",
	)
	X_val = base.align_feature_matrix(
		treatment.X_train[val_internal_idx],
		treatment.feature_names_out,
		val_raw_ids,
		truth_val_df["enrollment_id"].astype(str).tolist(),
		matrix_name="Neural-MTLR validation matrix",
	)
	X_train_full_all = base.align_feature_matrix(
		treatment.X_train,
		treatment.feature_names_out,
		treatment.train_df["enrollment_id"].astype(str).tolist(),
		treatment.truth_train_df["enrollment_id"].astype(str).tolist(),
		matrix_name="Neural-MTLR full-train matrix",
	)
	X_test_aligned = base.align_feature_matrix(
		treatment.X_test,
		treatment.feature_names_out,
		treatment.test_df["enrollment_id"].astype(str).tolist(),
		treatment.truth_test_df["enrollment_id"].astype(str).tolist(),
		matrix_name="Neural-MTLR test matrix",
	)

	search_row_cap = safe_mtlr_row_budget(X_subtrain_full.shape[0], X_subtrain_full.shape[1], MTLR_SEARCH_ROW_CAP)
	final_row_cap = safe_mtlr_row_budget(X_train_full_all.shape[0], X_train_full_all.shape[1], MTLR_FINAL_ROW_CAP)
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
		table_name=base.artifact_name(ctx, "table_mtlr_guardrails"),
		block_number="5.14.3",
		label="Stage 5.14.3 table_mtlr_guardrails - Neural-MTLR runtime guardrails",
	)
	log_stage_end("5.14.3")
	return MTLRGuardrailState(
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
# 5.14.4 - Tune, fit, evaluate, and persist the Neural-MTLR model
# ==============================================================
# What this block does:
# - Tunes the compact Neural-MTLR grid on the grouped validation split, fits the best model on the full training treatment, evaluates the resulting survival functions, and persists all outputs.
# Inputs:
# - files read: none
# - DuckDB tables read: enrollment_cox_ready_train, enrollment_cox_ready_test
# - configuration values used: benchmark.seed, validation split policy, tuning grid, benchmark horizons, calibration bins
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/mtlr_tuned_model_config.json, outputs_benchmark_survival/models/mtlr_tuned.pt, outputs_benchmark_survival/models/mtlr_preprocessor.joblib
# - DuckDB tables created or updated: table_mtlr_tuning_results, table_mtlr_tuned_test_predictions, table_mtlr_tuned_primary_metrics, table_mtlr_tuned_brier_by_horizon, table_mtlr_tuned_secondary_metrics, table_mtlr_tuned_td_auc_support_audit, table_mtlr_tuned_row_diagnostics, table_mtlr_tuned_support_by_horizon, table_mtlr_tuned_calibration_summary, table_mtlr_tuned_calibration_bins_by_horizon, table_mtlr_tuned_predicted_vs_observed_survival
# - objects returned in memory: none
def tune_and_evaluate_mtlr_model(
	ctx: PipelineContext,
	treatment: MTLRTreatmentState,
	guardrails: MTLRGuardrailState,
) -> None:
	log_stage_start("5.14.4", "Tune and evaluate the Neural-MTLR model")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	X_subtrain = np.asarray(guardrails.X_subtrain[guardrails.subtrain_sample_idx], dtype=np.float32)
	X_val = np.asarray(guardrails.X_val, dtype=np.float32)
	subtrain_truth_sampled_df = guardrails.truth_subtrain_df.iloc[guardrails.subtrain_sample_idx].reset_index(drop=True)

	tuning_rows: list[dict[str, Any]] = []
	best_candidate: dict[str, Any] | None = None
	best_history_df: pd.DataFrame | None = None
	for candidate in MTLR_TUNING_GRID:
		candidate_row, history_df, _ = fit_mtlr_candidate(
			X_subtrain=X_subtrain,
			truth_subtrain_df=subtrain_truth_sampled_df,
			X_val=X_val,
			truth_val_df=guardrails.truth_val_df,
			candidate=candidate,
			input_dim=int(X_subtrain.shape[1]),
			random_seed=ctx.random_seed,
			device=device,
		)
		tuning_rows.append(candidate_row)
		if best_candidate is None or float(candidate_row["val_ibs"]) < float(best_candidate["val_ibs"]):
			best_candidate = {
				"candidate_id": int(candidate["candidate_id"]),
				"params": {
					"num_durations": int(candidate["num_durations"]),
					"hidden_dims": list(candidate["hidden_dims"]),
					"dropout": float(candidate["dropout"]),
					"learning_rate": float(candidate["learning_rate"]),
					"weight_decay": float(candidate["weight_decay"]),
					"batch_size": int(candidate["batch_size"]),
				},
				"best_epoch": int(candidate_row["best_epoch"]),
				"best_val_loss": float(candidate_row["best_val_loss"]),
				"val_ibs": float(candidate_row["val_ibs"]),
				"val_c_index": float(candidate_row["val_c_index"]),
				"implementation": MODEL_IMPLEMENTATION,
			}
			best_history_df = history_df.copy()

	if best_candidate is None or best_history_df is None:
		raise RuntimeError("The Neural-MTLR tuning grid did not produce a best candidate.")

	tuning_results_df = pd.DataFrame(tuning_rows).sort_values(["val_ibs", "candidate_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

	train_full_truth_sampled_df = treatment.truth_train_df.iloc[guardrails.full_sample_idx].reset_index(drop=True)
	X_train_full = np.asarray(guardrails.X_train_full[guardrails.full_sample_idx], dtype=np.float32)
	train_durations, train_events = build_mtlr_target_arrays(train_full_truth_sampled_df)
	full_labtrans = MTLR.label_transform(int(best_candidate["params"]["num_durations"]))
	y_train_full = full_labtrans.fit_transform(train_durations, train_events)
	set_deterministic_state(ctx.random_seed)
	best_net = build_mtlr_network(
		input_dim=int(X_train_full.shape[1]),
		candidate=best_candidate["params"],
		out_features=int(full_labtrans.out_features),
	)
	best_model = MTLR(
		best_net,
		tt.optim.Adam(
			lr=float(best_candidate["params"]["learning_rate"]),
			weight_decay=float(best_candidate["params"]["weight_decay"]),
		),
		duration_index=full_labtrans.cuts,
		device=device,
	)
	final_training_log = best_model.fit(
		X_train_full,
		y_train_full,
		batch_size=int(best_candidate["params"]["batch_size"]),
		epochs=int(best_candidate["best_epoch"]),
		verbose=False,
	)
	training_history_df = final_training_log.to_pandas().reset_index(drop=True)
	training_history_df["epoch"] = np.arange(1, training_history_df.shape[0] + 1, dtype=int)
	training_history_df["candidate_id"] = int(best_candidate["candidate_id"])
	training_history_df["fit_stage"] = "full_train"

	max_test_week = int(max(float(treatment.truth_test_df["duration"].max()), float(max(BENCHMARK_HORIZONS))))
	survival_wide_df = normalize_mtlr_survival_frame(
		best_model.predict_surv_df(np.asarray(guardrails.X_test_aligned, dtype=np.float32)),
		treatment.truth_test_df["enrollment_id"].astype(str).tolist(),
		max_week=max_test_week,
	)

	artifacts = evaluate_continuous_survival(
		model_name=MODEL_NAME,
		survival_wide_df=survival_wide_df,
		truth_train_df=treatment.truth_train_df,
		truth_test_df=treatment.truth_test_df,
	)

	model_path = ctx.models_dir / base.artifact_filename(ctx, "mtlr_tuned.pt")
	preprocessor_path = ctx.models_dir / base.artifact_filename(ctx, "mtlr_preprocessor.joblib")
	config_path = ctx.metadata_dir / base.artifact_filename(ctx, "mtlr_tuned_model_config.json")
	best_model.save_net(str(model_path))
	joblib.dump(treatment.preprocessor, preprocessor_path)

	base.materialize_dataframe_table(
		ctx,
		df=tuning_results_df,
		table_name=base.artifact_name(ctx, "table_mtlr_tuning_results"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuning_results - Neural-MTLR tuning results",
	)
	base.materialize_dataframe_table(
		ctx,
		df=training_history_df,
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_training_history"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_training_history - Neural-MTLR training history",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["test_predictions_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_test_predictions"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_test_predictions - Neural-MTLR test predictions",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["primary_metrics_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_primary_metrics"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_primary_metrics - Neural-MTLR primary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["brier_by_horizon_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_brier_by_horizon"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_brier_by_horizon - Neural-MTLR Brier scores by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["secondary_metrics_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_secondary_metrics"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_secondary_metrics - Neural-MTLR secondary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["td_auc_support_audit_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_td_auc_support_audit"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_td_auc_support_audit - Neural-MTLR IPCW support audit",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["row_diagnostics_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_row_diagnostics"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_row_diagnostics - Neural-MTLR row-level diagnostics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["support_by_horizon_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_support_by_horizon"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_support_by_horizon - Neural-MTLR support by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_summary_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_calibration_summary"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_calibration_summary - Neural-MTLR calibration summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_bins_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_calibration_bins_by_horizon"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_calibration_bins_by_horizon - Neural-MTLR calibration bins",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["predicted_vs_observed_survival_df"],
		table_name=base.artifact_name(ctx, "table_mtlr_tuned_predicted_vs_observed_survival"),
		block_number="5.14.4",
		label="Stage 5.14.4 table_mtlr_tuned_predicted_vs_observed_survival - Neural-MTLR predicted versus observed survival",
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
				"num_durations": int(candidate["num_durations"]),
				"hidden_dims": list(candidate["hidden_dims"]),
				"dropout": float(candidate["dropout"]),
				"learning_rate": float(candidate["learning_rate"]),
				"weight_decay": float(candidate["weight_decay"]),
				"batch_size": int(candidate["batch_size"]),
			}
			for candidate in MTLR_TUNING_GRID
		],
		"benchmark_horizons": list(BENCHMARK_HORIZONS),
		"calibration_bins": int(CALIBRATION_BINS),
		"calibration_contract_version": base.CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": base.CALIBRATION_OBSERVED_RISK_METHOD,
		"variance_filter_threshold": float(LOW_VARIANCE_THRESHOLD),
		"min_duration": float(MTLR_MIN_DURATION),
		"max_epochs": int(MTLR_MAX_EPOCHS),
		"early_stopping_patience": int(MTLR_EARLY_STOPPING_PATIENCE),
		"device": device,
		"effective_window_weeks": int(ctx.effective_window_weeks),
		"input_train_table": ctx.input_train_table,
		"input_test_table": ctx.input_test_table,
		"best_candidate": best_candidate,
		"guardrails": guardrails.summary,
		"duration_index": [float(value) for value in full_labtrans.cuts],
		"out_features": int(full_labtrans.out_features),
		"preprocessing": {
			"categorical_features": CATEGORICAL_FEATURES,
			"numeric_features": treatment.feature_columns[len(CATEGORICAL_FEATURES):],
			"numeric_prefill_values": treatment.numeric_fill_values,
			"feature_names_before_filter": treatment.feature_names_before_filter,
			"feature_names_out": treatment.feature_names_out,
			"dropped_low_variance_features": treatment.dropped_feature_names,
		},
		"design_note": (
			"The comparable Neural-MTLR arm preserves the canonical early-window contract, "
			"uses deterministic train-only preprocessing, discretizes survival times through pycox LabTransDiscreteTime, "
			"and reconstructs the full weekly survival surface from the tuned MTLR network under validation IBS."
		),
	}
	base.save_json(config_payload, config_path)
	base.print_artifact("mtlr_model", str(model_path))
	base.print_artifact("mtlr_preprocessor", str(preprocessor_path))
	base.print_artifact("mtlr_tuned_model_config", str(config_path))
	print(config_path.read_text(encoding="utf-8"))
	print(pd.DataFrame([best_candidate]).to_string(index=False))

	log_stage_end("5.14.4")


# ==============================================================
# 5.14.5 - Close runtime resources
# ==============================================================
# What this block does:
# - Closes the DuckDB connection after all Neural-MTLR artifacts have been materialized.
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

	log_stage_start("5.14.5", "Close runtime resources")
	close_duckdb_connection(ctx.con)
	base.print_artifact("duckdb_connection", f"closed::{ctx.duckdb_path}")
	log_stage_end("5.14.5")


def run_single_window(window_weeks: int) -> None:
	ctx: PipelineContext | None = None
	try:
		ctx = initialize_context(window_weeks)
		treatment = build_mtlr_treatment(ctx)
		guardrails = build_mtlr_guardrails(ctx, treatment)
		tune_and_evaluate_mtlr_model(ctx, treatment, guardrails)
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
