from __future__ import annotations

"""
Production not-weighted CatBoost weekly hazard benchmark module for the official D5.8 arm.

What this file does:
- prepares the shared dynamic weekly treatment directly from DuckDB person-period tables
- sanitizes mixed categorical and numeric features for deterministic CatBoost training
- tunes and fits the CatBoost weekly hazard benchmark under the official grouped validation protocol
- evaluates the tuned model with the shared survival, calibration, discrimination, and row-level audit stack
- persists DuckDB audit tables, the trained CatBoost model artifact, and JSON metadata artifacts

Main processing purpose:
- materialize the full CatBoost weekly hazard benchmark arm deterministically from DuckDB-ready tables without notebook globals, CSV workflows, or hidden runtime state

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_catboost_weekly_hazard_not_weighted_preprocessing_summary,
  table_catboost_weekly_hazard_not_weighted_raw_feature_manifest,
  table_catboost_weekly_hazard_not_weighted_feature_names_out,
  table_catboost_weekly_hazard_not_weighted_guardrails,
  table_catboost_weekly_hazard_not_weighted_tuning_results,
  table_catboost_weekly_hazard_not_weighted_tuned_test_predictions,
  table_catboost_weekly_hazard_not_weighted_tuned_primary_metrics,
  table_catboost_weekly_hazard_not_weighted_tuned_brier_by_horizon,
  table_catboost_weekly_hazard_not_weighted_tuned_secondary_metrics,
  table_catboost_weekly_hazard_not_weighted_tuned_td_auc_support_audit,
  table_catboost_weekly_hazard_not_weighted_tuned_row_diagnostics,
  table_catboost_weekly_hazard_not_weighted_tuned_support_by_horizon,
  table_catboost_weekly_hazard_not_weighted_tuned_calibration_summary,
  table_catboost_weekly_hazard_not_weighted_tuned_calibration_bins_by_horizon,
  table_catboost_weekly_hazard_not_weighted_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_preprocessing_config.json
- outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/catboost_weekly_hazard_not_weighted_tuned.cbm

Main DuckDB tables used as inputs:
- pp_linear_hazard_ready_train
- pp_linear_hazard_ready_test
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_catboost_weekly_hazard_not_weighted_preprocessing_summary
- table_catboost_weekly_hazard_not_weighted_raw_feature_manifest
- table_catboost_weekly_hazard_not_weighted_feature_names_out
- table_catboost_weekly_hazard_not_weighted_guardrails
- table_catboost_weekly_hazard_not_weighted_tuning_results
- table_catboost_weekly_hazard_not_weighted_tuned_test_predictions
- table_catboost_weekly_hazard_not_weighted_tuned_primary_metrics
- table_catboost_weekly_hazard_not_weighted_tuned_brier_by_horizon
- table_catboost_weekly_hazard_not_weighted_tuned_secondary_metrics
- table_catboost_weekly_hazard_not_weighted_tuned_td_auc_support_audit
- table_catboost_weekly_hazard_not_weighted_tuned_row_diagnostics
- table_catboost_weekly_hazard_not_weighted_tuned_support_by_horizon
- table_catboost_weekly_hazard_not_weighted_tuned_calibration_summary
- table_catboost_weekly_hazard_not_weighted_tuned_calibration_bins_by_horizon
- table_catboost_weekly_hazard_not_weighted_tuned_predicted_vs_observed_survival
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_preprocessing_config.json
- outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/catboost_weekly_hazard_not_weighted_tuned.cbm

Failure policy:
- missing DuckDB tables, missing columns, invalid contracts, or missing dependencies raise immediately
- invalid feature values, invalid grouped splits, or non-finite predictions raise immediately
- no fallback behavior, silent degradation, or CSV-based workflows are permitted
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import dropout_bench_v3_D_06_A_dynamic_weekly_poisson_piecewise_exponential as base
from dropout_bench_v3_D_00_common import (
	append_suffix_before_extension,
	apply_name_suffix,
	resolve_early_window_sensitivity_weeks,
)


SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_D_5_8_not_weighted.ipynb"
MODEL_NAME = "catboost_weekly_hazard_not_weighted_tuned"
MODEL_TYPE = "catboost_weekly_hazard"
PREPROCESSING_CONFIG_NAME = "catboost_weekly_hazard_not_weighted_preprocessing_config.json"
MODEL_CONFIG_NAME = "catboost_weekly_hazard_not_weighted_tuned_model_config.json"
MODEL_ARTIFACT_NAME = "catboost_weekly_hazard_not_weighted_tuned.cbm"
VALIDATION_TEST_SIZE = 0.20
EPSILON = 1e-8
CATBOOST_SEARCH_ROW_CAP = 120_000
CATBOOST_FINAL_ROW_CAP = 180_000
CANDIDATE_GRID = (
	{"depth": 4, "learning_rate": 0.05, "iterations": 250, "l2_leaf_reg": 3.0},
	{"depth": 6, "learning_rate": 0.06, "iterations": 350, "l2_leaf_reg": 3.0},
	{"depth": 4, "learning_rate": 0.10, "iterations": 250, "l2_leaf_reg": 10.0},
	{"depth": 6, "learning_rate": 0.05, "iterations": 450, "l2_leaf_reg": 10.0},
)
WINDOW_WEEK_COLUMN = "week"


def build_dynamic_window_suffix(window_weeks: int) -> str:
	return f"_w{int(window_weeks)}"


def apply_window_table_name(base_name: str, window_weeks: int) -> str:
	return apply_name_suffix(base_name, build_dynamic_window_suffix(window_weeks))


def build_window_artifact_path(directory: Path, filename: str, window_weeks: int) -> Path:
	return directory / append_suffix_before_extension(filename, build_dynamic_window_suffix(window_weeks))


def resolve_dynamic_window_execution_weeks(ctx: base.PipelineContext) -> list[int]:
	benchmark_config = ctx.shared_modeling_contract.get("benchmark")
	if not isinstance(benchmark_config, dict):
		raise TypeError("benchmark_modeling_contract.toml is missing the [benchmark] section.")
	return [int(window_weeks) for window_weeks in resolve_early_window_sensitivity_weeks(benchmark_config)]


@dataclass
class CatBoostTreatmentState:
	train_df: pd.DataFrame
	test_df: pd.DataFrame
	target_col: str
	feature_columns: list[str]
	categorical_features: list[str]
	numeric_features: list[str]
	expected_features_raw: list[str]
	expected_features_resolved: list[str]
	y_train: pd.Series
	y_test: pd.Series
	truth_train_df: pd.DataFrame
	truth_test_df: pd.DataFrame


@dataclass
class ValidationSplitState:
	subtrain_index: np.ndarray
	validation_index: np.ndarray
	subtrain_sample_idx: np.ndarray
	full_sample_idx: np.ndarray
	full_train_df: pd.DataFrame
	subtrain_df: pd.DataFrame
	validation_df: pd.DataFrame
	y_subtrain: np.ndarray
	y_validation: np.ndarray
	cat_feature_indices: list[int]
	summary: dict[str, Any]


# ==============================================================
# 5.8.1 - Lightweight runtime bootstrap
# ==============================================================
# What this block does:
# - Initializes shared paths, configuration objects, run metadata, and the DuckDB connection for stage D5.8.
# Inputs:
# - files read: benchmark_shared_config.toml, benchmark_modeling_contract.toml, outputs_benchmark_survival/metadata/run_metadata.json
# - DuckDB tables read: pp_linear_hazard_ready_train, pp_linear_hazard_ready_test
# - configuration values used: paths.*, benchmark.seed, benchmark.test_size, benchmark.early_window_weeks, benchmark.main_enrollment_window_weeks
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: pipeline_table_catalog, vw_pipeline_table_catalog_schema
# - objects returned in memory: base.PipelineContext
def initialize_context() -> base.PipelineContext:
	from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
	from util import ensure_pipeline_catalog, open_duckdb_connection

	base.log_stage_start("5.8.1", "Lightweight runtime bootstrap")

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
		shared_config = base.toml_reader.load(file_obj)
	with open(modeling_contract_toml_path, "rb") as file_obj:
		shared_modeling_contract = base.toml_reader.load(file_obj)
	with open(run_metadata_path, "r", encoding="utf-8") as file_obj:
		run_metadata = json.load(file_obj)

	base.require_mapping_keys(shared_config, ["paths"], "benchmark_shared_config.toml")
	base.require_mapping_keys(shared_modeling_contract, base.REQUIRED_MODELING_KEYS, "benchmark_modeling_contract.toml")
	base.require_mapping_keys(run_metadata, ["run_id"], "run_metadata.json")

	paths_config = shared_config["paths"]
	base.require_mapping_keys(paths_config, base.REQUIRED_SHARED_PATH_KEYS, "benchmark_shared_config.toml [paths]")
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

	con = open_duckdb_connection(duckdb_path)
	ensure_pipeline_catalog(con)
	base.require_tables(con, base.REQUIRED_INPUT_TABLES, block_number="5.8.1")

	ctx = base.PipelineContext(
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
	print(f"- BENCHMARK_HORIZONS: {list(base.BENCHMARK_HORIZONS)}")
	print(f"- CALIBRATION_BINS: {base.CALIBRATION_BINS}")
	base.print_artifact("shared_config", str(ctx.config_toml_path))
	base.print_artifact("modeling_contract", str(ctx.modeling_contract_toml_path))
	base.print_artifact("run_metadata", str(ctx.run_metadata_path))
	base.log_stage_end("5.8.1")
	return ctx


# ==============================================================
# 5.8.2 - Build the CatBoost weekly treatment contract
# ==============================================================
# What this block does:
# - Loads the person-period train/test tables from DuckDB, validates the canonical feature contract, and records preprocessing metadata for the raw CatBoost representation.
# Inputs:
# - files read: benchmark_modeling_contract.toml
# - DuckDB tables read: pp_linear_hazard_ready_train, pp_linear_hazard_ready_test
# - configuration values used: feature_contract.static_features, feature_contract.temporal_features_discrete
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_preprocessing_config.json
# - DuckDB tables created or updated: table_catboost_weekly_hazard_not_weighted_preprocessing_summary, table_catboost_weekly_hazard_not_weighted_raw_feature_manifest, table_catboost_weekly_hazard_not_weighted_feature_names_out
# - objects returned in memory: CatBoostTreatmentState
def build_catboost_treatment(ctx: base.PipelineContext, window_weeks: int) -> CatBoostTreatmentState:
	base.log_stage_start("5.8.2", "Prepare the shared dynamic weekly treatment for CatBoost")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	feature_contract = ctx.shared_modeling_contract["feature_contract"]
	static_features = base.require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
	temporal_features_discrete = base.require_list_of_strings(
		feature_contract["temporal_features_discrete"],
		"feature_contract.temporal_features_discrete",
	)
	expected_features_raw = static_features + temporal_features_discrete
	expected_features_resolved = [base.FEATURE_ALIAS_MAP.get(feature_name, feature_name) for feature_name in expected_features_raw]

	train_df = base.load_required_table(ctx, "pp_linear_hazard_ready_train", base.REQUIRED_PERSON_PERIOD_COLUMNS, "5.8.2")
	test_df = base.load_required_table(ctx, "pp_linear_hazard_ready_test", base.REQUIRED_PERSON_PERIOD_COLUMNS, "5.8.2")
	raw_train_rows = int(len(train_df))
	raw_test_rows = int(len(test_df))
	if WINDOW_WEEK_COLUMN not in train_df.columns or WINDOW_WEEK_COLUMN not in test_df.columns:
		raise KeyError(f"Dynamic CatBoost treatment requires the '{WINDOW_WEEK_COLUMN}' column in both train and test tables.")
	train_df = train_df.copy()
	test_df = test_df.copy()
	train_df[WINDOW_WEEK_COLUMN] = pd.to_numeric(train_df[WINDOW_WEEK_COLUMN], errors="raise").astype(int)
	test_df[WINDOW_WEEK_COLUMN] = pd.to_numeric(test_df[WINDOW_WEEK_COLUMN], errors="raise").astype(int)
	train_df = train_df.loc[train_df[WINDOW_WEEK_COLUMN] <= int(window_weeks)].reset_index(drop=True)
	test_df = test_df.loc[test_df[WINDOW_WEEK_COLUMN] <= int(window_weeks)].reset_index(drop=True)
	if train_df.empty or test_df.empty:
		raise ValueError(f"Window truncation at w={int(window_weeks)} produced an empty train/test dynamic CatBoost table.")

	categorical_features = list(base.CATEGORICAL_FEATURES)
	numeric_features = list(base.NUMERIC_FEATURES)
	feature_columns = categorical_features + numeric_features
	missing_from_treatment_spec = [
		feature_name for feature_name in expected_features_resolved if feature_name not in feature_columns
	]
	if missing_from_treatment_spec:
		raise ValueError(
			"Configured canonical features are not covered by the operational CatBoost treatment spec after alias resolution: "
			+ ", ".join(missing_from_treatment_spec)
		)

	missing_in_train = [column for column in feature_columns if column not in train_df.columns]
	missing_in_test = [column for column in feature_columns if column not in test_df.columns]
	if missing_in_train or missing_in_test:
		raise KeyError(
			"Operational CatBoost feature columns are missing from the materialized train/test tables. "
			f"Missing in train: {missing_in_train}. Missing in test: {missing_in_test}"
		)

	target_col = "event_t"
	y_train = base.ensure_binary_target(train_df[target_col], "pp_linear_hazard_ready_train.event_t")
	y_test = base.ensure_binary_target(test_df[target_col], "pp_linear_hazard_ready_test.event_t")
	truth_train_df = base.build_truth_by_enrollment(train_df, "pp_linear_hazard_ready_train")
	truth_test_df = base.build_truth_by_enrollment(test_df, "pp_linear_hazard_ready_test")

	preprocessing_summary_df = pd.DataFrame(
		[
			{
				"model_family": MODEL_NAME,
				"active_window_weeks": int(window_weeks),
				"raw_train_rows_before_truncation": raw_train_rows,
				"raw_test_rows_before_truncation": raw_test_rows,
				"train_rows": int(len(train_df)),
				"test_rows": int(len(test_df)),
				"n_input_features_raw": int(len(feature_columns)),
				"n_categorical_features": int(len(categorical_features)),
				"n_numeric_features": int(len(numeric_features)),
				"n_output_features_after_preprocessing": int(len(feature_columns)),
				"train_event_rate": float(y_train.mean()),
				"test_event_rate": float(y_test.mean()),
			}
		]
	)
	feature_manifest_df = pd.DataFrame(
		{
			"feature_name_operational": feature_columns,
			"feature_role": ["categorical" if feature_name in categorical_features else "numeric" for feature_name in feature_columns],
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
	feature_names_df = pd.DataFrame(
		{
			"model_feature_name": feature_columns,
			"feature_role": ["categorical" if feature_name in categorical_features else "numeric" for feature_name in feature_columns],
			"catboost_feature_index": list(range(len(feature_columns))),
			"is_categorical": [feature_name in categorical_features for feature_name in feature_columns],
		}
	)

	base.materialize_dataframe_table(
		ctx,
		df=preprocessing_summary_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_preprocessing_summary", window_weeks),
		block_number="5.8.2",
		label=f"Stage 5.8.2 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_preprocessing_summary', output_table_suffix)} — CatBoost preprocessing summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_manifest_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_raw_feature_manifest", window_weeks),
		block_number="5.8.2",
		label=f"Stage 5.8.2 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_raw_feature_manifest', output_table_suffix)} — CatBoost raw feature manifest",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_names_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_feature_names_out", window_weeks),
		block_number="5.8.2",
		label=f"Stage 5.8.2 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_feature_names_out', output_table_suffix)} — CatBoost model feature names",
	)

	preprocessing_config = {
		"model_name": MODEL_NAME,
		"model_type": MODEL_TYPE,
		"train_table": "pp_linear_hazard_ready_train",
		"test_table": "pp_linear_hazard_ready_test",
		"target_column": target_col,
		"categorical_features": categorical_features,
		"numeric_features": numeric_features,
		"operational_feature_columns": feature_columns,
		"canonical_expected_features_raw": expected_features_raw,
		"canonical_expected_features_resolved": expected_features_resolved,
		"feature_alias_map": base.FEATURE_ALIAS_MAP,
		"n_train_rows": int(len(train_df)),
		"n_test_rows": int(len(test_df)),
		"active_window_weeks": int(window_weeks),
		"raw_train_rows_before_truncation": raw_train_rows,
		"raw_test_rows_before_truncation": raw_test_rows,
		"n_output_features_after_preprocessing": int(len(feature_columns)),
		"train_event_rate": float(y_train.mean()),
		"test_event_rate": float(y_test.mean()),
		"design_note": "The CatBoost benchmark keeps the same weekly dynamic feature contract as the linear discrete-time arm while preserving categorical columns as native string-valued features.",
		"methodological_note": "No learned preprocessing transform is applied because CatBoost consumes the validated raw feature columns directly after deterministic type sanitization.",
	}
	preprocessing_config_path = build_window_artifact_path(ctx.metadata_dir, PREPROCESSING_CONFIG_NAME, window_weeks)
	base.save_json(preprocessing_config, preprocessing_config_path)
	base.print_artifact("catboost_weekly_hazard_preprocessing_config", str(preprocessing_config_path))
	print(preprocessing_config_path.read_text(encoding="utf-8"))

	base.log_stage_end("5.8.2")
	return CatBoostTreatmentState(
		train_df=train_df,
		test_df=test_df,
		target_col=target_col,
		feature_columns=feature_columns,
		categorical_features=categorical_features,
		numeric_features=numeric_features,
		expected_features_raw=expected_features_raw,
		expected_features_resolved=expected_features_resolved,
		y_train=y_train,
		y_test=y_test,
		truth_train_df=truth_train_df,
		truth_test_df=truth_test_df,
	)


# ==============================================================
# 5.8.3 - Compute grouped validation guardrails
# ==============================================================
# What this block does:
# - Creates the deterministic grouped train/validation split for CatBoost and records the split audit table used to trace tuning behavior.
# Inputs:
# - files read: none
# - DuckDB tables read: pp_linear_hazard_ready_train
# - configuration values used: benchmark.seed, validation_test_size
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: table_catboost_weekly_hazard_not_weighted_guardrails
# - objects returned in memory: ValidationSplitState
def safe_catboost_row_budget(n_rows: int, n_features: int, requested_cap: int) -> int:
	feature_adjusted_cap = max(60_000, int(4_500_000 / max(1, n_features)))
	return int(min(n_rows, requested_cap, feature_adjusted_cap))


def build_validation_split(ctx: base.PipelineContext, treatment: CatBoostTreatmentState, window_weeks: int) -> ValidationSplitState:
	base.log_stage_start("5.8.3", "Compute grouped CatBoost validation guardrails")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	enrollment_groups = treatment.train_df["enrollment_id"].astype(str).to_numpy()
	if len(np.unique(enrollment_groups)) < 2:
		raise ValueError("The CatBoost training table must contain at least two enrollment groups for validation splitting.")

	splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_TEST_SIZE, random_state=ctx.random_seed)
	subtrain_index, validation_index = next(
		splitter.split(treatment.train_df, treatment.y_train.to_numpy(dtype=np.int32), groups=enrollment_groups)
	)

	subtrain_df = treatment.train_df.iloc[subtrain_index].reset_index(drop=True).copy()
	full_train_df = treatment.train_df.reset_index(drop=True).copy()
	validation_df = treatment.train_df.iloc[validation_index].reset_index(drop=True).copy()
	y_subtrain = treatment.y_train.iloc[subtrain_index].to_numpy(dtype=np.int32)
	y_validation = treatment.y_train.iloc[validation_index].to_numpy(dtype=np.int32)
	base.ensure_binary_target(pd.Series(y_subtrain), "catboost subtrain target")
	base.ensure_binary_target(pd.Series(y_validation), "catboost validation target")

	search_row_cap = safe_catboost_row_budget(subtrain_df.shape[0], len(treatment.feature_columns), CATBOOST_SEARCH_ROW_CAP)
	final_row_cap = safe_catboost_row_budget(full_train_df.shape[0], len(treatment.feature_columns), CATBOOST_FINAL_ROW_CAP)
	subtrain_sample_idx = base.sample_glm_rows(subtrain_df, y_subtrain, search_row_cap, ctx.random_seed + 80)
	full_sample_idx = base.sample_glm_rows(
		full_train_df,
		treatment.y_train.to_numpy(dtype=np.int32),
		final_row_cap,
		ctx.random_seed + 81,
	)
	y_subtrain_sampled = y_subtrain[subtrain_sample_idx]
	y_full_sampled = treatment.y_train.to_numpy(dtype=np.int32)[full_sample_idx]

	cat_feature_indices = [treatment.feature_columns.index(column_name) for column_name in treatment.categorical_features]
	summary = {
		"active_window_weeks": int(window_weeks),
		"validation_test_size": float(VALIDATION_TEST_SIZE),
		"search_row_cap": int(search_row_cap),
		"final_row_cap": int(final_row_cap),
		"search_rows_selected": int(subtrain_sample_idx.shape[0]),
		"final_rows_selected": int(full_sample_idx.shape[0]),
		"subtrain_rows": int(subtrain_df.shape[0]),
		"validation_rows": int(validation_df.shape[0]),
		"full_train_rows": int(treatment.train_df.shape[0]),
		"subtrain_enrollments": int(subtrain_df["enrollment_id"].nunique()),
		"validation_enrollments": int(validation_df["enrollment_id"].nunique()),
		"full_train_enrollments": int(treatment.train_df["enrollment_id"].nunique()),
		"n_features": int(len(treatment.feature_columns)),
		"n_categorical_features": int(len(treatment.categorical_features)),
		"n_numeric_features": int(len(treatment.numeric_features)),
		"subtrain_event_rate": float(np.mean(y_subtrain)),
		"sampled_subtrain_event_rate": float(np.mean(y_subtrain_sampled)),
		"validation_event_rate": float(np.mean(y_validation)),
		"full_train_event_rate": float(np.mean(treatment.y_train.to_numpy(dtype=np.int32))),
		"sampled_full_train_event_rate": float(np.mean(y_full_sampled)),
		"sample_weight_strategy": "none",
		"random_seed": int(ctx.random_seed),
	}
	guardrails_df = pd.DataFrame([summary])

	base.materialize_dataframe_table(
		ctx,
		df=guardrails_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_guardrails", window_weeks),
		block_number="5.8.3",
		label=f"Stage 5.8.3 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_guardrails', output_table_suffix)} — CatBoost validation guardrails",
	)
	base.log_stage_end("5.8.3")
	return ValidationSplitState(
		subtrain_index=subtrain_index,
		validation_index=validation_index,
		subtrain_sample_idx=subtrain_sample_idx,
		full_sample_idx=full_sample_idx,
		full_train_df=full_train_df,
		subtrain_df=subtrain_df,
		validation_df=validation_df,
		y_subtrain=y_subtrain,
		y_validation=y_validation,
		cat_feature_indices=cat_feature_indices,
		summary=summary,
	)


# ==============================================================
# 5.8.4 - Sanitize raw CatBoost feature frames deterministically
# ==============================================================
# What this block does:
# - Converts categorical columns into stable explicit strings and validates numeric columns as strictly numeric inputs for CatBoost.
# Inputs:
# - files read: none
# - DuckDB tables read: pp_linear_hazard_ready_train, pp_linear_hazard_ready_test
# - configuration values used: categorical feature list, numeric feature list
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: none
# - objects returned in memory: sanitized pandas.DataFrame objects
def sanitize_catboost_frame(
	frame: pd.DataFrame,
	categorical_features: list[str],
	numeric_features: list[str],
	frame_name: str,
) -> pd.DataFrame:
	safe = frame.copy()
	missing_columns = [column_name for column_name in categorical_features + numeric_features if column_name not in safe.columns]
	if missing_columns:
		raise KeyError(f"{frame_name} is missing required CatBoost feature columns: {', '.join(missing_columns)}")

	for column_name in categorical_features:
		column = safe[column_name].astype("string")
		column = column.fillna("missing")
		column = column.replace({"<NA>": "missing", "nan": "missing", "None": "missing"})
		safe[column_name] = column.astype(str)

	for column_name in numeric_features:
		numeric_series = pd.to_numeric(safe[column_name], errors="raise")
		safe[column_name] = numeric_series.astype(float)

	if safe[categorical_features].isna().any().any():
		raise ValueError(f"{frame_name} still contains missing categorical values after CatBoost sanitization.")
	numeric_matrix = safe[numeric_features].to_numpy(dtype=float)
	if np.isinf(numeric_matrix).any():
		raise ValueError(f"{frame_name} contains infinite numeric values that CatBoost cannot consume deterministically.")
	return safe[categorical_features + numeric_features]


# ==============================================================
# 5.8.5 - Tune, fit, evaluate, and persist the CatBoost benchmark
# ==============================================================
# What this block does:
# - Tunes the CatBoost weekly hazard model on the grouped validation split, fits the best configuration on the full training table, evaluates the resulting survival outputs, and persists all auditable artifacts.
# Inputs:
# - files read: none
# - DuckDB tables read: pp_linear_hazard_ready_train, pp_linear_hazard_ready_test
# - configuration values used: candidate grid, benchmark.seed, benchmark horizons, calibration bins
# Outputs:
# - files generated: outputs_benchmark_survival/metadata/catboost_weekly_hazard_not_weighted_tuned_model_config.json, outputs_benchmark_survival/models/catboost_weekly_hazard_not_weighted_tuned.cbm
# - DuckDB tables created or updated: table_catboost_weekly_hazard_not_weighted_tuning_results, table_catboost_weekly_hazard_not_weighted_tuned_test_predictions, table_catboost_weekly_hazard_not_weighted_tuned_primary_metrics, table_catboost_weekly_hazard_not_weighted_tuned_brier_by_horizon, table_catboost_weekly_hazard_not_weighted_tuned_secondary_metrics, table_catboost_weekly_hazard_not_weighted_tuned_td_auc_support_audit, table_catboost_weekly_hazard_not_weighted_tuned_row_diagnostics, table_catboost_weekly_hazard_not_weighted_tuned_support_by_horizon, table_catboost_weekly_hazard_not_weighted_tuned_calibration_summary, table_catboost_weekly_hazard_not_weighted_tuned_calibration_bins_by_horizon, table_catboost_weekly_hazard_not_weighted_tuned_predicted_vs_observed_survival
# - objects returned in memory: none
def tune_fit_evaluate_catboost(
	ctx: base.PipelineContext,
	treatment: CatBoostTreatmentState,
	split_state: ValidationSplitState,
	window_weeks: int,
) -> None:
	base.log_stage_start("5.8.5", "Tune, fit, and evaluate the CatBoost weekly hazard model")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	cat_subtrain_raw_full = sanitize_catboost_frame(
		split_state.subtrain_df[treatment.feature_columns],
		treatment.categorical_features,
		treatment.numeric_features,
		"catboost subtrain frame",
	)
	cat_subtrain_raw = cat_subtrain_raw_full.iloc[split_state.subtrain_sample_idx].reset_index(drop=True)
	cat_validation_raw = sanitize_catboost_frame(
		split_state.validation_df[treatment.feature_columns],
		treatment.categorical_features,
		treatment.numeric_features,
		"catboost validation frame",
	)
	cat_train_raw_full = sanitize_catboost_frame(
		split_state.full_train_df[treatment.feature_columns],
		treatment.categorical_features,
		treatment.numeric_features,
		"catboost train frame",
	)
	cat_train_raw = cat_train_raw_full.iloc[split_state.full_sample_idx].reset_index(drop=True)
	cat_test_raw = sanitize_catboost_frame(
		treatment.test_df[treatment.feature_columns],
		treatment.categorical_features,
		treatment.numeric_features,
		"catboost test frame",
	)

	candidate_rows: list[dict[str, Any]] = []
	best_candidate: dict[str, Any] | None = None

	for candidate_id, params in enumerate(CANDIDATE_GRID, start=1):
		model = CatBoostClassifier(
			loss_function="Logloss",
			eval_metric="Logloss",
			random_seed=ctx.random_seed,
			verbose=False,
			allow_writing_files=False,
			thread_count=ctx.cpu_cores,
			**params,
		)
		model.fit(
			cat_subtrain_raw,
			split_state.y_subtrain[split_state.subtrain_sample_idx],
			cat_features=split_state.cat_feature_indices,
			eval_set=(cat_validation_raw, split_state.y_validation),
			use_best_model=False,
		)
		validation_pred = np.clip(
			np.asarray(model.predict_proba(cat_validation_raw)[:, 1], dtype=float),
			EPSILON,
			1.0 - EPSILON,
		)
		if not np.isfinite(validation_pred).all():
			raise ValueError(f"Candidate {candidate_id} produced non-finite validation probabilities.")

		validation_log_loss = float(log_loss(split_state.y_validation, validation_pred, labels=[0, 1]))
		validation_brier = float(brier_score_loss(split_state.y_validation, validation_pred))
		validation_roc_auc = (
			float(roc_auc_score(split_state.y_validation, validation_pred))
			if np.unique(split_state.y_validation).shape[0] >= 2
			else np.nan
		)
		candidate_record = {
			"candidate_id": int(candidate_id),
			"depth": int(params["depth"]),
			"learning_rate": float(params["learning_rate"]),
			"iterations": int(params["iterations"]),
			"l2_leaf_reg": float(params["l2_leaf_reg"]),
			"fit_rows": int(cat_subtrain_raw.shape[0]),
			"fit_features": int(cat_subtrain_raw.shape[1]),
			"val_log_loss": validation_log_loss,
			"val_brier": validation_brier,
			"val_roc_auc": validation_roc_auc,
		}
		candidate_rows.append(candidate_record)
		if best_candidate is None or validation_log_loss < float(best_candidate["val_log_loss"]):
			best_candidate = candidate_record.copy()

		del model
		gc.collect()

	if best_candidate is None:
		raise RuntimeError("All CatBoost candidates failed in stage 5.8.5.")

	tuning_results_df = (
		pd.DataFrame(candidate_rows)
		.sort_values(["val_log_loss", "candidate_id"], ascending=[True, True], kind="mergesort")
		.reset_index(drop=True)
	)

	final_model = CatBoostClassifier(
		loss_function="Logloss",
		eval_metric="Logloss",
		random_seed=ctx.random_seed,
		verbose=False,
		allow_writing_files=False,
		thread_count=ctx.cpu_cores,
		depth=int(best_candidate["depth"]),
		learning_rate=float(best_candidate["learning_rate"]),
		iterations=int(best_candidate["iterations"]),
		l2_leaf_reg=float(best_candidate["l2_leaf_reg"]),
	)
	final_model.fit(
		cat_train_raw,
		treatment.y_train.to_numpy(dtype=np.int32)[split_state.full_sample_idx],
		cat_features=split_state.cat_feature_indices,
	)
	test_hazard = np.clip(
		np.asarray(final_model.predict_proba(cat_test_raw)[:, 1], dtype=float),
		EPSILON,
		1.0 - EPSILON,
	)
	if not np.isfinite(test_hazard).all():
		raise ValueError("The tuned CatBoost model produced non-finite test probabilities.")

	test_predictions_df = treatment.test_df.copy().sort_values(["enrollment_id", "week"]).reset_index(drop=True)
	test_predictions_df["pred_hazard"] = test_hazard
	test_predictions_df["pred_survival"] = test_predictions_df.groupby("enrollment_id")["pred_hazard"].transform(
		lambda series: (1.0 - series).cumprod()
	)
	test_predictions_df["pred_risk"] = 1.0 - test_predictions_df["pred_survival"]
	if not np.isfinite(test_predictions_df[["pred_hazard", "pred_survival", "pred_risk"]].to_numpy()).all():
		raise ValueError("The CatBoost prediction surface contains non-finite values.")

	artifacts = base.evaluate_discrete_predictions(
		model_name=MODEL_NAME,
		test_predictions_df=test_predictions_df,
		truth_train_df=treatment.truth_train_df,
		truth_test_df=treatment.truth_test_df,
		y_test_row=treatment.y_test,
	)

	model_path = build_window_artifact_path(ctx.models_dir, MODEL_ARTIFACT_NAME, window_weeks)
	config_path = build_window_artifact_path(ctx.metadata_dir, MODEL_CONFIG_NAME, window_weeks)
	final_model.save_model(str(model_path))

	base.materialize_dataframe_table(
		ctx,
		df=tuning_results_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuning_results", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuning_results', output_table_suffix)} — CatBoost tuning results",
	)
	base.materialize_dataframe_table(
		ctx,
		df=test_predictions_df,
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_test_predictions", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_test_predictions', output_table_suffix)} — CatBoost test predictions",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["primary_metrics_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_primary_metrics", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_primary_metrics', output_table_suffix)} — CatBoost primary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["brier_by_horizon_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_brier_by_horizon", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_brier_by_horizon', output_table_suffix)} — CatBoost Brier scores by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["secondary_metrics_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_secondary_metrics", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_secondary_metrics', output_table_suffix)} — CatBoost secondary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["td_auc_support_audit_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_td_auc_support_audit", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_td_auc_support_audit', output_table_suffix)} — CatBoost IPCW support audit",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["row_diagnostics_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_row_diagnostics", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_row_diagnostics', output_table_suffix)} — CatBoost row-level diagnostics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["support_by_horizon_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_support_by_horizon", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_support_by_horizon', output_table_suffix)} — CatBoost support by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_summary_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_calibration_summary", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_calibration_summary', output_table_suffix)} — CatBoost calibration summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_bins_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_calibration_bins_by_horizon", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_calibration_bins_by_horizon', output_table_suffix)} — CatBoost calibration bins",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["predicted_vs_observed_survival_df"],
		table_name=apply_window_table_name("table_catboost_weekly_hazard_not_weighted_tuned_predicted_vs_observed_survival", window_weeks),
		block_number="5.8.5",
		label=f"Stage 5.8.5 {apply_name_suffix('table_catboost_weekly_hazard_not_weighted_tuned_predicted_vs_observed_survival', output_table_suffix)} — CatBoost predicted versus observed survival",
	)

	config_payload = {
		"model_name": MODEL_NAME,
		"family_name": "discrete_time_dynamic",
		"model_type": MODEL_TYPE,
		"selection_metric": "val_log_loss",
		"candidate_grid": [dict(candidate) for candidate in CANDIDATE_GRID],
		"validation_split": {
			"method": "GroupShuffleSplit",
			"test_size": VALIDATION_TEST_SIZE,
			"group_column": "enrollment_id",
			"random_seed": ctx.random_seed,
		},
		"benchmark_horizons": list(base.BENCHMARK_HORIZONS),
		"calibration_bins": base.CALIBRATION_BINS,
		"calibration_contract_version": base.CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": base.CALIBRATION_OBSERVED_RISK_METHOD,
		"active_window_weeks": int(window_weeks),
		"categorical_features": treatment.categorical_features,
		"numeric_features": treatment.numeric_features,
		"guardrails": {
			key: (
				float(value)
				if isinstance(value, np.floating)
				else int(value)
				if isinstance(value, np.integer)
				else value
			)
			for key, value in split_state.summary.items()
		},
		"best_candidate": {
			"depth": int(best_candidate["depth"]),
			"learning_rate": float(best_candidate["learning_rate"]),
			"iterations": int(best_candidate["iterations"]),
			"l2_leaf_reg": float(best_candidate["l2_leaf_reg"]),
			"val_log_loss": float(best_candidate["val_log_loss"]),
			"val_brier": float(best_candidate["val_brier"]),
			"val_roc_auc": (
				float(best_candidate["val_roc_auc"])
				if pd.notna(best_candidate["val_roc_auc"])
				else None
			),
		},
		"implementation": {
			"library": "catboost",
			"class_name": "CatBoostClassifier",
			"loss_function": "Logloss",
			"eval_metric": "Logloss",
			"thread_count": 1,
			"allow_writing_files": False,
		},
		"weighting_strategy": "none; this official D5.8 variant preserves native categorical handling and the same dynamic feature contract without class weights.",
		"design_note": "The not-weighted CatBoost benchmark preserves native categorical columns, grouped validation, and the shared evaluation stack so the weighted script remains a pure Item 6 sensitivity counterpart.",
	}
	base.save_json(config_payload, config_path)
	base.print_artifact("catboost_weekly_hazard_model", str(model_path))
	base.print_artifact("catboost_weekly_hazard_not_weighted_tuned_model_config", str(config_path))
	print(config_path.read_text(encoding="utf-8"))
	print(tuning_results_df.head(base.PREVIEW_ROWS).to_string(index=False))

	base.log_stage_end("5.8.5")


# ==============================================================
# 5.8.6 - Close the DuckDB runtime cleanly
# ==============================================================
# What this block does:
# - Closes the DuckDB connection with a checkpoint after all CatBoost outputs have been materialized.
# Inputs:
# - files read: none
# - DuckDB tables read: none
# - configuration values used: none
# Outputs:
# - files generated: none
# - DuckDB tables created or updated: none
# - objects returned in memory: none
def close_context(ctx: base.PipelineContext) -> None:
	from util import close_duckdb_connection

	base.log_stage_start("5.8.6", "Close the DuckDB runtime cleanly")
	ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
	base.log_stage_end("5.8.6")


def main() -> None:
	ctx = initialize_context()
	try:
		for window_weeks in resolve_dynamic_window_execution_weeks(ctx):
			print(f"[RUN_PLAN] dynamic_window_weeks={int(window_weeks)} script={SCRIPT_NAME}")
			treatment = build_catboost_treatment(ctx, int(window_weeks))
			split_state = build_validation_split(ctx, treatment, int(window_weeks))
			tune_fit_evaluate_catboost(ctx, treatment, split_state, int(window_weeks))
	finally:
		close_context(ctx)


if __name__ == "__main__":
	main()
