from __future__ import annotations

"""
Production weighted Poisson piecewise-exponential benchmark module for Item 6 sensitivity.

What this file does:
- reuses the weekly dynamic treatment contract from D5.6 without touching the original cloglog module
- fits a penalized Poisson GLM on the same person-period design used by the cloglog benchmark
- converts predicted weekly event rates into discrete hazards and survival trajectories
- evaluates the tuned model with the same survival, calibration, discrimination, and row-level diagnostics used by D5.6
- persists dedicated DuckDB tables, model artifacts, and metadata under a separate namespace

Main processing purpose:
- provide a minimally disruptive alternative to the cloglog benchmark while preserving the same benchmark arm, feature contract, split discipline, and evaluation outputs

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB tables table_poisson_pexp_weighted_preprocessing_summary, table_poisson_pexp_weighted_raw_feature_manifest,
  table_poisson_pexp_weighted_feature_names_out, table_poisson_pexp_weighted_guardrails,
  table_poisson_pexp_weighted_tuning_results, table_poisson_pexp_weighted_tuned_test_predictions,
  table_poisson_pexp_weighted_tuned_primary_metrics, table_poisson_pexp_weighted_tuned_brier_by_horizon,
  table_poisson_pexp_weighted_tuned_secondary_metrics, table_poisson_pexp_weighted_tuned_td_auc_support_audit,
  table_poisson_pexp_weighted_tuned_row_diagnostics, table_poisson_pexp_weighted_tuned_support_by_horizon,
  table_poisson_pexp_weighted_tuned_calibration_summary, table_poisson_pexp_weighted_tuned_calibration_bins_by_horizon,
  table_poisson_pexp_weighted_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/poisson_piecewise_exponential_weighted_preprocessing_config.json
- outputs_benchmark_survival/metadata/poisson_piecewise_exponential_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/poisson_piecewise_exponential_weighted_tuned.joblib
- outputs_benchmark_survival/models/poisson_piecewise_exponential_weighted_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, invalid contracts, or missing dependencies raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, or CSV-based workflows are permitted
"""

import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from statsmodels.genmod.families import Poisson

import dropout_bench_v3_D_06_A_dynamic_weekly_poisson_piecewise_exponential as base
from dropout_bench_v3_D_00_common import (
	append_suffix_before_extension,
	apply_name_suffix,
	resolve_early_window_sensitivity_weeks,
)


SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_D_5_6_weighted.ipynb"
MODEL_NAME = "poisson_piecewise_exponential_weighted_tuned"
TABLE_PREFIX = "poisson_pexp"
PREPROCESSING_CONFIG_NAME = "poisson_piecewise_exponential_weighted_preprocessing_config.json"
MODEL_CONFIG_NAME = "poisson_piecewise_exponential_weighted_tuned_model_config.json"
MODEL_ARTIFACT_NAME = "poisson_piecewise_exponential_weighted_tuned.joblib"
PREPROCESSOR_ARTIFACT_NAME = "poisson_piecewise_exponential_weighted_preprocessor.joblib"
ALPHA_GRID = (2e-5, 5e-5, 1e-4, 1e-3, 1e-2, 5e-2)
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


def initialize_context() -> base.PipelineContext:
	from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores, resolve_runtime_tuning_parallel_backend
	from util import ensure_pipeline_catalog, open_duckdb_connection

	base.log_stage_start("5.6b.1", "Lightweight runtime bootstrap for the Poisson benchmark")

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
		run_metadata = base.json.load(file_obj)

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
	tuning_parallel_backend = resolve_runtime_tuning_parallel_backend(shared_config)

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
	base.require_tables(con, base.REQUIRED_INPUT_TABLES, block_number="5.6b.1")

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

	print(f"- CPU_CORES: {ctx.cpu_cores}")
	print(f"- TUNING_BACKEND: {tuning_parallel_backend}")

	print(f"- SCRIPT_NAME: {ctx.script_name}")
	print(f"- RUN_ID: {ctx.run_id}")
	print(f"- DUCKDB_PATH: {ctx.duckdb_path}")
	print(f"- BENCHMARK_HORIZONS: {list(base.BENCHMARK_HORIZONS)}")
	print(f"- CALIBRATION_BINS: {base.CALIBRATION_BINS}")
	base.print_artifact("shared_config", str(ctx.config_toml_path))
	base.print_artifact("modeling_contract", str(ctx.modeling_contract_toml_path))
	base.print_artifact("run_metadata", str(ctx.run_metadata_path))
	base.log_stage_end("5.6b.1")
	return ctx


def build_dynamic_treatment(ctx: base.PipelineContext, window_weeks: int) -> base.DynamicTreatmentState:
	base.log_stage_start("5.6b.2", "Prepare the shared dynamic weekly treatment for Poisson")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	feature_contract = ctx.shared_modeling_contract["feature_contract"]
	static_features = base.require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
	temporal_features_discrete = base.require_list_of_strings(
		feature_contract["temporal_features_discrete"],
		"feature_contract.temporal_features_discrete",
	)
	expected_features_raw = static_features + temporal_features_discrete
	expected_features_resolved = [base.FEATURE_ALIAS_MAP.get(feature_name, feature_name) for feature_name in expected_features_raw]

	train_df = base.load_required_table(ctx, "pp_linear_hazard_ready_train", base.REQUIRED_PERSON_PERIOD_COLUMNS, "5.6b.2")
	test_df = base.load_required_table(ctx, "pp_linear_hazard_ready_test", base.REQUIRED_PERSON_PERIOD_COLUMNS, "5.6b.2")
	raw_train_rows = int(len(train_df))
	raw_test_rows = int(len(test_df))
	if WINDOW_WEEK_COLUMN not in train_df.columns or WINDOW_WEEK_COLUMN not in test_df.columns:
		raise KeyError(f"Dynamic Poisson treatment requires the '{WINDOW_WEEK_COLUMN}' column in both train and test tables.")
	train_df = train_df.copy()
	test_df = test_df.copy()
	train_df[WINDOW_WEEK_COLUMN] = pd.to_numeric(train_df[WINDOW_WEEK_COLUMN], errors="raise").astype(int)
	test_df[WINDOW_WEEK_COLUMN] = pd.to_numeric(test_df[WINDOW_WEEK_COLUMN], errors="raise").astype(int)
	train_df = train_df.loc[train_df[WINDOW_WEEK_COLUMN] <= int(window_weeks)].reset_index(drop=True)
	test_df = test_df.loc[test_df[WINDOW_WEEK_COLUMN] <= int(window_weeks)].reset_index(drop=True)
	if train_df.empty or test_df.empty:
		raise ValueError(f"Window truncation at w={int(window_weeks)} produced an empty train/test dynamic Poisson table.")

	missing_from_treatment_spec = [
		feature_name for feature_name in expected_features_resolved if feature_name not in base.CATEGORICAL_FEATURES + base.NUMERIC_FEATURES
	]
	if missing_from_treatment_spec:
		raise ValueError(
			"Configured canonical features are not covered by the operational Poisson treatment spec after alias resolution: "
			+ ", ".join(missing_from_treatment_spec)
		)

	feature_columns = base.CATEGORICAL_FEATURES + base.NUMERIC_FEATURES
	missing_in_train = [column for column in feature_columns if column not in train_df.columns]
	missing_in_test = [column for column in feature_columns if column not in test_df.columns]
	if missing_in_train or missing_in_test:
		raise KeyError(
			"Operational Poisson feature columns are missing from the materialized train/test tables. "
			f"Missing in train: {missing_in_train}. Missing in test: {missing_in_test}"
		)

	target_col = "event_t"
	y_train = base.ensure_binary_target(train_df[target_col], "pp_linear_hazard_ready_train.event_t")
	y_test = base.ensure_binary_target(test_df[target_col], "pp_linear_hazard_ready_test.event_t")

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
				base.NUMERIC_FEATURES,
			),
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
						("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
					]
				),
				base.CATEGORICAL_FEATURES,
			),
		],
		remainder="drop",
		sparse_threshold=1.0,
		verbose_feature_names_out=True,
	)

	X_train = preprocessor.fit_transform(train_df[feature_columns].copy())
	X_test = preprocessor.transform(test_df[feature_columns].copy())
	feature_names_out = preprocessor.get_feature_names_out().tolist()
	if len(feature_names_out) == 0:
		raise ValueError("Poisson preprocessing produced zero output features.")

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
				"n_numeric_features": int(len(base.NUMERIC_FEATURES)),
				"n_categorical_features": int(len(base.CATEGORICAL_FEATURES)),
				"n_output_features_after_preprocessing": int(len(feature_names_out)),
				"train_event_rate": float(y_train.mean()),
				"test_event_rate": float(y_test.mean()),
			}
		]
	)
	feature_manifest_df = pd.DataFrame(
		{
			"feature_name_operational": feature_columns,
			"feature_role": ["categorical" if feature_name in base.CATEGORICAL_FEATURES else "numeric" for feature_name in feature_columns],
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
	feature_names_df = pd.DataFrame({"preprocessed_feature_name": feature_names_out})

	base.materialize_dataframe_table(
		ctx,
		df=preprocessing_summary_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_preprocessing_summary", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_weighted_preprocessing_summary', output_table_suffix)} — Poisson preprocessing summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_manifest_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_raw_feature_manifest", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_weighted_raw_feature_manifest', output_table_suffix)} — Poisson raw feature manifest",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_names_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_feature_names_out", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_weighted_feature_names_out', output_table_suffix)} — Poisson transformed feature names",
	)

	preprocessing_config = {
		"model_name": MODEL_NAME,
		"train_table": "pp_linear_hazard_ready_train",
		"test_table": "pp_linear_hazard_ready_test",
		"target_column": target_col,
		"categorical_features": base.CATEGORICAL_FEATURES,
		"numeric_features": base.NUMERIC_FEATURES,
		"operational_feature_columns": feature_columns,
		"canonical_expected_features_raw": expected_features_raw,
		"canonical_expected_features_resolved": expected_features_resolved,
		"feature_alias_map": base.FEATURE_ALIAS_MAP,
		"n_train_rows": int(len(train_df)),
		"n_test_rows": int(len(test_df)),
		"active_window_weeks": int(window_weeks),
		"raw_train_rows_before_truncation": raw_train_rows,
		"raw_test_rows_before_truncation": raw_test_rows,
		"n_output_features_after_preprocessing": int(len(feature_names_out)),
		"train_event_rate": float(y_train.mean()),
		"test_event_rate": float(y_test.mean()),
		"design_note": "The Poisson piecewise-exponential benchmark reuses the same weekly dynamic treatment contract as D5.6 and only swaps the GLM family.",
		"methodological_note": "All learned preprocessing operations were fit on training data only and then applied unchanged to test data.",
	}
	preprocessing_config_path = build_window_artifact_path(ctx.metadata_dir, PREPROCESSING_CONFIG_NAME, window_weeks)
	base.save_json(preprocessing_config, preprocessing_config_path)
	base.print_artifact("poisson_pexp_preprocessing_config", str(preprocessing_config_path))
	print(preprocessing_config_path.read_text(encoding="utf-8"))

	base.log_stage_end("5.6b.2")
	return base.DynamicTreatmentState(
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
		truth_train_df=truth_train_df,
		truth_test_df=truth_test_df,
	)


def build_guardrails(ctx: base.PipelineContext, treatment: base.DynamicTreatmentState, window_weeks: int) -> dict[str, object]:
	base.log_stage_start("5.6b.3", "Compute runtime guardrails for the Poisson GLM")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	enrollment_groups = treatment.train_df["enrollment_id"].astype(str).to_numpy()
	if len(np.unique(enrollment_groups)) < 2:
		raise ValueError("The Poisson training table must contain at least two enrollment groups for validation splitting.")

	splitter = GroupShuffleSplit(n_splits=1, test_size=base.VALIDATION_TEST_SIZE, random_state=ctx.random_seed)
	subtrain_index, validation_index = next(
		splitter.split(treatment.X_train, treatment.y_train.to_numpy(), groups=enrollment_groups)
	)

	subtrain_df = treatment.train_df.iloc[subtrain_index].reset_index(drop=True).copy()
	full_train_df = treatment.train_df.reset_index(drop=True).copy()
	y_subtrain = treatment.y_train.iloc[subtrain_index].to_numpy(dtype=np.int32)
	y_validation = treatment.y_train.iloc[validation_index].to_numpy(dtype=np.int32)
	base.ensure_binary_target(pd.Series(y_validation), "poisson validation target")

	X_subtrain = treatment.X_train[subtrain_index]
	X_validation = treatment.X_train[validation_index]
	X_full = treatment.X_train
	X_test = treatment.X_test

	search_row_cap = base.safe_glm_row_budget(X_subtrain.shape[0], X_subtrain.shape[1], base.GLM_SEARCH_ROW_CAP)
	final_row_cap = base.safe_glm_row_budget(X_full.shape[0], X_full.shape[1], base.GLM_FINAL_ROW_CAP)
	subtrain_sample_idx = base.sample_glm_rows(subtrain_df, y_subtrain, search_row_cap, ctx.random_seed)
	full_sample_idx = base.sample_glm_rows(full_train_df, treatment.y_train.to_numpy(dtype=np.int32), final_row_cap, ctx.random_seed + 1)
	y_subtrain_sampled = y_subtrain[subtrain_sample_idx]
	y_full_sampled = treatment.y_train.to_numpy(dtype=np.int32)[full_sample_idx]
	positive_weight_subtrain = float(max(1, int(y_subtrain_sampled.shape[0] - y_subtrain_sampled.sum())) / max(1, int(y_subtrain_sampled.sum())))
	positive_weight_full = float(max(1, int(y_full_sampled.shape[0] - y_full_sampled.sum())) / max(1, int(y_full_sampled.sum())))

	guardrails = {
		"validation_index": validation_index,
		"subtrain_index": subtrain_index,
		"X_subtrain": X_subtrain,
		"X_validation": X_validation,
		"X_full": X_full,
		"X_test": X_test,
		"y_subtrain": y_subtrain,
		"y_validation": y_validation,
		"subtrain_df": subtrain_df,
		"full_train_df": full_train_df,
		"subtrain_sample_idx": subtrain_sample_idx,
		"full_sample_idx": full_sample_idx,
		"summary": {
			"active_window_weeks": int(window_weeks),
			"search_row_cap": int(search_row_cap),
			"final_row_cap": int(final_row_cap),
			"search_rows_selected": int(subtrain_sample_idx.shape[0]),
			"final_rows_selected": int(full_sample_idx.shape[0]),
			"subtrain_total_rows": int(X_subtrain.shape[0]),
			"full_total_rows": int(X_full.shape[0]),
			"n_features": int(X_full.shape[1]),
			"subtrain_event_rate": float(np.mean(y_subtrain)),
			"full_event_rate": float(np.mean(treatment.y_train.to_numpy(dtype=np.int32))),
			"sampled_subtrain_event_rate": float(np.mean(y_subtrain_sampled)),
			"sampled_full_event_rate": float(np.mean(y_full_sampled)),
			"positive_weight_subtrain": positive_weight_subtrain,
			"positive_weight_full": positive_weight_full,
			"sample_weight_strategy": "class-frequency frequency weights",
		},
	}
	guardrails_df = pd.DataFrame([guardrails["summary"]])

	base.materialize_dataframe_table(
		ctx,
		df=guardrails_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_guardrails", window_weeks),
		block_number="5.6b.3",
		label=f"Stage 5.6b.3 {apply_name_suffix('table_poisson_pexp_weighted_guardrails', output_table_suffix)} — Poisson runtime guardrails",
	)
	base.log_stage_end("5.6b.3")
	return guardrails


def tune_and_evaluate_poisson_model(
	ctx: base.PipelineContext,
	treatment: base.DynamicTreatmentState,
	guardrails: dict[str, object],
	window_weeks: int,
) -> None:
	from dropout_bench_v3_A_2_runtime_config import resolve_runtime_tuning_parallel_backend

	base.log_stage_start("5.6b.4", "Tune and evaluate the Poisson piecewise-exponential model")
	output_table_suffix = build_dynamic_window_suffix(window_weeks)

	X_subtrain = base.matrix_to_float64_dense(guardrails["X_subtrain"][guardrails["subtrain_sample_idx"]])
	X_validation = base.matrix_to_float64_dense(guardrails["X_validation"])
	X_full = base.matrix_to_float64_dense(guardrails["X_full"][guardrails["full_sample_idx"]])
	X_test = base.matrix_to_float64_dense(guardrails["X_test"])
	y_subtrain = np.asarray(guardrails["y_subtrain"][guardrails["subtrain_sample_idx"]], dtype=np.int32)
	y_validation = np.asarray(guardrails["y_validation"], dtype=np.int32)
	y_full = np.asarray(treatment.y_train.to_numpy(dtype=np.int32)[guardrails["full_sample_idx"]], dtype=np.int32)
	subtrain_weights = np.where(y_subtrain == 1, float(guardrails["summary"]["positive_weight_subtrain"]), 1.0)
	full_weights = np.where(y_full == 1, float(guardrails["summary"]["positive_weight_full"]), 1.0)

	X_subtrain_const = sm.add_constant(X_subtrain, prepend=True, has_constant="add")
	X_validation_const = sm.add_constant(X_validation, prepend=True, has_constant="add")
	X_full_const = sm.add_constant(X_full, prepend=True, has_constant="add")
	X_test_const = sm.add_constant(X_test, prepend=True, has_constant="add")

	tuning_workers = max(1, min(int(ctx.cpu_cores), len(ALPHA_GRID)))
	tuning_parallel_backend = resolve_runtime_tuning_parallel_backend(ctx.shared_config)
	print(f"- TUNING_WORKERS: {tuning_workers}")
	print(f"- TUNING_BACKEND: {tuning_parallel_backend}")

	def evaluate_alpha(candidate_id: int, alpha: float) -> dict[str, object]:
		glm = sm.GLM(y_subtrain, X_subtrain_const, family=Poisson(), freq_weights=subtrain_weights)
		try:
			result = glm.fit_regularized(alpha=float(alpha), L1_wt=0.0, maxiter=base.REGULARIZED_MAXITER_SEARCH)
			validation_rate = np.clip(np.asarray(result.predict(X_validation_const), dtype=float), base.EPSILON, None)
			validation_pred = np.clip(1.0 - np.exp(-validation_rate), base.EPSILON, 1.0 - base.EPSILON)
			validation_log_loss = float(log_loss(y_validation, validation_pred, labels=[0, 1]))
			validation_brier = float(brier_score_loss(y_validation, validation_pred))
			validation_roc_auc = float(roc_auc_score(y_validation, validation_pred))
			status = "ok"
		except Exception as exc:
			result = None
			validation_log_loss = np.nan
			validation_brier = np.nan
			validation_roc_auc = np.nan
			status = f"failed: {str(exc)}"

		candidate_row = {
			"candidate_id": int(candidate_id),
			"alpha": float(alpha),
			"fit_rows": int(X_subtrain.shape[0]),
			"fit_features": int(X_subtrain.shape[1]),
			"positive_weight": float(guardrails["summary"]["positive_weight_subtrain"]),
			"val_log_loss": validation_log_loss,
			"val_brier": validation_brier,
			"val_roc_auc": validation_roc_auc,
			"fit_status": status,
		}

		del glm
		if result is not None:
			del result
		gc.collect()
		return candidate_row

	candidate_rows = joblib.Parallel(n_jobs=tuning_workers, prefer=tuning_parallel_backend)(
		joblib.delayed(evaluate_alpha)(candidate_id, float(alpha))
		for candidate_id, alpha in enumerate(ALPHA_GRID, start=1)
	)
	valid_candidates = [row for row in candidate_rows if row["fit_status"] == "ok"]
	best_row = min(valid_candidates, key=lambda row: (float(row["val_log_loss"]), int(row["candidate_id"]))) if valid_candidates else None
	best_alpha = float(best_row["alpha"]) if best_row is not None else None
	best_validation_log_loss = float(best_row["val_log_loss"]) if best_row is not None else None

	if best_alpha is None or best_validation_log_loss is None:
		raise RuntimeError("All guarded Poisson candidates failed in stage 5.6b.4.")

	tuning_results_df = (
		pd.DataFrame(candidate_rows)
		.sort_values(["fit_status", "val_log_loss", "candidate_id"], ascending=[True, True, True], kind="mergesort")
		.reset_index(drop=True)
	)

	final_glm = sm.GLM(y_full, X_full_const, family=Poisson(), freq_weights=full_weights)
	final_model = final_glm.fit_regularized(alpha=best_alpha, L1_wt=0.0, maxiter=base.REGULARIZED_MAXITER_FINAL)
	test_rate = np.clip(np.asarray(final_model.predict(X_test_const), dtype=float), base.EPSILON, None)
	test_hazard = np.clip(1.0 - np.exp(-test_rate), base.EPSILON, 1.0 - base.EPSILON)

	test_predictions_df = treatment.test_df.copy().sort_values(["enrollment_id", "week"]).reset_index(drop=True)
	test_predictions_df["pred_rate"] = test_rate
	test_predictions_df["pred_hazard"] = test_hazard
	test_predictions_df["pred_survival"] = test_predictions_df.groupby("enrollment_id")["pred_hazard"].transform(
		lambda series: (1.0 - series).cumprod()
	)
	test_predictions_df["pred_risk"] = 1.0 - test_predictions_df["pred_survival"]
	if not np.isfinite(test_predictions_df[["pred_rate", "pred_hazard", "pred_survival", "pred_risk"]].to_numpy()).all():
		raise ValueError("The tuned Poisson model produced non-finite predictions.")

	artifacts = base.evaluate_discrete_predictions(
		model_name=MODEL_NAME,
		test_predictions_df=test_predictions_df,
		truth_train_df=treatment.truth_train_df,
		truth_test_df=treatment.truth_test_df,
		y_test_row=treatment.y_test,
	)

	model_path = build_window_artifact_path(ctx.models_dir, MODEL_ARTIFACT_NAME, window_weeks)
	preprocessor_path = build_window_artifact_path(ctx.models_dir, PREPROCESSOR_ARTIFACT_NAME, window_weeks)
	config_path = build_window_artifact_path(ctx.metadata_dir, MODEL_CONFIG_NAME, window_weeks)

	joblib.dump(final_model, model_path)
	joblib.dump(treatment.preprocessor, preprocessor_path)

	base.materialize_dataframe_table(
		ctx,
		df=tuning_results_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuning_results", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuning_results', output_table_suffix)} — Poisson tuning results",
	)
	base.materialize_dataframe_table(
		ctx,
		df=test_predictions_df,
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_test_predictions", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_test_predictions', output_table_suffix)} — Poisson test predictions",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["primary_metrics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_primary_metrics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_primary_metrics', output_table_suffix)} — Poisson primary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["brier_by_horizon_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_brier_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_brier_by_horizon', output_table_suffix)} — Poisson Brier scores by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["secondary_metrics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_secondary_metrics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_secondary_metrics', output_table_suffix)} — Poisson secondary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["td_auc_support_audit_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_td_auc_support_audit", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_td_auc_support_audit', output_table_suffix)} — Poisson IPCW support audit",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["row_diagnostics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_row_diagnostics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_row_diagnostics', output_table_suffix)} — Poisson row-level diagnostics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["support_by_horizon_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_support_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_support_by_horizon', output_table_suffix)} — Poisson support by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_summary_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_calibration_summary", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_calibration_summary', output_table_suffix)} — Poisson calibration summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_bins_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_calibration_bins_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_calibration_bins_by_horizon', output_table_suffix)} — Poisson calibration bins",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["predicted_vs_observed_survival_df"],
		table_name=apply_window_table_name("table_poisson_pexp_weighted_tuned_predicted_vs_observed_survival", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_weighted_tuned_predicted_vs_observed_survival', output_table_suffix)} — Poisson predicted versus observed survival",
	)

	config_payload = {
		"model_name": MODEL_NAME,
		"family_name": "discrete_time_dynamic",
		"model_type": "poisson_piecewise_exponential",
		"alpha_grid": list(ALPHA_GRID),
		"selection_metric": "val_log_loss_on_discrete_hazard",
		"validation_split": {
			"method": "GroupShuffleSplit",
			"test_size": base.VALIDATION_TEST_SIZE,
			"group_column": "enrollment_id",
			"random_seed": ctx.random_seed,
		},
		"guardrails": {
			key: (
				float(value)
				if isinstance(value, np.floating)
				else int(value)
				if isinstance(value, np.integer)
				else value
			)
			for key, value in guardrails["summary"].items()
		},
		"benchmark_horizons": list(base.BENCHMARK_HORIZONS),
		"calibration_bins": base.CALIBRATION_BINS,
		"active_window_weeks": int(window_weeks),
		"best_candidate": {
			"alpha": float(best_alpha),
			"val_log_loss": float(best_validation_log_loss),
		},
		"preprocessing": {
			"categorical_features": base.CATEGORICAL_FEATURES,
			"numeric_features": base.NUMERIC_FEATURES,
			"categorical_encoder": "OneHotEncoder(handle_unknown=ignore, sparse_output=True)",
			"numeric_scaler": "MaxAbsScaler",
		},
		"event_rate_to_hazard_mapping": "pred_hazard = 1 - exp(-pred_rate)",
		"weighting_strategy": "Poisson GLM frequency weights are derived from the guarded positive-class imbalance ratio on sampled person-period rows.",
		"design_note": "This weighted Poisson variant preserves the dynamic weekly contract and changes only the fitting weights for Item 6 sensitivity.",
	}
	base.save_json(config_payload, config_path)
	base.print_artifact("poisson_pexp_model", str(model_path))
	base.print_artifact("poisson_pexp_preprocessor", str(preprocessor_path))
	base.print_artifact("poisson_pexp_tuned_model_config", str(config_path))
	print(config_path.read_text(encoding="utf-8"))
	print(tuning_results_df.head(base.PREVIEW_ROWS).to_string(index=False))

	base.log_stage_end("5.6b.4")


def close_context(ctx: base.PipelineContext) -> None:
	from util import close_duckdb_connection

	base.log_stage_start("5.6b.5", "Close the DuckDB runtime cleanly")
	ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
	base.log_stage_end("5.6b.5")


def main() -> None:
	ctx = initialize_context()
	try:
		for window_weeks in resolve_dynamic_window_execution_weeks(ctx):
			print(f"[RUN_PLAN] dynamic_window_weeks={int(window_weeks)} script={SCRIPT_NAME}")
			treatment_state = build_dynamic_treatment(ctx, int(window_weeks))
			guardrails = build_guardrails(ctx, treatment_state, int(window_weeks))
			tune_and_evaluate_poisson_model(ctx, treatment_state, guardrails, int(window_weeks))
	finally:
		close_context(ctx)


if __name__ == "__main__":
	main()