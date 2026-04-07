from __future__ import annotations

"""
Production not-weighted Poisson piecewise-exponential benchmark module for the official D5.6 arm.

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
- DuckDB tables table_poisson_pexp_not_weighted_preprocessing_summary, table_poisson_pexp_not_weighted_raw_feature_manifest,
  table_poisson_pexp_not_weighted_feature_names_out, table_poisson_pexp_not_weighted_guardrails,
  table_poisson_pexp_not_weighted_tuning_results, table_poisson_pexp_not_weighted_tuned_test_predictions,
  table_poisson_pexp_not_weighted_tuned_primary_metrics, table_poisson_pexp_not_weighted_tuned_brier_by_horizon,
  table_poisson_pexp_not_weighted_tuned_secondary_metrics, table_poisson_pexp_not_weighted_tuned_td_auc_support_audit,
  table_poisson_pexp_not_weighted_tuned_row_diagnostics, table_poisson_pexp_not_weighted_tuned_support_by_horizon,
  table_poisson_pexp_not_weighted_tuned_calibration_summary, table_poisson_pexp_not_weighted_tuned_calibration_bins_by_horizon,
  table_poisson_pexp_not_weighted_tuned_predicted_vs_observed_survival
- outputs_benchmark_survival/metadata/poisson_piecewise_exponential_not_weighted_preprocessing_config.json
- outputs_benchmark_survival/metadata/poisson_piecewise_exponential_not_weighted_tuned_model_config.json
- outputs_benchmark_survival/models/poisson_piecewise_exponential_not_weighted_tuned.joblib
- outputs_benchmark_survival/models/poisson_piecewise_exponential_not_weighted_preprocessor.joblib

Failure policy:
- missing DuckDB tables, missing columns, invalid contracts, or missing dependencies raise immediately
- invalid preprocessing outputs, unsupported evaluation subsets, or non-finite model outputs raise immediately
- no fallback behavior, silent degradation, or CSV-based workflows are permitted
"""

import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pycox.evaluation import EvalSurv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.util import Surv
from statsmodels.genmod.families import Poisson

import dropout_bench_v3_D_02_A_dynamic_weekly_linear_discrete_time_hazard as shared_base
from dropout_bench_v3_D_00_common import (
	append_suffix_before_extension,
	apply_name_suffix,
	resolve_early_window_sensitivity_weeks,
)


SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_D_5_6_not_weighted.ipynb"
MODEL_NAME = "poisson_piecewise_exponential_not_weighted_tuned"
TABLE_PREFIX = "poisson_pexp"
PREPROCESSING_CONFIG_NAME = "poisson_piecewise_exponential_not_weighted_preprocessing_config.json"
MODEL_CONFIG_NAME = "poisson_piecewise_exponential_not_weighted_tuned_model_config.json"
MODEL_ARTIFACT_NAME = "poisson_piecewise_exponential_not_weighted_tuned.joblib"
PREPROCESSOR_ARTIFACT_NAME = "poisson_piecewise_exponential_not_weighted_preprocessor.joblib"
ALPHA_GRID = (2e-5, 5e-5, 1e-4, 1e-3, 1e-2, 5e-2)
WINDOW_WEEK_COLUMN = "week"
STAGE_PREFIX = "5.6b"
PREVIEW_ROWS = shared_base.PREVIEW_ROWS
BENCHMARK_HORIZONS = shared_base.BENCHMARK_HORIZONS
CALIBRATION_BINS = shared_base.CALIBRATION_BINS
CALIBRATION_CONTRACT_VERSION = shared_base.CALIBRATION_CONTRACT_VERSION
CALIBRATION_OBSERVED_RISK_METHOD = shared_base.CALIBRATION_OBSERVED_RISK_METHOD
REQUIRED_SHARED_PATH_KEYS = shared_base.REQUIRED_SHARED_PATH_KEYS
REQUIRED_MODELING_KEYS = shared_base.REQUIRED_MODELING_KEYS
REQUIRED_INPUT_TABLES = shared_base.REQUIRED_INPUT_TABLES
REQUIRED_PERSON_PERIOD_COLUMNS = shared_base.REQUIRED_PERSON_PERIOD_COLUMNS
CATEGORICAL_FEATURES = shared_base.CATEGORICAL_FEATURES
NUMERIC_FEATURES = shared_base.NUMERIC_FEATURES
FEATURE_ALIAS_MAP = shared_base.FEATURE_ALIAS_MAP
EPSILON = 1e-8
VALIDATION_TEST_SIZE = shared_base.VALIDATION_TEST_SIZE
GLM_SEARCH_ROW_CAP = 120_000
GLM_FINAL_ROW_CAP = 220_000
GLM_MAX_EVENT_SHARE = 0.50
GLM_MIN_NEGATIVE_SHARE = 0.25
GLM_WEEK_COL = "week"
GLM_TARGET_COL = "event_t"
REGULARIZED_MAXITER_SEARCH = 120
REGULARIZED_MAXITER_FINAL = 160
PipelineContext = shared_base.PipelineContext
json = shared_base.json
toml_reader = shared_base.toml_reader
print_artifact = shared_base.print_artifact
require_mapping_keys = shared_base.require_mapping_keys
require_list_of_strings = shared_base.require_list_of_strings
save_json = shared_base.save_json
require_tables = shared_base.require_tables
ensure_binary_target = shared_base.ensure_binary_target
build_truth_by_enrollment = shared_base.build_truth_by_enrollment
materialize_dataframe_table = shared_base.materialize_dataframe_table


@dataclass
class DynamicTreatmentState:
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
	truth_train_df: pd.DataFrame
	truth_test_df: pd.DataFrame


def log_stage_start(block_number: str, title: str) -> None:
	print(f"[START] {STAGE_PREFIX} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("# ==============================================================")
	print(f"# {block_number} - {title}")
	print("# ==============================================================")


def log_stage_end(block_number: str) -> None:
	print(f"[END] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def load_required_table(
	ctx: PipelineContext,
	table_name: str,
	required_columns: list[str],
	block_number: str,
) -> pd.DataFrame:
	require_tables(ctx.con, [table_name], block_number=block_number)
	shared_base.require_columns(ctx.con, table_name, required_columns)
	df = ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()
	if df.empty:
		raise ValueError(f"{table_name} is empty.")
	return df


def matrix_to_float64_dense(matrix: Any) -> np.ndarray:
	if hasattr(matrix, "toarray"):
		matrix = matrix.toarray()
	return np.asarray(matrix, dtype=np.float64)


def safe_glm_row_budget(n_rows: int, n_features: int, requested_cap: int) -> int:
	feature_adjusted_cap = max(25_000, int(3_500_000 / max(1, n_features)))
	return int(min(n_rows, requested_cap, feature_adjusted_cap))


def sample_glm_rows(frame: pd.DataFrame, target: np.ndarray, row_cap: int, seed: int) -> np.ndarray:
	target = np.asarray(target, dtype=np.int32)
	if frame.shape[0] != target.shape[0]:
		raise ValueError("Frame and target length mismatch in Poisson GLM sampling helper.")
	if frame.shape[0] <= row_cap:
		return np.arange(frame.shape[0], dtype=np.int32)

	rng = np.random.default_rng(seed)
	work = frame[[GLM_WEEK_COL]].copy()
	work[GLM_TARGET_COL] = target
	work["row_idx"] = np.arange(frame.shape[0], dtype=np.int32)
	work["week_bucket"] = pd.cut(
		work[GLM_WEEK_COL].astype(float),
		bins=[-np.inf, 4, 8, 16, np.inf],
		labels=["w01_04", "w05_08", "w09_16", "w17_plus"],
		include_lowest=True,
	).astype(str)
	work["stratum"] = work[GLM_TARGET_COL].astype(str) + "__" + work["week_bucket"]

	pos_idx = work.loc[work[GLM_TARGET_COL] == 1, "row_idx"].to_numpy(dtype=np.int32)
	neg_idx = work.loc[work[GLM_TARGET_COL] == 0, "row_idx"].to_numpy(dtype=np.int32)

	max_positive = int(row_cap * GLM_MAX_EVENT_SHARE)
	if pos_idx.shape[0] > max_positive:
		pos_sample = rng.choice(pos_idx, size=max_positive, replace=False).astype(np.int32)
	else:
		pos_sample = pos_idx

	remaining_budget = max(0, row_cap - pos_sample.shape[0])
	min_negative = int(row_cap * GLM_MIN_NEGATIVE_SHARE)
	remaining_budget = max(min_negative, remaining_budget)
	remaining_budget = min(remaining_budget, neg_idx.shape[0])

	neg_pool = work.loc[work[GLM_TARGET_COL] == 0, ["row_idx", "stratum"]].copy()
	neg_chunks: list[np.ndarray] = []
	if remaining_budget > 0 and not neg_pool.empty:
		stratum_counts = neg_pool["stratum"].value_counts().sort_index()
		raw_alloc = (stratum_counts / stratum_counts.sum()) * remaining_budget
		alloc = np.floor(raw_alloc).astype(int)
		residual = int(remaining_budget - alloc.sum())
		if residual > 0:
			residual_order = (raw_alloc - alloc).sort_values(ascending=False)
			for stratum in residual_order.index[:residual]:
				alloc.loc[stratum] += 1
		for stratum, n_take in alloc.items():
			if n_take <= 0:
				continue
			stratum_idx = neg_pool.loc[neg_pool["stratum"] == stratum, "row_idx"].to_numpy(dtype=np.int32)
			if stratum_idx.shape[0] <= n_take:
				neg_chunks.append(stratum_idx)
			else:
				neg_chunks.append(rng.choice(stratum_idx, size=int(n_take), replace=False).astype(np.int32))

	if neg_chunks:
		neg_sample = np.concatenate(neg_chunks).astype(np.int32)
	else:
		neg_sample = np.empty(0, dtype=np.int32)

	sampled = np.concatenate([pos_sample, neg_sample]).astype(np.int32)
	if sampled.shape[0] > row_cap:
		sampled = rng.choice(sampled, size=row_cap, replace=False).astype(np.int32)
	sampled.sort()
	return sampled


def evaluate_discrete_predictions(
	model_name: str,
	test_predictions_df: pd.DataFrame,
	truth_train_df: pd.DataFrame,
	truth_test_df: pd.DataFrame,
	y_test_row: pd.Series,
) -> dict[str, pd.DataFrame]:
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
		raise ValueError("The Poisson survival surface contains missing values after enrollment alignment.")

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
		pred_survival_h = shared_base.get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_survival").rename(
			columns={"pred_survival": "pred_survival_h"}
		)
		pred_risk_h = shared_base.get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_risk").rename(
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
		calibration_table, predicted_vs_observed_row = shared_base.build_ipcw_calibration_artifacts(
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

	calibration_summary_df = shared_base.summarize_calibration_by_horizon(calibration_bins_df)

	max_ipcw_time = float(truth_train_df.loc[truth_train_df["event"] == 0, "duration"].max())
	if not np.isfinite(max_ipcw_time):
		raise ValueError("Could not determine a valid censoring-support horizon for the Poisson model.")
	td_auc_rows: list[dict[str, Any]] = []
	td_auc_audit_rows: list[dict[str, Any]] = []
	for horizon_week in BENCHMARK_HORIZONS:
		pred_risk_h = shared_base.get_prediction_at_horizon(test_predictions_df, horizon_week, "pred_risk")
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
			y_test_supported_surv = Surv.from_arrays(
				event=supported_df["event"].astype(bool).to_numpy(),
				time=supported_df["duration"].astype(float).to_numpy(),
			)
			horizon_auc = shared_base.compute_ipcw_time_dependent_auc(
				y_train_surv,
				y_test_supported_surv,
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
	td_auc_support_audit_df = pd.DataFrame(td_auc_audit_rows)
	secondary_metrics_df = pd.concat([risk_auc_by_horizon_df, time_dependent_auc_df], ignore_index=True).sort_values(
		["horizon_week", "metric_name"]
	).reset_index(drop=True)

	ensure_binary_target(y_test_row, "pp_linear_hazard_ready_test.event_t")
	row_diagnostics_df = pd.DataFrame(
		[
			{
				"model_name": model_name,
				"row_level_roc_auc": float(roc_auc_score(y_test_row, test_predictions_df["pred_hazard"])),
				"row_level_pr_auc": float(average_precision_score(y_test_row, test_predictions_df["pred_hazard"])),
				"row_level_log_loss": float(log_loss(y_test_row, test_predictions_df["pred_hazard"], labels=[0, 1])),
				"row_level_brier": float(brier_score_loss(y_test_row, test_predictions_df["pred_hazard"])),
			}
		]
	)

	return {
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


class _PoissonBaseNamespace:
	pass


base = _PoissonBaseNamespace()
base.PipelineContext = PipelineContext
base.DynamicTreatmentState = DynamicTreatmentState
base.PREVIEW_ROWS = PREVIEW_ROWS
base.BENCHMARK_HORIZONS = BENCHMARK_HORIZONS
base.CALIBRATION_BINS = CALIBRATION_BINS
base.CALIBRATION_CONTRACT_VERSION = CALIBRATION_CONTRACT_VERSION
base.CALIBRATION_OBSERVED_RISK_METHOD = CALIBRATION_OBSERVED_RISK_METHOD
base.REQUIRED_SHARED_PATH_KEYS = REQUIRED_SHARED_PATH_KEYS
base.REQUIRED_MODELING_KEYS = REQUIRED_MODELING_KEYS
base.REQUIRED_INPUT_TABLES = REQUIRED_INPUT_TABLES
base.REQUIRED_PERSON_PERIOD_COLUMNS = REQUIRED_PERSON_PERIOD_COLUMNS
base.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
base.NUMERIC_FEATURES = NUMERIC_FEATURES
base.FEATURE_ALIAS_MAP = FEATURE_ALIAS_MAP
base.EPSILON = EPSILON
base.VALIDATION_TEST_SIZE = VALIDATION_TEST_SIZE
base.GLM_SEARCH_ROW_CAP = GLM_SEARCH_ROW_CAP
base.GLM_FINAL_ROW_CAP = GLM_FINAL_ROW_CAP
base.REGULARIZED_MAXITER_SEARCH = REGULARIZED_MAXITER_SEARCH
base.REGULARIZED_MAXITER_FINAL = REGULARIZED_MAXITER_FINAL
base.json = json
base.toml_reader = toml_reader
base.print_artifact = print_artifact
base.require_mapping_keys = require_mapping_keys
base.require_list_of_strings = require_list_of_strings
base.save_json = save_json
base.require_tables = require_tables
base.ensure_binary_target = ensure_binary_target
base.build_truth_by_enrollment = build_truth_by_enrollment
base.materialize_dataframe_table = materialize_dataframe_table
base.load_required_table = load_required_table
base.matrix_to_float64_dense = matrix_to_float64_dense
base.safe_glm_row_budget = safe_glm_row_budget
base.sample_glm_rows = sample_glm_rows
base.evaluate_discrete_predictions = evaluate_discrete_predictions
base.log_stage_start = log_stage_start
base.log_stage_end = log_stage_end


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
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_preprocessing_summary", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_not_weighted_preprocessing_summary', output_table_suffix)} — Poisson preprocessing summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_manifest_df,
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_raw_feature_manifest", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_not_weighted_raw_feature_manifest', output_table_suffix)} — Poisson raw feature manifest",
	)
	base.materialize_dataframe_table(
		ctx,
		df=feature_names_df,
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_feature_names_out", window_weeks),
		block_number="5.6b.2",
		label=f"Stage 5.6b.2 {apply_name_suffix('table_poisson_pexp_not_weighted_feature_names_out', output_table_suffix)} — Poisson transformed feature names",
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
			"sample_weight_strategy": "none",
		},
	}
	guardrails_df = pd.DataFrame([guardrails["summary"]])

	base.materialize_dataframe_table(
		ctx,
		df=guardrails_df,
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_guardrails", window_weeks),
		block_number="5.6b.3",
		label=f"Stage 5.6b.3 {apply_name_suffix('table_poisson_pexp_not_weighted_guardrails', output_table_suffix)} — Poisson runtime guardrails",
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

	X_subtrain_const = sm.add_constant(X_subtrain, prepend=True, has_constant="add")
	X_validation_const = sm.add_constant(X_validation, prepend=True, has_constant="add")
	X_full_const = sm.add_constant(X_full, prepend=True, has_constant="add")
	X_test_const = sm.add_constant(X_test, prepend=True, has_constant="add")

	tuning_workers = max(1, min(int(ctx.cpu_cores), len(ALPHA_GRID)))
	tuning_parallel_backend = resolve_runtime_tuning_parallel_backend(ctx.shared_config)
	print(f"- TUNING_WORKERS: {tuning_workers}")
	print(f"- TUNING_BACKEND: {tuning_parallel_backend}")

	def evaluate_alpha(candidate_id: int, alpha: float) -> dict[str, object]:
		glm = sm.GLM(y_subtrain, X_subtrain_const, family=Poisson())
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

	final_glm = sm.GLM(y_full, X_full_const, family=Poisson())
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
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuning_results", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuning_results', output_table_suffix)} — Poisson tuning results",
	)
	base.materialize_dataframe_table(
		ctx,
		df=test_predictions_df,
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_test_predictions", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_test_predictions', output_table_suffix)} — Poisson test predictions",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["primary_metrics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_primary_metrics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_primary_metrics', output_table_suffix)} — Poisson primary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["brier_by_horizon_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_brier_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_brier_by_horizon', output_table_suffix)} — Poisson Brier scores by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["secondary_metrics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_secondary_metrics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_secondary_metrics', output_table_suffix)} — Poisson secondary metrics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["td_auc_support_audit_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_td_auc_support_audit", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_td_auc_support_audit', output_table_suffix)} — Poisson IPCW support audit",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["row_diagnostics_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_row_diagnostics", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_row_diagnostics', output_table_suffix)} — Poisson row-level diagnostics",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["support_by_horizon_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_support_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_support_by_horizon', output_table_suffix)} — Poisson support by horizon",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_summary_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_calibration_summary", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_calibration_summary', output_table_suffix)} — Poisson calibration summary",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["calibration_bins_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_calibration_bins_by_horizon", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_calibration_bins_by_horizon', output_table_suffix)} — Poisson calibration bins",
	)
	base.materialize_dataframe_table(
		ctx,
		df=artifacts["predicted_vs_observed_survival_df"],
		table_name=apply_window_table_name("table_poisson_pexp_not_weighted_tuned_predicted_vs_observed_survival", window_weeks),
		block_number="5.6b.4",
		label=f"Stage 5.6b.4 {apply_name_suffix('table_poisson_pexp_not_weighted_tuned_predicted_vs_observed_survival', output_table_suffix)} — Poisson predicted versus observed survival",
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
		"calibration_contract_version": base.CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": base.CALIBRATION_OBSERVED_RISK_METHOD,
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
		"weighting_strategy": "none; this official D5.6 variant fits the same sampled person-period rows without frequency weights.",
		"design_note": "This not-weighted Poisson variant preserves the dynamic weekly contract, split discipline, and evaluation stack so the weighted script remains a pure Item 6 sensitivity counterpart.",
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