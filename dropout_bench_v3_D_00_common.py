from __future__ import annotations

from pathlib import Path
import json
import os
import warnings

import duckdb
import numpy as np
import pandas as pd
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
try:
	import tomllib
except ModuleNotFoundError:  # pragma: no cover
	import tomli as tomllib

from util import shutdown_duckdb_connection_from_globals

try:
	from scipy.linalg import LinAlgWarning
except Exception:
	LinAlgWarning = None


CALIBRATION_CONTRACT_VERSION = "v2_ipcw_km_observed_risk"
CALIBRATION_OBSERVED_RISK_METHOD = "ipcw_km_pseudo_event_rate"

CANONICAL_STAGE_MODEL_CATALOG = [
	{
		"stage_id": "D5.1",
		"stage_order": 1,
		"model_name": "runtime_contract_materialization",
		"display_name": "Runtime Contract Materialization",
		"family_group": "contract",
		"paper_methodological_arm": "contract_stage",
		"input_representation": "contract_materialization",
		"primary_metrics_table": None,
		"alias_prefix": None,
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": None,
		"cross_arm_name": None,
		"cross_arm_table_prefix": None,
		"cross_arm_uses_unsuffixed": None,
	},
	{
		"stage_id": "D5.2",
		"stage_order": 2,
		"model_name": "linear_discrete_time_hazard",
		"display_name": "Linear Discrete-Time Hazard",
		"family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
		"primary_metrics_table": "table_linear_tuned_primary_metrics",
		"alias_prefix": "table_linear",
		"source_prefix": "table_linear_not_weighted",
		"weight_not_weighted_prefix": "table_linear_not_weighted",
		"weight_weighted_prefix": "table_linear_weighted",
		"comparable_window_table_prefix": None,
		"cross_arm_name": "dynamic",
		"cross_arm_table_prefix": "table_linear_not_weighted",
		"cross_arm_uses_unsuffixed": False,
	},
	{
		"stage_id": "D5.3",
		"stage_order": 3,
		"model_name": "neural_discrete_time_hazard",
		"display_name": "Neural Discrete-Time Hazard",
		"family_group": "dynamic_neural",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
		"primary_metrics_table": "table_neural_tuned_primary_metrics",
		"alias_prefix": "table_neural",
		"source_prefix": "table_neural_not_weighted",
		"weight_not_weighted_prefix": "table_neural_not_weighted",
		"weight_weighted_prefix": "table_neural_weighted",
		"comparable_window_table_prefix": None,
		"cross_arm_name": "dynamic",
		"cross_arm_table_prefix": "table_neural_not_weighted",
		"cross_arm_uses_unsuffixed": False,
	},
	{
		"stage_id": "D5.4",
		"stage_order": 4,
		"model_name": "cox_comparable",
		"display_name": "Cox Comparable",
		"family_group": "comparable_continuous_time",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_cox_tuned_primary_metrics",
		"alias_prefix": "table_cox",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_cox",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_cox",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.5",
		"stage_order": 5,
		"model_name": "deepsurv",
		"display_name": "DeepSurv",
		"family_group": "comparable_neural",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_deepsurv_tuned_primary_metrics",
		"alias_prefix": "table_deepsurv",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_deepsurv",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_deepsurv",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.6",
		"stage_order": 6,
		"model_name": "poisson_piecewise_exponential",
		"display_name": "Poisson Piecewise-Exponential",
		"family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
		"primary_metrics_table": "table_poisson_pexp_tuned_primary_metrics",
		"alias_prefix": "table_poisson_pexp",
		"source_prefix": "table_poisson_pexp_not_weighted",
		"weight_not_weighted_prefix": "table_poisson_pexp_not_weighted",
		"weight_weighted_prefix": "table_poisson_pexp_weighted",
		"comparable_window_table_prefix": None,
		"cross_arm_name": "dynamic",
		"cross_arm_table_prefix": "table_poisson_pexp_not_weighted",
		"cross_arm_uses_unsuffixed": False,
	},
	{
		"stage_id": "D5.7",
		"stage_order": 7,
		"model_name": "gb_weekly_hazard_unweighted",
		"display_name": "GB Weekly Hazard Unweighted",
		"family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
		"primary_metrics_table": "table_gb_weekly_hazard_unweighted_tuned_primary_metrics",
		"alias_prefix": "table_gb_weekly_hazard_unweighted",
		"source_prefix": "table_gb_weekly_hazard_not_weighted",
		"weight_not_weighted_prefix": "table_gb_weekly_hazard_not_weighted",
		"weight_weighted_prefix": "table_gb_weekly_hazard_weighted",
		"comparable_window_table_prefix": None,
		"cross_arm_name": "dynamic",
		"cross_arm_table_prefix": "table_gb_weekly_hazard_not_weighted",
		"cross_arm_uses_unsuffixed": False,
	},
	{
		"stage_id": "D5.8",
		"stage_order": 8,
		"model_name": "catboost_weekly_hazard",
		"display_name": "CatBoost Weekly Hazard",
		"family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
		"primary_metrics_table": "table_catboost_weekly_hazard_tuned_primary_metrics",
		"alias_prefix": "table_catboost_weekly_hazard",
		"source_prefix": "table_catboost_weekly_hazard_not_weighted",
		"weight_not_weighted_prefix": "table_catboost_weekly_hazard_not_weighted",
		"weight_weighted_prefix": "table_catboost_weekly_hazard_weighted",
		"comparable_window_table_prefix": None,
		"cross_arm_name": "dynamic",
		"cross_arm_table_prefix": "table_catboost_weekly_hazard_not_weighted",
		"cross_arm_uses_unsuffixed": False,
	},
	{
		"stage_id": "D5.9",
		"stage_order": 9,
		"model_name": "random_survival_forest",
		"display_name": "Random Survival Forest",
		"family_group": "comparable_tree_survival",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_rsf_tuned_primary_metrics",
		"alias_prefix": "table_rsf",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_rsf",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_rsf",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.10",
		"stage_order": 10,
		"model_name": "gradient_boosted_cox",
		"display_name": "Gradient-Boosted Cox",
		"family_group": "comparable_tree_survival",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_gb_cox_tuned_primary_metrics",
		"alias_prefix": "table_gb_cox",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_gb_cox",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_gb_cox",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.11",
		"stage_order": 11,
		"model_name": "weibull_aft",
		"display_name": "Weibull AFT",
		"family_group": "comparable_parametric",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_weibull_aft_tuned_primary_metrics",
		"alias_prefix": "table_weibull_aft",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_weibull_aft",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_weibull_aft",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.12",
		"stage_order": 12,
		"model_name": "royston_parmar",
		"display_name": "Royston-Parmar",
		"family_group": "comparable_parametric",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_royston_parmar_tuned_primary_metrics",
		"alias_prefix": "table_royston_parmar",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_royston_parmar",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_royston_parmar",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.13",
		"stage_order": 13,
		"model_name": "xgboost_aft",
		"display_name": "XGBoost AFT",
		"family_group": "comparable_tree_survival",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_xgb_aft_tuned_primary_metrics",
		"alias_prefix": "table_xgb_aft",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_xgb_aft",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_xgb_aft",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.14",
		"stage_order": 14,
		"model_name": "neural_mtlr",
		"display_name": "Neural-MTLR",
		"family_group": "comparable_neural",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_mtlr_tuned_primary_metrics",
		"alias_prefix": "table_mtlr",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_mtlr",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_mtlr",
		"cross_arm_uses_unsuffixed": True,
	},
	{
		"stage_id": "D5.15",
		"stage_order": 15,
		"model_name": "deephit",
		"display_name": "DeepHit",
		"family_group": "comparable_neural",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
		"primary_metrics_table": "table_deephit_tuned_primary_metrics",
		"alias_prefix": "table_deephit",
		"source_prefix": None,
		"weight_not_weighted_prefix": None,
		"weight_weighted_prefix": None,
		"comparable_window_table_prefix": "table_deephit",
		"cross_arm_name": "comparable",
		"cross_arm_table_prefix": "table_deephit",
		"cross_arm_uses_unsuffixed": True,
	},
]


def get_canonical_model_catalog(include_contract: bool = True) -> list[dict]:
	catalog = [dict(spec) for spec in CANONICAL_STAGE_MODEL_CATALOG]
	if include_contract:
		return catalog
	return [spec for spec in catalog if spec["family_group"] != "contract"]


def get_d16_model_specs(include_contract: bool = True) -> list[dict]:
	fields = ["stage_id", "stage_order", "model_name", "family_group", "primary_metrics_table", "alias_prefix", "source_prefix"]
	return [{field: spec.get(field) for field in fields} for spec in get_canonical_model_catalog(include_contract=include_contract)]


def get_d16_weight_sensitivity_specs() -> list[dict]:
	fields = ["model_name", "family_group"]
	rows = []
	for spec in get_canonical_model_catalog(include_contract=False):
		if not spec.get("weight_not_weighted_prefix"):
			continue
		row = {field: spec.get(field) for field in fields}
		row["not_weighted_prefix"] = spec.get("weight_not_weighted_prefix")
		row["weighted_prefix"] = spec.get("weight_weighted_prefix")
		rows.append(row)
	return rows


def get_d16_comparable_window_specs() -> list[dict]:
	fields = ["model_name", "family_group"]
	rows = []
	for spec in get_canonical_model_catalog(include_contract=False):
		table_prefix = spec.get("comparable_window_table_prefix")
		if not table_prefix:
			continue
		row = {field: spec.get(field) for field in fields}
		row["table_prefix"] = table_prefix
		rows.append(row)
	return rows


def get_d16_cross_arm_model_specs() -> list[dict]:
	fields = ["stage_id", "stage_order", "model_name", "family_group"]
	rows = []
	for spec in get_canonical_model_catalog(include_contract=False):
		arm_name = spec.get("cross_arm_name")
		table_prefix = spec.get("cross_arm_table_prefix")
		if not arm_name or not table_prefix:
			continue
		row = {field: spec.get(field) for field in fields}
		row["arm_name"] = arm_name
		row["table_prefix"] = table_prefix
		row["canonical_uses_unsuffixed"] = bool(spec.get("cross_arm_uses_unsuffixed"))
		rows.append(row)
	return rows


def get_benchmark_horizon_model_specs() -> list[dict]:
	return [
		{
			"model_key": spec["model_name"],
			"display_name": spec["display_name"],
			"family_group": spec["family_group"],
		}
		for spec in get_canonical_model_catalog(include_contract=False)
	]


def get_canonical_model_catalog_df(include_contract: bool = True) -> pd.DataFrame:
	return pd.DataFrame(get_canonical_model_catalog(include_contract=include_contract))


def benchmark_family_from_group(family_group: str) -> str:
	family_group = str(family_group)
	if family_group in {"dynamic_weekly", "dynamic_neural"}:
		return "dynamic_weekly_person_period"
	if family_group == "contract":
		return "contract_stage"
	if family_group in {"comparable_continuous_time", "comparable_tree_survival", "comparable_parametric", "comparable_neural"}:
		return "comparable_continuous_time_early_window"
	return "other"


def paper_arm_from_family_group(family_group: str) -> str:
	return benchmark_family_from_group(family_group)


def representation_from_family_group(family_group: str) -> str:
	family_group = str(family_group)
	if family_group == "contract":
		return "contract_materialization"
	if family_group in {"dynamic_weekly", "dynamic_neural"}:
		return "weekly_person_period"
	if family_group in {"comparable_continuous_time", "comparable_tree_survival", "comparable_parametric", "comparable_neural"}:
		return "early_window_enrollment"
	return "other"

CANONICAL_MANUSCRIPT_AUDIT_SUBSET = [
	# --- Dynamic arm (4 models, weekly person-period) ---
	{
		"model_id": "poisson_pexp_tuned",
		"stage_id": "D5.6",
		"canonical_model_name": "poisson_piecewise_exponential",
		"display_name": "Poisson Piecewise-Exponential",
		"operational_family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
	},
	{
		"model_id": "linear_tuned",
		"stage_id": "D5.2",
		"canonical_model_name": "linear_discrete_time_hazard",
		"display_name": "Linear Discrete-Time Hazard",
		"operational_family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
	},
	{
		"model_id": "gb_weekly_tuned",
		"stage_id": "D5.7",
		"canonical_model_name": "gb_weekly_hazard",
		"display_name": "GB Weekly Hazard",
		"operational_family_group": "dynamic_weekly",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
	},
	{
		"model_id": "neural_tuned",
		"stage_id": "D5.3",
		"canonical_model_name": "neural_discrete_time_survival",
		"display_name": "Neural Discrete-Time Survival",
		"operational_family_group": "dynamic_neural",
		"paper_methodological_arm": "dynamic_weekly_person_period",
		"input_representation": "weekly_person_period",
	},
	# --- Comparable arm (4 models, early-window enrollment) ---
	{
		"model_id": "cox_tuned",
		"stage_id": "D5.4",
		"canonical_model_name": "cox_comparable",
		"display_name": "Cox Comparable",
		"operational_family_group": "comparable_continuous_time",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
	},
	{
		"model_id": "deepsurv_tuned",
		"stage_id": "D5.5",
		"canonical_model_name": "deepsurv",
		"display_name": "DeepSurv",
		"operational_family_group": "comparable_neural",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
	},
	{
		"model_id": "rsf_tuned",
		"stage_id": "D5.9",
		"canonical_model_name": "random_survival_forest",
		"display_name": "Random Survival Forest",
		"operational_family_group": "comparable_tree_survival",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
	},
	{
		"model_id": "mtlr_tuned",
		"stage_id": "D5.14",
		"canonical_model_name": "neural_mtlr",
		"display_name": "Neural-MTLR",
		"operational_family_group": "comparable_neural",
		"paper_methodological_arm": "comparable_continuous_time_early_window",
		"input_representation": "early_window_enrollment",
	},
]

CANONICAL_EXPLAINABILITY_BLOCKS = [
	{
		"block_id": "static_structural",
		"block_label": "Static structural covariates",
		"applies_to": "all_families",
	},
	{
		"block_id": "early_window_behavior",
		"block_label": "Early-window behavioral summaries",
		"applies_to": "comparable_quartet",
	},
	{
		"block_id": "dynamic_temporal_behavioral",
		"block_label": "Weekly temporal-behavioral features",
		"applies_to": "dynamic_arm",
	},
	{
		"block_id": "discrete_time_index",
		"block_label": "Discrete time index (week)",
		"applies_to": "dynamic_arm",
	},
]

def get_manuscript_audit_subset_specs() -> list[dict]:
	return [dict(spec) for spec in CANONICAL_MANUSCRIPT_AUDIT_SUBSET]

def get_manuscript_audit_subset_by_model_id() -> dict[str, dict]:
	return {spec["model_id"]: dict(spec) for spec in get_manuscript_audit_subset_specs()}

def get_manuscript_explainability_model_order() -> list[str]:
	return [spec["display_name"] for spec in get_manuscript_audit_subset_specs()]

def get_canonical_explainability_blocks() -> list[dict]:
	return [dict(spec) for spec in CANONICAL_EXPLAINABILITY_BLOCKS]

def get_canonical_explainability_block_label_map() -> dict[str, str]:
	return {spec["block_id"]: spec["block_label"] for spec in get_canonical_explainability_blocks()}


def save_json(obj, pathlike) -> None:
	path = Path(pathlike)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as file_obj:
		json.dump(obj, file_obj, indent=2, ensure_ascii=False)


def parse_int_list(raw_values, field_name: str) -> list[int]:
	if not isinstance(raw_values, list) or any(not isinstance(value, int) for value in raw_values):
		raise TypeError(f"{field_name} must be a list of integers.")
	values = [int(value) for value in raw_values]
	if any(value <= 0 for value in values):
		raise ValueError(f"{field_name} must contain only positive integers.")
	if len(set(values)) != len(values):
		raise ValueError(f"{field_name} must not contain duplicate values.")
	return values


def resolve_early_window_sensitivity_weeks(
	benchmark_config: dict,
	default: tuple[int, ...] = (2, 4, 6, 8, 10),
) -> list[int]:
	raw_values = benchmark_config.get("early_window_sensitivity_weeks")
	if raw_values is None:
		return [int(value) for value in default]
	return parse_int_list(raw_values, "benchmark.early_window_sensitivity_weeks")


def resolve_benchmark_horizons(
	benchmark_config: dict,
	default: tuple[int, ...] = (10, 20, 30),
) -> list[int]:
	raw_values = benchmark_config.get("benchmark_horizons")
	if raw_values is None:
		return [int(value) for value in default]
	return parse_int_list(raw_values, "benchmark.benchmark_horizons")


def resolve_calibration_bins(benchmark_config: dict, default: int = 10) -> int:
	raw_value = benchmark_config.get("calibration_bins")
	if raw_value is None:
		return int(default)
	if not isinstance(raw_value, int):
		raise TypeError("benchmark.calibration_bins must be an integer.")
	if int(raw_value) <= 1:
		raise ValueError("benchmark.calibration_bins must be greater than 1.")
	return int(raw_value)


def build_calibration_contract(benchmark_config: dict) -> dict[str, object]:
	return {
		"benchmark_horizons": resolve_benchmark_horizons(benchmark_config),
		"calibration_bins": resolve_calibration_bins(benchmark_config),
		"calibration_contract_version": CALIBRATION_CONTRACT_VERSION,
		"calibration_observed_risk_method": CALIBRATION_OBSERVED_RISK_METHOD,
	}


def get_comparable_window_override(env_var: str = "BENCHMARK_COMPARABLE_WINDOW_WEEKS") -> int | None:
	raw_value = os.environ.get(env_var, "").strip()
	if not raw_value:
		return None
	try:
		window_weeks = int(raw_value)
	except ValueError as exc:
		raise ValueError(f"{env_var} must be an integer when provided. Received: {raw_value!r}") from exc
	if window_weeks <= 0:
		raise ValueError(f"{env_var} must be a positive integer. Received: {window_weeks}.")
	return window_weeks


def resolve_comparable_window_execution_plan(
	benchmark_config: dict,
	env_override_var: str = "BENCHMARK_COMPARABLE_WINDOW_WEEKS",
) -> list[int]:
	override_window = get_comparable_window_override(env_override_var)
	if override_window is not None:
		return [int(override_window)]
	return resolve_early_window_sensitivity_weeks(benchmark_config)


def build_window_suffix(canonical_window_weeks: int, active_window_weeks: int) -> str:
	return "" if int(active_window_weeks) == int(canonical_window_weeks) else f"_w{int(active_window_weeks)}"


def apply_name_suffix(name: str, suffix: str) -> str:
	return f"{name}{suffix}" if suffix else name


def append_suffix_before_extension(filename: str, suffix: str) -> str:
	path = Path(filename)
	if not suffix:
		return str(path)
	return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def comparable_window_feature_names(window_weeks: int) -> dict[str, str]:
	window = int(window_weeks)
	if window <= 0:
		raise ValueError(f"window_weeks must be positive. Received: {window}.")
	return {
		"main_clicks_feature": f"clicks_first_{window}_weeks",
		"main_active_feature": f"active_weeks_first_{window}",
		"main_mean_clicks_feature": f"mean_clicks_first_{window}_weeks",
	}


def comparable_numeric_features(window_weeks: int, static_numeric_features: list[str]) -> list[str]:
	feature_names = comparable_window_feature_names(window_weeks)
	return list(static_numeric_features) + [
		feature_names["main_clicks_feature"],
		feature_names["main_active_feature"],
		feature_names["main_mean_clicks_feature"],
	]


def comparable_required_columns(base_columns: list[str], window_weeks: int) -> list[str]:
	return list(base_columns) + comparable_numeric_features(
		window_weeks,
		static_numeric_features=["num_of_prev_attempts", "studied_credits"],
	)


def resolve_variant_table_name(base_table_name: str, canonical_window_weeks: int, active_window_weeks: int) -> str:
	if int(active_window_weeks) == int(canonical_window_weeks):
		return base_table_name
	return f"{base_table_name}_w{int(active_window_weeks)}"


def ensure_pipeline_table_catalog(con, catalog_table: str = "pipeline_table_catalog") -> None:
	con.execute(
		f"""
		CREATE TABLE IF NOT EXISTS {catalog_table} (
			table_name VARCHAR,
			created_by_notebook VARCHAR,
			created_in_cell VARCHAR,
			created_at TIMESTAMP,
			run_id VARCHAR
		)
		"""
	)


def register_duckdb_table(
	con,
	table_name: str,
	notebook_name: str,
	stage_id: str,
	run_id: str,
	catalog_table: str = "pipeline_table_catalog",
) -> None:
	ensure_pipeline_table_catalog(con, catalog_table=catalog_table)
	con.execute(
		f"""
		INSERT INTO {catalog_table}
		(table_name, created_by_notebook, created_in_cell, created_at, run_id)
		VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
		""",
		[table_name, notebook_name, stage_id, run_id],
	)


def _sanitize_dataframe_for_duckdb(df: pd.DataFrame) -> pd.DataFrame:
	safe = df.copy()
	for col in safe.columns:
		series = safe[col]
		if str(series.dtype) == "category":
			safe[col] = series.astype(str)
		elif series.dtype == "object":
			safe[col] = series.map(
				lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray)
				else json.dumps(list(x)) if isinstance(x, (list, tuple, set))
				else json.dumps(x) if isinstance(x, dict)
				else str(x) if isinstance(x, pd.Interval)
				else x
			)
	return safe


def materialize_dataframe(
	con,
	df: pd.DataFrame,
	table_name: str,
	stage_id: str,
	notebook_name: str,
	run_id: str,
) -> None:
	if not isinstance(df, pd.DataFrame):
		raise TypeError(f"{stage_id}: expected a pandas DataFrame for materialization into {table_name}.")
	safe_df = _sanitize_dataframe_for_duckdb(df)
	con.register("_tmp_materialize_dataframe", safe_df)
	con.execute(f"DROP TABLE IF EXISTS {table_name}")
	con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_materialize_dataframe")
	con.unregister("_tmp_materialize_dataframe")
	register_duckdb_table(con, table_name, notebook_name, stage_id, run_id)


def infer_table_name_from_pathlike(pathlike) -> str:
	return Path(pathlike).stem


def duckdb_table_exists(con, table_name: str) -> bool:
	available = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
	return table_name in available


def load_duckdb_table_or_raise(con, table_name: str) -> pd.DataFrame:
	if not duckdb_table_exists(con, table_name):
		raise FileNotFoundError(f"Required DuckDB table not found: {table_name}")
	return con.execute(f"SELECT * FROM {table_name}").fetchdf()


def validate_duckdb_tables(con, table_names: list[str], stage_id: str) -> None:
	missing = [table_name for table_name in table_names if not duckdb_table_exists(con, table_name)]
	if missing:
		raise FileNotFoundError(
			f"{stage_id}: missing required DuckDB table(s): {', '.join(missing)}"
		)


def persist_table_artifact(
	con,
	df: pd.DataFrame,
	pathlike,
	stage_id: str,
	notebook_name: str,
	run_id: str,
) -> str:
	table_name = infer_table_name_from_pathlike(pathlike)
	materialize_dataframe(con, df, table_name, stage_id, notebook_name, run_id)
	return table_name


def format_output_target(pathlike) -> str:
	candidate = Path(pathlike)
	if str(pathlike).startswith("duckdb://"):
		return str(pathlike)
	if not candidate.suffix:
		return f"duckdb://{candidate.name}"
	return str(candidate.resolve())


def configure_benchmark_warning_policy() -> None:
	warnings.filterwarnings(
		"ignore",
		message=r"This overload of add is deprecated:.*",
		category=UserWarning,
		module=r"torchtuples\.callbacks",
	)
	if LinAlgWarning is not None:
		warnings.filterwarnings("error", category=LinAlgWarning)


def _predict_censoring_survival(estimator: CensoringDistributionEstimator, times: np.ndarray) -> np.ndarray:
	times_array = np.asarray(times, dtype=float)
	times_array = np.maximum(times_array, 0.0)
	return np.asarray(estimator.predict_proba(times_array), dtype=float)


def build_ipcw_calibration_artifacts(
	survival_train,
	horizon_prediction_df: pd.DataFrame,
	horizon_week: int | float,
	calibration_bins: int,
	*,
	enrollment_id_col: str = "enrollment_id",
	duration_col: str = "duration",
	event_col: str = "event",
	risk_col: str = "pred_risk_h",
) -> tuple[pd.DataFrame, dict[str, float]]:
	if horizon_prediction_df.empty:
		raise ValueError(f"No prediction rows are available at horizon {horizon_week} for calibration.")
	for required_column in [enrollment_id_col, duration_col, event_col, risk_col]:
		if required_column not in horizon_prediction_df.columns:
			raise KeyError(f"Missing required calibration column: {required_column}")

	horizon_df = horizon_prediction_df[[enrollment_id_col, duration_col, event_col, risk_col]].copy()
	horizon_df[duration_col] = pd.to_numeric(horizon_df[duration_col], errors="coerce")
	horizon_df[event_col] = pd.to_numeric(horizon_df[event_col], errors="coerce")
	horizon_df[risk_col] = pd.to_numeric(horizon_df[risk_col], errors="coerce")
	if horizon_df[[duration_col, event_col, risk_col]].isna().any().any():
		raise ValueError(f"Calibration inputs contain missing values at horizon {horizon_week}.")
	if not np.isfinite(horizon_df[risk_col].to_numpy(dtype=float)).all():
		raise ValueError(f"Calibration risks contain non-finite values at horizon {horizon_week}.")

	horizon_value = float(horizon_week)
	durations = horizon_df[duration_col].to_numpy(dtype=float)
	events = horizon_df[event_col].astype(bool).to_numpy()
	survival_test = Surv.from_arrays(event=events, time=durations)

	censoring_estimator = CensoringDistributionEstimator()
	censoring_estimator.fit(survival_train)

	left_limit_times = np.where(durations <= 0.0, 0.0, np.nextafter(durations, -np.inf))
	censor_survival_at_event = np.clip(_predict_censoring_survival(censoring_estimator, left_limit_times), 1e-12, None)
	censor_survival_at_horizon = np.clip(
		_predict_censoring_survival(censoring_estimator, np.repeat(horizon_value, horizon_df.shape[0])),
		1e-12,
		None,
	)

	raw_event_by_horizon = ((events) & (durations <= horizon_value)).astype(float)
	raw_survival_by_horizon = (durations > horizon_value).astype(float)
	horizon_df["observed_event_by_h_ipcw"] = np.where(
		raw_event_by_horizon == 1.0,
		1.0 / censor_survival_at_event,
		0.0,
	)
	horizon_df["observed_survival_by_h_ipcw"] = np.where(
		raw_survival_by_horizon == 1.0,
		1.0 / censor_survival_at_horizon,
		0.0,
	)

	n_bins = int(min(int(calibration_bins), max(1, horizon_df.shape[0])))
	ranked_scores = horizon_df[risk_col].rank(method="first")
	horizon_df["calibration_bin"] = pd.qcut(ranked_scores, q=n_bins, labels=False)

	calibration_table = (
		horizon_df.groupby("calibration_bin", as_index=False)
		.agg(
			n=(enrollment_id_col, "count"),
			mean_predicted_risk=(risk_col, "mean"),
			observed_event_rate=("observed_event_by_h_ipcw", "mean"),
		)
		.sort_values("calibration_bin")
		.reset_index(drop=True)
	)
	calibration_table["horizon_week"] = int(horizon_week)
	calibration_table["abs_calibration_gap"] = (
		calibration_table["mean_predicted_risk"] - calibration_table["observed_event_rate"]
	).abs()

	predicted_vs_observed_row = {
		"horizon_week": int(horizon_week),
		"n_evaluable_enrollments": int(horizon_df.shape[0]),
		"mean_predicted_survival": float((1.0 - horizon_df[risk_col]).mean()),
		"mean_observed_survival": float(horizon_df["observed_survival_by_h_ipcw"].mean()),
		"abs_gap": float(abs((1.0 - horizon_df[risk_col]).mean() - horizon_df["observed_survival_by_h_ipcw"].mean())),
	}
	return calibration_table, predicted_vs_observed_row


def summarize_calibration_by_horizon(
	calibration_bins_df: pd.DataFrame,
	metric_name: str = "calibration_at_horizon",
	notes: str = "Weighted absolute calibration gap across bins",
) -> pd.DataFrame:
	if calibration_bins_df.empty:
		return pd.DataFrame(columns=["horizon_week", "metric_name", "metric_category", "metric_value", "notes"])
	summary_rows: list[dict[str, object]] = []
	for horizon_week, group_df in calibration_bins_df.groupby("horizon_week", sort=True):
		summary_rows.append(
			{
				"horizon_week": int(horizon_week),
				"metric_name": metric_name,
				"metric_category": "primary",
				"metric_value": float(np.average(group_df["abs_calibration_gap"], weights=group_df["n"])),
				"notes": notes,
			}
		)
	return pd.DataFrame(summary_rows)


def open_notebook_runtime(notebook_name: str) -> dict:
	project_root = Path.cwd()
	config_toml_path = project_root / "benchmark_shared_config.toml"
	modeling_contract_toml_path = project_root / "benchmark_modeling_contract.toml"
	run_metadata_json_path = project_root / "outputs_benchmark_survival" / "metadata" / "run_metadata.json"

	if not config_toml_path.exists():
		raise FileNotFoundError(f"Missing shared benchmark config TOML: {config_toml_path}")
	if not modeling_contract_toml_path.exists():
		raise FileNotFoundError(
			f"Missing modeling contract TOML exported by notebook B: {modeling_contract_toml_path}"
		)
	if not run_metadata_json_path.exists():
		raise FileNotFoundError(f"Missing execution metadata JSON: {run_metadata_json_path}")

	with open(config_toml_path, "rb") as file_obj:
		shared_config = tomllib.load(file_obj)
	with open(modeling_contract_toml_path, "rb") as file_obj:
		shared_modeling_contract = tomllib.load(file_obj)
	with open(run_metadata_json_path, "r", encoding="utf-8") as file_obj:
		run_metadata = json.load(file_obj)

	run_id = str(run_metadata["run_id"]).strip()
	paths_cfg = shared_config.get("paths", {})

	def _resolve_project_path(raw_path: str) -> Path:
		path = Path(raw_path)
		return path if path.is_absolute() else project_root / path

	data_dir = _resolve_project_path(paths_cfg.get("data_dir", "content"))
	output_dir = _resolve_project_path(paths_cfg.get("output_dir", "outputs_benchmark_survival"))
	tables_dir = output_dir / paths_cfg.get("tables_subdir", "tables")
	metadata_dir = output_dir / paths_cfg.get("metadata_subdir", "metadata")
	data_output_dir = output_dir / paths_cfg.get("data_output_subdir", "data")
	duckdb_path = output_dir / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")

	for path in [output_dir, tables_dir, metadata_dir, data_output_dir]:
		path.mkdir(parents=True, exist_ok=True)

	con = duckdb.connect(str(duckdb_path))
	ensure_pipeline_table_catalog(con)
	configure_benchmark_warning_policy()

	runtime = {
		"PROJECT_ROOT": project_root,
		"NOTEBOOK_NAME": notebook_name,
		"CONFIG_TOML_PATH": config_toml_path,
		"MODELING_CONTRACT_TOML_PATH": modeling_contract_toml_path,
		"RUN_METADATA_JSON_PATH": run_metadata_json_path,
		"SHARED_CONFIG": shared_config,
		"SHARED_MODELING_CONTRACT": shared_modeling_contract,
		"RUN_METADATA": run_metadata,
		"RUN_ID": run_id,
		"DATA_DIR": data_dir,
		"OUTPUT_DIR": output_dir,
		"TABLES_DIR": tables_dir,
		"METADATA_DIR": metadata_dir,
		"DATA_OUTPUT_DIR": data_output_dir,
		"DUCKDB_PATH": duckdb_path,
		"con": con,
		"save_json": save_json,
		"register_duckdb_table": register_duckdb_table,
		"materialize_dataframe": materialize_dataframe,
		"infer_table_name_from_pathlike": infer_table_name_from_pathlike,
		"load_duckdb_table_or_raise": load_duckdb_table_or_raise,
		"duckdb_table_exists": duckdb_table_exists,
		"validate_duckdb_tables": validate_duckdb_tables,
		"persist_table_artifact": persist_table_artifact,
		"format_output_target": format_output_target,
		"shutdown_duckdb_connection_from_globals": shutdown_duckdb_connection_from_globals,
	}
	return runtime


def close_notebook_runtime(runtime: dict) -> None:
	shutdown_duckdb_connection_from_globals(runtime)


def load_modeling_aliases(shared_modeling_contract: dict) -> dict:
	benchmark_contract = shared_modeling_contract.get("benchmark", {})
	modeling_contract = shared_modeling_contract.get("modeling", {})
	feature_contract = shared_modeling_contract.get("feature_contract", {})

	aliases = {
		"RANDOM_SEED": int(benchmark_contract["seed"]),
		"TEST_SIZE": float(benchmark_contract["test_size"]),
		"EARLY_WINDOW_WEEKS": int(benchmark_contract["early_window_weeks"]),
		"MAIN_ENROLLMENT_WINDOW_WEEKS": int(benchmark_contract["main_enrollment_window_weeks"]),
		"MAIN_CLICKS_FEATURE": str(modeling_contract["main_clicks_feature"]),
		"MAIN_ACTIVE_FEATURE": str(modeling_contract["main_active_feature"]),
		"MAIN_MEAN_CLICKS_FEATURE": str(modeling_contract["main_mean_clicks_feature"]),
		"STATIC_FEATURES": list(feature_contract["static_features"]),
		"TEMPORAL_FEATURES_DISCRETE": list(feature_contract["temporal_features_discrete"]),
		"MAIN_ENROLLMENT_WINDOW_FEATURES": list(feature_contract["main_enrollment_window_features"]),
		"OPTIONAL_COMPARABLE_WINDOW_FEATURES": list(feature_contract.get("optional_comparable_window_features", [])),
		"DYNAMIC_ARM_FEATURES_LINEAR": list(feature_contract["dynamic_arm_features_linear"]),
		"DYNAMIC_ARM_FEATURES_NEURAL": list(feature_contract["dynamic_arm_features_neural"]),
	}
	aliases["MODELING_CONFIG"] = {
		"random_seed": aliases["RANDOM_SEED"],
		"test_size": aliases["TEST_SIZE"],
		"early_window_weeks": aliases["EARLY_WINDOW_WEEKS"],
		"main_enrollment_window_weeks": aliases["MAIN_ENROLLMENT_WINDOW_WEEKS"],
		"main_clicks_feature": aliases["MAIN_CLICKS_FEATURE"],
		"main_active_feature": aliases["MAIN_ACTIVE_FEATURE"],
		"main_mean_clicks_feature": aliases["MAIN_MEAN_CLICKS_FEATURE"],
		"static_features": aliases["STATIC_FEATURES"],
		"temporal_features_discrete": aliases["TEMPORAL_FEATURES_DISCRETE"],
		"main_enrollment_window_features": aliases["MAIN_ENROLLMENT_WINDOW_FEATURES"],
		"optional_comparable_window_features": aliases["OPTIONAL_COMPARABLE_WINDOW_FEATURES"],
		"dynamic_arm_features_linear": aliases["DYNAMIC_ARM_FEATURES_LINEAR"],
		"dynamic_arm_features_neural": aliases["DYNAMIC_ARM_FEATURES_NEURAL"],
	}
	return aliases
