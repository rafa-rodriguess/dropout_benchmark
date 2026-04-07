# Auto-generated from dropout_bench_v3_G_explainability_paper_refatorado_v7.ipynb

from __future__ import annotations

# %% Cell 2
from datetime import datetime as _dt
print(f"[START] G0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G0 - Runtime / DuckDB bootstrap
# ==============================================================
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as package_version
import sys
import duckdb
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import atexit
import json
import pandas as pd
import numpy as np
import dropout_bench_v3_D_00_common as base
from util import shutdown_duckdb_connection_from_globals

try:
    from IPython.display import Image, Markdown, display
except ImportError:
    class Markdown(str):
        pass

    class Image:
        def __init__(self, filename: str | None = None, **_: object) -> None:
            self.filename = filename

        def __repr__(self) -> str:
            return f"Image(filename={self.filename!r})"

    def display(*objects: object) -> None:
        for obj in objects:
            print(obj)

PROJECT_ROOT = Path.cwd()
NOTEBOOK_NAME = "dropout_bench_v3_G_explainability_paper.ipynb"
CONFIG_TOML_PATH = PROJECT_ROOT / "benchmark_shared_config.toml"
RUN_METADATA_JSON_PATH = PROJECT_ROOT / "outputs_benchmark_survival" / "metadata" / "run_metadata.json"
MONOLITHIC_NOTEBOOK_PATH = PROJECT_ROOT / "dropout_bench.ipynb"

if CONFIG_TOML_PATH.exists():
    with open(CONFIG_TOML_PATH, "rb") as f:
        SHARED_CONFIG = tomllib.load(f)
else:
    SHARED_CONFIG = {
        "paths": {
            "data_dir": "content",
            "output_dir": "outputs_benchmark_survival",
            "tables_subdir": "tables",
            "metadata_subdir": "metadata",
            "data_output_subdir": "data",
            "duckdb_filename": "benchmark_survival.duckdb",
        },
        "benchmark": {
            "seed": 42,
            "test_size": 0.30,
            "early_window_weeks": 4,
            "main_enrollment_window_weeks": 4,
            "calibration_bins": 10,
        },
    }

if RUN_METADATA_JSON_PATH.exists():
    with open(RUN_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        RUN_METADATA = json.load(f)
    RUN_ID = str(RUN_METADATA.get("run_id", "unknown")).strip()
else:
    RUN_METADATA = {"run_id": "unknown"}
    RUN_ID = "unknown"

paths_cfg = SHARED_CONFIG.get("paths", {})
benchmark_cfg = SHARED_CONFIG.get("benchmark", {})

def _resolve_project_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else PROJECT_ROOT / p

DATA_DIR = _resolve_project_path(paths_cfg.get("data_dir", "content"))
OUTPUT_DIR = _resolve_project_path(paths_cfg.get("output_dir", "outputs_benchmark_survival"))
TABLES_DIR = OUTPUT_DIR / paths_cfg.get("tables_subdir", "tables")
METADATA_DIR = OUTPUT_DIR / paths_cfg.get("metadata_subdir", "metadata")
DATA_OUTPUT_DIR = OUTPUT_DIR / paths_cfg.get("data_output_subdir", "data")
DUCKDB_PATH = OUTPUT_DIR / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")
ENVIRONMENT_SUMMARY_PATH = METADATA_DIR / "environment_summary.json"

RANDOM_SEED = int(benchmark_cfg.get("seed", 42))
HORIZONS_WEEKS = [10, 20, 30]
HORIZON_WEEKS = list(HORIZONS_WEEKS)
CALIBRATION_BINS = int(benchmark_cfg.get("calibration_bins", 10))

MANUSCRIPT_TEX_PATH = PROJECT_ROOT / "paper" / "dropout_benchmark_v3.tex"
MANUSCRIPT_EXPLAINABILITY_MODEL_ORDER = base.get_manuscript_explainability_model_order()
MANUSCRIPT_EXPLAINABILITY_FALLBACK_ROWS = [
    {
        "Model": "Poisson Piecewise-Exponential",
        "Family": "discrete_time_poisson",
        "Top driver": "pending_runtime_materialization",
        "Dominant block": "dynamic_temporal_behavioral",
    },
    {
        "Model": "Linear Discrete-Time Hazard",
        "Family": "discrete_time_linear",
        "Top driver": "num__total_clicks_week",
        "Dominant block": "dynamic_temporal_behavioral",
    },
    {
        "Model": "GB Weekly Hazard",
        "Family": "discrete_time_gb",
        "Top driver": "pending_runtime_materialization",
        "Dominant block": "dynamic_temporal_behavioral",
    },
    {
        "Model": "Neural Discrete-Time Survival",
        "Family": "discrete_time_neural",
        "Top driver": "pending_runtime_materialization",
        "Dominant block": "dynamic_temporal_behavioral",
    },
    {
        "Model": "Cox Comparable",
        "Family": "continuous_time_cox",
        "Top driver": "num__active_weeks_first_4",
        "Dominant block": "early_window_behavior",
    },
    {
        "Model": "DeepSurv",
        "Family": "continuous_time_deepsurv",
        "Top driver": "active_weeks_first_4",
        "Dominant block": "early_window_behavior",
    },
    {
        "Model": "Random Survival Forest",
        "Family": "continuous_time_tree_ensemble",
        "Top driver": "pending_runtime_materialization",
        "Dominant block": "early_window_behavior",
    },
    {
        "Model": "Neural-MTLR",
        "Family": "continuous_time_neural_mtlr",
        "Top driver": "pending_runtime_materialization",
        "Dominant block": "early_window_behavior",
    },
]

for p in [OUTPUT_DIR, TABLES_DIR, METADATA_DIR, DATA_OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if "con" in globals():
    try:
        con.close()
    except Exception:
        pass

con = duckdb.connect(str(DUCKDB_PATH))

def _close_duckdb_connection() -> None:
    if "con" in globals():
        try:
            con.close()
        except Exception:
            pass

if "_duckdb_close_registered" not in globals():
    atexit.register(_close_duckdb_connection)
    _duckdb_close_registered = True

def _normalize_major_minor(version_text: str | None) -> str | None:
    if version_text is None:
        return None
    parts = str(version_text).strip().split(".")
    if len(parts) < 2:
        return str(version_text).strip() or None
    return ".".join(parts[:2])

def _package_version_or_none(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None

def validate_serialized_model_runtime() -> None:
    if not ENVIRONMENT_SUMMARY_PATH.exists():
        return

    with open(ENVIRONMENT_SUMMARY_PATH, "r", encoding="utf-8") as file_handle:
        expected_env = json.load(file_handle)

    expected_python = str(expected_env.get("python_version", "")).strip() or None
    expected_sklearn = str(expected_env.get("scikit_learn_version", "")).strip() or None
    expected_sksurv = str(expected_env.get("scikit_survival_version", "")).strip() or None

    current_python = ".".join(str(part) for part in sys.version_info[:3])
    current_sklearn = _package_version_or_none("scikit-learn")
    current_sksurv = _package_version_or_none("scikit-survival")

    mismatches = []
    if expected_python and _normalize_major_minor(current_python) != _normalize_major_minor(expected_python):
        mismatches.append(f"python expected {expected_python}, found {current_python}")
    if expected_sklearn and current_sklearn != expected_sklearn:
        mismatches.append(f"scikit-learn expected {expected_sklearn}, found {current_sklearn or 'not installed'}")
    if expected_sksurv and current_sksurv not in {None, expected_sksurv}:
        mismatches.append(f"scikit-survival expected {expected_sksurv}, found {current_sksurv}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            "Serialized benchmark artifacts were created under a different ML runtime and cannot be safely reused in the current interpreter. "
            f"Compatibility contract from {ENVIRONMENT_SUMMARY_PATH}: {mismatch_text}. "
            "Activate the validated benchmark environment before running stage G."
        )

print("Runtime context ready.")
print("- NOTEBOOK_NAME:", NOTEBOOK_NAME)
print("- RUN_ID       :", RUN_ID)
print("- DUCKDB_PATH  :", DUCKDB_PATH)
print("- MONOLITHIC   :", MONOLITHIC_NOTEBOOK_PATH)

print(f"[END] G0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 4
from datetime import datetime as _dt
print(f"[START] G0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

import json
from util import shutdown_duckdb_connection_from_globals

validate_serialized_model_runtime()

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(obj, file_handle, indent=2, ensure_ascii=False)

def ensure_active_duckdb_connection() -> duckdb.DuckDBPyConnection:
    global con
    connection = globals().get("con")
    if connection is not None:
        try:
            connection.execute("SELECT 1")
            return connection
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
    con = duckdb.connect(str(DUCKDB_PATH))
    return con

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

def register_duckdb_table(con, table_name: str, notebook_name: str, stage_id: str, run_id: str, catalog_table: str = "pipeline_table_catalog") -> None:
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

def materialize_dataframe(con, df: pd.DataFrame, table_name: str, stage_id: str, notebook_name: str = NOTEBOOK_NAME, run_id: str = RUN_ID) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{stage_id}: expected a pandas DataFrame for materialization into {table_name}.")
    safe_df = _sanitize_dataframe_for_duckdb(df)
    con.register("_tmp_materialize_dataframe", safe_df)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_materialize_dataframe")
    con.unregister("_tmp_materialize_dataframe")
    register_duckdb_table(con, table_name, notebook_name, stage_id, run_id)
    print(f"{stage_id}: materialized DuckDB table -> {table_name}")

def infer_table_name_from_pathlike(pathlike) -> str:
    return Path(pathlike).stem

def load_duckdb_table_or_raise(table_name: str) -> pd.DataFrame:
    connection = ensure_active_duckdb_connection()
    available = set(connection.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if table_name not in available:
        raise FileNotFoundError(f"Required DuckDB table not found: {table_name}")
    return connection.execute(f"SELECT * FROM {table_name}").fetchdf()

def load_duckdb_table_optional(table_name: str):
    connection = ensure_active_duckdb_connection()
    available = set(connection.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if table_name not in available:
        return None
    return connection.execute(f"SELECT * FROM {table_name}").fetchdf()

def print_duckdb_table(con, table_name: str, title: str | None = None, limit: int = 20) -> None:
    title = title or table_name
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    preview = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
    print(f"\n{title}")
    print(f"Table: {table_name}")
    print(f"Row count: {n_rows}")
    print(preview.to_string(index=False) if not preview.empty else "[empty table]")

ensure_pipeline_table_catalog(ensure_active_duckdb_connection())
required_runtime = ["NOTEBOOK_NAME", "RUN_ID", "con", "save_json", "materialize_dataframe", "register_duckdb_table", "print_duckdb_table", "infer_table_name_from_pathlike", "load_duckdb_table_or_raise"]
missing_runtime = [name for name in required_runtime if name not in globals()]
if missing_runtime:
    raise NameError("G0.1 runtime contract failed. Missing required object(s): " + ", ".join(missing_runtime))
print("Runtime contract validated.")

print(f"[END] G0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 8
from datetime import datetime as _dt
print(f"[START] G0.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Lightweight harmonization of shared globals only. No file rewriting here.

OUTPUT_DIR = OUTPUT_DIR
TABLES_DIR = TABLES_DIR
METADATA_DIR = METADATA_DIR
DATA_DIR = DATA_OUTPUT_DIR if DATA_OUTPUT_DIR.exists() else DATA_DIR

RANDOM_SEED = RANDOM_SEED
HORIZONS_WEEKS = list(HORIZONS_WEEKS)
HORIZON_WEEKS = list(HORIZON_WEEKS)
CALIBRATION_BINS = CALIBRATION_BINS

print("Shared globals harmonized for G notebook.")
print("- DATA_DIR    :", DATA_DIR)
print("- TABLES_DIR  :", TABLES_DIR)
print("- METADATA_DIR:", METADATA_DIR)

print(f"[END] G0.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 10
from datetime import datetime as _dt
print(f"[START] G1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G1 — Define Explainability Protocol
# --------------------------------------------------------------
# Purpose:
#   Define the explainability study design for the tuned models
#   already established in the benchmark.
#
# Methodological note:
#   This step does not train any model and does not compute any
#   explanation yet. It only formalizes:
#     - which models are included,
#     - which explainability methods will be used,
#     - which outputs are expected,
#     - and how the results should be interpreted.
#
# Scope:
#   The manuscript-facing explainability layer is anchored in the
#   retained comparable quartet used downstream by stage F and by the
#   paper-facing freeze layer:
#     - cox_tuned
#     - deepsurv_tuned
#     - rsf_tuned
#     - mtlr_tuned
#
#   The notebook still preserves legacy raw explainability cells for
#   the historical linear/neural anchors, but the manuscript contract
#   is defined by the comparable quartet above.
# ==============================================================

print("\n" + "=" * 70)
print("G1 — Define Explainability Protocol")
print("=" * 70)
print("Methodological note: this step defines the explainability study only.")
print("No model is trained and no explanation is computed here.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

import pandas as pd

# ------------------------------
# 2) Model registry
# ------------------------------
EXPLAINABILITY_MODEL_REGISTRY = [
    # --- Dynamic arm ---
    {
        "model_id": "poisson_pexp_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["poisson_pexp_tuned"]["display_name"],
        "family": "discrete_time_poisson",
        "data_level": "person_period",
        "primary_explainability_method": "coefficient_ranking",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Dynamic GLM anchor with log-rate coefficients; intrinsically interpretable per-week model."
    },
    {
        "model_id": "linear_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["linear_tuned"]["display_name"],
        "family": "discrete_time_linear",
        "data_level": "person_period",
        "primary_explainability_method": "coefficient_ranking",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Dynamic logistic discrete-time anchor with directly interpretable log-odds coefficients."
    },
    {
        "model_id": "gb_weekly_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["gb_weekly_tuned"]["display_name"],
        "family": "discrete_time_gb",
        "data_level": "person_period",
        "primary_explainability_method": "feature_importances",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Dynamic GB benchmark using MDI (mean decrease in impurity) feature importances."
    },
    {
        "model_id": "neural_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["neural_tuned"]["display_name"],
        "family": "discrete_time_neural",
        "data_level": "person_period",
        "primary_explainability_method": "grouped_permutation_importance",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Dynamic neural benchmark using grouped permutation importance at the original-feature level."
    },
    # --- Comparable arm ---
    {
        "model_id": "cox_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["cox_tuned"]["display_name"],
        "family": "continuous_time_cox",
        "data_level": "enrollment",
        "primary_explainability_method": "hazard_ratio_analysis",
        "secondary_explainability_method": "coefficient_ranking",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Classical comparable continuous-time anchor with directly interpretable coefficients."
    },
    {
        "model_id": "deepsurv_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["deepsurv_tuned"]["display_name"],
        "family": "continuous_time_deepsurv",
        "data_level": "enrollment",
        "primary_explainability_method": "grouped_permutation_importance",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Nonlinear comparable continuous-time benchmark over the same early-window enrollment representation."
    },
    {
        "model_id": "rsf_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["rsf_tuned"]["display_name"],
        "family": "continuous_time_tree_ensemble",
        "data_level": "enrollment",
        "primary_explainability_method": "grouped_permutation_importance",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Tree-based comparable benchmark and the current stable champion of the continuous-time window sensitivity layer."
    },
    {
        "model_id": "mtlr_tuned",
        "display_name": base.get_manuscript_audit_subset_by_model_id()["mtlr_tuned"]["display_name"],
        "family": "continuous_time_neural_mtlr",
        "data_level": "enrollment",
        "primary_explainability_method": "grouped_permutation_importance",
        "secondary_explainability_method": "feature_block_importance_summary",
        "local_explanation_planned": False,
        "global_explanation_planned": True,
        "positioning_note": "Discrete-time-inspired continuous-time comparable benchmark retained in the ablation manuscript subset."
    },
]

explainability_model_registry_df = pd.DataFrame(EXPLAINABILITY_MODEL_REGISTRY)

# ------------------------------
# 3) Feature-block registry
# ------------------------------
def _enrich_block(spec: dict) -> dict:
    bid = spec["block_id"]
    if bid == "static_structural":
        return {**spec, "examples": "gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability", "interpretation_goal": "Assess contribution of background and structural covariates."}
    if bid == "early_window_behavior":
        return {**spec, "examples": "clicks_first_4_weeks, active_weeks_first_4, mean_clicks_first_4_weeks", "interpretation_goal": "Assess contribution of compressed early-course behavior."}
    if bid == "dynamic_temporal_behavioral":
        return {**spec, "examples": "total_clicks_week, active_this_week, n_vle_rows_week, cum_clicks_until_t, recency, streak", "interpretation_goal": "Assess contribution of dynamic weekly behavioral signals."}
    if bid == "discrete_time_index":
        return {**spec, "examples": "week", "interpretation_goal": "Assess contribution of elapsed time (week index) as a standalone signal."}
    return {**spec, "examples": "", "interpretation_goal": ""}

EXPLAINABILITY_FEATURE_BLOCKS = [_enrich_block(spec) for spec in base.get_canonical_explainability_blocks()]

explainability_feature_blocks_df = pd.DataFrame(EXPLAINABILITY_FEATURE_BLOCKS)

def _enrich_taxonomy(spec: dict) -> dict:
    bid = spec["block_id"]
    if bid == "static_structural":
        return {**spec, "taxonomy_role": "baseline_context", "manuscript_reason": "Captures background and enrollment structure that anchor interpretation across retained families."}
    if bid == "early_window_behavior":
        return {**spec, "taxonomy_role": "behavioral_signal", "manuscript_reason": "Captures compressed early-course engagement patterns expected to explain much of the retained comparable-arm performance."}
    if bid == "dynamic_temporal_behavioral":
        return {**spec, "taxonomy_role": "dynamic_behavioral_signal", "manuscript_reason": "Captures rich week-by-week engagement; primary driver for dynamic-arm models."}
    if bid == "discrete_time_index":
        return {**spec, "taxonomy_role": "temporal_covariate", "manuscript_reason": "Week index encodes hazard shape; important for discrete-time hazard models."}
    return {**spec, "taxonomy_role": "unknown", "manuscript_reason": ""}

EXPLAINABILITY_TAXONOMY = [_enrich_taxonomy(spec) for spec in base.get_canonical_explainability_blocks()]

explainability_taxonomy_df = pd.DataFrame(EXPLAINABILITY_TAXONOMY)

# ------------------------------
# 4) Expected outputs
# ------------------------------
EXPLAINABILITY_OUTPUTS = [
    {
        "output_id": "global_feature_ranking",
        "description": "Rank features by global importance within each tuned family.",
        "planned_for": "all_models"
    },
    {
        "output_id": "feature_block_summary",
        "description": "Summarize importance patterns at the feature-block level.",
        "planned_for": "all_models"
    },
    {
        "output_id": "signed_effect_table",
        "description": "Report signed coefficient or hazard-ratio direction when directly available.",
        "planned_for": "cox_tuned"
    },
    {
        "output_id": "cross_family_explainability_summary",
        "description": "Compare whether the strongest drivers are consistent across families.",
        "planned_for": "final_consolidation"
    },
]

explainability_outputs_df = pd.DataFrame(EXPLAINABILITY_OUTPUTS)

# ------------------------------
# 5) Protocol
# ------------------------------
EXPLAINABILITY_PROTOCOL = {
    "scope": "tuned_models_only",
    "included_models": [row["model_id"] for row in EXPLAINABILITY_MODEL_REGISTRY],
    "main_goals": [
        "Identify which individual features are most influential within each retained comparable model family.",
        "Compare the dominant explanatory signals across Cox, DeepSurv, Random Survival Forest, and Neural-MTLR.",
        "Connect explainability findings back to the ablation results for the manuscript-facing comparable quartet."
    ],
    "global_vs_local_policy": {
        "global_explanations": True,
        "local_explanations": False,
        "rationale": (
            "The main goal of this benchmark stage is model-level interpretation, "
            "not case-level explanation."
        )
    },
    "method_policy_by_family": {
        "continuous_time_cox": {
            "primary": "hazard_ratio_analysis",
            "secondary": "coefficient_ranking"
        },
        "continuous_time_deepsurv": {
            "primary": "grouped_permutation_importance",
            "secondary": "feature_block_importance_summary"
        },
        "continuous_time_tree_ensemble": {
            "primary": "grouped_permutation_importance",
            "secondary": "feature_block_importance_summary"
        },
        "continuous_time_neural_mtlr": {
            "primary": "grouped_permutation_importance",
            "secondary": "feature_block_importance_summary"
        }
    },
    "interpretation_rules": {
        "large_positive_signed_effect": "Associated with increased risk when sign-based methods apply.",
        "large_negative_signed_effect": "Associated with reduced risk when sign-based methods apply.",
        "high_permutation_importance": "Model performance depends strongly on that feature.",
        "consistency_with_ablation": (
            "If an important feature belongs to a behavior-derived block, the result should align "
            "with the ablation conclusion that behavioral information dominates."
        )
    },
    "limitations_to_document": [
        "Permutation importance is global and model-dependent, not causal.",
        "Coefficient and hazard-ratio interpretations apply only to model families with directly interpretable parameters.",
        "Explainability describes how the model uses information, not whether the relationships are causal.",
        "The manuscript-facing quartet is editorially retained even when the raw feature-level explainability implementations are richer for some families than for others in the current notebook revision."
    ],
    "paper_positioning_note": (
        "Explainability is treated here as a post-benchmark interpretive layer, intended to explain "
        "why the retained comparable manuscript models behave as they do after the benchmark hierarchy, "
        "window sensitivity, and cross-arm layers have already been separated."
    )
}

# ------------------------------
# 6) Save outputs
# ------------------------------
model_registry_path = TABLES_DIR / "table_explainability_model_registry.csv"
feature_blocks_path = TABLES_DIR / "table_explainability_feature_blocks.csv"
taxonomy_path = TABLES_DIR / "table_explainability_block_taxonomy.csv"
outputs_path = TABLES_DIR / "table_explainability_outputs.csv"
config_path = METADATA_DIR / "explainability_config.json"

explainability_model_registry_df.to_csv(model_registry_path, index=False)
materialize_dataframe(con, explainability_model_registry_df, infer_table_name_from_pathlike(model_registry_path), "G1")
explainability_feature_blocks_df.to_csv(feature_blocks_path, index=False)
materialize_dataframe(con, explainability_feature_blocks_df, infer_table_name_from_pathlike(feature_blocks_path), "G1")
explainability_taxonomy_df.to_csv(taxonomy_path, index=False)
materialize_dataframe(con, explainability_taxonomy_df, infer_table_name_from_pathlike(taxonomy_path), "G1")
explainability_outputs_df.to_csv(outputs_path, index=False)
materialize_dataframe(con, explainability_outputs_df, infer_table_name_from_pathlike(outputs_path), "G1")
save_json({**EXPLAINABILITY_PROTOCOL, "taxonomy": EXPLAINABILITY_TAXONOMY}, config_path)

# ------------------------------
# 7) Output for feedback
# ------------------------------
print("\nExplainability model registry:")
display(explainability_model_registry_df)

print("\nExplainability feature blocks:")
display(explainability_feature_blocks_df)

print("\nExplainability taxonomy:")
display(explainability_taxonomy_df)

print("\nExplainability planned outputs:")
display(explainability_outputs_df)

print("\nExplainability protocol summary:")
display(pd.DataFrame([{
    "scope": EXPLAINABILITY_PROTOCOL["scope"],
    "included_models": ", ".join(EXPLAINABILITY_PROTOCOL["included_models"]),
    "global_explanations": EXPLAINABILITY_PROTOCOL["global_vs_local_policy"]["global_explanations"],
    "local_explanations": EXPLAINABILITY_PROTOCOL["global_vs_local_policy"]["local_explanations"],
    "paper_positioning_note": EXPLAINABILITY_PROTOCOL["paper_positioning_note"],
}]))

print("\nSaved:")
print("-", model_registry_path.resolve())
print("-", feature_blocks_path.resolve())
print("-", taxonomy_path.resolve())
print("-", outputs_path.resolve())
print("-", config_path.resolve())

print(f"[END] G1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 13
from datetime import datetime as _dt
print(f"[START] G2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G2 — Explainability for Linear Tuned
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned linear
#   discrete-time hazard model.
#
# Methodological note:
#   This step uses direct parameter interpretation because the
#   tuned linear model is intrinsically interpretable.
#
# Outputs:
#   - feature-level coefficient table
#   - odds-ratio table
#   - block-level summary
# ==============================================================

print("\n" + "=" * 70)
print("G2 — Explainability for Linear Tuned")
print("=" * 70)
print("Methodological note: this step computes global explainability")
print("for the tuned linear discrete-time hazard model.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# 2) Paths
# ------------------------------
MODEL_DIR = OUTPUT_DIR / "models"

model_path = MODEL_DIR / "linear_discrete_time_hazard_not_weighted_tuned_w4.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"Tuned model file not found: {model_path}")

preprocessor_path = MODEL_DIR / "linear_discrete_time_not_weighted_preprocessor_w4.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

# ------------------------------
# 3) Load artifacts
# ------------------------------
linear_model = joblib.load(model_path)
linear_preprocessor = joblib.load(preprocessor_path)

# ------------------------------
# 4) Recover transformed feature names
# ------------------------------
if not hasattr(linear_preprocessor, "get_feature_names_out"):
    raise AttributeError("The loaded preprocessor does not expose get_feature_names_out().")

feature_names_out = list(linear_preprocessor.get_feature_names_out())

if not hasattr(linear_model, "coef_"):
    raise AttributeError("The loaded linear model does not expose coef_.")

coefs = linear_model.coef_.reshape(-1)

if len(feature_names_out) != len(coefs):
    raise ValueError(
        f"Mismatch between transformed feature names ({len(feature_names_out)}) "
        f"and coefficients ({len(coefs)})."
    )

# ------------------------------
# 5) Feature-level explainability table
# ------------------------------
explain_df = pd.DataFrame({
    "feature_name_out": feature_names_out,
    "coefficient": coefs,
    "abs_coefficient": np.abs(coefs),
    "odds_ratio": np.exp(coefs),
})

def infer_block(feature_name: str) -> str:
    if feature_name.startswith("num__week"):
        return "discrete_time_index"
    if feature_name.startswith("num__total_clicks_week"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__active_this_week"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__n_vle_rows_week"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__n_distinct_sites_week"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__cum_clicks_until_t"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__recency"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__streak"):
        return "dynamic_temporal_behavioral"
    if feature_name.startswith("num__num_of_prev_attempts"):
        return "static_structural"
    if feature_name.startswith("num__studied_credits"):
        return "static_structural"
    if feature_name.startswith("cat__gender_"):
        return "static_structural"
    if feature_name.startswith("cat__region_"):
        return "static_structural"
    if feature_name.startswith("cat__highest_education_"):
        return "static_structural"
    if feature_name.startswith("cat__imd_band_"):
        return "static_structural"
    if feature_name.startswith("cat__age_band_"):
        return "static_structural"
    if feature_name.startswith("cat__disability_"):
        return "static_structural"
    return "other"

explain_df["feature_block"] = explain_df["feature_name_out"].apply(infer_block)

explain_df["effect_direction"] = np.where(
    explain_df["coefficient"] > 0,
    "increases_log_odds_of_weekly_event",
    np.where(
        explain_df["coefficient"] < 0,
        "decreases_log_odds_of_weekly_event",
        "neutral"
    )
)

explain_df_sorted_abs = explain_df.sort_values(
    by="abs_coefficient", ascending=False
).reset_index(drop=True)

explain_df_sorted_signed = explain_df.sort_values(
    by="coefficient", ascending=False
).reset_index(drop=True)

# ------------------------------
# 6) Block-level summary
# ------------------------------
block_summary_df = (
    explain_df.groupby("feature_block", as_index=False)
    .agg(
        n_features=("feature_name_out", "count"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        median_abs_coefficient=("abs_coefficient", "median"),
        max_abs_coefficient=("abs_coefficient", "max"),
        mean_coefficient=("coefficient", "mean"),
    )
    .sort_values(by="mean_abs_coefficient", ascending=False)
    .reset_index(drop=True)
)

# ------------------------------
# 7) Top positive / negative effects
# ------------------------------
top_positive_df = explain_df.sort_values(
    by="coefficient", ascending=False
).head(15).reset_index(drop=True)

top_negative_df = explain_df.sort_values(
    by="coefficient", ascending=True
).head(15).reset_index(drop=True)

# ------------------------------
# 8) Save outputs
# ------------------------------
feature_table_path = TABLES_DIR / "table_linear_explainability_feature_coefficients.csv"
signed_table_path = TABLES_DIR / "table_linear_explainability_signed_effects.csv"
block_summary_path = TABLES_DIR / "table_linear_explainability_block_summary.csv"
top_positive_path = TABLES_DIR / "table_linear_explainability_top_positive.csv"
top_negative_path = TABLES_DIR / "table_linear_explainability_top_negative.csv"
config_path = METADATA_DIR / "linear_explainability_summary.json"

explain_df_sorted_abs.to_csv(feature_table_path, index=False)
materialize_dataframe(con, explain_df_sorted_abs, infer_table_name_from_pathlike(feature_table_path), "G2")
explain_df_sorted_signed.to_csv(signed_table_path, index=False)
materialize_dataframe(con, explain_df_sorted_signed, infer_table_name_from_pathlike(signed_table_path), "G2")
block_summary_df.to_csv(block_summary_path, index=False)
materialize_dataframe(con, block_summary_df, infer_table_name_from_pathlike(block_summary_path), "G2")
top_positive_df.to_csv(top_positive_path, index=False)
materialize_dataframe(con, top_positive_df, infer_table_name_from_pathlike(top_positive_path), "G2")
top_negative_df.to_csv(top_negative_path, index=False)
materialize_dataframe(con, top_negative_df, infer_table_name_from_pathlike(top_negative_path), "G2")

save_json(
    {
        "model_id": "linear_tuned",
        "model_file_used": str(model_path),
        "preprocessor_file_used": str(preprocessor_path),
        "n_transformed_features": int(len(feature_names_out)),
        "top_feature_by_abs_coef": explain_df_sorted_abs.iloc[0]["feature_name_out"],
        "top_feature_block_by_mean_abs_coef": block_summary_df.iloc[0]["feature_block"],
    },
    config_path,
)

# ------------------------------
# 9) Output for feedback
# ------------------------------
print("\nTop features by absolute coefficient:")
display(explain_df_sorted_abs.head(20))

print("\nTop positive effects:")
display(top_positive_df)

print("\nTop negative effects:")
display(top_negative_df)

print("\nFeature-block summary:")
display(block_summary_df)

print("\nSaved:")
print("-", feature_table_path.resolve())
print("-", signed_table_path.resolve())
print("-", block_summary_path.resolve())
print("-", top_positive_path.resolve())
print("-", top_negative_path.resolve())
print("-", config_path.resolve())

print(f"[END] G2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 15
from datetime import datetime as _dt
print(f"[START] G3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G3 — Explainability for Neural Tuned (Revised v2)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned neural
#   discrete-time survival model using GROUPED permutation
#   importance at the original-feature level.
#
# Methodological note:
#   This revised version corrects the prior approach by avoiding
#   permutation at the one-hot encoded column level.
#
#   Instead, permutation is applied by original feature/group:
#   - each numeric feature is permuted as one group
#   - each categorical feature is permuted jointly across all
#     derived one-hot columns through raw-data permutation followed
#     by preprocessing
#
# This yields feature-level explainability that is better aligned
# with the original data semantics and with the ablation study.
# ==============================================================

print("\n" + "=" * 70)
print("G3 — Explainability for Neural Tuned (Revised v2)")
print("=" * 70)
print("Methodological note: this step computes global explainability")
print("for the tuned neural discrete-time survival model using GROUPED")
print("permutation importance at the original-feature level.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = [
    "OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR",
    "RANDOM_SEED", "HORIZONS_WEEKS", "save_json"
]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
import torchtuples as tt
import joblib

from sklearn.metrics import roc_auc_score

try:
    from pycox.evaluation import EvalSurv
    PYCOX_AVAILABLE = True
except Exception:
    PYCOX_AVAILABLE = False

if not PYCOX_AVAILABLE:
    raise ImportError("pycox is required for P34.")

# ------------------------------
# 2) Compatibility patch for SciPy / PyCox
# ------------------------------
try:
    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        def _simps_compat(y, x=None, dx=1.0, axis=-1, even=None):
            return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
        scipy.integrate.simps = _simps_compat
except Exception:
    pass

# ------------------------------
# 3) Paths
# ------------------------------
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = OUTPUT_DIR / "data"

preprocessor_path = MODEL_DIR / "neural_discrete_time_not_weighted_preprocessor_w4.joblib"
train_data_table = "pp_neural_hazard_ready_train"
test_data_table = "pp_neural_hazard_ready_test"


if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

# ------------------------------
# 4) Load artifacts
# ------------------------------
neural_preprocessor = joblib.load(preprocessor_path)

neural_train_df = load_duckdb_table_or_raise(train_data_table)

neural_test_df = load_duckdb_table_or_raise(test_data_table)

# ------------------------------
# 5) Column definitions
# ------------------------------
AUX_DISCRETE = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "event_observed",
    "t_event_week",
    "t_final_week",
    "used_zero_week_fallback_for_censoring",
    "split",
    "time_for_split",
    "time_bucket",
    "event_time_bucket_label",
]

TARGET_COL = "event_t"

FEATURE_GROUPS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
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

print(f"[END] G3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 17
from datetime import datetime as _dt
print(f"[START] G3.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_feature_columns(df: pd.DataFrame):
    excluded = set(AUX_DISCRETE + [TARGET_COL])
    return [c for c in df.columns if c not in excluded]

print(f"[END] G3.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 19
from datetime import datetime as _dt

import warnings



print(f"[START] G3.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")



feature_cols_raw = get_feature_columns(neural_train_df)



missing_expected_features = [c for c in FEATURE_GROUPS if c not in feature_cols_raw]

if missing_expected_features:

    raise ValueError(

        f"Missing expected feature groups in test/train data: {missing_expected_features}"

    )



# transformed design

X_train = neural_preprocessor.transform(neural_train_df[feature_cols_raw])

X_test = neural_preprocessor.transform(neural_test_df[feature_cols_raw])



if hasattr(X_train, "toarray"):

    X_train_dense = X_train.toarray().astype(np.float32)

    X_test_dense = X_test.toarray().astype(np.float32)

else:

    X_train_dense = np.asarray(X_train).astype(np.float32)

    X_test_dense = np.asarray(X_test).astype(np.float32)



y_train = neural_train_df[TARGET_COL].to_numpy().astype(np.float32)



# ------------------------------

# 6) Refit tuned neural model

# ------------------------------

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)



net = tt.practical.MLPVanilla(

    in_features=X_train_dense.shape[1],

    num_nodes=[128, 64],

    out_features=1,

    batch_norm=True,

    dropout=0.1,

    output_bias=False,

)



model = tt.Model(net, torch.nn.BCEWithLogitsLoss(), tt.optim.AdamW)

model.optimizer.set_lr(5e-4)



with warnings.catch_warnings():

    warnings.filterwarnings(

        "ignore",

        message=r"This overload of add is deprecated:.*",

        category=UserWarning,

        module=r"torchtuples\.callbacks",

    )

    _ = model.fit(

        X_train_dense,

        y_train.reshape(-1, 1),

        batch_size=256,

        epochs=25,

        verbose=False,

    )



print(f"[END] G3.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 21
from datetime import datetime as _dt

import numpy as np

import pandas as pd



print(f"[START] G3.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")



# ------------------------------

# 7) Evaluation helper

# ------------------------------

def get_surv_at_horizon(surv_frame: pd.DataFrame, h: int) -> pd.Series:

    idx = np.asarray(surv_frame.index, dtype=float)

    pos = np.searchsorted(idx, float(h), side="right") - 1

    if pos < 0:

        return pd.Series(np.ones(surv_frame.shape[1]), index=surv_frame.columns)

    return surv_frame.iloc[pos]



print(f"[END] G3.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 23
from datetime import datetime as _dt
print(f"[START] G3.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def evaluate_discrete_survival_from_hazard(test_pred_df: pd.DataFrame, horizons: list[int]):
    test_pred_df = test_pred_df.sort_values(["enrollment_id", "week"]).copy()
    test_pred_df["pred_survival"] = test_pred_df.groupby("enrollment_id")["pred_hazard"].transform(
        lambda s: (1.0 - s).cumprod()
    )
    test_pred_df["pred_risk"] = 1.0 - test_pred_df["pred_survival"]

    duration_source_col = "time_for_split" if "time_for_split" in test_pred_df.columns else "t_final_week" if "t_final_week" in test_pred_df.columns else "week"

    truth_test = (
        test_pred_df.groupby("enrollment_id", as_index=False)
        .agg(
            event=("event_observed", "max") if "event_observed" in test_pred_df.columns else ("event_t", "max"),
            duration=(duration_source_col, "max"),
        )
    )

    surv_wide = (
        test_pred_df[["enrollment_id", "week", "pred_survival"]]
        .drop_duplicates(subset=["enrollment_id", "week"])
        .pivot(index="week", columns="enrollment_id", values="pred_survival")
        .sort_index()
    )

    max_week_test = int(pd.to_numeric(test_pred_df["week"], errors="coerce").max())
    full_week_index = pd.Index(np.arange(0, max_week_test + 1), name="week")
    surv_wide = surv_wide.reindex(full_week_index).ffill().fillna(1.0)

    surv_df = surv_wide.copy()
    surv_df.columns.name = "enrollment_id"

    durations_test = pd.to_numeric(truth_test["duration"], errors="coerce").fillna(0).astype(int).to_numpy()
    events_test = pd.to_numeric(truth_test["event"], errors="coerce").fillna(0).astype(int).to_numpy()

    eval_surv = EvalSurv(
        surv=surv_df,
        durations=durations_test,
        events=events_test,
        censor_surv="km",
    )

    try:
        max_requested_horizon = int(max(horizons))
        ibs_grid = np.arange(1, max_requested_horizon + 1, dtype=int)
        ibs_value = float(eval_surv.integrated_brier_score(ibs_grid))
    except Exception:
        ibs_value = np.nan

    risk_auc_rows = []
    for h in horizons:
        pred_surv_h = get_surv_at_horizon(surv_df, h)
        pred_risk_h = 1.0 - pred_surv_h

        eval_df = truth_test.copy()
        eval_df["pred_risk_h"] = eval_df["enrollment_id"].map(pred_risk_h.to_dict())

        eval_df["is_evaluable_at_h"] = (
            ((pd.to_numeric(eval_df["event"], errors="coerce") == 1) & (pd.to_numeric(eval_df["duration"], errors="coerce") <= h)) |
            (pd.to_numeric(eval_df["duration"], errors="coerce") >= h)
        ).astype(int)

        eval_df = eval_df[eval_df["is_evaluable_at_h"] == 1].copy()
        eval_df["observed_event_by_h"] = ((pd.to_numeric(eval_df["event"], errors="coerce") == 1) & (pd.to_numeric(eval_df["duration"], errors="coerce") <= h)).astype(int)

        if eval_df["observed_event_by_h"].nunique() >= 2:
            risk_auc = roc_auc_score(eval_df["observed_event_by_h"], eval_df["pred_risk_h"])
        else:
            risk_auc = np.nan

        risk_auc_rows.append({
            "horizon_week": h,
            "risk_auc_at_horizon": float(risk_auc) if pd.notna(risk_auc) else np.nan,
        })

    return {
        "ibs": ibs_value,
        "risk_auc_df": pd.DataFrame(risk_auc_rows),
    }

print(f"[END] G3.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 25
from datetime import datetime as _dt
print(f"[START] G3.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def predict_hazard_from_raw_df(raw_df: pd.DataFrame) -> np.ndarray:
    X = neural_preprocessor.transform(raw_df[feature_cols_raw])
    if hasattr(X, "toarray"):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X).astype(np.float32)

    logits = model.predict(X_dense).reshape(-1)
    pred_hazard = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(pred_hazard, 1e-8, 1 - 1e-8)

print(f"[END] G3.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 27
from datetime import datetime as _dt
print(f"[START] G3.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def infer_block(feature_name: str) -> str:
    if feature_name == "week":
        return "discrete_time_index"
    if feature_name in [
        "total_clicks_week",
        "active_this_week",
        "n_vle_rows_week",
        "n_distinct_sites_week",
        "cum_clicks_until_t",
        "recency",
        "streak",
    ]:
        return "dynamic_temporal_behavioral"
    return "static_structural"

print(f"[END] G3.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 29
from datetime import datetime as _dt
print(f"[START] G3.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 8) Baseline performance
# ------------------------------
baseline_test_pred_df = neural_test_df.copy()
baseline_test_pred_df["pred_hazard"] = predict_hazard_from_raw_df(neural_test_df)

baseline_eval = evaluate_discrete_survival_from_hazard(
    baseline_test_pred_df,
    HORIZONS_WEEKS
)

baseline_ibs = baseline_eval["ibs"]
baseline_risk_auc_df = baseline_eval["risk_auc_df"].copy()

baseline_risk_auc_h10 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 10, "risk_auc_at_horizon"].iloc[0]
)
baseline_risk_auc_h20 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 20, "risk_auc_at_horizon"].iloc[0]
)
baseline_risk_auc_h30 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 30, "risk_auc_at_horizon"].iloc[0]
)

# ------------------------------
# 9) GROUPED permutation importance
# ------------------------------
rng = np.random.default_rng(RANDOM_SEED)
importance_rows = []

for feat in FEATURE_GROUPS:
    perm_df = neural_test_df.copy()

    shuffled = perm_df[feat].to_numpy(copy=True)
    rng.shuffle(shuffled)
    perm_df[feat] = shuffled

    perm_test_pred_df = perm_df.copy()
    perm_test_pred_df["pred_hazard"] = predict_hazard_from_raw_df(perm_df)

    perm_eval = evaluate_discrete_survival_from_hazard(
        perm_test_pred_df,
        HORIZONS_WEEKS
    )

    perm_ibs = perm_eval["ibs"]
    perm_risk_auc_df = perm_eval["risk_auc_df"].copy()

    perm_risk_auc_h10 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 10, "risk_auc_at_horizon"].iloc[0]
    )
    perm_risk_auc_h20 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 20, "risk_auc_at_horizon"].iloc[0]
    )
    perm_risk_auc_h30 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 30, "risk_auc_at_horizon"].iloc[0]
    )

    importance_rows.append({
        "feature_name_original": feat,
        "baseline_ibs": baseline_ibs,
        "permuted_ibs": perm_ibs,
        "delta_ibs": perm_ibs - baseline_ibs if pd.notna(perm_ibs) and pd.notna(baseline_ibs) else np.nan,
        "baseline_risk_auc_h10": baseline_risk_auc_h10,
        "permuted_risk_auc_h10": perm_risk_auc_h10,
        "delta_risk_auc_h10": perm_risk_auc_h10 - baseline_risk_auc_h10 if pd.notna(perm_risk_auc_h10) else np.nan,
        "baseline_risk_auc_h20": baseline_risk_auc_h20,
        "permuted_risk_auc_h20": perm_risk_auc_h20,
        "delta_risk_auc_h20": perm_risk_auc_h20 - baseline_risk_auc_h20 if pd.notna(perm_risk_auc_h20) else np.nan,
        "baseline_risk_auc_h30": baseline_risk_auc_h30,
        "permuted_risk_auc_h30": perm_risk_auc_h30,
        "delta_risk_auc_h30": perm_risk_auc_h30 - baseline_risk_auc_h30 if pd.notna(perm_risk_auc_h30) else np.nan,
    })

importance_df = pd.DataFrame(importance_rows)
importance_df["importance_score_ibs"] = importance_df["delta_ibs"]
importance_df["importance_score_risk_auc_h10"] = -importance_df["delta_risk_auc_h10"]
importance_df["importance_score_risk_auc_h20"] = -importance_df["delta_risk_auc_h20"]
importance_df["importance_score_risk_auc_h30"] = -importance_df["delta_risk_auc_h30"]
importance_df["mean_importance_score_auc"] = importance_df[
    ["importance_score_risk_auc_h10", "importance_score_risk_auc_h20", "importance_score_risk_auc_h30"]
].mean(axis=1)
importance_df["feature_block"] = importance_df["feature_name_original"].apply(infer_block)

importance_sorted_df = importance_df.sort_values(
    by=["importance_score_ibs", "mean_importance_score_auc"],
    ascending=[False, False]
).reset_index(drop=True)

# ------------------------------
# 10) Block summary
# ------------------------------
block_summary_df = (
    importance_df.groupby("feature_block", as_index=False)
    .agg(
        n_features=("feature_name_original", "count"),
        mean_importance_score_ibs=("importance_score_ibs", "mean"),
        median_importance_score_ibs=("importance_score_ibs", "median"),
        max_importance_score_ibs=("importance_score_ibs", "max"),
        mean_importance_score_auc=("mean_importance_score_auc", "mean"),
    )
    .sort_values(by="mean_importance_score_ibs", ascending=False)
    .reset_index(drop=True)
)

top_features_df = importance_sorted_df.head(20).reset_index(drop=True)

# ------------------------------
# 11) Save outputs
# ------------------------------
feature_table_path = TABLES_DIR / "table_neural_explainability_grouped_permutation_importance.csv"
block_summary_path = TABLES_DIR / "table_neural_explainability_grouped_block_summary.csv"
top_features_path = TABLES_DIR / "table_neural_explainability_grouped_top_features.csv"
config_path = METADATA_DIR / "neural_explainability_grouped_summary.json"

importance_sorted_df.to_csv(feature_table_path, index=False)
materialize_dataframe(con, importance_sorted_df, infer_table_name_from_pathlike(feature_table_path), "G3")
block_summary_df.to_csv(block_summary_path, index=False)
materialize_dataframe(con, block_summary_df, infer_table_name_from_pathlike(block_summary_path), "G3")
top_features_df.to_csv(top_features_path, index=False)
materialize_dataframe(con, top_features_df, infer_table_name_from_pathlike(top_features_path), "G3")

save_json(
    {
        "model_id": "neural_tuned",
        "preprocessor_file_used": str(preprocessor_path),
        "train_data_used": train_data_table,
        "test_data_used": test_data_table,
        "n_feature_groups": int(len(FEATURE_GROUPS)),
        "baseline_ibs_refit_model": float(baseline_ibs) if pd.notna(baseline_ibs) else None,
        "top_feature_by_grouped_permutation_ibs": top_features_df.iloc[0]["feature_name_original"],
        "top_feature_block_by_mean_importance_ibs": block_summary_df.iloc[0]["feature_block"],
    },
    config_path,
)

# ------------------------------
# 12) Output for feedback
# ------------------------------
print("\nTop neural features by GROUPED permutation importance:")
display(top_features_df)

print("\nNeural grouped feature-block summary:")
display(block_summary_df)

print("\nSaved:")
print("-", feature_table_path.resolve())
print("-", block_summary_path.resolve())
print("-", top_features_path.resolve())
print("-", config_path.resolve())

print(f"[END] G3.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 31
from datetime import datetime as _dt
print(f"[START] G4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G4 — Explainability for Cox Tuned (Revised)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned Cox
#   comparable benchmark.
#
# Methodological note:
#   This step uses direct parameter interpretation because the
#   tuned Cox model is intrinsically interpretable.
# ==============================================================

print("\n" + "=" * 70)
print("G4 — Explainability for Cox Tuned (Revised)")
print("=" * 70)
print("Methodological note: this step computes global explainability")
print("for the tuned Cox comparable benchmark.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# 2) Paths
# ------------------------------
MODEL_DIR = OUTPUT_DIR / "models"

model_path = MODEL_DIR / "cox_early_window_tuned.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"Tuned model file not found: {model_path}")

preprocessor_path = MODEL_DIR / "cox_preprocessor.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

# ------------------------------
# 3) Load artifacts
# ------------------------------
cox_model = joblib.load(model_path)
cox_preprocessor = joblib.load(preprocessor_path)

# ------------------------------
# 4) Recover transformed feature names
# ------------------------------
if not hasattr(cox_preprocessor, "get_feature_names_out"):
    raise AttributeError("The loaded preprocessor does not expose get_feature_names_out().")

feature_names_out = list(cox_preprocessor.get_feature_names_out())

if not hasattr(cox_model, "params_"):
    raise AttributeError("Loaded Cox model does not expose params_.")

coef_series = cox_model.params_.copy()
param_names = coef_series.index.tolist()

print(f"[END] G4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 33
from datetime import datetime as _dt
print(f"[START] G4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def map_param_name(name, feature_names_out):
    if isinstance(name, str) and name.startswith("x"):
        suffix = name[1:]
        if suffix.isdigit():
            idx = int(suffix)
            if idx < len(feature_names_out):
                return feature_names_out[idx]
    return name

print(f"[END] G4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 35
from datetime import datetime as _dt
print(f"[START] G4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

mapped_feature_names = [map_param_name(name, feature_names_out) for name in param_names]

# ------------------------------
# 5) Robustly load summary and rename first column
# ------------------------------
summary_df = cox_model.summary.copy().reset_index()

if summary_df.shape[1] == 0:
    raise ValueError("Cox model summary is empty after reset_index().")

first_col_name = summary_df.columns[0]
summary_df = summary_df.rename(columns={first_col_name: "raw_feature_name"})

if "coef" not in summary_df.columns:
    raise ValueError("Cox model summary does not contain 'coef' column.")

raw_feature_names_summary = summary_df["raw_feature_name"].tolist()
mapped_summary_names = [map_param_name(name, feature_names_out) for name in raw_feature_names_summary]
summary_df["feature_name_out"] = mapped_summary_names

# ------------------------------
# 6) Build explainability table
# ------------------------------
keep_cols = [
    "feature_name_out",
    "coef",
    "exp(coef)",
    "se(coef)",
    "coef lower 95%",
    "coef upper 95%",
    "exp(coef) lower 95%",
    "exp(coef) upper 95%",
    "z",
    "p",
]

available_keep_cols = [c for c in keep_cols if c in summary_df.columns]
explain_df = summary_df[available_keep_cols].copy()

rename_map = {
    "coef": "coefficient",
    "exp(coef)": "hazard_ratio",
    "se(coef)": "se_coefficient",
    "coef lower 95%": "coef_lower_95",
    "coef upper 95%": "coef_upper_95",
    "exp(coef) lower 95%": "hazard_ratio_lower_95",
    "exp(coef) upper 95%": "hazard_ratio_upper_95",
    "p": "p_value",
}
explain_df.rename(columns=rename_map, inplace=True)

explain_df["abs_coefficient"] = explain_df["coefficient"].abs()

print(f"[END] G4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 37
from datetime import datetime as _dt
print(f"[START] G4.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def infer_block(feature_name: str) -> str:
    if str(feature_name).startswith("num__clicks_first_4_weeks"):
        return "early_window_behavior"
    if str(feature_name).startswith("num__active_weeks_first_4"):
        return "early_window_behavior"
    if str(feature_name).startswith("num__mean_clicks_first_4_weeks"):
        return "early_window_behavior"
    if str(feature_name).startswith("num__num_of_prev_attempts"):
        return "static_structural"
    if str(feature_name).startswith("num__studied_credits"):
        return "static_structural"
    if str(feature_name).startswith("cat__gender_"):
        return "static_structural"
    if str(feature_name).startswith("cat__region_"):
        return "static_structural"
    if str(feature_name).startswith("cat__highest_education_"):
        return "static_structural"
    if str(feature_name).startswith("cat__imd_band_"):
        return "static_structural"
    if str(feature_name).startswith("cat__age_band_"):
        return "static_structural"
    if str(feature_name).startswith("cat__disability_"):
        return "static_structural"
    return "other"

print(f"[END] G4.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 39
from datetime import datetime as _dt
print(f"[START] G4.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

explain_df["feature_block"] = explain_df["feature_name_out"].apply(infer_block)

explain_df["effect_direction"] = np.where(
    explain_df["coefficient"] > 0,
    "increases_hazard",
    np.where(
        explain_df["coefficient"] < 0,
        "decreases_hazard",
        "neutral"
    )
)

explain_df_sorted_abs = explain_df.sort_values(
    by="abs_coefficient", ascending=False
).reset_index(drop=True)

explain_df_sorted_signed = explain_df.sort_values(
    by="coefficient", ascending=False
).reset_index(drop=True)

# ------------------------------
# 7) Block-level summary
# ------------------------------
block_summary_df = (
    explain_df.groupby("feature_block", as_index=False)
    .agg(
        n_features=("feature_name_out", "count"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        median_abs_coefficient=("abs_coefficient", "median"),
        max_abs_coefficient=("abs_coefficient", "max"),
        mean_coefficient=("coefficient", "mean"),
        mean_hazard_ratio=("hazard_ratio", "mean"),
    )
    .sort_values(by="mean_abs_coefficient", ascending=False)
    .reset_index(drop=True)
)

# ------------------------------
# 8) Top positive / negative effects
# ------------------------------
top_positive_df = explain_df.sort_values(
    by="coefficient", ascending=False
).head(15).reset_index(drop=True)

top_negative_df = explain_df.sort_values(
    by="coefficient", ascending=True
).head(15).reset_index(drop=True)

# ------------------------------
# 9) Significant effects only
# ------------------------------
if "p_value" in explain_df.columns:
    significant_df = explain_df[explain_df["p_value"] < 0.05].copy()
    significant_df = significant_df.sort_values(
        by="abs_coefficient", ascending=False
    ).reset_index(drop=True)
else:
    significant_df = explain_df.iloc[0:0].copy()

# ------------------------------
# 10) Save outputs
# ------------------------------
feature_table_path = TABLES_DIR / "table_cox_explainability_feature_coefficients.csv"
signed_table_path = TABLES_DIR / "table_cox_explainability_signed_effects.csv"
block_summary_path = TABLES_DIR / "table_cox_explainability_block_summary.csv"
top_positive_path = TABLES_DIR / "table_cox_explainability_top_positive.csv"
top_negative_path = TABLES_DIR / "table_cox_explainability_top_negative.csv"
significant_path = TABLES_DIR / "table_cox_explainability_significant_effects.csv"
config_path = METADATA_DIR / "cox_explainability_summary.json"

explain_df_sorted_abs.to_csv(feature_table_path, index=False)
materialize_dataframe(con, explain_df_sorted_abs, infer_table_name_from_pathlike(feature_table_path), "G4")
explain_df_sorted_signed.to_csv(signed_table_path, index=False)
materialize_dataframe(con, explain_df_sorted_signed, infer_table_name_from_pathlike(signed_table_path), "G4")
block_summary_df.to_csv(block_summary_path, index=False)
materialize_dataframe(con, block_summary_df, infer_table_name_from_pathlike(block_summary_path), "G4")
top_positive_df.to_csv(top_positive_path, index=False)
materialize_dataframe(con, top_positive_df, infer_table_name_from_pathlike(top_positive_path), "G4")
top_negative_df.to_csv(top_negative_path, index=False)
materialize_dataframe(con, top_negative_df, infer_table_name_from_pathlike(top_negative_path), "G4")
significant_df.to_csv(significant_path, index=False)
materialize_dataframe(con, significant_df, infer_table_name_from_pathlike(significant_path), "G4")

save_json(
    {
        "model_id": "cox_tuned",
        "model_file_used": str(model_path),
        "preprocessor_file_used": str(preprocessor_path),
        "n_transformed_features_in_summary": int(explain_df.shape[0]),
        "top_feature_by_abs_coef": explain_df_sorted_abs.iloc[0]["feature_name_out"],
        "top_feature_block_by_mean_abs_coef": block_summary_df.iloc[0]["feature_block"],
        "n_significant_features_p_lt_0_05": int(significant_df.shape[0]),
    },
    config_path,
)

# ------------------------------
# 11) Output for feedback
# ------------------------------
print("\nTop Cox features by absolute coefficient:")
display(explain_df_sorted_abs.head(20))

print("\nTop positive Cox effects:")
display(top_positive_df)

print("\nTop negative Cox effects:")
display(top_negative_df)

print("\nSignificant Cox effects (p < 0.05):")
display(significant_df.head(20))

print("\nCox feature-block summary:")
display(block_summary_df)

print("\nSaved:")
print("-", feature_table_path.resolve())
print("-", signed_table_path.resolve())
print("-", block_summary_path.resolve())
print("-", top_positive_path.resolve())
print("-", top_negative_path.resolve())
print("-", significant_path.resolve())
print("-", config_path.resolve())

print(f"[END] G4.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 41
from datetime import datetime as _dt
print(f"[START] G5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G5 — Explainability for DeepSurv Tuned
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned DeepSurv
#   model using GROUPED permutation importance at the original-
#   feature level.
#
# Methodological note:
#   This step follows the same corrected explainability logic
#   used for the tuned neural model:
#   - permutation is applied by original feature/group
#   - not by individual one-hot encoded transformed column
#
# Scope:
#   Enrollment-level continuous-time DeepSurv benchmark.
# ==============================================================

print("\n" + "=" * 70)
print("G5 — Explainability for DeepSurv Tuned")
print("=" * 70)
print("Methodological note: this step computes global explainability")
print("for the tuned DeepSurv model using GROUPED permutation")
print("importance at the original-feature level.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = [
    "OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR",
    "RANDOM_SEED", "HORIZONS_WEEKS", "save_json"
]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
import torchtuples as tt
import joblib

from sklearn.metrics import roc_auc_score

try:
    from pycox.evaluation import EvalSurv
    PYCOX_AVAILABLE = True
except Exception:
    PYCOX_AVAILABLE = False

if not PYCOX_AVAILABLE:
    raise ImportError("pycox is required for P36.")

# ------------------------------
# 2) Compatibility patch for SciPy / PyCox
# ------------------------------
try:
    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        def _simps_compat(y, x=None, dx=1.0, axis=-1, even=None):
            return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
        scipy.integrate.simps = _simps_compat
except Exception:
    pass

# ------------------------------
# 3) Paths
# ------------------------------
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = OUTPUT_DIR / "data"

preprocessor_path = MODEL_DIR / "deepsurv_preprocessor.joblib"
train_data_table = "enrollment_deepsurv_ready_train"
test_data_table = "enrollment_deepsurv_ready_test"


if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

# ------------------------------
# 4) Load artifacts
# ------------------------------
deepsurv_preprocessor = joblib.load(preprocessor_path)

deepsurv_train_df = load_duckdb_table_or_raise(train_data_table)

deepsurv_test_df = load_duckdb_table_or_raise(test_data_table)

# ------------------------------
# 5) Column definitions
# ------------------------------
AUX_CONTINUOUS = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "split",
    "time_for_split",
    "time_bucket",
    "event_time_bucket_label",
]

TARGET_EVENT_COL = "event"
TARGET_DURATION_COL = "duration"

FEATURE_GROUPS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
    "clicks_first_4_weeks",
    "active_weeks_first_4",
    "mean_clicks_first_4_weeks",
]

print(f"[END] G5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 43
from datetime import datetime as _dt
print(f"[START] G5.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_feature_columns(df: pd.DataFrame):
    excluded = set(AUX_CONTINUOUS + [TARGET_EVENT_COL, TARGET_DURATION_COL, "duration_raw", "used_zero_week_fallback_for_censoring"])
    return [c for c in df.columns if c not in excluded]

print(f"[END] G5.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 45
from datetime import datetime as _dt
print(f"[START] G5.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

feature_cols_raw = get_feature_columns(deepsurv_train_df)

missing_expected_features = [c for c in FEATURE_GROUPS if c not in feature_cols_raw]
if missing_expected_features:
    raise ValueError(
        f"Missing expected feature groups in DeepSurv data: {missing_expected_features}"
    )

# transformed design
X_train = deepsurv_preprocessor.transform(deepsurv_train_df[feature_cols_raw])
X_test = deepsurv_preprocessor.transform(deepsurv_test_df[feature_cols_raw])

if hasattr(X_train, "toarray"):
    X_train_dense = X_train.toarray().astype(np.float32)
    X_test_dense = X_test.toarray().astype(np.float32)
else:
    X_train_dense = np.asarray(X_train).astype(np.float32)
    X_test_dense = np.asarray(X_test).astype(np.float32)

duration_train = deepsurv_train_df[TARGET_DURATION_COL].to_numpy().astype(np.float32)
event_train = deepsurv_train_df[TARGET_EVENT_COL].to_numpy().astype(np.float32)

duration_test = deepsurv_test_df[TARGET_DURATION_COL].to_numpy().astype(np.float32)
event_test = deepsurv_test_df[TARGET_EVENT_COL].to_numpy().astype(np.float32)

# ------------------------------
# 6) Refit tuned DeepSurv model
# ------------------------------
# Based on tuned winner family size from P25:
# hidden_dims=[64, 32], dropout=0.3, lr=5e-4, weight_decay=1e-4

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

net = tt.practical.MLPVanilla(
    in_features=X_train_dense.shape[1],
    num_nodes=[64, 32],
    out_features=1,
    batch_norm=True,
    dropout=0.3,
    output_bias=False,
)

model = tt.Model(net, torch.nn.BCEWithLogitsLoss(), tt.optim.AdamW)
model.optimizer.set_lr(5e-4)

# NOTE:
# This is a practical benchmark-side refit for explainability.
# We keep the same tuned architecture family and optimizer scale.
_ = model.fit(
    X_train_dense,
    event_train.reshape(-1, 1),
    batch_size=256,
    epochs=55,
    verbose=False,
)

print(f"[END] G5.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 47
from datetime import datetime as _dt
print(f"[START] G5.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 7) Survival helper
# ------------------------------
def build_survival_df_from_risk_scores(
    risk_scores: np.ndarray,
    train_duration: np.ndarray,
    train_event: np.ndarray,
    test_duration: np.ndarray
) -> pd.DataFrame:
    """
    Approximate survival curves from relative risk scores using a
    Breslow-style baseline hazard estimated from the TRAIN set.
    """

    risk_scores = np.asarray(risk_scores).reshape(-1)
    train_duration_i = np.asarray(train_duration).astype(int)
    train_event_i = np.asarray(train_event).astype(int)
    test_duration_i = np.asarray(test_duration).astype(int)

    # approximate train scores from current model
    train_logits = model.predict(X_train_dense.astype(np.float32)).reshape(-1)
    train_risk_scores = np.exp(train_logits)

    unique_event_times = np.sort(np.unique(train_duration_i[train_event_i == 1]))
    if len(unique_event_times) == 0:
        raise ValueError("No event times found in training set for DeepSurv baseline estimation.")

    baseline_hazard = []
    for t in unique_event_times:
        d_t = np.sum((train_duration_i == t) & (train_event_i == 1))
        at_risk = train_duration_i >= t
        denom = np.sum(train_risk_scores[at_risk])
        h0_t = d_t / denom if denom > 0 else 0.0
        baseline_hazard.append(h0_t)

    baseline_hazard = np.asarray(baseline_hazard, dtype=float)
    baseline_cum_hazard = np.cumsum(baseline_hazard)

    max_t = int(max(np.max(test_duration_i), np.max(unique_event_times)))
    full_times = np.arange(0, max_t + 1, dtype=int)

    cumhaz_full = np.zeros_like(full_times, dtype=float)
    time_to_ch = {int(t): ch for t, ch in zip(unique_event_times, baseline_cum_hazard)}
    running = 0.0
    for i, t in enumerate(full_times):
        if t in time_to_ch:
            running = time_to_ch[t]
        cumhaz_full[i] = running

    surv_cols = {}
    for i in range(len(risk_scores)):
        rs = np.exp(risk_scores[i])
        surv_cols[i] = np.exp(-cumhaz_full * rs)

    surv_df = pd.DataFrame(surv_cols, index=full_times)
    surv_df.index.name = "time"
    return surv_df

print(f"[END] G5.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 49
from datetime import datetime as _dt
print(f"[START] G5.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_surv_at_horizon(surv_frame: pd.DataFrame, h: int) -> pd.Series:
    idx = np.asarray(surv_frame.index, dtype=float)
    pos = np.searchsorted(idx, float(h), side="right") - 1
    if pos < 0:
        return pd.Series(np.ones(surv_frame.shape[1]), index=surv_frame.columns)
    return surv_frame.iloc[pos]

print(f"[END] G5.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 51
from datetime import datetime as _dt
print(f"[START] G5.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def evaluate_continuous_survival_from_risk_scores(
    risk_scores_test: np.ndarray,
    duration_test: np.ndarray,
    event_test: np.ndarray,
    horizons: list[int]
):
    surv_df = build_survival_df_from_risk_scores(
        risk_scores=risk_scores_test,
        train_duration=duration_train,
        train_event=event_train,
        test_duration=duration_test,
    )

    eval_surv = EvalSurv(
        surv=surv_df,
        durations=duration_test.astype(int),
        events=event_test.astype(int),
        censor_surv="km",
    )

    try:
        max_requested_horizon = int(max(horizons))
        ibs_grid = np.arange(1, max_requested_horizon + 1, dtype=int)
        ibs_value = float(eval_surv.integrated_brier_score(ibs_grid))
    except Exception:
        ibs_value = np.nan

    truth_test = pd.DataFrame({
        "duration": duration_test.astype(int),
        "event": event_test.astype(int),
    })

    risk_auc_rows = []
    for h in horizons:
        pred_surv_h = get_surv_at_horizon(surv_df, h)
        pred_risk_h = 1.0 - pred_surv_h

        eval_df = truth_test.copy()
        eval_df["pred_risk_h"] = pred_risk_h.values

        eval_df["is_evaluable_at_h"] = (
            ((eval_df["event"] == 1) & (eval_df["duration"] <= h)) |
            (eval_df["duration"] >= h)
        ).astype(int)

        eval_df = eval_df[eval_df["is_evaluable_at_h"] == 1].copy()
        eval_df["observed_event_by_h"] = ((eval_df["event"] == 1) & (eval_df["duration"] <= h)).astype(int)

        if eval_df["observed_event_by_h"].nunique() >= 2:
            risk_auc = roc_auc_score(eval_df["observed_event_by_h"], eval_df["pred_risk_h"])
        else:
            risk_auc = np.nan

        risk_auc_rows.append({
            "horizon_week": h,
            "risk_auc_at_horizon": float(risk_auc) if pd.notna(risk_auc) else np.nan,
        })

    return {
        "ibs": ibs_value,
        "risk_auc_df": pd.DataFrame(risk_auc_rows),
        "surv_df": surv_df,
    }

print(f"[END] G5.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 53
from datetime import datetime as _dt
print(f"[START] G5.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def predict_risk_score_from_raw_df(raw_df: pd.DataFrame) -> np.ndarray:
    X = deepsurv_preprocessor.transform(raw_df[feature_cols_raw])
    if hasattr(X, "toarray"):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X).astype(np.float32)

    logits = model.predict(X_dense).reshape(-1)
    return logits

print(f"[END] G5.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 55
from datetime import datetime as _dt
print(f"[START] G5.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def infer_block(feature_name: str) -> str:
    if feature_name in [
        "clicks_first_4_weeks",
        "active_weeks_first_4",
        "mean_clicks_first_4_weeks",
    ]:
        return "early_window_behavior"
    return "static_structural"

print(f"[END] G5.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 57
from datetime import datetime as _dt
print(f"[START] G5.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 8) Baseline performance
# ------------------------------
baseline_risk_scores = predict_risk_score_from_raw_df(deepsurv_test_df)

baseline_eval = evaluate_continuous_survival_from_risk_scores(
    risk_scores_test=baseline_risk_scores,
    duration_test=duration_test,
    event_test=event_test,
    horizons=HORIZONS_WEEKS,
)

baseline_ibs = baseline_eval["ibs"]
baseline_risk_auc_df = baseline_eval["risk_auc_df"].copy()

baseline_risk_auc_h10 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 10, "risk_auc_at_horizon"].iloc[0]
)
baseline_risk_auc_h20 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 20, "risk_auc_at_horizon"].iloc[0]
)
baseline_risk_auc_h30 = float(
    baseline_risk_auc_df.loc[baseline_risk_auc_df["horizon_week"] == 30, "risk_auc_at_horizon"].iloc[0]
)

# ------------------------------
# 9) GROUPED permutation importance
# ------------------------------
rng = np.random.default_rng(RANDOM_SEED)
importance_rows = []

for feat in FEATURE_GROUPS:
    perm_df = deepsurv_test_df.copy()

    shuffled = perm_df[feat].to_numpy(copy=True)
    rng.shuffle(shuffled)
    perm_df[feat] = shuffled

    perm_risk_scores = predict_risk_score_from_raw_df(perm_df)

    perm_eval = evaluate_continuous_survival_from_risk_scores(
        risk_scores_test=perm_risk_scores,
        duration_test=duration_test,
        event_test=event_test,
        horizons=HORIZONS_WEEKS,
    )

    perm_ibs = perm_eval["ibs"]
    perm_risk_auc_df = perm_eval["risk_auc_df"].copy()

    perm_risk_auc_h10 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 10, "risk_auc_at_horizon"].iloc[0]
    )
    perm_risk_auc_h20 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 20, "risk_auc_at_horizon"].iloc[0]
    )
    perm_risk_auc_h30 = float(
        perm_risk_auc_df.loc[perm_risk_auc_df["horizon_week"] == 30, "risk_auc_at_horizon"].iloc[0]
    )

    importance_rows.append({
        "feature_name_original": feat,
        "baseline_ibs": baseline_ibs,
        "permuted_ibs": perm_ibs,
        "delta_ibs": perm_ibs - baseline_ibs if pd.notna(perm_ibs) and pd.notna(baseline_ibs) else np.nan,
        "baseline_risk_auc_h10": baseline_risk_auc_h10,
        "permuted_risk_auc_h10": perm_risk_auc_h10,
        "delta_risk_auc_h10": perm_risk_auc_h10 - baseline_risk_auc_h10 if pd.notna(perm_risk_auc_h10) else np.nan,
        "baseline_risk_auc_h20": baseline_risk_auc_h20,
        "permuted_risk_auc_h20": perm_risk_auc_h20,
        "delta_risk_auc_h20": perm_risk_auc_h20 - baseline_risk_auc_h20 if pd.notna(perm_risk_auc_h20) else np.nan,
        "baseline_risk_auc_h30": baseline_risk_auc_h30,
        "permuted_risk_auc_h30": perm_risk_auc_h30,
        "delta_risk_auc_h30": perm_risk_auc_h30 - baseline_risk_auc_h30 if pd.notna(perm_risk_auc_h30) else np.nan,
    })

importance_df = pd.DataFrame(importance_rows)
importance_df["importance_score_ibs"] = importance_df["delta_ibs"]
importance_df["importance_score_risk_auc_h10"] = -importance_df["delta_risk_auc_h10"]
importance_df["importance_score_risk_auc_h20"] = -importance_df["delta_risk_auc_h20"]
importance_df["importance_score_risk_auc_h30"] = -importance_df["delta_risk_auc_h30"]
importance_df["mean_importance_score_auc"] = importance_df[
    ["importance_score_risk_auc_h10", "importance_score_risk_auc_h20", "importance_score_risk_auc_h30"]
].mean(axis=1)
importance_df["feature_block"] = importance_df["feature_name_original"].apply(infer_block)

importance_sorted_df = importance_df.sort_values(
    by=["importance_score_ibs", "mean_importance_score_auc"],
    ascending=[False, False]
).reset_index(drop=True)

# ------------------------------
# 10) Block summary
# ------------------------------
block_summary_df = (
    importance_df.groupby("feature_block", as_index=False)
    .agg(
        n_features=("feature_name_original", "count"),
        mean_importance_score_ibs=("importance_score_ibs", "mean"),
        median_importance_score_ibs=("importance_score_ibs", "median"),
        max_importance_score_ibs=("importance_score_ibs", "max"),
        mean_importance_score_auc=("mean_importance_score_auc", "mean"),
    )
    .sort_values(by="mean_importance_score_ibs", ascending=False)
    .reset_index(drop=True)
)

top_features_df = importance_sorted_df.head(20).reset_index(drop=True)

# ------------------------------
# 11) Save outputs
# ------------------------------
feature_table_path = TABLES_DIR / "table_deepsurv_explainability_grouped_permutation_importance.csv"
block_summary_path = TABLES_DIR / "table_deepsurv_explainability_grouped_block_summary.csv"
top_features_path = TABLES_DIR / "table_deepsurv_explainability_grouped_top_features.csv"
config_path = METADATA_DIR / "deepsurv_explainability_grouped_summary.json"

importance_sorted_df.to_csv(feature_table_path, index=False)
materialize_dataframe(con, importance_sorted_df, infer_table_name_from_pathlike(feature_table_path), "G5")
block_summary_df.to_csv(block_summary_path, index=False)
materialize_dataframe(con, block_summary_df, infer_table_name_from_pathlike(block_summary_path), "G5")
top_features_df.to_csv(top_features_path, index=False)
materialize_dataframe(con, top_features_df, infer_table_name_from_pathlike(top_features_path), "G5")

save_json(
    {
        "model_id": "deepsurv_tuned",
        "preprocessor_file_used": str(preprocessor_path),
        "train_data_used": train_data_table,
        "test_data_used": test_data_table,
        "n_feature_groups": int(len(FEATURE_GROUPS)),
        "baseline_ibs_refit_model": float(baseline_ibs) if pd.notna(baseline_ibs) else None,
        "top_feature_by_grouped_permutation_ibs": top_features_df.iloc[0]["feature_name_original"],
        "top_feature_block_by_mean_importance_ibs": block_summary_df.iloc[0]["feature_block"],
    },
    config_path,
)

# ------------------------------
# 12) Output for feedback
# ------------------------------
print("\nTop DeepSurv features by GROUPED permutation importance:")
display(top_features_df)

print("\nDeepSurv grouped feature-block summary:")
display(block_summary_df)

print("\nSaved:")
print("-", feature_table_path.resolve())
print("-", block_summary_path.resolve())
print("-", top_features_path.resolve())
print("-", config_path.resolve())

print(f"[END] G5.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 59a
from datetime import datetime as _dt
print(f"[START] G5.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G5.9 — Explainability for Poisson Piecewise-Exponential (Tuned)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned Poisson
#   PPE model using coefficient-based interpretation.
#
# Methodological note:
#   PoissonRegressor is a GLM with log link; coef_ gives
#   log-rate coefficients which are directly interpretable.
#   Positive coef_ → higher weekly dropout rate.
# ==============================================================

print("\n" + "=" * 70)
print("G5.9 — Explainability for Poisson PPE")
print("=" * 70)

import numpy as np
import pandas as pd
import joblib

MODEL_DIR = OUTPUT_DIR / "models"

poisson_model_path = MODEL_DIR / "poisson_piecewise_exponential_not_weighted_tuned_w4.joblib"
poisson_prep_path  = MODEL_DIR / "poisson_piecewise_exponential_not_weighted_preprocessor_w4.joblib"

if not poisson_model_path.exists():
    raise FileNotFoundError(str(poisson_model_path))
if not poisson_prep_path.exists():
    raise FileNotFoundError(str(poisson_prep_path))

poisson_pipeline  = joblib.load(poisson_model_path)
poisson_prep      = joblib.load(poisson_prep_path)

# Recover feature names from the saved preprocessor
if hasattr(poisson_prep, "get_feature_names_out"):
    poisson_feature_names_out = list(poisson_prep.get_feature_names_out())
elif hasattr(poisson_pipeline, "named_steps"):
    prep_step = poisson_pipeline.named_steps.get("preprocessor") or poisson_pipeline.named_steps.get("prep")
    poisson_feature_names_out = list(prep_step.get_feature_names_out()) if prep_step else []
else:
    poisson_feature_names_out = []

# Recover coefficients from the estimator step
# statsmodels RegularizedResultsWrapper has .params (not .coef_)
# params[0] is the intercept (const); remaining match preprocessor output features
if hasattr(poisson_pipeline, "params"):
    _raw_params = np.array(poisson_pipeline.params)
    # Drop intercept: model.exog_names starts with 'const'; params[1:] align with prep features
    if hasattr(poisson_pipeline, "model") and hasattr(poisson_pipeline.model, "exog_names"):
        _exog = list(poisson_pipeline.model.exog_names)
        if _exog and _exog[0] == "const":
            poisson_coefs = _raw_params[1:]  # strip intercept
        else:
            poisson_coefs = _raw_params
    elif len(_raw_params) == len(poisson_feature_names_out) + 1:
        poisson_coefs = _raw_params[1:]  # assume first is intercept
    else:
        poisson_coefs = _raw_params
elif hasattr(poisson_pipeline, "coef_"):
    poisson_coefs = poisson_pipeline.coef_.reshape(-1)
elif hasattr(poisson_pipeline, "named_steps"):
    est_step = None
    for k in ["model", "estimator", "classifier", "regressor"]:
        if k in poisson_pipeline.named_steps:
            est_step = poisson_pipeline.named_steps[k]
            break
    if est_step is None:
        est_step = list(poisson_pipeline.named_steps.values())[-1]
    poisson_coefs = est_step.coef_.reshape(-1) if hasattr(est_step, "coef_") else np.array([])
else:
    poisson_coefs = np.array([])

def _infer_block_pp(fn):
    if fn.startswith("num__week"):                         return "discrete_time_index"
    if fn.startswith(("num__total_clicks", "num__active_this", "num__n_vle", "num__n_distinct", "num__cum_clicks", "num__recency", "num__streak")): return "dynamic_temporal_behavioral"
    if fn.startswith(("num__num_of_prev", "num__studied_credits", "cat__gender", "cat__region", "cat__highest_ed", "cat__imd_band", "cat__age_band", "cat__disability")): return "static_structural"
    return "other"

if len(poisson_coefs) > 0 and len(poisson_feature_names_out) == len(poisson_coefs):
    poisson_exp_df = pd.DataFrame({
        "feature_name_out": poisson_feature_names_out,
        "coefficient": poisson_coefs,
        "abs_coefficient": np.abs(poisson_coefs),
        "rate_ratio": np.exp(poisson_coefs),
    })
    poisson_exp_df["feature_block"] = poisson_exp_df["feature_name_out"].apply(_infer_block_pp)
    poisson_exp_df["effect_direction"] = np.where(poisson_exp_df["coefficient"] > 0, "increases_log_rate", np.where(poisson_exp_df["coefficient"] < 0, "decreases_log_rate", "neutral"))
    poisson_exp_df_sorted = poisson_exp_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    poisson_block_df = (
        poisson_exp_df.groupby("feature_block", as_index=False).agg(
            n_features=("feature_name_out", "count"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            median_abs_coefficient=("abs_coefficient", "median"),
            max_abs_coefficient=("abs_coefficient", "max"),
            mean_coefficient=("coefficient", "mean"),
        ).sort_values("mean_abs_coefficient", ascending=False).reset_index(drop=True)
    )
else:
    # Fallback: schema-stable empty tables
    print("WARNING: Could not recover Poisson coefficients. Emitting schema-stable fallback.")
    poisson_exp_df_sorted = pd.DataFrame(columns=["feature_name_out", "coefficient", "abs_coefficient", "rate_ratio", "feature_block", "effect_direction"])
    poisson_block_df = pd.DataFrame(columns=["feature_block", "n_features", "mean_abs_coefficient", "median_abs_coefficient", "max_abs_coefficient", "mean_coefficient"])

feat_path_pois  = TABLES_DIR / "table_poisson_explainability_feature_coefficients.csv"
block_path_pois = TABLES_DIR / "table_poisson_explainability_block_summary.csv"
poisson_exp_df_sorted.to_csv(feat_path_pois, index=False)
materialize_dataframe(con, poisson_exp_df_sorted, infer_table_name_from_pathlike(feat_path_pois), "G5.9")
poisson_block_df.to_csv(block_path_pois, index=False)
materialize_dataframe(con, poisson_block_df, infer_table_name_from_pathlike(block_path_pois), "G5.9")
save_json({"model_id": "poisson_pexp_tuned", "model_file": str(poisson_model_path), "n_feats": int(len(poisson_exp_df_sorted))}, METADATA_DIR / "poisson_explainability_summary.json")
print("G5.9 done:", feat_path_pois.name, block_path_pois.name)

print(f"[END] G5.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 59b
from datetime import datetime as _dt
print(f"[START] G5.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G5.10 — Explainability for GB Weekly Hazard (Tuned)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned GB model
#   using MDI (mean decrease in impurity) feature importances.
# ==============================================================

print("\n" + "=" * 70)
print("G5.10 — Explainability for GB Weekly Hazard")
print("=" * 70)

import numpy as np
import pandas as pd
import joblib

gb_model_path = MODEL_DIR / "gb_weekly_hazard_not_weighted_tuned_w4.joblib"
gb_prep_path  = MODEL_DIR / "gb_weekly_hazard_not_weighted_preprocessor_w4.joblib"

if not gb_model_path.exists():
    raise FileNotFoundError(str(gb_model_path))
if not gb_prep_path.exists():
    raise FileNotFoundError(str(gb_prep_path))

gb_pipeline = joblib.load(gb_model_path)
gb_prep     = joblib.load(gb_prep_path)

# Recover feature names
if hasattr(gb_prep, "get_feature_names_out"):
    gb_feat_names = list(gb_prep.get_feature_names_out())
else:
    gb_feat_names = []

# HistGradientBoostingClassifier has no feature_importances_ attribute.
# Use grouped permutation importance (delta-AUC) on the person-period test set.
from sklearn.metrics import roc_auc_score as _roc_auc

# Raw feature groups for GB (same 16 features seen by gb_prep)
_GB_FEATURE_GROUPS_PP = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
    "num_of_prev_attempts", "studied_credits",
    "week",
    "total_clicks_week", "active_this_week", "n_vle_rows_week", "n_distinct_sites_week",
    "cum_clicks_until_t", "recency", "streak",
]

def _infer_block_pp_raw(fn):
    if fn == "week":
        return "discrete_time_index"
    if fn in {"total_clicks_week", "active_this_week", "n_vle_rows_week",
              "n_distinct_sites_week", "cum_clicks_until_t", "recency", "streak"}:
        return "dynamic_temporal_behavioral"
    return "static_structural"

gb_test_df_perm = load_duckdb_table_or_raise("pp_linear_hazard_ready_test")
_GB_AUX = {"enrollment_id", "id_student", "code_module", "code_presentation",
           "event_t", "t_event_week", "t_final_week",
           "used_zero_week_fallback_for_censoring", "split"}
_GB_TARGET = "event_observed"
_gb_fcols = [c for c in gb_test_df_perm.columns
             if c not in _GB_AUX and c != _GB_TARGET]

X_test_gb_base = gb_prep.transform(gb_test_df_perm[_gb_fcols])
if hasattr(X_test_gb_base, "toarray"):
    X_test_gb_base = X_test_gb_base.toarray()
y_test_gb       = gb_test_df_perm[_GB_TARGET].to_numpy()
y_pred_gb_base  = gb_pipeline.predict_proba(X_test_gb_base)[:, 1]
baseline_auc_gb = float(_roc_auc(y_test_gb, y_pred_gb_base))
print(f"GB baseline AUC: {baseline_auc_gb:.4f}")

rng_gb = np.random.default_rng(RANDOM_SEED)
gb_imp_rows = []
for _feat in _GB_FEATURE_GROUPS_PP:
    if _feat not in gb_test_df_perm.columns:
        continue
    _perm_df = gb_test_df_perm.copy()
    _shuf = _perm_df[_feat].to_numpy(copy=True)
    rng_gb.shuffle(_shuf)
    _perm_df[_feat] = _shuf
    X_perm_gb = gb_prep.transform(_perm_df[_gb_fcols])
    if hasattr(X_perm_gb, "toarray"):
        X_perm_gb = X_perm_gb.toarray()
    try:
        _perm_auc = float(_roc_auc(y_test_gb, gb_pipeline.predict_proba(X_perm_gb)[:, 1]))
    except Exception:
        _perm_auc = float("nan")
    _delta = (baseline_auc_gb - _perm_auc) if not np.isnan(_perm_auc) else 0.0
    gb_imp_rows.append({
        "feature_name_out": _feat,
        "feature_block": _infer_block_pp_raw(_feat),
        "importance_mdi": float(_delta),
    })

gb_exp_df_sorted = (
    pd.DataFrame(gb_imp_rows)
    .sort_values("importance_mdi", ascending=False)
    .reset_index(drop=True)
)
gb_block_df = (
    gb_exp_df_sorted.groupby("feature_block", as_index=False).agg(
        n_features=("feature_name_out", "count"),
        mean_abs_coefficient=("importance_mdi", "mean"),
        median_abs_coefficient=("importance_mdi", "median"),
        max_abs_coefficient=("importance_mdi", "max"),
        mean_coefficient=("importance_mdi", "mean"),
    ).sort_values("mean_abs_coefficient", ascending=False).reset_index(drop=True)
)

feat_path_gb  = TABLES_DIR / "table_gb_explainability_feature_importances.csv"
block_path_gb = TABLES_DIR / "table_gb_explainability_block_summary.csv"
gb_exp_df_sorted.to_csv(feat_path_gb, index=False)
materialize_dataframe(con, gb_exp_df_sorted, infer_table_name_from_pathlike(feat_path_gb), "G5.10")
gb_block_df.to_csv(block_path_gb, index=False)
materialize_dataframe(con, gb_block_df, infer_table_name_from_pathlike(block_path_gb), "G5.10")
save_json({"model_id": "gb_weekly_tuned", "model_file": str(gb_model_path), "n_feats": int(len(gb_exp_df_sorted))}, METADATA_DIR / "gb_explainability_summary.json")
print("G5.10 done:", feat_path_gb.name, block_path_gb.name)

print(f"[END] G5.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 59c
from datetime import datetime as _dt
print(f"[START] G5.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G5.11 — Explainability for RSF (Tuned)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned Random
#   Survival Forest using grouped permutation importance.
#
# Methodological note:
#   Permutation is applied at the original-feature level on the
#   enrollment-level test set. IBS change is the primary metric.
# ==============================================================

print("\n" + "=" * 70)
print("G5.11 — Explainability for RSF")
print("=" * 70)

import numpy as np
import pandas as pd
import joblib
import scipy

try:
    from pycox.evaluation import EvalSurv
    _PYCOX_RSF = True
except Exception:
    _PYCOX_RSF = False

try:
    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        def _simps_compat(y, x=None, dx=1.0, axis=-1, even=None):
            return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
        scipy.integrate.simps = _simps_compat
except Exception:
    pass

rsf_model_path = MODEL_DIR / "rsf_tuned.joblib"
rsf_prep_path  = MODEL_DIR / "rsf_preprocessor.joblib"

if not rsf_model_path.exists():
    raise FileNotFoundError(str(rsf_model_path))
if not rsf_prep_path.exists():
    raise FileNotFoundError(str(rsf_prep_path))

rsf_model = joblib.load(rsf_model_path)
rsf_prep  = joblib.load(rsf_prep_path)

rsf_test_df = load_duckdb_table_or_raise("enrollment_cox_ready_test")

AUX_ENR = ["enrollment_id", "id_student", "code_module", "code_presentation", "split",
           "time_for_split", "time_bucket", "event_time_bucket_label"]
TGT_EVENT = "event"
TGT_DUR   = "duration"

def _rsf_feature_cols(df):
    excluded = set(AUX_ENR + [TGT_EVENT, TGT_DUR, "duration_raw", "used_zero_week_fallback_for_censoring"])
    return [c for c in df.columns if c not in excluded]

rsf_test_fcols = _rsf_feature_cols(rsf_test_df)

FEATURE_GROUPS_ENR = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
    "num_of_prev_attempts", "studied_credits",
    "clicks_first_4_weeks", "active_weeks_first_4", "mean_clicks_first_4_weeks",
]

def _infer_block_enr(fn):
    if fn in ["clicks_first_4_weeks", "active_weeks_first_4", "mean_clicks_first_4_weeks"]:
        return "early_window_behavior"
    return "static_structural"

def _get_rsf_ibs(model, prep, test_df, fcols, dur_arr, ev_arr):
    X = prep.transform(test_df[fcols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    surv_funcs = model.predict_survival_function(X)
    time_grid = np.arange(1, int(dur_arr.max()) + 1)
    surv_matrix = np.column_stack([fn(time_grid) for fn in surv_funcs])
    surv_df_local = pd.DataFrame(surv_matrix, index=time_grid)
    ev = EvalSurv(surv=surv_df_local, durations=dur_arr, events=ev_arr, censor_surv="km")
    return float(ev.integrated_brier_score(time_grid))

dur_test_rsf = rsf_test_df[TGT_DUR].astype(int).to_numpy()
ev_test_rsf  = rsf_test_df[TGT_EVENT].astype(int).to_numpy()

try:
    baseline_ibs_rsf = _get_rsf_ibs(rsf_model, rsf_prep, rsf_test_df, rsf_test_fcols, dur_test_rsf, ev_test_rsf)
except Exception as e:
    print(f"WARNING: RSF baseline IBS failed: {e}")
    baseline_ibs_rsf = np.nan

rng_rsf = np.random.default_rng(RANDOM_SEED)
rsf_imp_rows = []

for feat in FEATURE_GROUPS_ENR:
    if feat not in rsf_test_df.columns:
        continue
    perm_df = rsf_test_df.copy()
    shuf = perm_df[feat].to_numpy(copy=True)
    rng_rsf.shuffle(shuf)
    perm_df[feat] = shuf
    try:
        perm_ibs = _get_rsf_ibs(rsf_model, rsf_prep, perm_df, rsf_test_fcols, dur_test_rsf, ev_test_rsf)
    except Exception as e:
        perm_ibs = np.nan
    delta = float(perm_ibs - baseline_ibs_rsf) if pd.notna(perm_ibs) and pd.notna(baseline_ibs_rsf) else np.nan
    rsf_imp_rows.append({
        "feature_name_original": feat,
        "feature_block": _infer_block_enr(feat),
        "baseline_ibs": float(baseline_ibs_rsf) if pd.notna(baseline_ibs_rsf) else np.nan,
        "permuted_ibs": float(perm_ibs) if pd.notna(perm_ibs) else np.nan,
        "importance_score_ibs": delta,
        "mean_importance_score_auc": delta,  # no AUC separate for RSF here
    })

rsf_exp_df = pd.DataFrame(rsf_imp_rows).sort_values("importance_score_ibs", ascending=False).reset_index(drop=True)
rsf_block_df = (
    rsf_exp_df.groupby("feature_block", as_index=False).agg(
        n_features=("feature_name_original", "count"),
        mean_importance_score_ibs=("importance_score_ibs", "mean"),
        median_importance_score_ibs=("importance_score_ibs", "median"),
        max_importance_score_ibs=("importance_score_ibs", "max"),
        mean_importance_score_auc=("mean_importance_score_auc", "mean"),
    ).sort_values("mean_importance_score_ibs", ascending=False).reset_index(drop=True)
)

feat_path_rsf  = TABLES_DIR / "table_rsf_explainability_grouped_permutation_importance.csv"
block_path_rsf = TABLES_DIR / "table_rsf_explainability_grouped_block_summary.csv"
rsf_exp_df.to_csv(feat_path_rsf, index=False)
materialize_dataframe(con, rsf_exp_df, infer_table_name_from_pathlike(feat_path_rsf), "G5.11")
rsf_block_df.to_csv(block_path_rsf, index=False)
materialize_dataframe(con, rsf_block_df, infer_table_name_from_pathlike(block_path_rsf), "G5.11")
save_json({"model_id": "rsf_tuned", "n_feats": int(len(rsf_imp_rows)), "baseline_ibs": float(baseline_ibs_rsf) if pd.notna(baseline_ibs_rsf) else None}, METADATA_DIR / "rsf_explainability_summary.json")
print("G5.11 done:", feat_path_rsf.name, block_path_rsf.name)

print(f"[END] G5.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 59d
from datetime import datetime as _dt
print(f"[START] G5.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G5.12 — Explainability for MTLR (Tuned)
# --------------------------------------------------------------
# Purpose:
#   Produce global explainability outputs for the tuned MTLR
#   model using grouped permutation importance.
#
# Methodological note:
#   Same approach as G5 (DeepSurv): permute each original feature
#   and measure IBS degradation on the enrollment-level test set.
# ==============================================================

print("\n" + "=" * 70)
print("G5.12 — Explainability for MTLR")
print("=" * 70)

import numpy as np
import pandas as pd
import json as _json
import joblib
import torch
import torchtuples as tt
import scipy

try:
    from pycox.models import MTLR as PyMTLR
    from pycox.evaluation import EvalSurv
    _PYCOX_MTLR = True
except Exception:
    _PYCOX_MTLR = False

if not _PYCOX_MTLR:
    raise ImportError("pycox is required for G5.12.")

try:
    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        def _simps_compat2(y, x=None, dx=1.0, axis=-1, even=None):
            return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
        scipy.integrate.simps = _simps_compat2
except Exception:
    pass

mtlr_model_path = MODEL_DIR / "mtlr_tuned.pt"
mtlr_prep_path  = MODEL_DIR / "mtlr_preprocessor.joblib"
# The MTLR config is written to metadata/, not models/
mtlr_cfg_path   = (MODEL_DIR.parent / "metadata" / "mtlr_tuned_model_config.json")

for p in [mtlr_model_path, mtlr_prep_path, mtlr_cfg_path]:
    if not p.exists():
        raise FileNotFoundError(str(p))

with open(mtlr_cfg_path, "r") as _fh:
    _mtlr_cfg = _json.load(_fh)

mtlr_prep     = joblib.load(mtlr_prep_path)
mtlr_test_df  = load_duckdb_table_or_raise("enrollment_cox_ready_test")

def _rsf_feature_cols_g(df):
    excluded = set(AUX_ENR + [TGT_EVENT, TGT_DUR, "duration_raw", "used_zero_week_fallback_for_censoring"])
    return [c for c in df.columns if c not in excluded]

mtlr_test_fcols = _rsf_feature_cols_g(mtlr_test_df)

def _get_X_mtlr(prep, df, fcols):
    X = prep.transform(df[fcols])
    if hasattr(X, "toarray"):
        X = X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)

mtlr_X_test = _get_X_mtlr(mtlr_prep, mtlr_test_df, mtlr_test_fcols)
dur_test_mtlr = mtlr_test_df[TGT_DUR].astype(int).to_numpy()
ev_test_mtlr  = mtlr_test_df[TGT_EVENT].astype(int).to_numpy()

# Rebuild MTLR from config
_in_feats = mtlr_X_test.shape[1]
_hidden = list(_mtlr_cfg.get("best_candidate", {}).get("hidden_dims", [128, 64]))
_num_dur = (
    int(_mtlr_cfg.get("best_candidate", {}).get("num_durations", 20))
    if "best_candidate" in _mtlr_cfg
    else int(_mtlr_cfg.get("num_durations", 20))
)
_dropout = float((_mtlr_cfg.get("best_candidate") or _mtlr_cfg).get("dropout", 0.1))
_lr = float((_mtlr_cfg.get("best_candidate") or _mtlr_cfg).get("learning_rate", 0.001))

_mtlr_net = tt.practical.MLPVanilla(
    in_features=_in_feats,
    num_nodes=_hidden,
    out_features=_num_dur,
    batch_norm=True,
    dropout=_dropout,
    output_bias=False,
)

_mtlr_model = PyMTLR(_mtlr_net, tt.optim.Adam(lr=_lr))
# PyTorch ≥ 2.4 requires weights_only=False for legacy pickled state dicts
try:
    _mtlr_model.load_net(str(mtlr_model_path))
except Exception:
    import torch as _torch
    _state = _torch.load(str(mtlr_model_path), map_location="cpu", weights_only=False)
    if isinstance(_state, dict):
        _mtlr_model.net.load_state_dict(_state)
    else:
        _mtlr_model.net = _state

def _get_mtlr_ibs(model, X_arr, dur_arr, ev_arr):
    surv = model.predict_surv_df(X_arr)
    ev = EvalSurv(surv=surv, durations=dur_arr, events=ev_arr, censor_surv="km")
    return float(ev.integrated_brier_score(np.arange(1, int(dur_arr.max()) + 1)))

try:
    baseline_ibs_mtlr = _get_mtlr_ibs(_mtlr_model, mtlr_X_test, dur_test_mtlr, ev_test_mtlr)
except Exception as e:
    print(f"WARNING: MTLR baseline IBS failed: {e}")
    baseline_ibs_mtlr = np.nan

rng_mtlr = np.random.default_rng(RANDOM_SEED)
mtlr_imp_rows = []

for feat in FEATURE_GROUPS_ENR:
    if feat not in mtlr_test_df.columns:
        continue
    perm_df = mtlr_test_df.copy()
    shuf = perm_df[feat].to_numpy(copy=True)
    rng_mtlr.shuffle(shuf)
    perm_df[feat] = shuf
    X_perm = _get_X_mtlr(mtlr_prep, perm_df, mtlr_test_fcols)
    try:
        perm_ibs = _get_mtlr_ibs(_mtlr_model, X_perm, dur_test_mtlr, ev_test_mtlr)
    except Exception as e:
        perm_ibs = np.nan
    delta = float(perm_ibs - baseline_ibs_mtlr) if pd.notna(perm_ibs) and pd.notna(baseline_ibs_mtlr) else np.nan
    mtlr_imp_rows.append({
        "feature_name_original": feat,
        "feature_block": _infer_block_enr(feat),
        "baseline_ibs": float(baseline_ibs_mtlr) if pd.notna(baseline_ibs_mtlr) else np.nan,
        "permuted_ibs": float(perm_ibs) if pd.notna(perm_ibs) else np.nan,
        "importance_score_ibs": delta,
        "mean_importance_score_auc": delta,
    })

mtlr_exp_df = pd.DataFrame(mtlr_imp_rows).sort_values("importance_score_ibs", ascending=False).reset_index(drop=True)
mtlr_block_df = (
    mtlr_exp_df.groupby("feature_block", as_index=False).agg(
        n_features=("feature_name_original", "count"),
        mean_importance_score_ibs=("importance_score_ibs", "mean"),
        median_importance_score_ibs=("importance_score_ibs", "median"),
        max_importance_score_ibs=("importance_score_ibs", "max"),
        mean_importance_score_auc=("mean_importance_score_auc", "mean"),
    ).sort_values("mean_importance_score_ibs", ascending=False).reset_index(drop=True)
)

feat_path_mtlr  = TABLES_DIR / "table_mtlr_explainability_grouped_permutation_importance.csv"
block_path_mtlr = TABLES_DIR / "table_mtlr_explainability_grouped_block_summary.csv"
mtlr_exp_df.to_csv(feat_path_mtlr, index=False)
materialize_dataframe(con, mtlr_exp_df, infer_table_name_from_pathlike(feat_path_mtlr), "G5.12")
mtlr_block_df.to_csv(block_path_mtlr, index=False)
materialize_dataframe(con, mtlr_block_df, infer_table_name_from_pathlike(block_path_mtlr), "G5.12")
save_json({"model_id": "mtlr_tuned", "n_feats": int(len(mtlr_imp_rows)), "baseline_ibs": float(baseline_ibs_mtlr) if pd.notna(baseline_ibs_mtlr) else None}, METADATA_DIR / "mtlr_explainability_summary.json")
print("G5.12 done:", feat_path_mtlr.name, block_path_mtlr.name)

print(f"[END] G5.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 60
from datetime import datetime as _dt
print(f"[START] G6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G6 — Consolidate Explainability Across All Tuned Families
# --------------------------------------------------------------
# Purpose:
#   Consolidate explainability outputs across all tuned model
#   families into cross-family comparison tables.
#
# Inputs expected from previous cells:
#   - OUTPUT_DIR
#   - TABLES_DIR
#   - METADATA_DIR
#   - save_json
#
# Expected existing outputs:
#   - Linear tuned explainability tables (G2)
#   - Neural tuned grouped explainability tables (G3 revised v2)
#   - Cox tuned explainability tables (G4)
#   - DeepSurv tuned grouped explainability tables (G5)
#
# Outputs:
#   - consolidated explainability summary by model
#   - cross-family feature-block comparison
#   - top drivers by model
#   - convergence summary across families
# ==============================================================

print("\n" + "=" * 70)
print("G6 — Consolidate Explainability Across All Tuned Families")
print("=" * 70)
print("Methodological note: this step consolidates explainability outputs only.")
print("No model is trained and no new explanation is computed here.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------
# 2) Paths
# ------------------------------
# Linear
linear_feature_path = TABLES_DIR / "table_linear_explainability_feature_coefficients.csv"
linear_block_path = TABLES_DIR / "table_linear_explainability_block_summary.csv"

# Neural
neural_feature_path = TABLES_DIR / "table_neural_explainability_grouped_permutation_importance.csv"
neural_block_path = TABLES_DIR / "table_neural_explainability_grouped_block_summary.csv"

# Cox
cox_feature_path = TABLES_DIR / "table_cox_explainability_feature_coefficients.csv"
cox_block_path = TABLES_DIR / "table_cox_explainability_block_summary.csv"

# DeepSurv
deepsurv_feature_path = TABLES_DIR / "table_deepsurv_explainability_grouped_permutation_importance.csv"
deepsurv_block_path = TABLES_DIR / "table_deepsurv_explainability_grouped_block_summary.csv"

# Poisson (coefficient-based)
poisson_feature_path = TABLES_DIR / "table_poisson_explainability_feature_coefficients.csv"
poisson_block_path = TABLES_DIR / "table_poisson_explainability_block_summary.csv"

# GB (MDI importances)
gb_feature_path = TABLES_DIR / "table_gb_explainability_feature_importances.csv"
gb_block_path = TABLES_DIR / "table_gb_explainability_block_summary.csv"

# RSF (grouped permutation)
rsf_feature_path = TABLES_DIR / "table_rsf_explainability_grouped_permutation_importance.csv"
rsf_block_path = TABLES_DIR / "table_rsf_explainability_grouped_block_summary.csv"

# MTLR (grouped permutation)
mtlr_feature_path = TABLES_DIR / "table_mtlr_explainability_grouped_permutation_importance.csv"
mtlr_block_path = TABLES_DIR / "table_mtlr_explainability_grouped_block_summary.csv"

required_paths = [
    linear_feature_path, linear_block_path,
    neural_feature_path, neural_block_path,
    cox_feature_path, cox_block_path,
    deepsurv_feature_path, deepsurv_block_path,
    poisson_feature_path, poisson_block_path,
    gb_feature_path, gb_block_path,
    rsf_feature_path, rsf_block_path,
    mtlr_feature_path, mtlr_block_path,
]
missing_paths = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Missing explainability input files:\n- " + "\n- ".join(missing_paths)
    )

# ------------------------------
# 3) Load tables
# ------------------------------
linear_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(linear_feature_path))
linear_block_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(linear_block_path))

neural_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(neural_feature_path))
neural_block_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(neural_block_path))

cox_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(cox_feature_path))
cox_block_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(cox_block_path))

deepsurv_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(deepsurv_feature_path))
deepsurv_block_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(deepsurv_block_path))

poisson_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(poisson_feature_path))
poisson_block_df_g6 = load_duckdb_table_or_raise(infer_table_name_from_pathlike(poisson_block_path))

gb_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(gb_feature_path))
gb_block_df_g6 = load_duckdb_table_or_raise(infer_table_name_from_pathlike(gb_block_path))

rsf_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(rsf_feature_path))
rsf_block_df_g6 = load_duckdb_table_or_raise(infer_table_name_from_pathlike(rsf_block_path))

mtlr_feature_df = load_duckdb_table_or_raise(infer_table_name_from_pathlike(mtlr_feature_path))
mtlr_block_df_g6 = load_duckdb_table_or_raise(infer_table_name_from_pathlike(mtlr_block_path))

# ------------------------------
# 4) Normalize feature-level tables
# ------------------------------
# Linear / Cox are coefficient-based
linear_norm = linear_feature_df.copy()
linear_norm["model_id"] = "linear_tuned"
linear_norm["display_name"] = "Linear Discrete-Time Hazard"
linear_norm["family"] = "discrete_time_linear"
linear_norm["explainability_type"] = "coefficient_based"
linear_norm["feature_name"] = linear_norm["feature_name_out"]
linear_norm["importance_primary"] = linear_norm["abs_coefficient"]
linear_norm["importance_secondary"] = linear_norm["coefficient"]
linear_norm["importance_primary_label"] = "abs_coefficient"
linear_norm["importance_secondary_label"] = "signed_coefficient"

cox_norm = cox_feature_df.copy()
cox_norm["model_id"] = "cox_tuned"
cox_norm["display_name"] = "Cox Comparable"
cox_norm["family"] = "continuous_time_cox"
cox_norm["explainability_type"] = "coefficient_based"
cox_norm["feature_name"] = cox_norm["feature_name_out"]
cox_norm["importance_primary"] = cox_norm["abs_coefficient"]
cox_norm["importance_secondary"] = cox_norm["coefficient"]
cox_norm["importance_primary_label"] = "abs_coefficient"
cox_norm["importance_secondary_label"] = "signed_coefficient"

# Neural / DeepSurv are grouped permutation-based
neural_norm = neural_feature_df.copy()
neural_norm["model_id"] = "neural_tuned"
neural_norm["display_name"] = "Neural Discrete-Time Survival"
neural_norm["family"] = "discrete_time_neural"
neural_norm["explainability_type"] = "grouped_permutation"
neural_norm["feature_name"] = neural_norm["feature_name_original"]
neural_norm["importance_primary"] = neural_norm["importance_score_ibs"]
neural_norm["importance_secondary"] = neural_norm["mean_importance_score_auc"]
neural_norm["importance_primary_label"] = "delta_ibs_after_grouped_permutation"
neural_norm["importance_secondary_label"] = "mean_auc_importance_after_grouped_permutation"

deepsurv_norm = deepsurv_feature_df.copy()
deepsurv_norm["model_id"] = "deepsurv_tuned"
deepsurv_norm["display_name"] = "DeepSurv"
deepsurv_norm["family"] = "continuous_time_deepsurv"
deepsurv_norm["explainability_type"] = "grouped_permutation"
deepsurv_norm["feature_name"] = deepsurv_norm["feature_name_original"]
deepsurv_norm["importance_primary"] = deepsurv_norm["importance_score_ibs"]
deepsurv_norm["importance_secondary"] = deepsurv_norm["mean_importance_score_auc"]
deepsurv_norm["importance_primary_label"] = "delta_ibs_after_grouped_permutation"
deepsurv_norm["importance_secondary_label"] = "mean_auc_importance_after_grouped_permutation"

# Poisson (coefficient-based, same schema as Linear/Cox)
poisson_norm = poisson_feature_df.copy()
poisson_norm["model_id"] = "poisson_pexp_tuned"
poisson_norm["display_name"] = "Poisson Piecewise-Exponential"
poisson_norm["family"] = "discrete_time_poisson"
poisson_norm["explainability_type"] = "coefficient_based"
poisson_norm["feature_name"] = poisson_norm["feature_name_out"]
poisson_norm["importance_primary"] = poisson_norm["abs_coefficient"]
poisson_norm["importance_secondary"] = poisson_norm["coefficient"]
poisson_norm["importance_primary_label"] = "abs_coefficient"
poisson_norm["importance_secondary_label"] = "signed_coefficient"

# GB (MDI feature importances)
gb_norm = gb_feature_df.copy()
gb_norm["model_id"] = "gb_weekly_tuned"
gb_norm["display_name"] = "GB Weekly Hazard"
gb_norm["family"] = "discrete_time_gb"
gb_norm["explainability_type"] = "permutation_importance_auc"
gb_norm["feature_name"] = gb_norm["feature_name_out"]
gb_norm["importance_primary"] = gb_norm["importance_mdi"]
gb_norm["importance_secondary"] = gb_norm["importance_mdi"]
gb_norm["importance_primary_label"] = "delta_auc_after_permutation"
gb_norm["importance_secondary_label"] = "delta_auc_after_permutation"

# RSF (grouped permutation, same schema as Neural/DeepSurv)
rsf_norm = rsf_feature_df.copy()
rsf_norm["model_id"] = "rsf_tuned"
rsf_norm["display_name"] = "Random Survival Forest"
rsf_norm["family"] = "continuous_time_tree_ensemble"
rsf_norm["explainability_type"] = "grouped_permutation"
rsf_norm["feature_name"] = rsf_norm["feature_name_original"]
rsf_norm["importance_primary"] = rsf_norm["importance_score_ibs"]
rsf_norm["importance_secondary"] = rsf_norm["mean_importance_score_auc"]
rsf_norm["importance_primary_label"] = "delta_ibs_after_grouped_permutation"
rsf_norm["importance_secondary_label"] = "mean_delta_ibs_approx"

# MTLR (grouped permutation)
mtlr_norm = mtlr_feature_df.copy()
mtlr_norm["model_id"] = "mtlr_tuned"
mtlr_norm["display_name"] = "Neural-MTLR"
mtlr_norm["family"] = "continuous_time_neural_mtlr"
mtlr_norm["explainability_type"] = "grouped_permutation"
mtlr_norm["feature_name"] = mtlr_norm["feature_name_original"]
mtlr_norm["importance_primary"] = mtlr_norm["importance_score_ibs"]
mtlr_norm["importance_secondary"] = mtlr_norm["mean_importance_score_auc"]
mtlr_norm["importance_primary_label"] = "delta_ibs_after_grouped_permutation"
mtlr_norm["importance_secondary_label"] = "mean_delta_ibs_approx"

feature_compare_cols = [
    "model_id", "display_name", "family", "explainability_type",
    "feature_name", "feature_block",
    "importance_primary", "importance_secondary",
    "importance_primary_label", "importance_secondary_label"
]

all_features_df = pd.concat([
    linear_norm[feature_compare_cols],
    neural_norm[feature_compare_cols],
    cox_norm[feature_compare_cols],
    deepsurv_norm[feature_compare_cols],
    poisson_norm[feature_compare_cols],
    gb_norm[feature_compare_cols],
    rsf_norm[feature_compare_cols],
    mtlr_norm[feature_compare_cols],
], ignore_index=True)

# ------------------------------
# 5) Normalize block-level tables
# ------------------------------
linear_block_norm = linear_block_df.copy()
linear_block_norm["model_id"] = "linear_tuned"
linear_block_norm["display_name"] = "Linear Discrete-Time Hazard"
linear_block_norm["family"] = "discrete_time_linear"
linear_block_norm["block_importance_primary"] = linear_block_norm["mean_abs_coefficient"]
linear_block_norm["block_importance_secondary"] = linear_block_norm["mean_coefficient"]
linear_block_norm["block_primary_label"] = "mean_abs_coefficient"
linear_block_norm["block_secondary_label"] = "mean_signed_coefficient"

neural_block_norm = neural_block_df.copy()
neural_block_norm["model_id"] = "neural_tuned"
neural_block_norm["display_name"] = "Neural Discrete-Time Survival"
neural_block_norm["family"] = "discrete_time_neural"
neural_block_norm["block_importance_primary"] = neural_block_norm["mean_importance_score_ibs"]
neural_block_norm["block_importance_secondary"] = neural_block_norm["mean_importance_score_auc"]
neural_block_norm["block_primary_label"] = "mean_delta_ibs_after_grouped_permutation"
neural_block_norm["block_secondary_label"] = "mean_auc_importance_after_grouped_permutation"

cox_block_norm = cox_block_df.copy()
cox_block_norm["model_id"] = "cox_tuned"
cox_block_norm["display_name"] = "Cox Comparable"
cox_block_norm["family"] = "continuous_time_cox"
cox_block_norm["block_importance_primary"] = cox_block_norm["mean_abs_coefficient"]
cox_block_norm["block_importance_secondary"] = cox_block_norm["mean_coefficient"]
cox_block_norm["block_primary_label"] = "mean_abs_coefficient"
cox_block_norm["block_secondary_label"] = "mean_signed_coefficient"

deepsurv_block_norm = deepsurv_block_df.copy()
deepsurv_block_norm["model_id"] = "deepsurv_tuned"
deepsurv_block_norm["display_name"] = "DeepSurv"
deepsurv_block_norm["family"] = "continuous_time_deepsurv"
deepsurv_block_norm["block_importance_primary"] = deepsurv_block_norm["mean_importance_score_ibs"]
deepsurv_block_norm["block_importance_secondary"] = deepsurv_block_norm["mean_importance_score_auc"]
deepsurv_block_norm["block_primary_label"] = "mean_delta_ibs_after_grouped_permutation"
deepsurv_block_norm["block_secondary_label"] = "mean_auc_importance_after_grouped_permutation"

poisson_block_norm = poisson_block_df_g6.copy()
poisson_block_norm["model_id"] = "poisson_pexp_tuned"
poisson_block_norm["display_name"] = "Poisson Piecewise-Exponential"
poisson_block_norm["family"] = "discrete_time_poisson"
poisson_block_norm["block_importance_primary"] = poisson_block_norm["mean_abs_coefficient"]
poisson_block_norm["block_importance_secondary"] = poisson_block_norm["mean_abs_coefficient"]
poisson_block_norm["block_primary_label"] = "mean_abs_log_rate_coefficient"
poisson_block_norm["block_secondary_label"] = "mean_abs_log_rate_coefficient"

gb_block_norm = gb_block_df_g6.copy()
gb_block_norm["model_id"] = "gb_weekly_tuned"
gb_block_norm["display_name"] = "GB Weekly Hazard"
gb_block_norm["family"] = "discrete_time_gb"
gb_block_norm["block_importance_primary"] = gb_block_norm["mean_abs_coefficient"]
gb_block_norm["block_importance_secondary"] = gb_block_norm["mean_abs_coefficient"]
gb_block_norm["block_primary_label"] = "mean_delta_auc_after_permutation"
gb_block_norm["block_secondary_label"] = "mean_delta_auc_after_permutation"

rsf_block_norm = rsf_block_df_g6.copy()
rsf_block_norm["model_id"] = "rsf_tuned"
rsf_block_norm["display_name"] = "Random Survival Forest"
rsf_block_norm["family"] = "continuous_time_tree_ensemble"
rsf_block_norm["block_importance_primary"] = rsf_block_norm["mean_importance_score_ibs"]
rsf_block_norm["block_importance_secondary"] = rsf_block_norm["mean_importance_score_auc"]
rsf_block_norm["block_primary_label"] = "mean_delta_ibs_after_grouped_permutation"
rsf_block_norm["block_secondary_label"] = "mean_delta_ibs_approx"

mtlr_block_norm = mtlr_block_df_g6.copy()
mtlr_block_norm["model_id"] = "mtlr_tuned"
mtlr_block_norm["display_name"] = "Neural-MTLR"
mtlr_block_norm["family"] = "continuous_time_neural_mtlr"
mtlr_block_norm["block_importance_primary"] = mtlr_block_norm["mean_importance_score_ibs"]
mtlr_block_norm["block_importance_secondary"] = mtlr_block_norm["mean_importance_score_auc"]
mtlr_block_norm["block_primary_label"] = "mean_delta_ibs_after_grouped_permutation"
mtlr_block_norm["block_secondary_label"] = "mean_delta_ibs_approx"

block_compare_cols = [
    "model_id", "display_name", "family",
    "feature_block", "n_features",
    "block_importance_primary", "block_importance_secondary",
    "block_primary_label", "block_secondary_label"
]

all_blocks_df = pd.concat([
    linear_block_norm[block_compare_cols],
    neural_block_norm[block_compare_cols],
    cox_block_norm[block_compare_cols],
    deepsurv_block_norm[block_compare_cols],
    poisson_block_norm[block_compare_cols],
    gb_block_norm[block_compare_cols],
    rsf_block_norm[block_compare_cols],
    mtlr_block_norm[block_compare_cols],
], ignore_index=True)

# ------------------------------
# 6) Top drivers by model
# ------------------------------
top_k = 5
top_drivers_df = (
    all_features_df.sort_values(
        by=["model_id", "importance_primary", "importance_secondary"],
        ascending=[True, False, False]
    )
    .groupby("model_id", as_index=False, group_keys=False)
    .head(top_k)
    .reset_index(drop=True)
)

top_drivers_df["driver_rank_within_model"] = (
    top_drivers_df.groupby("model_id").cumcount() + 1
)

# ------------------------------
# 7) Block ranking within model
# ------------------------------
block_rank_df = all_blocks_df.copy()
block_rank_df["block_rank_within_model"] = (
    block_rank_df.groupby("model_id")["block_importance_primary"]
    .rank(method="dense", ascending=False)
)

# wide comparison by block
block_comparison_wide = (
    all_blocks_df.pivot_table(
        index="model_id",
        columns="feature_block",
        values="block_importance_primary",
        aggfunc="first"
    )
    .reset_index()
)

# ------------------------------
# 8) Dominant block summary by model
# ------------------------------
dominant_block_df = (
    all_blocks_df.sort_values(
        by=["model_id", "block_importance_primary"],
        ascending=[True, False]
    )
    .groupby("model_id", as_index=False, group_keys=False)
    .head(1)
    .reset_index(drop=True)
)

dominant_block_df = dominant_block_df[
    ["model_id", "display_name", "family", "feature_block", "block_importance_primary", "block_primary_label"]
].rename(columns={
    "feature_block": "dominant_feature_block",
    "block_importance_primary": "dominant_block_importance_value",
    "block_primary_label": "dominant_block_importance_label",
})

taxonomy_label_map = explainability_taxonomy_df.set_index("block_id")["block_label"].to_dict()
taxonomy_role_map = explainability_taxonomy_df.set_index("block_id")["taxonomy_role"].to_dict()
taxonomy_reason_map = explainability_taxonomy_df.set_index("block_id")["manuscript_reason"].to_dict()

dominant_block_df["dominant_feature_block_label"] = dominant_block_df["dominant_feature_block"].map(taxonomy_label_map)
dominant_block_df["dominant_feature_block_role"] = dominant_block_df["dominant_feature_block"].map(taxonomy_role_map)
dominant_block_df["dominant_feature_block_reason"] = dominant_block_df["dominant_feature_block"].map(taxonomy_reason_map)

# ------------------------------
# 9) Cross-family convergence summary
# ------------------------------
# We create a compact summary of recurring top drivers
top_driver_frequency_df = (
    top_drivers_df.groupby("feature_name", as_index=False)
    .agg(
        n_models_appearing=("model_id", "nunique"),
        models=("model_id", lambda s: ", ".join(sorted(set(s))))
    )
    .sort_values(
        by=["n_models_appearing", "feature_name"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

# recurring drivers only
recurring_top_drivers_df = top_driver_frequency_df[top_driver_frequency_df["n_models_appearing"] >= 2].copy()

# model-level summary
model_summary_rows = []

def safe_top_feature(model_id):
    subset = top_drivers_df[top_drivers_df["model_id"] == model_id].copy()
    if subset.empty:
        return None
    return subset.iloc[0]["feature_name"]

for model_id in sorted(all_features_df["model_id"].unique()):
    disp = all_features_df.loc[all_features_df["model_id"] == model_id, "display_name"].iloc[0]
    fam = all_features_df.loc[all_features_df["model_id"] == model_id, "family"].iloc[0]
    dom_block = dominant_block_df.loc[dominant_block_df["model_id"] == model_id, "dominant_feature_block"].iloc[0]
    top_feat = safe_top_feature(model_id)

    model_summary_rows.append({
        "model_id": model_id,
        "display_name": disp,
        "family": fam,
        "top_driver_feature": top_feat,
        "dominant_feature_block": dom_block,
        "dominant_feature_block_label": taxonomy_label_map.get(dom_block, dom_block),
        "dominant_feature_block_role": taxonomy_role_map.get(dom_block, "unknown"),
    })

explainability_summary_by_model_df = pd.DataFrame(model_summary_rows)

# ------------------------------
# 10) Save outputs
# ------------------------------
all_features_path = TABLES_DIR / "table_explainability_all_features_normalized.csv"
all_blocks_path = TABLES_DIR / "table_explainability_all_blocks_normalized.csv"
top_drivers_path = TABLES_DIR / "table_explainability_top_drivers_by_model.csv"
block_rank_path = TABLES_DIR / "table_explainability_block_rank_within_model.csv"
block_wide_path = TABLES_DIR / "table_explainability_block_comparison_wide.csv"
dominant_block_path = TABLES_DIR / "table_explainability_dominant_block_by_model.csv"
recurring_drivers_path = TABLES_DIR / "table_explainability_recurring_top_drivers.csv"
summary_by_model_path = TABLES_DIR / "table_explainability_summary_by_model.csv"
taxonomy_summary_path = TABLES_DIR / "table_appendix_explainability_block_taxonomy.csv"
config_path = METADATA_DIR / "explainability_consolidated_summary.json"

all_features_df.to_csv(all_features_path, index=False)
materialize_dataframe(con, all_features_df, infer_table_name_from_pathlike(all_features_path), "G6")
all_blocks_df.to_csv(all_blocks_path, index=False)
materialize_dataframe(con, all_blocks_df, infer_table_name_from_pathlike(all_blocks_path), "G6")
top_drivers_df.to_csv(top_drivers_path, index=False)
materialize_dataframe(con, top_drivers_df, infer_table_name_from_pathlike(top_drivers_path), "G6")
block_rank_df.to_csv(block_rank_path, index=False)
materialize_dataframe(con, block_rank_df, infer_table_name_from_pathlike(block_rank_path), "G6")
block_comparison_wide.to_csv(block_wide_path, index=False)
materialize_dataframe(con, block_comparison_wide, infer_table_name_from_pathlike(block_wide_path), "G6")
dominant_block_df.to_csv(dominant_block_path, index=False)
materialize_dataframe(con, dominant_block_df, infer_table_name_from_pathlike(dominant_block_path), "G6")
recurring_top_drivers_df.to_csv(recurring_drivers_path, index=False)
materialize_dataframe(con, recurring_top_drivers_df, infer_table_name_from_pathlike(recurring_drivers_path), "G6")
explainability_summary_by_model_df.to_csv(summary_by_model_path, index=False)
materialize_dataframe(con, explainability_summary_by_model_df, infer_table_name_from_pathlike(summary_by_model_path), "G6")
explainability_taxonomy_df.to_csv(taxonomy_summary_path, index=False)
materialize_dataframe(con, explainability_taxonomy_df, infer_table_name_from_pathlike(taxonomy_summary_path), "G6")

save_json(
    {
        "included_models": sorted(all_features_df["model_id"].unique().tolist()),
        "n_total_normalized_feature_rows": int(all_features_df.shape[0]),
        "n_total_block_rows": int(all_blocks_df.shape[0]),
        "dominant_blocks_by_model": dominant_block_df.set_index("model_id")["dominant_feature_block"].to_dict(),
        "dominant_block_labels_by_model": dominant_block_df.set_index("model_id")["dominant_feature_block_label"].to_dict(),
        "taxonomy_roles_by_block": explainability_taxonomy_df.set_index("block_id")["taxonomy_role"].to_dict(),
        "recurring_top_drivers_count": int(recurring_top_drivers_df.shape[0]),
        "top_driver_by_model": explainability_summary_by_model_df.set_index("model_id")["top_driver_feature"].to_dict(),
    },
    config_path,
)

# ------------------------------
# 11) Output for feedback
# ------------------------------
print("\nExplainability summary by model:")
display(explainability_summary_by_model_df)

print("\nDominant feature block by model:")
display(dominant_block_df)

print("\nExplainability block taxonomy:")
display(explainability_taxonomy_df)

print("\nTop drivers by model:")
display(top_drivers_df)

print("\nRecurring top drivers across families:")
display(recurring_top_drivers_df)

print("\nFeature-block comparison (wide):")
display(block_comparison_wide)

print("\nSaved:")
print("-", all_features_path.resolve())
print("-", all_blocks_path.resolve())
print("-", top_drivers_path.resolve())
print("-", block_rank_path.resolve())
print("-", block_wide_path.resolve())
print("-", dominant_block_path.resolve())
print("-", recurring_drivers_path.resolve())
print("-", summary_by_model_path.resolve())
print("-", taxonomy_summary_path.resolve())
print("-", config_path.resolve())

print(f"[END] G6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G6.CONV — Cross-Paradigm Explainability Convergence (§5.7)
# --------------------------------------------------------------
# Diagnoses whether early-window/temporal-behavioral blocks
# dominate consistently across all 8 model paradigms.
# Writes table_explainability_convergence_paradigm.
# ==============================================================

TEMPORAL_BLOCKS_G = {"dynamic_temporal_behavioral", "early_window_behavior", "discrete_time_index"}

conv_exp = explainability_summary_by_model_df[[
    "model_id", "display_name", "family",
    "dominant_feature_block", "dominant_feature_block_label", "dominant_feature_block_role",
]].copy()

conv_exp["temporal_block_dominates"] = conv_exp["dominant_feature_block"].isin(TEMPORAL_BLOCKS_G)
conv_exp["convergence_verdict"] = conv_exp["temporal_block_dominates"].map(
    {True: "TEMPORAL_DOMINANT", False: "STATIC_DOMINANT"}
)

# Cross-paradigm recurring top drivers
top_recurring_crossparadigm_df = (
    top_driver_frequency_df[top_driver_frequency_df["n_models_appearing"] >= 4].copy()
    .sort_values(["n_models_appearing", "feature_name"], ascending=[False, True])
    .reset_index(drop=True)
)

n_temp_dominant = int(conv_exp["temporal_block_dominates"].sum())
n_total_exp = int(conv_exp.shape[0])

conv_exp_meta = {
    "n_models_in_subset": n_total_exp,
    "n_temporal_dominant": n_temp_dominant,
    "pct_temporal_dominant": round(100.0 * n_temp_dominant / max(n_total_exp, 1), 1),
    "unanimous": n_temp_dominant == n_total_exp,
    "top_recurring_threshold": 4,
    "n_recurring_top_drivers": int(top_recurring_crossparadigm_df.shape[0]),
}

conv_exp_table = "table_explainability_convergence_paradigm"
conv_exp_csv   = TABLES_DIR / "table_explainability_convergence_paradigm.csv"
conv_exp_meta_path = METADATA_DIR / "explainability_convergence_paradigm.json"
conv_top_csv   = TABLES_DIR / "table_explainability_top_recurring_crossparadigm.csv"

conv_exp.to_csv(conv_exp_csv, index=False)
materialize_dataframe(con, conv_exp, conv_exp_table, "G6.CONV")
top_recurring_crossparadigm_df.to_csv(conv_top_csv, index=False)
materialize_dataframe(con, top_recurring_crossparadigm_df, infer_table_name_from_pathlike(conv_top_csv), "G6.CONV")
save_json(conv_exp_meta, conv_exp_meta_path)

print(f"\nG6.CONV — Explainability convergence: {n_temp_dominant}/{n_total_exp} models have temporal-dominant block")
display(conv_exp[["model_id", "family", "dominant_feature_block", "temporal_block_dominates", "convergence_verdict"]])

# %% Cell 62
from datetime import datetime as _dt
print(f"[START] G6.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path
import pandas as pd

summary_table = "table_explainability_summary_by_model"
summary_df = load_duckdb_table_or_raise(summary_table)
if "Top driver" not in summary_df.columns and "top_driver_feature" in summary_df.columns:
    summary_df["Top driver"] = summary_df["top_driver_feature"]
if "Dominant block" not in summary_df.columns and "dominant_feature_block" in summary_df.columns:
    summary_df["Dominant block"] = summary_df["dominant_feature_block"]
if "top_feature" not in summary_df.columns and "top_driver_feature" in summary_df.columns:
    summary_df["top_feature"] = summary_df["top_driver_feature"]
if "dominant_block" not in summary_df.columns and "dominant_feature_block" in summary_df.columns:
    summary_df["dominant_block"] = summary_df["dominant_feature_block"]
summary_path = Path("outputs_benchmark_survival/tables/table_explainability_summary_by_model.csv")
summary_df.to_csv(summary_path, index=False)
materialize_dataframe(con, summary_df, infer_table_name_from_pathlike(summary_path), "G6.1")
materialize_dataframe(con, summary_df, summary_table, "G6.1")
print("Explainability summary aliases materialized for G7:")
print("-", summary_path.resolve())

all_blocks_table = "table_explainability_all_blocks_normalized"
all_blocks_df = load_duckdb_table_or_raise(all_blocks_table)
if "Normalized importance" not in all_blocks_df.columns and "block_importance_primary" in all_blocks_df.columns:
    all_blocks_df["Normalized importance"] = all_blocks_df["block_importance_primary"]
if "normalized_importance" not in all_blocks_df.columns and "block_importance_primary" in all_blocks_df.columns:
    all_blocks_df["normalized_importance"] = all_blocks_df["block_importance_primary"]
if "normalized_value" not in all_blocks_df.columns and "block_importance_primary" in all_blocks_df.columns:
    all_blocks_df["normalized_value"] = all_blocks_df["block_importance_primary"]
if "value" not in all_blocks_df.columns and "block_importance_primary" in all_blocks_df.columns:
    all_blocks_df["value"] = all_blocks_df["block_importance_primary"]
all_blocks_path = Path("outputs_benchmark_survival/tables/table_explainability_all_blocks_normalized.csv")
all_blocks_df.to_csv(all_blocks_path, index=False)
materialize_dataframe(con, all_blocks_df, infer_table_name_from_pathlike(all_blocks_path), "G6.1")
materialize_dataframe(con, all_blocks_df, all_blocks_table, "G6.1")
print("-", all_blocks_path.resolve())

print(f"[END] G6.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 64
from datetime import datetime as _dt
print(f"[START] G7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path
import json
import re
import shutil

import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

print("=" * 70)
print("G7 — Freeze Curated Paper Artifacts")
print("=" * 70)
print("Methodological note: this step freezes TeX-facing assets using the manuscript as the source-of-truth contract.")
print()

OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", "outputs_benchmark_survival"))
TABLES_DIR = Path(globals().get("TABLES_DIR", OUTPUT_DIR / "tables"))
FIGURES_DIR = Path(globals().get("FIGURES_DIR", OUTPUT_DIR / "figures"))
METADATA_DIR = Path(globals().get("METADATA_DIR", OUTPUT_DIR / "metadata"))
PAPER_MAIN_DIR = OUTPUT_DIR / "paper_main"
PAPER_APPENDIX_DIR = OUTPUT_DIR / "paper_appendix"
TEX_PATH = MANUSCRIPT_TEX_PATH if MANUSCRIPT_TEX_PATH.exists() else Path("dropout_benchmark_v2.tex")

for target_dir in [PAPER_MAIN_DIR, PAPER_APPENDIX_DIR, METADATA_DIR]:
    target_dir.mkdir(parents=True, exist_ok=True)

for target_dir in [PAPER_MAIN_DIR, PAPER_APPENDIX_DIR]:
    for child in target_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

print(f"[END] G7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 66
from datetime import datetime as _dt
print(f"[START] G7.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def resolve_first_existing(candidates):
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return None

def load_table_duckdb_first(source_candidates=None, preferred_table_names=None):
    candidate_paths = [Path(candidate) for candidate in (source_candidates or [])]
    table_candidates = [str(table_name) for table_name in (preferred_table_names or []) if table_name]
    table_candidates.extend(infer_table_name_from_pathlike(candidate) for candidate in candidate_paths)

    seen_tables = set()
    for table_name in table_candidates:
        if table_name in seen_tables:
            continue
        seen_tables.add(table_name)
        loaded_df = load_duckdb_table_optional(table_name)
        if loaded_df is not None:
            return loaded_df, table_name, "duckdb"

    source_path = resolve_first_existing(candidate_paths)
    if source_path is not None:
        return pd.read_csv(source_path), source_path, "csv"

    return None, None, None

print(f"[END] G7.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 68
from datetime import datetime as _dt
print(f"[START] G7.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def checked_candidates_note(candidates):
    return "Checked: " + " | ".join(str(Path(candidate)) for candidate in candidates)

print(f"[END] G7.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 70
from datetime import datetime as _dt
print(f"[START] G7.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def export_table_from_df(dataframe, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    materialize_dataframe(con, dataframe, infer_table_name_from_pathlike(output_path), "G6")
    return output_path

print(f"[END] G7.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 72
from datetime import datetime as _dt
print(f"[START] G7.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def normalize_model_names(model_name):
    # Strip " (Tuned)" suffix before any lookup
    raw = str(model_name).replace(" (Tuned)", "").strip()
    mapping = {
        "linear_discrete_time_hazard": "Linear Discrete-Time Hazard",
        "neural_discrete_time_hazard": "Neural Discrete-Time Survival",
        "poisson_piecewise_exponential": "Poisson Piecewise-Exponential",
        "gb_weekly_hazard_unweighted": "Gradient-Boosted Weekly Hazard",
        "catboost_weekly_hazard": "CatBoost Weekly Hazard",
        "cox_comparable": "Cox Comparable",
        "deepsurv": "DeepSurv",
        "random_survival_forest": "Random Survival Forest",
        "gradient_boosted_cox": "Gradient-Boosted Cox",
        "weibull_aft": "Weibull AFT",
        "royston_parmar": "Royston-Parmar",
        "xgboost_aft": "XGBoost AFT",
        "neural_mtlr": "Neural-MTLR",
        "deephit": "DeepHit",
        "Linear Discrete-Time": "Linear Discrete-Time Hazard",
        "Neural Discrete-Time": "Neural Discrete-Time Survival",
        "Cloglog Discrete-Time": "Cloglog Discrete-Time Hazard",
        "Cox": "Cox Comparable",
        # Aliases already without Tuned (from previous sed pass)
        "RSF": "Random Survival Forest",
        "GB-Cox": "Gradient-Boosted Cox",
        "GB Weekly Hazard": "Gradient-Boosted Weekly Hazard",
    }
    return mapping.get(raw, raw)

print(f"[END] G7.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 74
from datetime import datetime as _dt
print(f"[START] G7.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def normalize_family_names(family_name):
    mapping = {
        "dynamic_weekly": "discrete_time_dynamic",
        "dynamic_neural": "discrete_time_dynamic",
        "comparable_continuous_time": "continuous_time_comparable",
        "comparable_tree_survival": "continuous_time_comparable",
        "comparable_parametric": "continuous_time_comparable",
        "comparable_neural": "continuous_time_comparable",
        "contract": "contract_stage",
        "discrete_time_linear": "discrete_time_dynamic",
        "discrete_time_cloglog": "discrete_time_dynamic",
        "discrete_time_neural": "discrete_time_dynamic",
        "discrete_time_boosted": "discrete_time_dynamic",
        "continuous_time_cox": "continuous_time_comparable",
        "continuous_time_deepsurv": "continuous_time_comparable",
        "continuous_time_neural": "continuous_time_comparable",
        "continuous_time_tree_ensemble": "continuous_time_comparable",
        "continuous_time_boosted": "continuous_time_comparable",
    }
    normalized_value = str(family_name).replace(" ", "_")
    return mapping.get(normalized_value, normalized_value)

print(f"[END] G7.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 76
from datetime import datetime as _dt
print(f"[START] G7.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def model_sort_key(model_name):
    order = {
        "Linear Discrete-Time Hazard": 1,
        "Neural Discrete-Time Survival": 2,
        "Poisson Piecewise-Exponential": 3,
        "Gradient-Boosted Weekly Hazard": 4,
        "CatBoost Weekly Hazard": 5,
        "Cox Comparable": 6,
        "DeepSurv": 7,
        "Random Survival Forest": 8,
        "Gradient-Boosted Cox": 9,
        "Weibull AFT": 10,
        "Royston-Parmar": 11,
        "XGBoost AFT": 12,
        "Neural-MTLR": 13,
        "DeepHit": 14,
    }
    return order.get(str(model_name), 99)

print(f"[END] G7.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 78
from datetime import datetime as _dt
print(f"[START] G7.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def pick_column(dataframe, candidates, required=True):
    normalized_lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        lookup_key = str(candidate).strip().lower()
        if lookup_key in normalized_lookup:
            return normalized_lookup[lookup_key]
    if required:
        raise KeyError(f"Could not find any of the candidate columns: {candidates}")
    return None

print(f"[END] G7.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 80
from datetime import datetime as _dt
print(f"[START] G7.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def append_asset_row(container, tex_label, tex_caption, scope, artifact_type, status, file_name, output_path, source_path, notes):
    container.append(
        {
            "tex_label": tex_label,
            "tex_caption": tex_caption,
            "scope": scope,
            "artifact_type": artifact_type,
            "status": status,
            "file_name": file_name,
            "output_path": str(output_path),
            "source_path": "" if source_path is None else str(source_path),
            "notes": notes,
        }
    )

print(f"[END] G7.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 82
from datetime import datetime as _dt
print(f"[START] G7.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def append_supporting_row(container, related_tex_label, scope, artifact_type, status, file_name, output_path, source_path, notes):
    container.append(
        {
            "related_tex_label": related_tex_label,
            "scope": scope,
            "artifact_type": artifact_type,
            "status": status,
            "file_name": file_name,
            "output_path": str(output_path),
            "source_path": "" if source_path is None else str(source_path),
            "notes": notes,
        }
    )

print(f"[END] G7.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 84
from datetime import datetime as _dt
print(f"[START] G7.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def export_direct_table_asset(tex_label, output_path, source_candidates=None, dataframe=None, note=None, missing_note=None):
    spec = tex_contract[tex_label]
    source_candidates = [Path(candidate) for candidate in (source_candidates or [])]
    if dataframe is not None:
        _, source_ref, source_kind = load_table_duckdb_first(source_candidates)
        source_provenance = DUCKDB_PATH if source_kind == "duckdb" else source_ref
        export_table_from_df(dataframe, output_path)
        append_asset_row(
            tex_asset_rows,
            tex_label=tex_label,
            tex_caption=spec["caption"],
            scope=spec["scope"],
            artifact_type="table",
            status="exported",
            file_name=Path(output_path).name,
            output_path=output_path,
            source_path=TEX_PATH if source_provenance is None else source_provenance,
            notes=note or "Exported from a curated dataframe.",
        )
        return dataframe
    exported_df, source_ref, source_kind = load_table_duckdb_first(source_candidates)
    if exported_df is not None:
        source_provenance = DUCKDB_PATH if source_kind == "duckdb" else source_ref
        export_table_from_df(exported_df, output_path)
        append_asset_row(
            tex_asset_rows,
            tex_label=tex_label,
            tex_caption=spec["caption"],
            scope=spec["scope"],
            artifact_type="table",
            status="exported",
            file_name=Path(output_path).name,
            output_path=output_path,
            source_path=source_provenance,
            notes=note or (f"Loaded from DuckDB table {source_ref}." if source_kind == "duckdb" else "Copied from an existing notebook table artifact."),
        )
        return exported_df
    append_asset_row(
        tex_asset_rows,
        tex_label=tex_label,
        tex_caption=spec["caption"],
        scope=spec["scope"],
        artifact_type="table",
        status="missing",
        file_name=Path(output_path).name,
        output_path=output_path,
        source_path=None,
        notes=missing_note or checked_candidates_note(source_candidates),
    )
    return None

print(f"[END] G7.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 86
from datetime import datetime as _dt
print(f"[START] G7.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def export_supporting_table_asset(related_tex_label, output_path, source_candidates=None, dataframe=None, note=None, missing_note=None):
    scope = tex_contract[related_tex_label]["scope"]
    source_candidates = [Path(candidate) for candidate in (source_candidates or [])]
    if dataframe is not None:
        _, source_ref, source_kind = load_table_duckdb_first(source_candidates)
        source_provenance = DUCKDB_PATH if source_kind == "duckdb" else source_ref
        export_table_from_df(dataframe, output_path)
        append_supporting_row(
            supporting_asset_rows,
            related_tex_label=related_tex_label,
            scope=scope,
            artifact_type="support_table",
            status="exported",
            file_name=Path(output_path).name,
            output_path=output_path,
            source_path=TEX_PATH if source_provenance is None else source_provenance,
            notes=note or "Exported from a curated dataframe.",
        )
        return dataframe
    exported_df, source_ref, source_kind = load_table_duckdb_first(source_candidates)
    if exported_df is not None:
        source_provenance = DUCKDB_PATH if source_kind == "duckdb" else source_ref
        export_table_from_df(exported_df, output_path)
        append_supporting_row(
            supporting_asset_rows,
            related_tex_label=related_tex_label,
            scope=scope,
            artifact_type="support_table",
            status="exported",
            file_name=Path(output_path).name,
            output_path=output_path,
            source_path=source_provenance,
            notes=note or (f"Loaded from DuckDB table {source_ref}." if source_kind == "duckdb" else "Copied from an existing notebook support artifact."),
        )
        return exported_df
    append_supporting_row(
        supporting_asset_rows,
        related_tex_label=related_tex_label,
        scope=scope,
        artifact_type="support_table",
        status="missing",
        file_name=Path(output_path).name,
        output_path=output_path,
        source_path=None,
        notes=missing_note or checked_candidates_note(source_candidates),
    )
    return None

print(f"[END] G7.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 88
from datetime import datetime as _dt
print(f"[START] G7.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def export_direct_figure_asset(tex_label, output_path, source_candidates=None, generator=None, note=None, missing_note=None):
    spec = tex_contract[tex_label]
    source_candidates = [Path(candidate) for candidate in (source_candidates or [])]
    source_path = resolve_first_existing(source_candidates)
    if source_path is not None:
        shutil.copy2(source_path, output_path)
        append_asset_row(
            tex_asset_rows,
            tex_label=tex_label,
            tex_caption=spec["caption"],
            scope=spec["scope"],
            artifact_type="figure",
            status="exported",
            file_name=Path(output_path).name,
            output_path=output_path,
            source_path=source_path,
            notes=note or "Copied from an existing notebook figure artifact.",
        )
        return True
    if generator is not None:
        generated = bool(generator(output_path))
        if generated and Path(output_path).exists():
            append_asset_row(
                tex_asset_rows,
                tex_label=tex_label,
                tex_caption=spec["caption"],
                scope=spec["scope"],
                artifact_type="figure",
                status="exported",
                file_name=Path(output_path).name,
                output_path=output_path,
                source_path=TEX_PATH,
                notes=note or "Generated inside the paper freeze layer.",
            )
            return True
    append_asset_row(
        tex_asset_rows,
        tex_label=tex_label,
        tex_caption=spec["caption"],
        scope=spec["scope"],
        artifact_type="figure",
        status="missing",
        file_name=Path(output_path).name,
        output_path=output_path,
        source_path=None,
        notes=missing_note or checked_candidates_note(source_candidates),
    )
    return False

print(f"[END] G7.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 90
from datetime import datetime as _dt
print(f"[START] G7.13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_main_benchmark_table():
    rows = [
        {"Model": "Linear Discrete-Time Hazard", "Family": "discrete_time_dynamic", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Neural Discrete-Time Survival", "Family": "discrete_time_dynamic", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Poisson Piecewise-Exponential", "Family": "discrete_time_dynamic", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Gradient-Boosted Weekly Hazard", "Family": "discrete_time_dynamic", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "CatBoost Weekly Hazard", "Family": "discrete_time_dynamic", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Cox Comparable", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "DeepSurv", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Random Survival Forest", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Gradient-Boosted Cox", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Weibull AFT", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Royston-Parmar", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "XGBoost AFT", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "Neural-MTLR", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
        {"Model": "DeepHit", "Family": "continuous_time_comparable", "IBS": float("nan"), "TD Concordance": float("nan"), "Brier@10": float("nan"), "Brier@20": float("nan"), "Brier@30": float("nan")},
    ]
    return pd.DataFrame(rows)


EXPANDED_BENCHMARK_MODEL_SPECS = [
    {"model_key": "linear_discrete_time_hazard", "display_name": "Linear Discrete-Time Hazard", "family_group": "dynamic_weekly", "brier_table": "table_linear_tuned_brier_by_horizon", "model_type": "linear_discrete_time_hazard"},
    {"model_key": "neural_discrete_time_hazard", "display_name": "Neural Discrete-Time Survival", "family_group": "dynamic_neural", "brier_table": "table_neural_tuned_brier_by_horizon", "model_type": "neural_discrete_time_hazard"},
    {"model_key": "poisson_piecewise_exponential", "display_name": "Poisson Piecewise-Exponential", "family_group": "dynamic_weekly", "brier_table": "table_poisson_pexp_tuned_brier_by_horizon", "model_type": "poisson_piecewise_exponential"},
    {"model_key": "gb_weekly_hazard_unweighted", "display_name": "Gradient-Boosted Weekly Hazard", "family_group": "dynamic_weekly", "brier_table": "table_gb_weekly_hazard_unweighted_tuned_brier_by_horizon", "model_type": "gradient_boosted_weekly_hazard"},
    {"model_key": "catboost_weekly_hazard", "display_name": "CatBoost Weekly Hazard", "family_group": "dynamic_weekly", "brier_table": "table_catboost_weekly_hazard_tuned_brier_by_horizon", "model_type": "catboost_weekly_hazard"},
    {"model_key": "cox_comparable", "display_name": "Cox Comparable", "family_group": "comparable_continuous_time", "brier_table": "table_cox_tuned_brier_by_horizon", "model_type": "cox_comparable"},
    {"model_key": "deepsurv", "display_name": "DeepSurv", "family_group": "comparable_neural", "brier_table": "table_deepsurv_tuned_brier_by_horizon", "model_type": "deepsurv"},
    {"model_key": "random_survival_forest", "display_name": "Random Survival Forest", "family_group": "comparable_tree_survival", "brier_table": "table_rsf_tuned_brier_by_horizon", "model_type": "random_survival_forest"},
    {"model_key": "gradient_boosted_cox", "display_name": "Gradient-Boosted Cox", "family_group": "comparable_tree_survival", "brier_table": "table_gb_cox_tuned_brier_by_horizon", "model_type": "gradient_boosted_cox"},
    {"model_key": "weibull_aft", "display_name": "Weibull AFT", "family_group": "comparable_parametric", "brier_table": "table_weibull_aft_tuned_brier_by_horizon", "model_type": "weibull_aft"},
    {"model_key": "royston_parmar", "display_name": "Royston-Parmar", "family_group": "comparable_parametric", "brier_table": "table_royston_parmar_tuned_brier_by_horizon", "model_type": "royston_parmar"},
    {"model_key": "xgboost_aft", "display_name": "XGBoost AFT", "family_group": "comparable_tree_survival", "brier_table": "table_xgb_aft_tuned_brier_by_horizon", "model_type": "xgboost_aft"},
    {"model_key": "neural_mtlr", "display_name": "Neural-MTLR", "family_group": "comparable_neural", "brier_table": "table_mtlr_tuned_brier_by_horizon", "model_type": "neural_mtlr"},
    {"model_key": "deephit", "display_name": "DeepHit", "family_group": "comparable_neural", "brier_table": "table_deephit_tuned_brier_by_horizon", "model_type": "deephit"},
]

EXPANDED_BENCHMARK_MODEL_SPEC_BY_KEY = {
    spec["model_key"]: spec for spec in EXPANDED_BENCHMARK_MODEL_SPECS
}


def load_model_brier_values(table_name):
    brier_df = load_duckdb_table_optional(table_name)
    if brier_df is None or brier_df.empty:
        return {}
    horizon_col = pick_column(brier_df, ["horizon_week", "Horizon", "horizon"])
    value_col = pick_column(brier_df, ["metric_value", "value", "brier_value"])
    metric_col = pick_column(brier_df, ["metric_name", "metric"], required=False)
    working_df = brier_df.copy()
    if metric_col is not None:
        working_df = working_df[
            working_df[metric_col].astype(str).str.lower().eq("brier_at_horizon")
        ].copy()
    return {
        int(horizon): float(metric_value)
        for horizon, metric_value in working_df[[horizon_col, value_col]].itertuples(index=False)
    }


def build_expanded_benchmark_tables_from_stage_catalog():
    primary_df = load_duckdb_table_optional("table_5_16_model_primary_summary")
    if primary_df is None or primary_df.empty:
        return None, None

    model_col = pick_column(primary_df, ["model_name", "model"])
    family_col = pick_column(primary_df, ["family_group", "family"])
    ibs_col = pick_column(primary_df, ["ibs", "IBS", "integrated_brier_score"])
    cindex_col = pick_column(primary_df, ["c_index", "TD Concordance", "td_concordance", "concordance_td"])

    primary_metrics_df = primary_df[[model_col, family_col, ibs_col, cindex_col]].copy()
    primary_metrics_df.columns = ["model_key", "family_group_live", "IBS", "TD Concordance"]

    spec_df = pd.DataFrame(EXPANDED_BENCHMARK_MODEL_SPECS)
    benchmark_df = spec_df.merge(primary_metrics_df, on="model_key", how="left")
    benchmark_df["family_group_live"] = benchmark_df["family_group_live"].fillna(benchmark_df["family_group"])
    benchmark_df["Model"] = benchmark_df["display_name"]
    benchmark_df["Family"] = benchmark_df["family_group_live"].map(normalize_family_names)

    brier_cache = {
        spec["model_key"]: load_model_brier_values(spec["brier_table"])
        for spec in EXPANDED_BENCHMARK_MODEL_SPECS
    }
    for horizon in [10, 20, 30]:
        benchmark_df[f"Brier@{horizon}"] = benchmark_df["model_key"].map(
            lambda model_key: brier_cache.get(model_key, {}).get(horizon, float("nan"))
        )

    benchmark_df = benchmark_df[
        ["Model", "Family", "IBS", "TD Concordance", "Brier@10", "Brier@20", "Brier@30"]
    ].copy()
    benchmark_df = benchmark_df.sort_values(
        by="Model", key=lambda series: series.map(model_sort_key)
    ).reset_index(drop=True)
    benchmark_figure_df = benchmark_df[["Model", "Family", "IBS", "TD Concordance"]].copy()
    return benchmark_df, benchmark_figure_df


def build_benchmark_membership_from_stage_inventory():
    inventory_df = load_duckdb_table_optional("table_5_16_model_inventory")
    if inventory_df is None or inventory_df.empty:
        return None

    working_df = inventory_df.sort_values("stage_order", kind="mergesort").copy()
    working_df["display_name"] = working_df["model_name"].map(normalize_model_names)
    working_df["family_name"] = working_df["family_group"].map(normalize_family_names)
    working_df["model_family"] = working_df["family_group"]
    working_df["input_representation"] = working_df["family_group"].map(
        lambda family_group: "runtime_contract_tables"
        if str(family_group) == "contract"
        else "weekly_person_period"
        if str(family_group) in {"dynamic_weekly", "dynamic_neural"}
        else "early_window_enrollment_summary"
    )
    working_df["training_contract"] = working_df["family_group"].map(
        lambda family_group: "runtime_contract_materialization"
        if str(family_group) == "contract"
        else "dynamic_weekly_person_period"
        if str(family_group) in {"dynamic_weekly", "dynamic_neural"}
        else "static_after_early_window"
    )
    working_df["model_type"] = working_df["model_name"].map(
        lambda model_name: EXPANDED_BENCHMARK_MODEL_SPEC_BY_KEY.get(str(model_name), {}).get("model_type", "contract_stage")
    )
    working_df["comparability_rule"] = working_df["family_group"].map(
        lambda family_group: "Not part of predictive comparability; contract stage only."
        if str(family_group) == "contract"
        else "Comparable within the dynamic weekly paper arm; cross-arm interpretation must remain harmonized rather than architecture-only."
        if str(family_group) in {"dynamic_weekly", "dynamic_neural"}
        else "Comparable within the early-window continuous-time paper arm; cross-arm interpretation must remain harmonized rather than architecture-only."
    )
    working_df["execution_status"] = working_df["artifact_status"]
    return working_df[
        [
            "stage_id",
            "stage_order",
            "model_name",
            "display_name",
            "family_name",
            "model_family",
            "input_representation",
            "training_contract",
            "model_type",
            "comparability_rule",
            "execution_status",
        ]
    ].copy()


def build_manuscript_explainability_fallback_table():
    fallback_df = pd.DataFrame(MANUSCRIPT_EXPLAINABILITY_FALLBACK_ROWS)
    return fallback_df.sort_values(
        by="Model",
        key=lambda series: series.map(model_sort_key),
    ).reset_index(drop=True)


def normalize_manuscript_explainability_summary(explainability_df: pd.DataFrame | None):
    if explainability_df is None or explainability_df.empty:
        return build_manuscript_explainability_fallback_table(), "fallback"

    working_df = explainability_df.copy()
    model_col = pick_column(working_df, ["display_name", "Model", "model", "model_name"])
    family_col = pick_column(working_df, ["family", "Family", "model_family"])
    top_driver_col = pick_column(working_df, ["Top driver", "top_driver", "top_feature", "top_driver_feature"])
    dominant_block_col = pick_column(working_df, ["Dominant block", "dominant_block", "top_block", "dominant_feature_block"])

    normalized_df = working_df[[model_col, family_col, top_driver_col, dominant_block_col]].copy()
    normalized_df.columns = ["Model", "Family", "Top driver", "Dominant block"]
    normalized_df["Model"] = normalized_df["Model"].map(normalize_model_names)
    normalized_df["Family"] = normalized_df["Family"].map(normalize_family_names)
    normalized_df = normalized_df[normalized_df["Model"].isin(MANUSCRIPT_EXPLAINABILITY_MODEL_ORDER)].copy()

    if normalized_df.empty:
        return build_manuscript_explainability_fallback_table(), "fallback"

    normalized_df = normalized_df.sort_values(
        by="Model",
        key=lambda series: series.map(model_sort_key),
    ).reset_index(drop=True)
    return normalized_df, "materialized"

print(f"[END] G7.13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 92
from datetime import datetime as _dt
print(f"[START] G7.14 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_ablation_table():
    # Try to load from runtime-persisted convergence table first
    try:
        conv_path = TABLES_DIR / "table_ablation_convergence_paradigm.csv"
        if conv_path.exists():
            import pandas as _pd
            conv_df = _pd.read_csv(conv_path)
            rows = []
            for _, r in conv_df.iterrows():
                rows.append({
                    "Model": r["display_name"],
                    "Family": r["family"],
                    "Delta IBS static": r.get("delta_ibs_drop_static", float("nan")),
                    "Delta IBS temporal": r.get("delta_ibs_drop_temporal", float("nan")),
                    "Delta TD concordance static": r.get("delta_c_index_drop_static", float("nan")),
                    "Delta TD concordance temporal": r.get("delta_c_index_drop_temporal", float("nan")),
                    "IBS ratio": r.get("ibs_temporal_vs_static_ratio", float("nan")),
                })
            return _pd.DataFrame(rows)
    except Exception:
        pass
    # Fallback: hardcoded values from last known run (pending re-run for dynamic arm)
    return pd.DataFrame(
        [
            {"Model": "Poisson PPE",                   "Family": "discrete_time_poisson",           "Delta IBS static": float("nan"), "Delta IBS temporal": float("nan"), "Delta TD concordance static": float("nan"), "Delta TD concordance temporal": float("nan"), "IBS ratio": float("nan")},
            {"Model": "Linear Discrete-Time Hazard",    "Family": "discrete_time_linear",            "Delta IBS static": 0.0040,        "Delta IBS temporal": 0.0149,        "Delta TD concordance static": -0.0346,        "Delta TD concordance temporal": -0.0997,        "IBS ratio": 3.7220},
            {"Model": "GB Weekly Hazard",               "Family": "discrete_time_gb",                "Delta IBS static": float("nan"), "Delta IBS temporal": float("nan"), "Delta TD concordance static": float("nan"), "Delta TD concordance temporal": float("nan"), "IBS ratio": float("nan")},
            {"Model": "Neural Discrete-Time Survival",  "Family": "discrete_time_neural",            "Delta IBS static": 0.0038,        "Delta IBS temporal": 0.0187,        "Delta TD concordance static": -0.0330,        "Delta TD concordance temporal": -0.0944,        "IBS ratio": 4.8539},
            {"Model": "Cox Comparable",                 "Family": "continuous_time_cox",             "Delta IBS static": 0.0069,        "Delta IBS temporal": 0.0282,        "Delta TD concordance static": -0.0211,        "Delta TD concordance temporal": -0.1121,        "IBS ratio": 4.0930},
            {"Model": "DeepSurv",                       "Family": "continuous_time_deepsurv",        "Delta IBS static": 0.0106,        "Delta IBS temporal": 0.0323,        "Delta TD concordance static": -0.0284,        "Delta TD concordance temporal": -0.1159,        "IBS ratio": 3.0540},
            {"Model": "Random Survival Forest",         "Family": "continuous_time_tree_ensemble",   "Delta IBS static": float("nan"), "Delta IBS temporal": float("nan"), "Delta TD concordance static": float("nan"), "Delta TD concordance temporal": float("nan"), "IBS ratio": float("nan")},
            {"Model": "Neural-MTLR",                    "Family": "continuous_time_neural_mtlr",     "Delta IBS static": float("nan"), "Delta IBS temporal": float("nan"), "Delta TD concordance static": float("nan"), "Delta TD concordance temporal": float("nan"), "IBS ratio": float("nan")},
        ]
    )

print(f"[END] G7.14 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 94
from datetime import datetime as _dt
print(f"[START] G7.15 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_explainability_summary_table():
    return build_manuscript_explainability_fallback_table()

print(f"[END] G7.15 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 96
from datetime import datetime as _dt
print(f"[START] G7.16 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_calibration_table():
    return pd.DataFrame(
        [
            {"Model": "Cox Comparable", "Family": "continuous_time_cox", "Calib@10": 0.0433, "Calib@20": 0.0268, "Calib@30": 0.0328},
            {"Model": "DeepSurv", "Family": "continuous_time_deepsurv", "Calib@10": 0.0304, "Calib@20": 0.0160, "Calib@30": 0.0295},
            {"Model": "Linear Discrete-Time Hazard", "Family": "discrete_time_linear", "Calib@10": 0.0790, "Calib@20": 0.1316, "Calib@30": 0.1542},
            {"Model": "Neural Discrete-Time Survival", "Family": "discrete_time_neural", "Calib@10": 0.0647, "Calib@20": 0.1056, "Calib@30": 0.1367},
        ]
    )

print(f"[END] G7.16 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 98
from datetime import datetime as _dt
print(f"[START] G7.17 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_protocol_audit_table():
    return pd.DataFrame(
        [
            {"Component": "Evaluation unit", "Value": "enrollment", "Details": "All final benchmark comparisons are performed at the enrollment level."},
            {"Component": "Shared horizons", "Value": "10, 20, 30", "Details": "Common benchmark horizons used for Brier and calibration summaries."},
            {"Component": "Brier / IBS censoring", "Value": "ipcw_km / pycox", "Details": "Brier score and IBS are computed with inverse-probability-of-censoring weighting using the Kaplan-Meier estimator for the censoring distribution through pycox."},
            {"Component": "Primary concordance", "Value": "TD concordance", "Details": "The benchmark co-primary discrimination metric is time-dependent concordance as returned by EvalSurv.concordance_td()."},
            {"Component": "Discrete-time prediction rule", "Value": "dynamic_weekly_updated", "Details": "For prediction at week t, the discrete-time families use only information observed up to week t, and enrollment-level survival is reconstructed from accumulated weekly hazards."},
            {"Component": "Continuous-time prediction rule", "Value": "static_baseline_from_early_window", "Details": "The continuous-time comparable families generate survival curves from fixed enrollment-level representations built from the early observation window only."},
            {"Component": "Identity leakage result", "Value": "enrollment level: none", "Details": "No enrollment identity leakage was detected between train and test."},
            {"Component": "Contextual split scope", "Value": "shared curricular context", "Details": "All modules, presentations, and module-presentation combinations appeared in both splits; the benchmark is therefore not context-disjoint."},
            {"Component": "Calibration metric", "Value": "weighted_mean_absolute_gap_across_bins", "Details": "At each horizon, predicted event risk is grouped into quantile-based bins and calibration error is summarized as the sample-size-weighted mean absolute gap between mean predicted risk and empirical event rate across bins."},
        ]
    )

print(f"[END] G7.17 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 100
from datetime import datetime as _dt
print(f"[START] G7.18 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_preproc_tuning_table():
    return pd.DataFrame(
        [
            {"Model": "Linear Discrete-Time Hazard", "Input level": "person-period weekly", "Preprocessing": "Imputation: median for numeric variables; constant-missing category for categorical variables. Encoding and scaling: one-hot encoding plus standard scaling. Imbalance handling: none (class_weight=None).", "Validation and tuning": "Enrollment-level GroupShuffleSplit (20%). 8 candidates. Selection by val_log_loss. No early stopping."},
            {"Model": "Neural Discrete-Time Survival", "Input level": "person-period weekly", "Preprocessing": "Imputation: median for numeric variables; constant-missing category for categorical variables. Encoding and scaling: one-hot encoding plus standard scaling. Imbalance handling: none (unweighted BCE loss).", "Validation and tuning": "Distinct-enrollment train/validation split (10%). 16 candidates. Selection by lowest validation loss. Early stopping used."},
            {"Model": "Cox Comparable", "Input level": "enrollment early window", "Preprocessing": "Imputation: median for numeric variables; constant-missing category for categorical variables. Encoding and scaling: one-hot encoding plus standard scaling. Imbalance handling: none.", "Validation and tuning": "Enrollment-level split with event stratification when possible (20%). 12 candidates. Selection by negative partial log-likelihood. No early stopping."},
            {"Model": "DeepSurv", "Input level": "enrollment early window", "Preprocessing": "Imputation: median for numeric variables; constant-missing category for categorical variables. Encoding and scaling: one-hot encoding plus standard scaling. Imbalance handling: none.", "Validation and tuning": "Internal validation fraction on training rows (20%). 24 candidates. Selection by best validation loss. Early stopping used."},
        ]
    )

print(f"[END] G7.18 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 102
from datetime import datetime as _dt
print(f"[START] G7.19 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_bootstrap_table():
    return pd.DataFrame(
        [
            {"Model": "DeepSurv", "IBS [95% CI]": "0.1110 [0.1080, 0.1160]", "Time-dependent concordance [95% CI]": "0.7300 [0.7200, 0.7410]"},
            {"Model": "Cox Comparable", "IBS [95% CI]": "0.1160 [0.1120, 0.1200]", "Time-dependent concordance [95% CI]": "0.7220 [0.7120, 0.7300]"},
            {"Model": "Neural Discrete-Time Survival", "IBS [95% CI]": "0.1560 [0.1510, 0.1610]", "Time-dependent concordance [95% CI]": "0.6760 [0.6650, 0.6860]"},
            {"Model": "Linear Discrete-Time Hazard", "IBS [95% CI]": "0.1620 [0.1570, 0.1680]", "Time-dependent concordance [95% CI]": "0.6580 [0.6490, 0.6680]"},
        ]
    )

print(f"[END] G7.19 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 104
from datetime import datetime as _dt
print(f"[START] G7.20 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_split_context_table():
    return pd.DataFrame(
        [
            {
                "Split unit": "enrollment",
                "Stratification": "event status + coarse event-time bucket",
                "Total": 32593,
                "Train": 22815,
                "Test": 9778,
                "Train event rate": 0.2266,
                "Test event rate": 0.2266,
                "Identity leakage": "no",
                "Shared modules": "7/7",
                "Shared presentations": "4/4",
                "Shared module-presentations": "22/22",
            }
        ]
    )

print(f"[END] G7.20 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 106
from datetime import datetime as _dt
print(f"[START] G7.21 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_cox_ph_summary_table():
    return pd.DataFrame(
        [
            {
                "Model": "Cox Comparable",
                "Covariates tested": 41,
                "Flagged": 5,
                "Global interpretation": "Localized departures rather than broad failure of proportional hazards",
            }
        ]
    )

print(f"[END] G7.21 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 108
from datetime import datetime as _dt
print(f"[START] G7.22 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_tex_cox_ph_audit_table():
    return pd.DataFrame(
        [
            {"Narrative rank": 1, "Covariate": "num__active_weeks_first_4", "Source": "tex_narrative"},
            {"Narrative rank": 2, "Covariate": "num__num_of_prev_attempts", "Source": "tex_narrative"},
            {"Narrative rank": 3, "Covariate": "num__mean_clicks_first_4_weeks", "Source": "tex_narrative"},
            {"Narrative rank": 4, "Covariate": "num__clicks_first_4_weeks", "Source": "tex_narrative"},
            {"Narrative rank": 5, "Covariate": "num__studied_credits", "Source": "tex_narrative"},
        ]
    )

print(f"[END] G7.22 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 110
from datetime import datetime as _dt
print(f"[START] G7.23 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def extract_tex_contract(tex_source):
    contract = {}
    table_pattern = re.compile(
        r"\\begin\{table\}.*?\\caption\{(?P<caption>.*?)\}.*?\\label\{(?P<label>[^}]+)\}.*?\\end\{table\}",
        flags=re.DOTALL,
    )
    figure_pattern = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{(?P<graphic>[^}]+)\}(?P<middle>.*?)\\(?:caption|captionof\{figure\})\{(?P<caption>.*?)\}(?P<after>.*?)\\label\{(?P<label>[^}]+)\}",
        flags=re.DOTALL,
    )
    for match in table_pattern.finditer(tex_source):
        label = match.group("label").strip()
        contract[label] = {
            "label": label,
            "caption": re.sub(r"\s+", " ", match.group("caption")).strip(),
            "artifact_type": "table",
            "scope": "appendix" if "appendix" in label else "main",
            "graphic_path": "",
            "file_name": "",
        }
    for match in figure_pattern.finditer(tex_source):
        label = match.group("label").strip()
        graphic_path = match.group("graphic").strip()
        file_name = Path(graphic_path).name
        if "." not in file_name:
            file_name = f"{file_name}.png"
        contract[label] = {
            "label": label,
            "caption": re.sub(r"\s+", " ", match.group("caption")).strip(),
            "artifact_type": "figure",
            "scope": "appendix" if "appendix" in label else "main",
            "graphic_path": graphic_path,
            "file_name": file_name,
        }
    return contract

print(f"[END] G7.23 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 112
from datetime import datetime as _dt
print(f"[START] G7.24 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

tex_source = TEX_PATH.read_text(encoding="utf-8") if TEX_PATH.exists() else ""
parsed_tex_contract = extract_tex_contract(tex_source)
required_tex_labels = [
    "tab:main_benchmark",
    "fig:benchmark_tuned_comparison",
    "tab:ablation_summary",
    "fig:ablation_impact",
    "tab:explainability_summary",
    "fig:explainability_block_dominance",
    "tab:calibration_summary",
    "tab:appendix_protocol_audit",
    "tab:appendix_preproc_tuning_audit",
    "tab:appendix_bootstrap_uncertainty",
    "tab:appendix_split_context_audit",
    "tab:appendix_cox_ph_summary",
    "fig:appendix_cox_ph_diagnostics",
    "tab:appendix_window_sensitivity",
]

tex_contract = {}
for label in required_tex_labels:
    fallback_file_name = f"{label.replace(':', '_')}.{'png' if label.startswith('fig:') else 'csv'}"
    tex_contract[label] = parsed_tex_contract.get(
        label,
        {
            "label": label,
            "caption": label,
            "artifact_type": "figure" if label.startswith("fig:") else "table",
            "scope": "appendix" if "appendix" in label else "main",
            "graphic_path": "",
            "file_name": fallback_file_name,
        },
    )

tex_asset_rows = []
supporting_asset_rows = []
manifest_rows = []

leaderboard_source_candidates = [TABLES_DIR / "table_benchmark_leaderboard_main.csv"]
brier_wide_source_candidates = [TABLES_DIR / "table_benchmark_brier_by_horizon_wide.csv"]
calibration_wide_source_candidates = [TABLES_DIR / "table_benchmark_calibration_by_horizon_wide.csv"]
leaderboard_source = resolve_first_existing(leaderboard_source_candidates)
brier_wide_source = resolve_first_existing(brier_wide_source_candidates)
calibration_wide_source = resolve_first_existing(calibration_wide_source_candidates)

leaderboard_df, _, _ = load_table_duckdb_first(leaderboard_source_candidates, preferred_table_names=["table_benchmark_leaderboard_main"])
brier_wide_df, _, _ = load_table_duckdb_first(brier_wide_source_candidates, preferred_table_names=["table_benchmark_brier_by_horizon_wide"])

benchmark_paper_df = None
benchmark_figure_df = None
if leaderboard_df is not None and brier_wide_df is not None:
    leaderboard_model_col = pick_column(leaderboard_df, ["display_name", "Model", "model", "model_name"])
    leaderboard_family_col = pick_column(leaderboard_df, ["family", "Family", "model_family"])
    ibs_col = pick_column(leaderboard_df, ["ibs", "IBS", "integrated_brier_score"])
    cindex_col = pick_column(leaderboard_df, ["c_index", "TD Concordance", "td_concordance", "concordance_td"])
    brier_model_col = pick_column(brier_wide_df, ["display_name", "Model", "model", "model_name"])
    benchmark_paper_df = leaderboard_df[[leaderboard_model_col, leaderboard_family_col, ibs_col, cindex_col]].copy()
    benchmark_paper_df.columns = ["Model", "Family", "IBS", "TD Concordance"]
    benchmark_paper_df["Model"] = benchmark_paper_df["Model"].map(normalize_model_names)
    benchmark_paper_df["Family"] = benchmark_paper_df["Family"].map(normalize_family_names)
    for horizon in [10, 20, 30]:
        brier_column = pick_column(
            brier_wide_df,
            [
                f"brier_h{horizon}",
                f"brier_hh{horizon}",
                f"Brier@{horizon}",
                f"brier@{horizon}",
                f"brier_{horizon}",
                f"brier_at_{horizon}",
            ],
        )
        brier_lookup = brier_wide_df[[brier_model_col, brier_column]].copy()
        brier_lookup.columns = ["Model", f"Brier@{horizon}"]
        brier_lookup["Model"] = brier_lookup["Model"].map(normalize_model_names)
        benchmark_paper_df = benchmark_paper_df.merge(brier_lookup, on="Model", how="left")
    benchmark_paper_df = benchmark_paper_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
    benchmark_figure_df = benchmark_paper_df[["Model", "Family", "IBS", "TD Concordance"]].copy()
else:
    benchmark_paper_df = build_tex_main_benchmark_table()
    benchmark_figure_df = benchmark_paper_df[["Model", "Family", "IBS", "TD Concordance"]].copy()

benchmark_table_output = PAPER_MAIN_DIR / "table_paper_main_benchmark_family_expanded.csv"
export_direct_table_asset(
    "tab:main_benchmark",
    benchmark_table_output,
    source_candidates=leaderboard_source_candidates + brier_wide_source_candidates,
    dataframe=benchmark_paper_df,
    note="Curated expanded benchmark table exported under the manuscript contract."
)
benchmark_support_output = PAPER_MAIN_DIR / "table_figure1_benchmark_family_expanded_summary.csv"
export_supporting_table_asset(
    "fig:benchmark_tuned_comparison",
    benchmark_support_output,
    source_candidates=leaderboard_source_candidates + brier_wide_source_candidates,
    dataframe=benchmark_figure_df,
    note="Support table for the expanded benchmark comparison figure."
)

print(f"[END] G7.24 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 114
from datetime import datetime as _dt
print(f"[START] G7.25 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_benchmark_figure(output_path):
    if not MATPLOTLIB_AVAILABLE or benchmark_figure_df is None:
        return False
    plot_df = benchmark_figure_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key), ascending=False)
    # Colour-code arms — same palette used in both panels so the shared legend is correct
    arm_colors = ["#2a9d8f" if "Discrete-Time" in m or "Weekly" in m or "CatBoost" in m or "Poisson" in m
                  else "#1a6b5f" for m in plot_df["Model"]]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    bars0 = axes[0].barh(plot_df["Model"], plot_df["IBS"], color=arm_colors, height=0.65)
    axes[0].set_title("Integrated Brier Score (lower is better)", fontsize=12)
    axes[0].set_xlabel("IBS", fontsize=11)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].xaxis.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_axisbelow(True)
    bars1 = axes[1].barh(plot_df["Model"], plot_df["TD Concordance"], color=arm_colors, height=0.65)
    axes[1].set_title("TD Concordance (higher is better)", fontsize=12)
    axes[1].set_xlabel("TD Concordance", fontsize=11)
    axes[1].tick_params(axis="y", labelsize=10)
    axes[1].xaxis.grid(True, linestyle="--", alpha=0.5)
    axes[1].set_axisbelow(True)
    # Shared legend for arm identity
    from matplotlib.patches import Patch
    legend_handles = [Patch(color="#2a9d8f", label="Dynamic arm"), Patch(color="#1a6b5f", label="Comparable arm")]
    figure.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
                  frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Benchmark comparison across all tuned model families", fontsize=14, y=1.02)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True

print(f"[END] G7.25 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 116
from datetime import datetime as _dt
print(f"[START] G7.26 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

export_direct_figure_asset(
    "fig:benchmark_tuned_comparison",
    PAPER_MAIN_DIR / tex_contract["fig:benchmark_tuned_comparison"]["file_name"],
    generator=generate_benchmark_figure,
    note="Generated from the benchmark table frozen for the manuscript."
)

ablation_source = resolve_first_existing([TABLES_DIR / "table_ablation_summary_by_model.csv", TABLES_DIR / "table_p31_paper_ablation_summary.csv"])
ablation_table_df = None
if ablation_source is not None:
    ablation_raw_df = load_duckdb_table_optional(infer_table_name_from_pathlike(ablation_source))
    if ablation_raw_df is None:
        ablation_raw_df = pd.read_csv(ablation_source)

    model_col = pick_column(ablation_raw_df, ["display_name", "Model", "model", "model_name"])
    family_col = pick_column(ablation_raw_df, ["family", "Family", "model_family"])
    delta_ibs_static_col = pick_column(ablation_raw_df, ["Delta IBS static", "delta_ibs_static", "ibs_delta_static", "delta_ibs_without_static"], required=False)
    delta_ibs_temporal_col = pick_column(ablation_raw_df, ["Delta IBS temporal", "delta_ibs_temporal", "ibs_delta_temporal", "delta_ibs_without_temporal"], required=False)
    delta_td_static_col = pick_column(ablation_raw_df, ["Delta TD concordance static", "delta_td_concordance_static", "delta_cindex_static"], required=False)
    delta_td_temporal_col = pick_column(ablation_raw_df, ["Delta TD concordance temporal", "delta_td_concordance_temporal", "delta_cindex_temporal"], required=False)
    ibs_ratio_col = pick_column(ablation_raw_df, ["IBS ratio", "ibs_ratio", "temporal_static_ibs_ratio"], required=False)

    if None in [delta_ibs_static_col, delta_ibs_temporal_col, delta_td_static_col, delta_td_temporal_col, ibs_ratio_col]:
        ablation_table_df = build_tex_ablation_table()
    else:
        ablation_table_df = ablation_raw_df[[model_col, family_col, delta_ibs_static_col, delta_ibs_temporal_col, delta_td_static_col, delta_td_temporal_col, ibs_ratio_col]].copy()
        ablation_table_df.columns = ["Model", "Family", "Delta IBS static", "Delta IBS temporal", "Delta TD concordance static", "Delta TD concordance temporal", "IBS ratio"]
        ablation_table_df["Model"] = ablation_table_df["Model"].map(normalize_model_names)
        ablation_table_df["Family"] = ablation_table_df["Family"].map(normalize_family_names)
        ablation_table_df = ablation_table_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
else:
    ablation_table_df = build_tex_ablation_table()

export_direct_table_asset(
    "tab:ablation_summary",
    PAPER_MAIN_DIR / "table_paper_ablation_summary.csv",
    source_candidates=[ablation_source] if ablation_source is not None else [],
    dataframe=ablation_table_df,
    note="Curated ablation table exported under the manuscript contract."
)
export_supporting_table_asset(
    "fig:ablation_impact",
    PAPER_MAIN_DIR / "table_figure2_ablation_delta_summary.csv",
    dataframe=ablation_table_df,
    note="Support table for the ablation impact figure."
)

print(f"[END] G7.26 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 117b
from datetime import datetime as _dt
print(f"[START] G7.26b - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Freeze convergence CSVs produced in F5.CONV and G6.CONV ---
_convergence_freeze_pairs = [
    (
        TABLES_DIR / "table_ablation_convergence_paradigm.csv",
        PAPER_MAIN_DIR / "table_paper_ablation_convergence_paradigm.csv",
    ),
    (
        TABLES_DIR / "table_explainability_convergence_paradigm.csv",
        PAPER_MAIN_DIR / "table_paper_explainability_convergence_paradigm.csv",
    ),
    (
        TABLES_DIR / "table_explainability_top_recurring_crossparadigm.csv",
        PAPER_MAIN_DIR / "table_paper_explainability_top_recurring_crossparadigm.csv",
    ),
]
for _src, _dst in _convergence_freeze_pairs:
    if Path(_src).exists():
        PAPER_MAIN_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_src, _dst)
        print(f"  [FREEZE] {Path(_src).name} → {Path(_dst).name}")
    else:
        print(f"  [SKIP] {Path(_src).name} not found — convergence not yet computed")

print(f"[END] G7.26b - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 118
from datetime import datetime as _dt
print(f"[START] G7.27 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_ablation_figure(output_path):
    if not MATPLOTLIB_AVAILABLE or ablation_table_df is None:
        return False
    plot_df = ablation_table_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key), ascending=False)
    # Replace --- with NaN so we can skip missing IBS values for the dynamic arm
    import numpy as _np
    ibs_static_all = pd.to_numeric(plot_df["Delta IBS static"].replace("---", _np.nan), errors="coerce")
    ibs_temporal_all = pd.to_numeric(plot_df["Delta IBS temporal"].replace("---", _np.nan), errors="coerce")
    td_static = pd.to_numeric(plot_df["Delta TD concordance static"].replace("---", _np.nan), errors="coerce").abs()
    td_temporal = pd.to_numeric(plot_df["Delta TD concordance temporal"].replace("---", _np.nan), errors="coerce").abs()
    # IBS panel: only rows where IBS data exists (comparable arm only)
    ibs_mask = ibs_static_all.notna() | ibs_temporal_all.notna()
    ibs_df = plot_df[ibs_mask].copy()
    ibs_static = ibs_static_all[ibs_mask]
    ibs_temporal = ibs_temporal_all[ibs_mask]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    height = 0.35
    # IBS panel — comparable arm only (filtered rows)
    y_ibs = list(range(len(ibs_df)))
    y_ibs_static = [y + height / 2 for y in y_ibs]
    y_ibs_temporal = [y - height / 2 for y in y_ibs]
    axes[0].barh(y_ibs_static, ibs_static.values, height=height, color="#8ecae6", label="Remove static")
    axes[0].barh(y_ibs_temporal, ibs_temporal.values, height=height, color="#219ebc", label="Remove temporal")
    axes[0].set_yticks(y_ibs)
    axes[0].set_yticklabels(ibs_df["Model"].values, fontsize=10)
    axes[0].set_title("IBS increase (comparable arm only)", fontsize=12)
    axes[0].set_xlabel("Delta IBS", fontsize=11)
    axes[0].xaxis.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=True)
    # TD concordance panel — both arms (all rows)
    y_td = list(range(len(plot_df)))
    y_td_static = [y + height / 2 for y in y_td]
    y_td_temporal = [y - height / 2 for y in y_td]
    axes[1].barh(y_td_static, td_static.values, height=height, color="#ffb703", label="Remove static")
    axes[1].barh(y_td_temporal, td_temporal.values, height=height, color="#fb8500", label="Remove temporal")
    axes[1].set_yticks(y_td)
    axes[1].set_yticklabels(plot_df["Model"].values, fontsize=10)
    axes[1].set_title("TD concordance drop (both arms)", fontsize=12)
    axes[1].set_xlabel("|\u0394 TD Concordance|", fontsize=11)
    axes[1].xaxis.grid(True, linestyle="--", alpha=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=True)
    figure.suptitle("Ablation impact — temporal vs. static signal removal", fontsize=14, y=1.02)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True

print(f"[END] G7.27 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 120
from datetime import datetime as _dt
print(f"[START] G7.28 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

export_direct_figure_asset(
    "fig:ablation_impact",
    PAPER_MAIN_DIR / "fig_paper_02_ablation_impact.png",
    generator=generate_ablation_figure,
    note="Generated from the ablation table frozen for the manuscript."
)

explainability_summary_source_candidates = [TABLES_DIR / "table_explainability_summary_by_model.csv", TABLES_DIR / "table_p37_explainability_summary_by_model.csv"]
explainability_summary_source = resolve_first_existing(explainability_summary_source_candidates)
explainability_raw_df, _, _ = load_table_duckdb_first(
    explainability_summary_source_candidates,
    preferred_table_names=["table_explainability_summary_by_model"],
)
explainability_summary_df, explainability_summary_status = normalize_manuscript_explainability_summary(explainability_raw_df)

export_direct_table_asset(
    "tab:explainability_summary",
    PAPER_MAIN_DIR / "table_paper_explainability_summary.csv",
    source_candidates=explainability_summary_source_candidates,
    dataframe=explainability_summary_df,
    note=(
        "Curated explainability table exported under the manuscript contract from a materialized summary."
        if explainability_summary_status == "materialized"
        else "Curated explainability table exported under the manuscript contract using the manuscript-facing fallback quartet."
    )
)

block_source_candidates = [TABLES_DIR / "table_explainability_all_blocks_normalized.csv"]
block_source = resolve_first_existing(block_source_candidates)
all_blocks_df, _, _ = load_table_duckdb_first(
    block_source_candidates,
    preferred_table_names=["table_explainability_all_blocks_normalized"],
)
block_plot_wide_df = None
explainability_figure_note = "Generated from normalized explainability block outputs when available."
if all_blocks_df is not None:
    model_col = pick_column(all_blocks_df, ["display_name", "Model", "model", "model_name"])
    block_col = pick_column(all_blocks_df, ["Block", "block", "feature_block", "dominant_block"])
    value_col = pick_column(all_blocks_df, ["Normalized importance", "normalized_importance", "normalized_value", "value", "importance"])
    block_long_df = all_blocks_df[[model_col, block_col, value_col]].copy()
    block_long_df.columns = ["Model", "Block", "Normalized value"]
    block_long_df["Model"] = block_long_df["Model"].map(normalize_model_names)
    block_long_df["Block"] = block_long_df["Block"].astype(str)
    block_plot_wide_df = (
        block_long_df
        .pivot_table(index="Model", columns="Block", values="Normalized value", aggfunc="mean", fill_value=0.0)
        .reset_index()
    )
    block_plot_wide_df = block_plot_wide_df[
        block_plot_wide_df["Model"].isin(MANUSCRIPT_EXPLAINABILITY_MODEL_ORDER)
    ].copy()
    if block_plot_wide_df.empty:
        block_fallback_long_df = explainability_summary_df[["Model", "Dominant block"]].copy()
        block_fallback_long_df["Normalized value"] = 1.0
        block_plot_wide_df = (
            block_fallback_long_df
            .pivot_table(index="Model", columns="Dominant block", values="Normalized value", aggfunc="max", fill_value=0.0)
            .reset_index()
        )
        block_plot_wide_df = block_plot_wide_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
        export_supporting_table_asset(
            "fig:explainability_block_dominance",
            PAPER_MAIN_DIR / "table_figure3_explainability_block_summary_normalized.csv",
            source_candidates=block_source_candidates,
            dataframe=block_plot_wide_df,
            note="Normalized explainability block output was present but did not contain the manuscript-facing quartet, so the support table was reconstructed from the curated explainability summary."
        )
        explainability_figure_note = "Generated from the curated explainability summary because the available normalized block table did not cover the manuscript-facing quartet."
    else:
        block_plot_wide_df = block_plot_wide_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
        export_supporting_table_asset(
            "fig:explainability_block_dominance",
            PAPER_MAIN_DIR / "table_figure3_explainability_block_summary_normalized.csv",
            source_candidates=block_source_candidates,
            dataframe=block_plot_wide_df,
            note="Support table derived from normalized explainability block outputs after filtering to the manuscript-facing explainability quartet."
        )
else:
    block_fallback_long_df = explainability_summary_df[["Model", "Dominant block"]].copy()
    block_fallback_long_df["Normalized value"] = 1.0
    block_plot_wide_df = (
        block_fallback_long_df
        .pivot_table(index="Model", columns="Dominant block", values="Normalized value", aggfunc="max", fill_value=0.0)
        .reset_index()
    )
    block_plot_wide_df = block_plot_wide_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
    export_supporting_table_asset(
        "fig:explainability_block_dominance",
        PAPER_MAIN_DIR / "table_figure3_explainability_block_summary_normalized.csv",
        source_candidates=block_source_candidates,
        dataframe=block_plot_wide_df,
        note="Fallback support table reconstructed from the dominant block reported in the explainability summary because the detailed normalized block output is not materialized."
    )
    explainability_figure_note = "Generated from the explainability dominant-block summary because table_explainability_all_blocks_normalized.csv is not currently materialized."

print(f"[END] G7.28 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 122
from datetime import datetime as _dt
print(f"[START] G7.29 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_explainability_figure(output_path):
    if not MATPLOTLIB_AVAILABLE or block_plot_wide_df is None:
        return False
    plot_df = block_plot_wide_df.copy()
    block_columns = [column for column in plot_df.columns if column != "Model"]
    if not block_columns:
        return False
    import numpy as _np
    # Normalize each row to proportions (row sums to 1)
    numeric = plot_df[block_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = numeric.sum(axis=1).replace(0, 1.0)
    norm_df = numeric.div(row_sums, axis=0)
    # Sort: comparable arm first, then dynamic arm
    comparable_order = ["Cox Comparable", "DeepSurv", "Random Survival Forest", "Neural-MTLR"]
    dynamic_order = ["Linear Discrete-Time Hazard", "Poisson Piecewise-Exponential",
                     "Neural Discrete-Time Survival", "Gradient-Boosted Weekly Hazard", "CatBoost Weekly Hazard"]
    desired_order = comparable_order + dynamic_order
    order_map = {m: i for i, m in enumerate(desired_order)}
    sort_key = plot_df["Model"].map(lambda m: order_map.get(m, 99))
    sorted_idx = sort_key.argsort().values
    norm_df = norm_df.iloc[sorted_idx].reset_index(drop=True)
    labels = plot_df["Model"].iloc[sorted_idx].values
    # Colour palette: one colour per block with intuitive mapping
    block_palette = {
        "early_window_behavior": "#2a9d8f",
        "dynamic_temporal_behavioral": "#264653",
        "discrete_time_index": "#7fb3c8",
        "static_structural": "#e9c46a",
    }
    n_models = len(norm_df)
    figure, axis = plt.subplots(figsize=(12, max(5, n_models * 0.75 + 1.5)), constrained_layout=True)
    lefts = _np.zeros(n_models)
    for col in block_columns:
        values = norm_df[col].values
        color = block_palette.get(col, "#aaa")
        label = col.replace("_", " ").title()
        axis.barh(range(n_models), values, left=lefts, height=0.65,
                  color=color, label=label, edgecolor="white", linewidth=0.4)
        lefts += values
    # Separator line between the two arms
    n_comp = sum(1 for m in labels if m in set(comparable_order))
    if 0 < n_comp < n_models:
        axis.axhline(n_comp - 0.5, color="#555", linewidth=1.0, linestyle="--")
        axis.text(1.01, n_comp - 0.5, "  Dynamic arm", va="center", ha="left",
                  transform=axis.get_yaxis_transform(), fontsize=9, color="#444")
        axis.text(1.01, n_comp - 0.5 - (n_models - n_comp) / 2, "", va="center", ha="left",
                  transform=axis.get_yaxis_transform(), fontsize=9, color="#444")
    axis.set_yticks(range(n_models))
    axis.set_yticklabels(labels, fontsize=11)
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Proportion of total importance", fontsize=11)
    axis.set_title("Feature block dominance by model family (row-normalized)", fontsize=13)
    axis.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    axis.xaxis.grid(True, linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    # Arm labels on Y axis side
    axis.annotate("Comparable arm", xy=(0, 0), xytext=(-0.32, (n_comp - 1) / 2.0),
                  xycoords="data", textcoords=("axes fraction", "data"),
                  fontsize=9, color="#333", rotation=90, ha="center", va="center")
    axis.annotate("Dynamic arm", xy=(0, 0), xytext=(-0.32, n_comp + (n_models - n_comp - 1) / 2.0),
                  xycoords="data", textcoords=("axes fraction", "data"),
                  fontsize=9, color="#333", rotation=90, ha="center", va="center")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=len(block_columns),
                fontsize=9, frameon=True, framealpha=0.9)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True

print(f"[END] G7.29 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 124
from datetime import datetime as _dt
print(f"[START] G7.30 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

export_direct_figure_asset(
    "fig:explainability_block_dominance",
    PAPER_MAIN_DIR / "fig_paper_03_explainability_block_dominance_normalized.png",
    generator=generate_explainability_figure,
    note=explainability_figure_note,
    missing_note="Neither the normalized block output nor the dominant-block summary was available to reconstruct the explainability figure."
)

calibration_source_candidates = [TABLES_DIR / "table_benchmark_calibration_by_horizon_wide.csv"]
calibration_wide_df, _, _ = load_table_duckdb_first(
    calibration_source_candidates,
    preferred_table_names=["table_benchmark_calibration_by_horizon_wide"],
)
calibration_paper_df = None
if calibration_wide_df is not None:
    model_col = pick_column(calibration_wide_df, ["display_name", "Model", "model", "model_name"])
    family_col = pick_column(calibration_wide_df, ["family", "Family", "model_family"])
    calibration_paper_df = calibration_wide_df[[
        model_col,
        family_col,
        pick_column(calibration_wide_df, ["calibration_h10", "calibration_hh10", "Calib@10", "calib@10", "calibration@10", "calibration_10"]),
        pick_column(calibration_wide_df, ["calibration_h20", "calibration_hh20", "Calib@20", "calib@20", "calibration@20", "calibration_20"]),
        pick_column(calibration_wide_df, ["calibration_h30", "calibration_hh30", "Calib@30", "calib@30", "calibration@30", "calibration_30"]),
    ]].copy()
    calibration_paper_df.columns = ["Model", "Family", "Calib@10", "Calib@20", "Calib@30"]
    calibration_paper_df["Model"] = calibration_paper_df["Model"].map(normalize_model_names)
    calibration_paper_df["Family"] = calibration_paper_df["Family"].map(normalize_family_names)
    calibration_paper_df = calibration_paper_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
else:
    calibration_paper_df = build_tex_calibration_table()

export_direct_table_asset(
    "tab:calibration_summary",
    PAPER_MAIN_DIR / "table_paper_calibration_summary_tuned_models.csv",
    source_candidates=calibration_source_candidates,
    dataframe=calibration_paper_df,
    note="Curated calibration table exported under the manuscript contract."
)

appendix_protocol_source_candidates = [TABLES_DIR / "table_evaluation_protocol_audit.csv"]
appendix_protocol_source = resolve_first_existing(appendix_protocol_source_candidates)
appendix_protocol_df, _, _ = load_table_duckdb_first(
    appendix_protocol_source_candidates,
    preferred_table_names=["table_evaluation_protocol_audit"],
)
if appendix_protocol_df is None:
    appendix_protocol_df = build_tex_protocol_audit_table()
export_direct_table_asset(
    "tab:appendix_protocol_audit",
    PAPER_APPENDIX_DIR / "table_paper_appendix_evaluation_protocol_audit.csv",
    source_candidates=appendix_protocol_source_candidates,
    dataframe=appendix_protocol_df,
    note="Appendix protocol audit exported under the manuscript contract."
)

preproc_source_candidates = [TABLES_DIR / "table_preprocessing_and_tuning_audit.csv", TABLES_DIR / "table_paper_appendix_preprocessing_and_tuning_audit.csv"]
preproc_source = resolve_first_existing(preproc_source_candidates)
preproc_df, _, _ = load_table_duckdb_first(
    preproc_source_candidates,
    preferred_table_names=["table_preprocessing_and_tuning_audit", "table_paper_appendix_preprocessing_and_tuning_audit"],
)
if preproc_df is None:
    preproc_df = build_tex_preproc_tuning_table()
export_direct_table_asset(
    "tab:appendix_preproc_tuning_audit",
    PAPER_APPENDIX_DIR / "table_paper_appendix_preprocessing_and_tuning_audit.csv",
    source_candidates=preproc_source_candidates,
    dataframe=preproc_df,
    note="Appendix preprocessing and tuning audit exported under the manuscript contract." if preproc_source is not None else "Fallback reconstructed from the inline TeX appendix table."
)

bootstrap_source_candidates = [TABLES_DIR / "table_appendix_bootstrap_uncertainty_compact.csv"]
bootstrap_source = resolve_first_existing(bootstrap_source_candidates)
bootstrap_df, _, _ = load_table_duckdb_first(
    bootstrap_source_candidates,
    preferred_table_names=["table_appendix_bootstrap_uncertainty_compact"],
)
if bootstrap_df is None:
    bootstrap_df = build_tex_bootstrap_table()
export_direct_table_asset(
    "tab:appendix_bootstrap_uncertainty",
    PAPER_APPENDIX_DIR / "table_appendix_bootstrap_uncertainty_compact.csv",
    source_candidates=bootstrap_source_candidates,
    dataframe=bootstrap_df,
    note="Appendix bootstrap uncertainty table exported under the manuscript contract." if bootstrap_source is not None else "Fallback reconstructed from the inline TeX appendix table."
)

split_context_source_candidates = [
    TABLES_DIR / "paper_appendix" / "table_paper_appendix_split_context_audit.csv",
    TABLES_DIR / "table_paper_appendix_split_context_audit.csv",
    TABLES_DIR / "table_appendix_split_context_audit_compact.csv",
    TABLES_DIR / "table_split_context_appendix.csv",
]
split_context_source = resolve_first_existing(split_context_source_candidates)
split_context_df, _, _ = load_table_duckdb_first(
    split_context_source_candidates,
    preferred_table_names=["table_paper_appendix_split_context_audit", "table_appendix_split_context_audit_compact", "table_split_context_appendix"],
)
if split_context_df is None:
    split_context_df = build_tex_split_context_table()
export_direct_table_asset(
    "tab:appendix_split_context_audit",
    PAPER_APPENDIX_DIR / "table_paper_appendix_split_context_audit.csv",
    source_candidates=split_context_source_candidates,
    dataframe=split_context_df,
    note="Appendix split/context audit exported under the manuscript contract." if split_context_source is not None else "Fallback reconstructed from the inline TeX appendix table."
)

cox_summary_source_candidates = [
    TABLES_DIR / "paper_appendix" / "table_paper_appendix_cox_ph_global_summary.csv",
    TABLES_DIR / "table_paper_appendix_cox_ph_global_summary.csv",
    TABLES_DIR / "table_appendix_cox_ph_global_summary.csv",
    TABLES_DIR / "table_cox_ph_global_summary.csv",
]
cox_summary_source = resolve_first_existing(cox_summary_source_candidates)
cox_summary_df, _, _ = load_table_duckdb_first(
    cox_summary_source_candidates,
    preferred_table_names=["table_paper_appendix_cox_ph_global_summary", "table_appendix_cox_ph_global_summary", "table_cox_ph_global_summary"],
)
if cox_summary_df is None:
    cox_summary_df = build_tex_cox_ph_summary_table()
export_direct_table_asset(
    "tab:appendix_cox_ph_summary",
    PAPER_APPENDIX_DIR / "table_paper_appendix_cox_ph_global_summary.csv",
    source_candidates=cox_summary_source_candidates,
    dataframe=cox_summary_df,
    note="Appendix Cox PH global summary exported under the manuscript contract." if cox_summary_source is not None else "Fallback reconstructed from the inline TeX appendix table."
)

cox_audit_source_candidates = [
    TABLES_DIR / "paper_appendix" / "table_paper_appendix_cox_ph_audit.csv",
    TABLES_DIR / "table_paper_appendix_cox_ph_audit.csv",
    TABLES_DIR / "table_appendix_cox_ph_audit.csv",
    TABLES_DIR / "table_cox_ph_assumption_audit.csv",
]
cox_audit_source = resolve_first_existing(cox_audit_source_candidates)
cox_ph_audit_df, _, _ = load_table_duckdb_first(
    cox_audit_source_candidates,
    preferred_table_names=["table_paper_appendix_cox_ph_audit", "table_appendix_cox_ph_audit", "table_cox_ph_assumption_audit"],
)
if cox_ph_audit_df is None:
    cox_ph_audit_df = build_tex_cox_ph_audit_table()
export_supporting_table_asset(
    "fig:appendix_cox_ph_diagnostics",
    PAPER_APPENDIX_DIR / "table_paper_appendix_cox_ph_audit.csv",
    source_candidates=cox_audit_source_candidates,
    dataframe=cox_ph_audit_df,
    note="Supporting PH audit table exported under the manuscript contract." if cox_audit_source is not None else "Fallback reconstructed from the manuscript narrative listing the flagged covariates."
)

# --- Window sensitivity export ---
# Loads table_5_16_comparable_window_sensitivity from DuckDB, normalises names,
# and exports the manuscript-facing table to paper_appendix.
window_sens_df, _, _ = load_table_duckdb_first(
    [],
    preferred_table_names=["table_5_16_comparable_window_sensitivity"],
)
if window_sens_df is not None:
    _rename_model = normalize_model_names if callable(normalize_model_names) else lambda x: x
    _rename_family = normalize_family_names if callable(normalize_family_names) else lambda x: x
    window_sens_export_df = window_sens_df.copy()
    window_sens_export_df["display_name"] = window_sens_export_df["model_name"].map(_rename_model).fillna(window_sens_export_df["model_name"])
    window_sens_export_df["family_display"] = window_sens_export_df["family_group"].map(_rename_family).fillna(window_sens_export_df["family_group"])
    window_sens_export_df = window_sens_export_df.rename(columns={"window_weeks": "window_w", "ibs": "IBS", "c_index": "TD_Concordance", "canonical_window": "is_canonical_window"})
    window_sens_export_df = window_sens_export_df[["display_name", "family_display", "window_w", "IBS", "TD_Concordance", "availability_status", "is_canonical_window"]]
    window_sens_export_df = window_sens_export_df.sort_values(["display_name", "window_w"]).reset_index(drop=True)
    export_direct_table_asset(
        "tab:appendix_window_sensitivity",
        PAPER_APPENDIX_DIR / "table_paper_appendix_window_sensitivity.csv",
        source_candidates=[],
        dataframe=window_sens_export_df,
        note="Comparable-arm window sensitivity (IBS and TD Concordance by model and window w in [2,4,6,8,10]). Exported from table_5_16_comparable_window_sensitivity in DuckDB.",
    )
else:
    print("[WARN] G7.30: table_5_16_comparable_window_sensitivity not found in DuckDB; tab:appendix_window_sensitivity skipped.")
    append_asset_row(
        tex_asset_rows,
        tex_label="tab:appendix_window_sensitivity",
        tex_caption="Comparable-arm window sensitivity table",
        scope="appendix",
        artifact_type="table",
        status="missing",
        file_name="table_paper_appendix_window_sensitivity.csv",
        output_path=PAPER_APPENDIX_DIR / "table_paper_appendix_window_sensitivity.csv",
        source_path=DUCKDB_PATH,
        notes="table_5_16_comparable_window_sensitivity not found in DuckDB.",
    )

print(f"[END] G7.30 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 126
from datetime import datetime as _dt
print(f"[START] G7.31 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_appendix_ph_figure(output_path):
    if not MATPLOTLIB_AVAILABLE or cox_ph_audit_df is None:
        return False
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.axis("off")
    axis.text(0.01, 0.94, "Comparable Cox PH diagnostic summary", fontsize=18, fontweight="bold", va="top")
    axis.text(
        0.01,
        0.84,
        "Covariates with the strongest evidence of possible non-proportionality",
        fontsize=12,
        color="#264653",
        va="top",
    )
    display_df = cox_ph_audit_df.copy()
    if "Narrative rank" in display_df.columns and "Covariate" in display_df.columns:
        display_df = display_df.sort_values("Narrative rank")
        lines = [f"{int(row['Narrative rank'])}. {row['Covariate']}" for _, row in display_df.iterrows()]
        footer = "Narrative fallback generated from the manuscript because the quantitative PH audit table is not currently materialized in outputs_benchmark_survival/tables."
    else:
        covariate_col = pick_column(display_df, ["Covariate", "covariate", "feature", "variable"])
        lines = [f"- {value}" for value in display_df[covariate_col].astype(str).head(5)]
        footer = "Generated from the notebook PH audit table."
    y_position = 0.72
    for line in lines:
        axis.text(0.03, y_position, line, fontsize=12, va="top")
        y_position -= 0.10
    axis.text(
        0.01,
        0.10,
        "Global interpretation: localized departures rather than broad failure of proportional hazards.",
        fontsize=11,
        color="#6b705c",
        va="top",
    )
    axis.text(0.01, 0.03, footer, fontsize=9.5, color="#666666", va="bottom")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True

print(f"[END] G7.31 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 128
from datetime import datetime as _dt
print(f"[START] G7.32 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

export_direct_figure_asset(
    "fig:appendix_cox_ph_diagnostics",
    PAPER_APPENDIX_DIR / tex_contract["fig:appendix_cox_ph_diagnostics"]["file_name"],
    source_candidates=[
        FIGURES_DIR / tex_contract["fig:appendix_cox_ph_diagnostics"]["file_name"],
        FIGURES_DIR / "paper_appendix" / tex_contract["fig:appendix_cox_ph_diagnostics"]["file_name"],
    ],
    generator=generate_appendix_ph_figure,
    note="Generated under the manuscript contract; falls back to a narrative diagnostic summary when the quantitative PH figure is unavailable."
)

for asset_row in tex_asset_rows:
    manifest_rows.append(
        {
            "tex_label": asset_row["tex_label"],
            "scope": asset_row["scope"],
            "artifact_type": asset_row["artifact_type"],
            "status": asset_row["status"],
            "file_name": asset_row["file_name"],
            "output_path": asset_row["output_path"],
        }
    )
for supporting_row in supporting_asset_rows:
    manifest_rows.append(
        {
            "tex_label": supporting_row["related_tex_label"],
            "scope": supporting_row["scope"],
            "artifact_type": supporting_row["artifact_type"],
            "status": supporting_row["status"],
            "file_name": supporting_row["file_name"],
            "output_path": supporting_row["output_path"],
        }
    )

tex_asset_registry_df = pd.DataFrame(tex_asset_rows).sort_values(["scope", "artifact_type", "tex_label"]).reset_index(drop=True)
supporting_asset_registry_df = pd.DataFrame(supporting_asset_rows).sort_values(["scope", "related_tex_label"]).reset_index(drop=True)
asset_manifest_df = pd.DataFrame(manifest_rows).sort_values(["scope", "artifact_type", "tex_label"]).reset_index(drop=True)

tex_asset_registry_path = METADATA_DIR / "paper_tex_asset_registry.csv"
supporting_registry_path = METADATA_DIR / "paper_tex_supporting_asset_registry.csv"
asset_manifest_path = METADATA_DIR / "paper_curated_asset_manifest.csv"
freeze_summary_path = METADATA_DIR / "paper_freeze_summary.json"

tex_asset_registry_df.to_csv(tex_asset_registry_path, index=False)
materialize_dataframe(con, tex_asset_registry_df, infer_table_name_from_pathlike(tex_asset_registry_path), "G6")
supporting_asset_registry_df.to_csv(supporting_registry_path, index=False)
materialize_dataframe(con, supporting_asset_registry_df, infer_table_name_from_pathlike(supporting_registry_path), "G6")
asset_manifest_df.to_csv(asset_manifest_path, index=False)
materialize_dataframe(con, asset_manifest_df, infer_table_name_from_pathlike(asset_manifest_path), "G6")

freeze_summary = {
    "section_id": "G7",
    "contract_source": str(TEX_PATH),
    "paper_main_dir": str(PAPER_MAIN_DIR),
    "paper_appendix_dir": str(PAPER_APPENDIX_DIR),
    "n_expected_tex_assets": int(len(tex_asset_registry_df)),
    "n_exported_tex_assets": int((tex_asset_registry_df["status"] == "exported").sum()),
    "n_missing_tex_assets": int((tex_asset_registry_df["status"] == "missing").sum()),
    "n_exported_supporting_assets": int((supporting_asset_registry_df["status"] == "exported").sum()),
}
with open(freeze_summary_path, "w", encoding="utf-8") as file_handle:
    json.dump(freeze_summary, file_handle, indent=2)

print("Curated TeX output directories:")
print("-", PAPER_MAIN_DIR.resolve())
print("-", PAPER_APPENDIX_DIR.resolve())
print()
print("Direct TeX-linked assets:")
display(tex_asset_registry_df)
print("\nSupporting assets for TeX figures/tables:")
display(supporting_asset_registry_df)
missing_tex_assets_df = tex_asset_registry_df.loc[tex_asset_registry_df["status"] == "missing", ["tex_label", "file_name", "notes"]]
if not missing_tex_assets_df.empty:
    print("\nAssets still missing from the dedicated TeX export layer:")
    display(missing_tex_assets_df)
else:
    print("\nAll TeX-linked assets were materialized.")
print("\nSaved metadata files:")
print("-", tex_asset_registry_path.resolve())
print("-", supporting_registry_path.resolve())
print("-", asset_manifest_path.resolve())
print("-", freeze_summary_path.resolve())

print(f"[END] G7.32 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 130
from datetime import datetime as _dt
print(f"[START] G8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G8 — Display Curated Paper Figures
# --------------------------------------------------------------
# Purpose:
#   Load and display the curated TeX-facing figures exported in G7
#   directly inside the notebook for visual inspection.
#
# Methodological note:
#   This step is display-only.
#   No model is trained and no metric is recomputed.
# ==============================================================

print("\n" + "=" * 70)
print("G8 — Display Curated Paper Figures")
print("=" * 70)
print("Methodological note: this step displays curated TeX-facing figures only.")

required_names = ["OUTPUT_DIR", "METADATA_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

import pandas as pd

registry_path = METADATA_DIR / "paper_tex_asset_registry.csv"
if not registry_path.exists():
    raise FileNotFoundError(
        f"TeX asset registry not found: {registry_path}. Please run Cell 136 first."
    )

registry_df = pd.read_csv(registry_path)
figure_registry_df = registry_df[
    (registry_df["artifact_type"] == "figure") & (registry_df["status"] == "exported")
].copy()

if figure_registry_df.empty:
    print("No exported TeX-facing figures are currently available.")
else:
    for _, row in figure_registry_df.sort_values(["scope", "tex_label"]).iterrows():
        display(Markdown(f"## {row['tex_label']}"))
        display(Markdown(row["tex_caption"]))
        display(Image(filename=row["output_path"]))
        print("-", Path(row["output_path"]).resolve())

missing_figures_df = registry_df[
    (registry_df["artifact_type"] == "figure") & (registry_df["status"] != "exported")
].copy()
if not missing_figures_df.empty:
    print("\nFigures still missing from the TeX export layer:")
    display(missing_figures_df[["tex_label", "file_name", "notes"]])

print(f"[END] G8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 132
from datetime import datetime as _dt
print(f"[START] G9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# G9 — Preview Curated Paper Evidence
# --------------------------------------------------------------
# Purpose:
#   Preview the manuscript-facing evidence directly from the curated
#   paper_main and paper_appendix directories created in G7.
#
# Methodological note:
#   This step is synthesis-only.
#   No model is trained and no metric is recomputed.
# ==============================================================

print("\n" + "=" * 70)
print("G9 — Preview Curated Paper Evidence")
print("=" * 70)
print("Methodological note: this step reads curated paper artifacts only.")

required_names = ["OUTPUT_DIR", "METADATA_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

import pandas as pd

PAPER_MAIN_DIR = OUTPUT_DIR / "paper_main"
PAPER_APPENDIX_DIR = OUTPUT_DIR / "paper_appendix"
tex_registry_path = METADATA_DIR / "paper_tex_asset_registry.csv"
supporting_registry_path = METADATA_DIR / "paper_tex_supporting_asset_registry.csv"

if not tex_registry_path.exists():
    raise FileNotFoundError(
        f"TeX asset registry not found: {tex_registry_path}. Please run Cell 136 first."
    )

tex_registry_df = pd.read_csv(tex_registry_path)
supporting_registry_df = pd.read_csv(supporting_registry_path) if supporting_registry_path.exists() else pd.DataFrame()

print("\nDirect TeX-linked assets:")
display(tex_registry_df)

if not supporting_registry_df.empty:
    print("\nSupporting assets for TeX figures/tables:")
    display(supporting_registry_df)

exported_main_tables_df = tex_registry_df[
    (tex_registry_df["scope"] == "main")
    & (tex_registry_df["artifact_type"] == "table")
    & (tex_registry_df["status"] == "exported")
].copy()

exported_appendix_tables_df = tex_registry_df[
    (tex_registry_df["scope"] == "appendix")
    & (tex_registry_df["artifact_type"] == "table")
    & (tex_registry_df["status"] == "exported")
].copy()

print("\nPreview of exported main-paper tables:")
if exported_main_tables_df.empty:
    print("- No main-paper tables are currently materialized.")
else:
    for _, row in exported_main_tables_df.sort_values("tex_label").iterrows():
        preview_df = pd.read_csv(row["output_path"])
        print(f"- {row['tex_label']} -> {Path(row['output_path']).name} ({preview_df.shape[0]} rows)")
        display(preview_df)

print("\nPreview of exported appendix tables:")
if exported_appendix_tables_df.empty:
    print("- No appendix tables are currently materialized.")
else:
    for _, row in exported_appendix_tables_df.sort_values("tex_label").iterrows():
        preview_df = pd.read_csv(row["output_path"])
        print(f"- {row['tex_label']} -> {Path(row['output_path']).name} ({preview_df.shape[0]} rows)")
        display(preview_df)

missing_assets_df = tex_registry_df[tex_registry_df["status"] != "exported"].copy()
print("\nAssets still missing from the dedicated TeX export layer:")
if missing_assets_df.empty:
    print("- None. All TeX-linked assets are materialized.")
else:
    display(missing_assets_df[["tex_label", "artifact_type", "file_name", "notes"]])

print("\nExclusive TeX output directories:")
print("-", PAPER_MAIN_DIR.resolve())
print("-", PAPER_APPENDIX_DIR.resolve())

print(f"[END] G9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 133b
from datetime import datetime as _dt
print(f"[START] G9.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# G9.5 — Reliability diagrams (calibration plots) exported to paper_appendix
# Reads calibration bin-level data from DuckDB for each arm and generates
# reliability diagrams (observed event rate vs. mean predicted risk per bin).

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COMPARABLE_BIN_TABLES = {
    "RSF": "table_rsf_tuned_calibration_bins_by_horizon",
    "Cox Comparable": "table_cox_tuned_calibration_bins_by_horizon",
    "DeepSurv": "table_deepsurv_tuned_calibration_bins_by_horizon",
    "Neural-MTLR": "table_mtlr_tuned_calibration_bins_by_horizon",
    "DeepHit": "table_deephit_tuned_calibration_bins_by_horizon",
    "GB-Cox": "table_gb_cox_tuned_calibration_bins_by_horizon",
    "Weibull AFT": "table_weibull_aft_tuned_calibration_bins_by_horizon",
    "Royston-Parmar": "table_royston_parmar_tuned_calibration_bins_by_horizon",
    "XGBoost AFT": "table_xgb_aft_tuned_calibration_bins_by_horizon",
}

_DYNAMIC_BIN_TABLES = {
    "Linear": "table_linear_tuned_calibration_bins_by_horizon",
    "Poisson": "table_poisson_pexp_tuned_calibration_bins_by_horizon",
    "GB Weekly": "table_gb_weekly_hazard_unweighted_tuned_calibration_bins_by_horizon",
    "Neural": "table_neural_tuned_calibration_bins_by_horizon",
    "CatBoost": "table_catboost_weekly_hazard_tuned_calibration_bins_by_horizon",
}

_HORIZONS = [10, 20, 30]
_RELIABILITY_COLORS = plt.cm.tab10.colors

def _load_calib_bins(con, table_name, horizon):
    try:
        df = con.execute(
            f"SELECT mean_predicted_risk, observed_event_rate, n "
            f"FROM {table_name} WHERE horizon_week = {horizon} ORDER BY calibration_bin"
        ).fetchdf()
        return df
    except Exception:
        return None


def _build_reliability_figure(arm_tables, arm_label, con):
    n_models = len(arm_tables)
    n_horizons = len(_HORIZONS)
    # Layout: rows = models, columns = horizons (9×3 for comparable, 5×3 for dynamic)
    fig, axes = plt.subplots(
        n_models, n_horizons,
        figsize=(3.5 * n_horizons, 2.8 * n_models),
        squeeze=False
    )
    fig.suptitle(f"Reliability Diagrams — {arm_label}", fontsize=11, y=1.01)
    for row_idx, (model_label, tbl) in enumerate(arm_tables.items()):
        for col_idx, hz in enumerate(_HORIZONS):
            ax = axes[row_idx][col_idx]
            df = _load_calib_bins(con, tbl, hz)
            if df is not None and not df.empty:
                ax.scatter(
                    df["mean_predicted_risk"], df["observed_event_rate"],
                    s=df["n"] / df["n"].max() * 200 + 20,
                    color=_RELIABILITY_COLORS[row_idx % len(_RELIABILITY_COLORS)],
                    alpha=0.8, zorder=3
                )
                ax.plot(
                    df["mean_predicted_risk"], df["observed_event_rate"],
                    color=_RELIABILITY_COLORS[row_idx % len(_RELIABILITY_COLORS)],
                    linewidth=1.0, alpha=0.6
                )
            lo = 0.0
            hi = max(
                (df["mean_predicted_risk"].max() if df is not None and not df.empty else 0.5),
                (df["observed_event_rate"].max() if df is not None and not df.empty else 0.5),
                0.4
            )
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
            ax.set_xlim(lo - 0.02, hi + 0.05)
            ax.set_ylim(lo - 0.02, hi + 0.05)
            if row_idx == 0:
                ax.set_title(f"Horizon {hz}", fontsize=9, pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"{model_label}\nObserved rate", fontsize=7)
            else:
                ax.set_ylabel("Observed rate", fontsize=7)
            if row_idx == n_models - 1:
                ax.set_xlabel("Mean predicted risk", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    return fig


try:
    _dcon = ensure_active_duckdb_connection()
    PAPER_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

    # Comparable arm
    _fig_comp = _build_reliability_figure(_COMPARABLE_BIN_TABLES, "Comparable Arm", _dcon)
    _out_comp = PAPER_APPENDIX_DIR / "fig_appendix_reliability_comparable_arm.png"
    _fig_comp.savefig(_out_comp, dpi=150, bbox_inches="tight")
    plt.close(_fig_comp)
    print(f"[G9.5] Reliability diagram (comparable arm) -> {_out_comp.name}")

    # Dynamic arm
    _fig_dyn = _build_reliability_figure(_DYNAMIC_BIN_TABLES, "Dynamic Arm", _dcon)
    _out_dyn = PAPER_APPENDIX_DIR / "fig_appendix_reliability_dynamic_arm.png"
    _fig_dyn.savefig(_out_dyn, dpi=150, bbox_inches="tight")
    plt.close(_fig_dyn)
    print(f"[G9.5] Reliability diagram (dynamic arm)     -> {_out_dyn.name}")

except Exception as _e_rel:
    print(f"[G9.5] WARNING: reliability diagram generation failed: {_e_rel}")

print(f"[END] G9.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")



shutdown_duckdb_connection_from_globals(globals())

print(f"[END] G10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 136
from datetime import datetime as _dt
print(f"[START] G8.0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path
import pandas as pd

required_names = ["OUTPUT_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

paper_main_dir = OUTPUT_DIR / "paper_main"
benchmark_df = pd.read_csv(paper_main_dir / "table_paper_main_benchmark_family_expanded.csv")
benchmark_support_df = pd.read_csv(paper_main_dir / "table_figure1_benchmark_family_expanded_summary.csv")
ablation_df = pd.read_csv(paper_main_dir / "table_paper_ablation_summary.csv")
ablation_support_df = pd.read_csv(paper_main_dir / "table_figure2_ablation_delta_summary.csv")

print("\nExpanded benchmark table frozen for the manuscript:")
display(benchmark_df)

print("\nBenchmark support table used by Figure 1:")
display(benchmark_support_df)

print("\nAblation summary frozen for the manuscript:")
display(ablation_df)

print("\nAblation support table used by Figure 2:")
display(ablation_support_df)

leader = benchmark_df.sort_values("IBS", ascending=True).iloc[0]
runner_up = benchmark_df.sort_values("IBS", ascending=True).iloc[1]
best_td = benchmark_df.sort_values("TD Concordance", ascending=False).iloc[0]
linear_row = benchmark_df.loc[benchmark_df["Model"] == "Linear Discrete-Time Hazard"].iloc[0]
neural_row = benchmark_df.loc[benchmark_df["Model"] == "Neural Discrete-Time Survival"].iloc[0]

print("\nEvidence summary:")
print(
    f"- IBS leader: {leader['Model']} ({leader['IBS']:.4f}); runner-up: {runner_up['Model']} ({runner_up['IBS']:.4f})."
)
print(
    f"- Best TD Concordance: {best_td['Model']} ({best_td['TD Concordance']:.4f})."
)
print(
    f"- Linear vs Neural TD Concordance: {linear_row['TD Concordance']:.4f} vs {neural_row['TD Concordance']:.4f}."
)
print(
    f"- Temporal-vs-static IBS ratios: min={ablation_df['IBS ratio'].min():.3f}, max={ablation_df['IBS ratio'].max():.3f}."
)

print(f"[END] G8.0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 138
from datetime import datetime as _dt
print(f"[START] G7.24 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

tex_source = TEX_PATH.read_text(encoding="utf-8") if TEX_PATH.exists() else ""
parsed_tex_contract = extract_tex_contract(tex_source)
required_tex_labels = [
    "tab:main_benchmark",
    "fig:benchmark_tuned_comparison",
    "tab:ablation_summary",
    "fig:ablation_impact",
    "tab:explainability_summary",
    "fig:explainability_block_dominance",
    "tab:calibration_summary",
    "tab:appendix_protocol_audit",
    "tab:appendix_preproc_tuning_audit",
    "tab:appendix_bootstrap_uncertainty",
    "tab:appendix_split_context_audit",
    "tab:appendix_cox_ph_summary",
    "fig:appendix_cox_ph_diagnostics",
    "tab:appendix_window_sensitivity",
]

tex_contract = {}
for label in required_tex_labels:
    fallback_file_name = f"{label.replace(':', '_')}.{'png' if label.startswith('fig:') else 'csv'}"
    tex_contract[label] = parsed_tex_contract.get(
        label,
        {
            "label": label,
            "caption": label,
            "artifact_type": "figure" if label.startswith("fig:") else "table",
            "scope": "appendix" if "appendix" in label else "main",
            "graphic_path": "",
            "file_name": fallback_file_name,
        },
    )

tex_asset_rows = []
supporting_asset_rows = []
manifest_rows = []

leaderboard_source_candidates = [TABLES_DIR / "table_benchmark_leaderboard_main.csv"]
brier_wide_source_candidates = [TABLES_DIR / "table_benchmark_brier_by_horizon_wide.csv"]
registry_source_candidates = [
    TABLES_DIR / "table_benchmark_family_membership_audit.csv",
    TABLES_DIR / "table_benchmark_model_registry.csv",
]

leaderboard_df, _, _ = load_table_duckdb_first(
    leaderboard_source_candidates,
    preferred_table_names=["table_benchmark_leaderboard_main"],
)
brier_wide_df, _, _ = load_table_duckdb_first(
    brier_wide_source_candidates,
    preferred_table_names=["table_benchmark_brier_by_horizon_wide"],
)
registry_df, _, _ = load_table_duckdb_first(
    registry_source_candidates,
    preferred_table_names=["table_benchmark_family_membership_audit", "table_benchmark_model_registry"],
)

benchmark_paper_df = None
benchmark_figure_df = None
if leaderboard_df is not None and brier_wide_df is not None:
    leaderboard_model_col = pick_column(leaderboard_df, ["display_name", "Model", "model", "model_name"])
    leaderboard_family_col = pick_column(leaderboard_df, ["family_name", "Family", "family", "model_family"])
    ibs_col = pick_column(leaderboard_df, ["ibs", "IBS", "integrated_brier_score"])
    cindex_col = pick_column(leaderboard_df, ["c_index", "TD Concordance", "td_concordance", "concordance_td"])
    brier_model_col = pick_column(brier_wide_df, ["display_name", "Model", "model", "model_name"])
    benchmark_paper_df = leaderboard_df[[leaderboard_model_col, leaderboard_family_col, ibs_col, cindex_col]].copy()
    benchmark_paper_df.columns = ["Model", "Family", "IBS", "TD Concordance"]
    benchmark_paper_df["Model"] = benchmark_paper_df["Model"].map(normalize_model_names)
    benchmark_paper_df["Family"] = benchmark_paper_df["Family"].map(normalize_family_names)
    for horizon in [10, 20, 30]:
        brier_column = pick_column(
            brier_wide_df,
            [
                f"brier_h{horizon}",
                f"brier_hh{horizon}",
                f"Brier@{horizon}",
                f"brier@{horizon}",
                f"brier_{horizon}",
                f"brier_at_{horizon}",
            ],
        )
        brier_lookup = brier_wide_df[[brier_model_col, brier_column]].copy()
        brier_lookup.columns = ["Model", f"Brier@{horizon}"]
        brier_lookup["Model"] = brier_lookup["Model"].map(normalize_model_names)
        benchmark_paper_df = benchmark_paper_df.merge(brier_lookup, on="Model", how="left")
    benchmark_paper_df = benchmark_paper_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
    benchmark_figure_df = benchmark_paper_df[["Model", "Family", "IBS", "TD Concordance"]].copy()
else:
    benchmark_paper_df, benchmark_figure_df = build_expanded_benchmark_tables_from_stage_catalog()

if benchmark_paper_df is None or benchmark_figure_df is None:
    benchmark_paper_df = build_tex_main_benchmark_table()
    benchmark_figure_df = benchmark_paper_df[["Model", "Family", "IBS", "TD Concordance"]].copy()

benchmark_table_output = PAPER_MAIN_DIR / "table_paper_main_benchmark_family_expanded.csv"
export_direct_table_asset(
    "tab:main_benchmark",
    benchmark_table_output,
    source_candidates=leaderboard_source_candidates + brier_wide_source_candidates,
    dataframe=benchmark_paper_df,
    note="Curated expanded benchmark table exported under the manuscript contract.",
)
benchmark_support_output = PAPER_MAIN_DIR / "table_figure1_benchmark_family_expanded_summary.csv"
export_supporting_table_asset(
    "fig:benchmark_tuned_comparison",
    benchmark_support_output,
    source_candidates=leaderboard_source_candidates + brier_wide_source_candidates,
    dataframe=benchmark_figure_df,
    note="Support table for the expanded benchmark comparison figure.",
)

benchmark_membership_df = None
if registry_df is not None:
    membership_columns = {
        "model_name": pick_column(registry_df, ["model_name", "model_key", "Model", "model"], required=False),
        "display_name": pick_column(registry_df, ["display_name", "Model", "model"], required=False),
        "family_name": pick_column(registry_df, ["family_name", "Family", "family"], required=False),
        "model_family": pick_column(registry_df, ["model_family", "family", "family_detail"], required=False),
        "input_representation": pick_column(registry_df, ["input_representation", "data_level", "input_level"], required=False),
        "training_contract": pick_column(registry_df, ["training_contract", "raw_input_design", "contract"], required=False),
        "model_type": pick_column(registry_df, ["model_type", "model_class", "type"], required=False),
        "comparability_rule": pick_column(registry_df, ["comparability_rule", "comparability_note", "notes"], required=False),
        "execution_status": pick_column(registry_df, ["execution_status", "status"], required=False),
    }
    selected = [source_name for source_name in membership_columns.values() if source_name is not None]
    benchmark_membership_df = registry_df[selected].copy()
    benchmark_membership_df.columns = [target_name for target_name, source_name in membership_columns.items() if source_name is not None]
    if "display_name" in benchmark_membership_df.columns:
        benchmark_membership_df["display_name"] = benchmark_membership_df["display_name"].map(normalize_model_names)
    if "family_name" in benchmark_membership_df.columns:
        benchmark_membership_df["family_name"] = benchmark_membership_df["family_name"].map(normalize_family_names)
    benchmark_membership_df = benchmark_membership_df.sort_values(
        by=benchmark_membership_df.columns.intersection(["display_name"]).tolist() or benchmark_membership_df.columns.tolist()[:1],
        kind="mergesort",
    ).reset_index(drop=True)
else:
    benchmark_membership_df = build_benchmark_membership_from_stage_inventory()

if benchmark_membership_df is not None:
    benchmark_membership_output = PAPER_APPENDIX_DIR / "table_paper_appendix_benchmark_family_membership_audit.csv"
    benchmark_membership_output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_membership_df.to_csv(benchmark_membership_output, index=False)
    if "materialize_dataframe" in globals() and "con" in globals():
        materialize_dataframe(con, benchmark_membership_df, "table_paper_appendix_benchmark_family_membership_audit", "G7.24")
    print("Exported benchmark family membership audit:")
    print("-", benchmark_membership_output)

print(f"[END] G7.24 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 140
from datetime import datetime as _dt
print(f"[START] G7.26 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

export_direct_figure_asset(
    "fig:benchmark_tuned_comparison",
    PAPER_MAIN_DIR / "fig_paper_01_benchmark_family_expanded_comparison.png",
    generator=generate_benchmark_figure,
    note="Generated from the expanded benchmark table frozen for the manuscript.",
)

ablation_source = resolve_first_existing([TABLES_DIR / "table_ablation_summary_by_model.csv", TABLES_DIR / "table_p31_paper_ablation_summary.csv"])
ablation_table_df = None
if ablation_source is not None:
    ablation_raw_df = load_duckdb_table_optional(infer_table_name_from_pathlike(ablation_source))
    if ablation_raw_df is None:
        ablation_raw_df = pd.read_csv(ablation_source)

    model_col = pick_column(ablation_raw_df, ["display_name", "Model", "model", "model_name"])
    family_col = pick_column(ablation_raw_df, ["family_name", "Family", "family", "model_family"])
    delta_ibs_static_col = pick_column(ablation_raw_df, ["Delta IBS static", "delta_ibs_static", "ibs_delta_static", "delta_ibs_without_static"], required=False)
    delta_ibs_temporal_col = pick_column(ablation_raw_df, ["Delta IBS temporal", "delta_ibs_temporal", "ibs_delta_temporal", "delta_ibs_without_temporal"], required=False)
    delta_td_static_col = pick_column(ablation_raw_df, ["Delta TD concordance static", "delta_td_concordance_static", "delta_cindex_static"], required=False)
    delta_td_temporal_col = pick_column(ablation_raw_df, ["Delta TD concordance temporal", "delta_td_concordance_temporal", "delta_cindex_temporal"], required=False)
    ibs_ratio_col = pick_column(ablation_raw_df, ["IBS ratio", "ibs_ratio", "temporal_static_ibs_ratio"], required=False)

    if None in [delta_ibs_static_col, delta_ibs_temporal_col, delta_td_static_col, delta_td_temporal_col, ibs_ratio_col]:
        ablation_table_df = build_tex_ablation_table()
    else:
        ablation_table_df = ablation_raw_df[[model_col, family_col, delta_ibs_static_col, delta_ibs_temporal_col, delta_td_static_col, delta_td_temporal_col, ibs_ratio_col]].copy()
        ablation_table_df.columns = ["Model", "Family", "Delta IBS static", "Delta IBS temporal", "Delta TD concordance static", "Delta TD concordance temporal", "IBS ratio"]
        ablation_table_df["Model"] = ablation_table_df["Model"].map(normalize_model_names)
        ablation_table_df["Family"] = ablation_table_df["Family"].map(normalize_family_names)
        ablation_table_df = ablation_table_df.sort_values(by="Model", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)
else:
    ablation_table_df = build_tex_ablation_table()

export_direct_table_asset(
    "tab:ablation_summary",
    PAPER_MAIN_DIR / "table_paper_ablation_summary.csv",
    source_candidates=[ablation_source] if ablation_source is not None else [],
    dataframe=ablation_table_df,
    note="Curated ablation table exported under the manuscript contract.",
)
export_supporting_table_asset(
    "fig:ablation_impact",
    PAPER_MAIN_DIR / "table_figure2_ablation_delta_summary.csv",
    dataframe=ablation_table_df,
    note="Support table for the ablation impact figure.",
)

print(f"[END] G7.26 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 142
from datetime import datetime as _dt
print(f"[START] G8.0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path
import pandas as pd

required_names = ["OUTPUT_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous cells: "
        + ", ".join(missing_names)
        + ". Please run prior setup cells first."
    )

paper_main_dir = OUTPUT_DIR / "paper_main"
paper_appendix_dir = OUTPUT_DIR / "paper_appendix"
benchmark_df = pd.read_csv(paper_main_dir / "table_paper_main_benchmark_family_expanded.csv")
benchmark_support_df = pd.read_csv(paper_main_dir / "table_figure1_benchmark_family_expanded_summary.csv")
benchmark_membership_df = pd.read_csv(paper_appendix_dir / "table_paper_appendix_benchmark_family_membership_audit.csv")
ablation_df = pd.read_csv(paper_main_dir / "table_paper_ablation_summary.csv")
ablation_support_df = pd.read_csv(paper_main_dir / "table_figure2_ablation_delta_summary.csv")

print("\nExpanded benchmark leaderboard frozen for the manuscript:")
display(benchmark_df)

print("\nExpanded benchmark support table used by Figure 1:")
display(benchmark_support_df)

print("\nBenchmark family membership audit:")
display(benchmark_membership_df)

print("\nAblation summary frozen for the manuscript:")
display(ablation_df)

print("\nAblation support table used by Figure 2:")
display(ablation_support_df)

leader = benchmark_df.sort_values("IBS", ascending=True).iloc[0]
runner_up = benchmark_df.sort_values("IBS", ascending=True).iloc[1]
best_td = benchmark_df.sort_values("TD Concordance", ascending=False).iloc[0]
family_counts = benchmark_df.groupby("Family", dropna=False).size().to_dict()

print("\nEvidence summary:")
print(
    f"- IBS leader: {leader['Model']} ({leader['IBS']:.4f}); runner-up: {runner_up['Model']} ({runner_up['IBS']:.4f})."
)
print(
    f"- Best TD Concordance: {best_td['Model']} ({best_td['TD Concordance']:.4f})."
)
print(
    f"- Models frozen by family: {family_counts}."
)
print(
    f"- Temporal-vs-static IBS ratios: min={ablation_df['IBS ratio'].min():.3f}, max={ablation_df['IBS ratio'].max():.3f}."
)

print(f"[END] G8.0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 145
from datetime import datetime as _dt
print(f"[START] G11.0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path

if "g11_display_heading" not in globals():
    def g11_display_heading(title: str, level: int = 3) -> None:
        display(Markdown(f"{'#' * level} {title}"))

if "g11_display_bullets" not in globals():
    def g11_display_bullets(lines: list[str]) -> None:
        clean_lines = [line for line in lines if str(line).strip()]
        display(Markdown("\n".join(f"- {line}" for line in clean_lines)))

paper_main_dir_value = globals().get("G11_PAPER_MAIN_DIR", globals().get("PAPER_MAIN_DIR"))
paper_appendix_dir_value = globals().get("G11_PAPER_APPENDIX_DIR", globals().get("PAPER_APPENDIX_DIR"))
if paper_main_dir_value is None or paper_appendix_dir_value is None:
    raise NameError("Neither G11 nor shared paper output directories are available in the current runtime.")
paper_main_dir = Path(paper_main_dir_value)
paper_appendix_dir = Path(paper_appendix_dir_value)

required_artifacts = [
    ("paper_main", "table", "table_paper_main_benchmark_family_expanded.csv", "Expanded benchmark table displayed before narrative."),
    ("paper_main", "table", "table_figure1_benchmark_family_expanded_summary.csv", "Analytical support table for the expanded Figure 1 benchmark view."),
    ("paper_main", "figure", "fig_paper_01_benchmark_family_expanded_comparison.png", "Expanded main benchmark comparison figure."),
    ("paper_main", "table", "table_paper_ablation_summary.csv", "Canonical ablation table."),
    ("paper_main", "table", "table_figure2_ablation_delta_summary.csv", "Analytical support table for Figure 2."),
    ("paper_main", "figure", "fig_paper_02_ablation_impact.png", "Ablation figure."),
    ("paper_main", "table", "table_paper_explainability_summary.csv", "Canonical explainability table."),
    ("paper_main", "table", "table_figure3_explainability_block_summary_normalized.csv", "Analytical support table for Figure 3."),
    ("paper_main", "figure", "fig_paper_03_explainability_block_dominance_normalized.png", "Explainability figure."),
    ("paper_main", "table", "table_paper_calibration_summary_tuned_models.csv", "Canonical calibration table."),
    ("paper_appendix", "table", "table_paper_appendix_benchmark_family_membership_audit.csv", "Benchmark family membership audit."),
    ("paper_appendix", "table", "table_paper_appendix_evaluation_protocol_audit.csv", "Evaluation protocol audit."),
    ("paper_appendix", "table", "table_paper_appendix_preprocessing_and_tuning_audit.csv", "Preprocessing and tuning audit."),
    ("paper_appendix", "table", "table_appendix_bootstrap_uncertainty_compact.csv", "Compact bootstrap uncertainty table."),
    ("paper_appendix", "table", "table_paper_appendix_bootstrap_inferential_scope_summary.csv", "Bootstrap inferential scope table."),
    ("paper_appendix", "table", "table_paper_appendix_split_context_audit.csv", "Split and contextual overlap audit."),
    ("paper_appendix", "table", "table_paper_appendix_discrete_time_diagnostic_summary.csv", "Discrete-time diagnostic summary."),
    ("paper_appendix", "table", "table_paper_appendix_discrete_time_hypothesis_audit.csv", "Discrete-time hypothesis audit."),
    ("paper_appendix", "table", "table_paper_appendix_cox_ph_global_summary.csv", "Global Cox PH summary."),
    ("paper_appendix", "table", "table_paper_appendix_cox_ph_audit.csv", "Cox PH audit details."),
    ("paper_appendix", "table", "table_paper_appendix_ph_scope_boundary.csv", "PH scope boundary table."),
    ("paper_appendix", "figure", "fig_appendix_cox_ph_diagnostics.png", "Appendix PH diagnostic figure."),
]

checklist_rows = []
for scope, artifact_kind, file_name, purpose in required_artifacts:
    base_dir = paper_main_dir if scope == "paper_main" else paper_appendix_dir
    artifact_path = base_dir / file_name
    checklist_rows.append({
        "scope": scope,
        "artifact_kind": artifact_kind,
        "file_name": file_name,
        "exists": artifact_path.exists(),
        "purpose": purpose,
    })

checklist_df = pd.DataFrame(checklist_rows)
g11_display_heading("Canonical manuscript checklist", level=3)
display(checklist_df)

present_count = int(checklist_df["exists"].sum())
missing_df = checklist_df.loc[~checklist_df["exists"]].copy()
g11_display_heading("Checklist interpretation", level=4)
summary_lines = [
    f"The checklist currently finds {present_count} of {len(checklist_df)} required manuscript-facing artifacts already frozen in paper_main or paper_appendix.",
    "Passing the checklist still requires more than file existence: each table must be printed in notebook output, each figure must be accompanied by numerical support, and each narrative block must stay tied to the displayed values.",
]
if missing_df.empty:
    summary_lines.append("No required artifact is currently missing from the canonical paper-facing directories.")
else:
    summary_lines.append("Missing artifacts that still block a fully closed manuscript-facing checklist: " + ", ".join(missing_df["file_name"].astype(str).tolist()) + ".")
g11_display_bullets(summary_lines)

g11_display_heading("Narrative quality requirements", level=4)
g11_display_bullets([
    "Do not write manuscript-facing text from generic templates.",
    "State who leads, who is closest, what the gap magnitude is, and where the hierarchy becomes metric-dependent.",
    "For every figure, show the support table or the equivalent analytical summary before the interpretation.",
    "State what the displayed evidence does not justify whenever the output only supports a cautious claim.",
])

print(f"[END] G11.0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 146
from datetime import datetime as _dt
print(f"[START] G7.29b - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path

paper_main_dir = Path(globals().get("PAPER_MAIN_DIR", Path(globals().get("OUTPUT_DIR", "outputs_benchmark_survival")) / "paper_main"))
output_path = paper_main_dir / "fig_paper_03_explainability_block_dominance_normalized.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

was_generated = generate_explainability_figure(output_path)
if not was_generated:
    raise RuntimeError("The explainability figure could not be regenerated from the current notebook state.")

print("Explainability figure regenerated:")
print(output_path.resolve())
print(f"[END] G7.29b - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 148
from datetime import datetime as _dt
print(f"[START] G11.0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path

if "g11_display_heading" not in globals():
    def g11_display_heading(title: str, level: int = 3) -> None:
        display(Markdown(f"{'#' * level} {title}"))

if "g11_display_bullets" not in globals():
    def g11_display_bullets(lines: list[str]) -> None:
        clean_lines = [line for line in lines if str(line).strip()]
        display(Markdown("\n".join(f"- {line}" for line in clean_lines)))

paper_main_dir_value = globals().get("G11_PAPER_MAIN_DIR", globals().get("PAPER_MAIN_DIR"))
paper_appendix_dir_value = globals().get("G11_PAPER_APPENDIX_DIR", globals().get("PAPER_APPENDIX_DIR"))
output_dir_value = globals().get("G11_OUTPUT_DIR", globals().get("OUTPUT_DIR"))
if paper_main_dir_value is None or paper_appendix_dir_value is None or output_dir_value is None:
    raise NameError("Neither G11 nor shared paper output directories are available in the current runtime.")
G11_PAPER_MAIN_DIR = Path(paper_main_dir_value)
G11_PAPER_APPENDIX_DIR = Path(paper_appendix_dir_value)
G11_OUTPUT_DIR = Path(output_dir_value)
G11_CON = ensure_active_duckdb_connection()

if "g11_pick_column" not in globals():
    def g11_pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
        available = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in available:
                return available[key]
        if required:
            raise KeyError(f"None of the expected columns were found: {candidates}")
        return None

if "g11_load_table" not in globals():
    def g11_load_table(file_name: str, paper_scope: str | None = None, duckdb_table: str | None = None):
        search_paths = []
        if paper_scope == "main":
            search_paths.append(G11_PAPER_MAIN_DIR / file_name)
        elif paper_scope == "appendix":
            search_paths.append(G11_PAPER_APPENDIX_DIR / file_name)
        else:
            search_paths.extend([
                G11_PAPER_MAIN_DIR / file_name,
                G11_PAPER_APPENDIX_DIR / file_name,
            ])
        search_paths.append(TABLES_DIR / file_name)

        seen_paths = set()
        for path in search_paths:
            resolved = str(path)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if path.exists():
                return pd.read_csv(path)

        candidate_tables = []
        if duckdb_table:
            candidate_tables.append(duckdb_table)
        inferred_name = Path(file_name).stem
        candidate_tables.append(inferred_name)
        if inferred_name.endswith(".csv"):
            candidate_tables.append(inferred_name[:-4])

        seen_tables = set()
        for table_name in candidate_tables:
            if not table_name or table_name in seen_tables:
                continue
            seen_tables.add(table_name)
            loaded_df = load_duckdb_table_optional(table_name)
            if loaded_df is not None:
                return loaded_df
        return None

if "g11_export_appendix" not in globals():
    def g11_export_appendix(df: pd.DataFrame, file_name: str) -> Path:
        output_path = G11_PAPER_APPENDIX_DIR / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

if "g11_export_main" not in globals():
    def g11_export_main(df: pd.DataFrame, file_name: str) -> Path:
        output_path = G11_PAPER_MAIN_DIR / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

if "g11_materialize_claim_table" not in globals():
    def g11_materialize_claim_table(df: pd.DataFrame, table_name: str, stage_id: str) -> None:
        if "materialize_dataframe" in globals() and "con" in globals():
            materialize_dataframe(con, df, table_name, stage_id)

if "g11_display_figure" not in globals():
    def g11_display_figure(path: Path, title: str | None = None) -> None:
        figure_path = Path(path)
        if title:
            g11_display_heading(title, level=5)
        if not figure_path.exists():
            raise FileNotFoundError(f"Expected figure not found: {figure_path}")
        display(Image(filename=str(figure_path)))
        print(f"- {figure_path}")

required_artifacts = [
    ("paper_main", "table", "table_paper_main_benchmark_family_expanded.csv", "Main benchmark table displayed before narrative."),
    ("paper_main", "table", "table_figure1_benchmark_family_expanded_summary.csv", "Analytical support table for Figure 1."),
    ("paper_main", "table", "table_text_claim_main_benchmark_summary.csv", "Canonical numeric claim table for the main benchmark narrative."),
    ("paper_main", "figure", "fig_paper_01_benchmark_family_expanded_comparison.png", "Main benchmark comparison figure."),
    ("paper_main", "table", "table_paper_ablation_summary.csv", "Canonical ablation table."),
    ("paper_main", "table", "table_figure2_ablation_delta_summary.csv", "Analytical support table for Figure 2."),
    ("paper_main", "table", "table_text_claim_ablation_summary.csv", "Canonical numeric claim table for the ablation narrative."),
    ("paper_main", "figure", "fig_paper_02_ablation_impact.png", "Ablation figure."),
    ("paper_main", "table", "table_paper_explainability_summary.csv", "Canonical explainability table."),
    ("paper_main", "table", "table_figure3_explainability_block_summary_normalized.csv", "Analytical support table for Figure 3."),
    ("paper_main", "figure", "fig_paper_03_explainability_block_dominance_normalized.png", "Explainability figure."),
    ("paper_main", "table", "table_paper_calibration_summary_tuned_models.csv", "Canonical calibration table."),
    ("paper_appendix", "table", "table_paper_appendix_evaluation_protocol_audit.csv", "Evaluation protocol audit."),
    ("paper_appendix", "table", "table_paper_appendix_preprocessing_and_tuning_audit.csv", "Preprocessing and tuning audit."),
    ("paper_appendix", "table", "table_appendix_bootstrap_uncertainty_compact.csv", "Compact bootstrap uncertainty table."),
    ("paper_appendix", "table", "table_paper_appendix_bootstrap_inferential_scope_summary.csv", "Bootstrap inferential scope table."),
    ("paper_appendix", "table", "table_paper_appendix_split_context_audit.csv", "Split and contextual overlap audit."),
    ("paper_appendix", "table", "table_paper_appendix_split_context_claims.csv", "Canonical numeric claim table for the split/context narrative."),
    ("paper_appendix", "table", "table_paper_appendix_weighting_sensitivity_summary.csv", "Dynamic-arm weighting sensitivity summary exported from D5.16."),
    ("paper_appendix", "table", "table_paper_appendix_comparable_window_champions.csv", "Comparable-arm champions by early-window sensitivity."),
    ("paper_appendix", "table", "table_paper_appendix_comparable_window_leadership_stability.csv", "Comparable-arm leadership stability across sensitivity windows."),
    ("paper_appendix", "table", "table_paper_appendix_comparable_window_claims.csv", "Canonical claim table for comparable-arm window sensitivity."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_parity_protocol.csv", "Cross-arm parity protocol exported from D5.16."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_execution_scope.csv", "Cross-arm execution scope exported from D5.16."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_decision_summary.csv", "Cross-arm decision summary exported from D5.16."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_window_champions.csv", "Cross-arm champions by sensitivity window."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_window_leadership_stability.csv", "Cross-arm leadership stability across windows."),
    ("paper_appendix", "table", "table_paper_appendix_cross_arm_claims.csv", "Canonical claim table for cross-arm parity interpretation."),
    ("paper_appendix", "table", "table_paper_appendix_discrete_time_diagnostic_summary.csv", "Discrete-time diagnostic summary."),
    ("paper_appendix", "table", "table_paper_appendix_discrete_time_hypothesis_audit.csv", "Discrete-time hypothesis audit."),
    ("paper_appendix", "table", "table_paper_appendix_discrete_time_bridge_claims.csv", "Canonical numeric claim table for the discrete-time bridge narrative."),
    ("paper_appendix", "table", "table_paper_appendix_cox_ph_global_summary.csv", "Global Cox PH summary."),
    ("paper_appendix", "table", "table_paper_appendix_cox_ph_audit.csv", "Cox PH audit details."),
    ("paper_appendix", "table", "table_paper_appendix_ph_scope_boundary.csv", "PH scope boundary table."),
    ("paper_appendix", "table", "table_paper_appendix_cox_ph_claims.csv", "Canonical numeric claim table for the PH narrative."),
    ("paper_appendix", "figure", "fig_appendix_cox_ph_diagnostics.png", "Appendix PH diagnostic figure."),
]

checklist_rows = []
for scope, artifact_kind, file_name, purpose in required_artifacts:
    base_dir = G11_PAPER_MAIN_DIR if scope == "paper_main" else G11_PAPER_APPENDIX_DIR
    artifact_path = base_dir / file_name
    checklist_rows.append({
        "scope": scope,
        "artifact_kind": artifact_kind,
        "file_name": file_name,
        "exists": artifact_path.exists(),
        "purpose": purpose,
    })

checklist_df = pd.DataFrame(checklist_rows)
g11_display_heading("Canonical manuscript checklist", level=3)
display(checklist_df)

present_count = int(checklist_df["exists"].sum())
missing_df = checklist_df.loc[~checklist_df["exists"]].copy()
g11_display_heading("Checklist interpretation", level=4)
summary_lines = [
    f"The checklist currently finds {present_count} of {len(checklist_df)} required manuscript-facing artifacts already frozen in paper_main or paper_appendix.",
    "Passing the checklist still requires more than file existence: each table must be printed in notebook output, each figure must be accompanied by numerical support, and each narrative block must stay tied to the displayed values.",
]
if missing_df.empty:
    summary_lines.append("No required artifact is currently missing from the canonical paper-facing directories.")
else:
    summary_lines.append("Missing artifacts that still block a fully closed manuscript-facing checklist: " + ", ".join(missing_df["file_name"].astype(str).tolist()) + ".")
g11_display_bullets(summary_lines)

g11_display_heading("Narrative quality requirements", level=4)
g11_display_bullets([
    "Do not write manuscript-facing text from generic templates.",
    "State who leads, who is closest, what the gap magnitude is, and where the hierarchy becomes metric-dependent.",
    "For every figure, show the support table or the equivalent analytical summary before the interpretation.",
    "State what the displayed evidence does not justify whenever the output only supports a cautious claim.",
])

print(f"[END] G11.0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 150
from datetime import datetime as _dt
print(f"[START] G11.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

benchmark_df = g11_load_table("table_paper_main_benchmark_family_expanded.csv", paper_scope="main")
benchmark_support_df = g11_load_table("table_figure1_benchmark_family_expanded_summary.csv", paper_scope="main")
benchmark_membership_df = g11_load_table("table_paper_appendix_benchmark_family_membership_audit.csv", paper_scope="appendix")
ablation_df = g11_load_table("table_paper_ablation_summary.csv", paper_scope="main")
ablation_support_df = g11_load_table("table_figure2_ablation_delta_summary.csv", paper_scope="main")
explainability_df = g11_load_table("table_paper_explainability_summary.csv", paper_scope="main")
block_df = g11_load_table("table_figure3_explainability_block_summary_normalized.csv", paper_scope="main")
calibration_df = g11_load_table("table_paper_calibration_summary_tuned_models.csv", paper_scope="main")

required_frames = {
    "benchmark_df": benchmark_df,
    "benchmark_support_df": benchmark_support_df,
    "benchmark_membership_df": benchmark_membership_df,
    "ablation_df": ablation_df,
    "ablation_support_df": ablation_support_df,
    "explainability_df": explainability_df,
    "block_df": block_df,
    "calibration_df": calibration_df,
}
missing_frames = [name for name, frame in required_frames.items() if frame is None]
if missing_frames:
    raise ValueError(f"G11.1 is missing required canonical paper-facing artifacts: {missing_frames}")

g11_display_heading("Main benchmark comparison", level=3)
display(benchmark_df)
g11_display_heading("Figure 1 support table", level=4)
display(benchmark_support_df)
g11_display_figure(G11_PAPER_MAIN_DIR / "fig_paper_01_benchmark_family_expanded_comparison.png", "Figure 1 preview")
g11_display_heading("Benchmark family membership audit", level=4)
display(benchmark_membership_df)

ibs_col = g11_pick_column(benchmark_df, ["IBS"])
td_col = g11_pick_column(benchmark_df, ["TD Concordance", "c_index_td"])
model_col = g11_pick_column(benchmark_df, ["Model", "model"])
family_col = g11_pick_column(benchmark_df, ["Family", "family"])
brier_cols = [
    col
    for col in [
        g11_pick_column(benchmark_df, ["Brier@10"]),
        g11_pick_column(benchmark_df, ["Brier@20"]),
        g11_pick_column(benchmark_df, ["Brier@30"]),
    ]
    if col is not None
]
leaderboard_sorted = benchmark_df.sort_values(ibs_col, ascending=True).reset_index(drop=True)
td_sorted = benchmark_df.sort_values(td_col, ascending=False).reset_index(drop=True)
ibs_leader = leaderboard_sorted.iloc[0]
ibs_runner_up = leaderboard_sorted.iloc[1]
td_leader = td_sorted.iloc[0]
td_runner_up = td_sorted.iloc[1]
ibs_gap = float(ibs_runner_up[ibs_col] - ibs_leader[ibs_col])
td_gap = float(td_leader[td_col] - td_runner_up[td_col])
family_counts = benchmark_df.groupby(family_col, dropna=False).size().to_dict()

benchmark_claim_rows = [
    {
        "claim_section": "main_benchmark",
        "claim_id": "ibs_leader",
        "source_table": "table_paper_main_benchmark_family_expanded.csv",
        "related_tex_label": "tab:main_benchmark",
        "model_1": str(ibs_leader[model_col]),
        "model_2": str(ibs_runner_up[model_col]),
        "metric": "IBS",
        "horizon": pd.NA,
        "value_1": float(ibs_leader[ibs_col]),
        "value_2": float(ibs_runner_up[ibs_col]),
        "gap_value": float(ibs_gap),
        "claim_text": f"IBS leader {ibs_leader[model_col]} vs runner-up {ibs_runner_up[model_col]} with gap {ibs_gap:.4f}",
    },
    {
        "claim_section": "main_benchmark",
        "claim_id": "td_concordance_leader",
        "source_table": "table_paper_main_benchmark_family_expanded.csv",
        "related_tex_label": "tab:main_benchmark",
        "model_1": str(td_leader[model_col]),
        "model_2": str(td_runner_up[model_col]),
        "metric": "TD Concordance",
        "horizon": pd.NA,
        "value_1": float(td_leader[td_col]),
        "value_2": float(td_runner_up[td_col]),
        "gap_value": float(td_gap),
        "claim_text": f"TD concordance leader {td_leader[model_col]} vs runner-up {td_runner_up[model_col]} with gap {td_gap:.4f}",
    },
]
for family_name, family_count in family_counts.items():
    benchmark_claim_rows.append(
        {
            "claim_section": "main_benchmark",
            "claim_id": f"family_count_{family_name}",
            "source_table": "table_paper_main_benchmark_family_expanded.csv",
            "related_tex_label": "tab:main_benchmark",
            "model_1": str(family_name),
            "model_2": pd.NA,
            "metric": "family_count",
            "horizon": pd.NA,
            "value_1": float(family_count),
            "value_2": pd.NA,
            "gap_value": pd.NA,
            "claim_text": f"Frozen benchmark family count for {family_name} = {family_count}",
        }
    )

brier_lines = []
for brier_col in brier_cols:
    ranked = benchmark_df.sort_values(brier_col, ascending=True).reset_index(drop=True)
    horizon_best = ranked.iloc[0]
    horizon_runner_up = ranked.iloc[1]
    horizon_value = int(str(brier_col).split("@")[1]) if "@" in str(brier_col) else pd.NA
    horizon_gap = float(horizon_runner_up[brier_col] - horizon_best[brier_col])
    brier_lines.append(
        f"At {brier_col}, the best value is {horizon_best[model_col]} = {horizon_best[brier_col]:.4f}, followed by {horizon_runner_up[model_col]} = {horizon_runner_up[brier_col]:.4f}, a gap of {horizon_gap:.4f}."
    )
    benchmark_claim_rows.append(
        {
            "claim_section": "main_benchmark",
            "claim_id": f"brier_h{horizon_value}_leader",
            "source_table": "table_paper_main_benchmark_family_expanded.csv",
            "related_tex_label": "tab:main_benchmark",
            "model_1": str(horizon_best[model_col]),
            "model_2": str(horizon_runner_up[model_col]),
            "metric": "Brier",
            "horizon": horizon_value,
            "value_1": float(horizon_best[brier_col]),
            "value_2": float(horizon_runner_up[brier_col]),
            "gap_value": horizon_gap,
            "claim_text": f"Best Brier at horizon {horizon_value}: {horizon_best[model_col]} vs {horizon_runner_up[model_col]} with gap {horizon_gap:.4f}",
        }
    )

benchmark_claims_df = pd.DataFrame(benchmark_claim_rows)
benchmark_claims_path = g11_export_main(benchmark_claims_df, "table_text_claim_main_benchmark_summary.csv")
g11_materialize_claim_table(benchmark_claims_df, "table_text_claim_main_benchmark_summary", "G11.1")

g11_display_heading("Benchmark interpretation", level=4)
g11_display_bullets([
    f"The current IBS leader is {ibs_leader[model_col]} with IBS = {ibs_leader[ibs_col]:.4f}, followed by {ibs_runner_up[model_col]} at {ibs_runner_up[ibs_col]:.4f}; the IBS gap between them is {ibs_gap:.4f}.",
    f"The best time-dependent concordance is {td_leader[model_col]} with TD Concordance = {td_leader[td_col]:.4f}, followed by {td_runner_up[model_col]} at {td_runner_up[td_col]:.4f}; the TD gap is {td_gap:.4f}.",
    f"The frozen benchmark now carries the full fourteen-model predictive roster under two paper-facing families, with counts by family = {family_counts}.",
    *brier_lines,
    "The manuscript-facing takeaway is therefore a family-structured benchmark with enough within-family diversity to answer the limited-palette critique without opening a third representation contract.",
])

g11_display_heading("Ablation results: static versus temporal-behavioral signal", level=3)
display(ablation_df)
g11_display_heading("Figure 2 support table", level=4)
display(ablation_support_df)
g11_display_figure(G11_PAPER_MAIN_DIR / "fig_paper_02_ablation_impact.png", "Figure 2 preview")

ibs_ratio_col = g11_pick_column(ablation_df, ["IBS ratio"])
delta_temporal_col = g11_pick_column(ablation_df, ["Delta IBS temporal", "Delta IBS: temporal removed", "delta_ibs_temporal_removed", "Delta IBS temporal removed"])
delta_static_col = g11_pick_column(ablation_df, ["Delta IBS static", "Delta IBS: static removed", "delta_ibs_static_removed", "Delta IBS static removed"])
ablation_model_col = g11_pick_column(ablation_df, ["Model", "model"])
if ibs_ratio_col is None or delta_temporal_col is None or delta_static_col is None:
    raise ValueError(
        "Ablation summary is missing one of the expected columns for the canonical narrative."
    )

max_ratio_row = ablation_df.sort_values(ibs_ratio_col, ascending=False).iloc[0]
min_ratio_row = ablation_df.sort_values(ibs_ratio_col, ascending=True).iloc[0]
mean_temporal_loss = float(pd.to_numeric(ablation_df[delta_temporal_col], errors="coerce").mean())
mean_static_loss = float(pd.to_numeric(ablation_df[delta_static_col], errors="coerce").mean())

ablation_claims_df = pd.DataFrame([
    {
        "claim_section": "ablation",
        "claim_id": "max_ibs_ratio",
        "source_table": "table_paper_ablation_summary.csv",
        "related_tex_label": "tab:ablation_summary",
        "model_1": str(max_ratio_row[ablation_model_col]),
        "model_2": pd.NA,
        "metric": "IBS ratio",
        "value_1": float(max_ratio_row[ibs_ratio_col]),
        "value_2": pd.NA,
        "gap_value": pd.NA,
        "claim_text": f"Largest temporal-vs-static IBS ratio = {float(max_ratio_row[ibs_ratio_col]):.4f} for {max_ratio_row[ablation_model_col]}",
    },
    {
        "claim_section": "ablation",
        "claim_id": "min_ibs_ratio",
        "source_table": "table_paper_ablation_summary.csv",
        "related_tex_label": "tab:ablation_summary",
        "model_1": str(min_ratio_row[ablation_model_col]),
        "model_2": pd.NA,
        "metric": "IBS ratio",
        "value_1": float(min_ratio_row[ibs_ratio_col]),
        "value_2": pd.NA,
        "gap_value": pd.NA,
        "claim_text": f"Smallest temporal-vs-static IBS ratio = {float(min_ratio_row[ibs_ratio_col]):.4f} for {min_ratio_row[ablation_model_col]}",
    },
    {
        "claim_section": "ablation",
        "claim_id": "mean_temporal_loss",
        "source_table": "table_paper_ablation_summary.csv",
        "related_tex_label": "tab:ablation_summary",
        "model_1": pd.NA,
        "model_2": pd.NA,
        "metric": "mean_delta_ibs_temporal_removed",
        "value_1": mean_temporal_loss,
        "value_2": pd.NA,
        "gap_value": pd.NA,
        "claim_text": f"Average IBS change after removing temporal signal = {mean_temporal_loss:.4f}",
    },
    {
        "claim_section": "ablation",
        "claim_id": "mean_static_loss",
        "source_table": "table_paper_ablation_summary.csv",
        "related_tex_label": "tab:ablation_summary",
        "model_1": pd.NA,
        "model_2": pd.NA,
        "metric": "mean_delta_ibs_static_removed",
        "value_1": mean_static_loss,
        "value_2": pd.NA,
        "gap_value": pd.NA,
        "claim_text": f"Average IBS change after removing static structure = {mean_static_loss:.4f}",
    },
])
ablation_claims_path = g11_export_main(ablation_claims_df, "table_text_claim_ablation_summary.csv")
g11_materialize_claim_table(ablation_claims_df, "table_text_claim_ablation_summary", "G11.1")

g11_display_heading("Ablation interpretation", level=4)
g11_display_bullets([
    f"Across the exported ablation table, the largest temporal-versus-static IBS ratio is {float(max_ratio_row[ibs_ratio_col]):.4f} for {max_ratio_row[ablation_model_col]}, while the smallest ratio is {float(min_ratio_row[ibs_ratio_col]):.4f} for {min_ratio_row[ablation_model_col]}.",
    f"On average, removing temporal signal changes IBS by {mean_temporal_loss:.4f}, compared with {mean_static_loss:.4f} when removing static structure.",
    "The manuscript-safe interpretation remains that temporal-behavioral information is the dominant predictive block, even after the benchmark roster is expanded.",
])

print(f"[END] G11.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 152
from datetime import datetime as _dt
print(f"[START] G11.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

diagnostic_df = g11_load_table(
    file_name="table_paper_appendix_discrete_time_diagnostic_summary.csv",
    paper_scope="appendix",
    duckdb_table="table_d6_4_discrete_time_diagnostic_summary",
)
hypothesis_df = g11_load_table(
    file_name="table_paper_appendix_discrete_time_hypothesis_audit.csv",
    paper_scope="appendix",
    duckdb_table="table_d6_4_discrete_time_hypothesis_audit",
)

if diagnostic_df is None or hypothesis_df is None:
    print("D6.4 manuscript-facing artifacts are not yet fully available in paper_appendix or DuckDB.")
    print("Notebook G is therefore refusing to synthesize a final discrete-time bridge claim without the upstream diagnostic freeze.")
else:
    exported_diag_path = g11_export_appendix(diagnostic_df, "table_paper_appendix_discrete_time_diagnostic_summary.csv")
    exported_hyp_path = g11_export_appendix(hypothesis_df, "table_paper_appendix_discrete_time_hypothesis_audit.csv")

    g11_display_heading("Discrete-time diagnostic summary", level=3)
    display(diagnostic_df)
    g11_display_heading("Discrete-time hypothesis audit", level=4)
    display(hypothesis_df)

    model_key_col = g11_pick_column(diagnostic_df, ["model_key", "model"])
    row_auc_col = g11_pick_column(diagnostic_df, ["row_level_roc_auc"])
    ibs_col = g11_pick_column(diagnostic_df, ["ibs"])
    rows_per_enrollment_col = g11_pick_column(diagnostic_df, ["mean_rows_per_enrollment"])
    event_rate_col = g11_pick_column(diagnostic_df, ["row_event_rate_test"])

    linear_diag = diagnostic_df.loc[diagnostic_df[model_key_col].astype(str).str.contains("linear", case=False, na=False)].iloc[0]
    neural_diag = diagnostic_df.loc[diagnostic_df[model_key_col].astype(str).str.contains("neural", case=False, na=False)].iloc[0]
    cox_diag = diagnostic_df.loc[diagnostic_df[model_key_col].astype(str).str.contains("cox", case=False, na=False)].iloc[0]
    deepsurv_diag = diagnostic_df.loc[diagnostic_df[model_key_col].astype(str).str.contains("deepsurv", case=False, na=False)].iloc[0]

    takeaway_col = g11_pick_column(hypothesis_df, ["manuscript_safe_takeaway"])
    status_col = g11_pick_column(hypothesis_df, ["status"])
    tested_col = g11_pick_column(hypothesis_df, ["tested_component", "hypothesis_key"])

    supported_rows = hypothesis_df[hypothesis_df[status_col].astype(str).str.contains("supported|best_supported|residual", case=False, na=False)].copy()

    g11_display_heading("Discrete-time bridge interpretation", level=4)
    lines = [
        f"The weekly dynamic arms retain non-trivial row-level signal, with row-level ROC-AUC {float(linear_diag[row_auc_col]):.4f} for the linear branch and {float(neural_diag[row_auc_col]):.4f} for the neural branch.",
        f"Even so, their final IBS values remain worse than the comparable continuous-time arms: linear = {float(linear_diag[ibs_col]):.4f}, neural = {float(neural_diag[ibs_col]):.4f}, Cox = {float(cox_diag[ibs_col]):.4f}, DeepSurv = {float(deepsurv_diag[ibs_col]):.4f}.",
        f"The weekly arms also carry a higher aggregation burden, with mean rows per enrollment of {float(linear_diag[rows_per_enrollment_col]):.2f} and {float(neural_diag[rows_per_enrollment_col]):.2f}, while the row-level event rate remains sparse at {float(linear_diag[event_rate_col]):.4f} and {float(neural_diag[event_rate_col]):.4f}.",
        "The manuscript-facing reading is therefore not that weekly signal is absent, but that the current weekly person-period representation appears less efficient than the comparable early-window design under the present benchmark contract.",
    ]
    g11_display_bullets(lines)

    if not supported_rows.empty:
        g11_display_heading("Hypotheses currently safe to use in the manuscript", level=4)
        g11_display_bullets([
            f"{row[tested_col]}: {row[takeaway_col]}"
            for _, row in supported_rows.iterrows()
        ])

    discrete_time_claims_rows = [
        {
            "claim_section": "discrete_time_bridge",
            "claim_id": "linear_row_level_auc",
            "source_table": "table_paper_appendix_discrete_time_diagnostic_summary.csv",
            "related_tex_label": "appendix:discrete_time_bridge",
            "model_1": str(linear_diag[model_key_col]),
            "model_2": pd.NA,
            "metric": "row_level_roc_auc",
            "value_1": float(linear_diag[row_auc_col]),
            "value_2": pd.NA,
            "gap_value": pd.NA,
            "claim_text": f"Linear weekly row-level ROC-AUC = {float(linear_diag[row_auc_col]):.4f}",
        },
        {
            "claim_section": "discrete_time_bridge",
            "claim_id": "neural_row_level_auc",
            "source_table": "table_paper_appendix_discrete_time_diagnostic_summary.csv",
            "related_tex_label": "appendix:discrete_time_bridge",
            "model_1": str(neural_diag[model_key_col]),
            "model_2": pd.NA,
            "metric": "row_level_roc_auc",
            "value_1": float(neural_diag[row_auc_col]),
            "value_2": pd.NA,
            "gap_value": pd.NA,
            "claim_text": f"Neural weekly row-level ROC-AUC = {float(neural_diag[row_auc_col]):.4f}",
        },
        {
            "claim_section": "discrete_time_bridge",
            "claim_id": "ibs_cross_arm_snapshot",
            "source_table": "table_paper_appendix_discrete_time_diagnostic_summary.csv",
            "related_tex_label": "appendix:discrete_time_bridge",
            "model_1": str(linear_diag[model_key_col]),
            "model_2": str(cox_diag[model_key_col]),
            "metric": "ibs_cross_arm_snapshot",
            "value_1": float(linear_diag[ibs_col]),
            "value_2": float(cox_diag[ibs_col]),
            "gap_value": float(linear_diag[ibs_col] - cox_diag[ibs_col]),
            "claim_text": f"Linear vs Cox IBS snapshot = {float(linear_diag[ibs_col]):.4f} vs {float(cox_diag[ibs_col]):.4f}",
        },
        {
            "claim_section": "discrete_time_bridge",
            "claim_id": "weekly_rows_per_enrollment",
            "source_table": "table_paper_appendix_discrete_time_diagnostic_summary.csv",
            "related_tex_label": "appendix:discrete_time_bridge",
            "model_1": str(linear_diag[model_key_col]),
            "model_2": str(neural_diag[model_key_col]),
            "metric": "mean_rows_per_enrollment",
            "value_1": float(linear_diag[rows_per_enrollment_col]),
            "value_2": float(neural_diag[rows_per_enrollment_col]),
            "gap_value": pd.NA,
            "claim_text": f"Weekly rows per enrollment = {float(linear_diag[rows_per_enrollment_col]):.2f} (linear) and {float(neural_diag[rows_per_enrollment_col]):.2f} (neural)",
        },
    ]
    if not supported_rows.empty:
        for _, row in supported_rows.iterrows():
            discrete_time_claims_rows.append(
                {
                    "claim_section": "discrete_time_bridge",
                    "claim_id": f"supported_hypothesis_{str(row[tested_col]).strip()}",
                    "source_table": "table_paper_appendix_discrete_time_hypothesis_audit.csv",
                    "related_tex_label": "appendix:discrete_time_bridge",
                    "model_1": pd.NA,
                    "model_2": pd.NA,
                    "metric": "supported_hypothesis",
                    "value_1": pd.NA,
                    "value_2": pd.NA,
                    "gap_value": pd.NA,
                    "claim_text": f"{row[tested_col]}: {row[takeaway_col]}",
                }
            )
    discrete_time_claims_df = pd.DataFrame(discrete_time_claims_rows)
    discrete_time_claims_path = g11_export_appendix(discrete_time_claims_df, "table_paper_appendix_discrete_time_bridge_claims.csv")
    g11_materialize_claim_table(discrete_time_claims_df, "table_paper_appendix_discrete_time_bridge_claims", "G11.2")

    print("Exported manuscript-facing appendix tables:")
    print("-", exported_diag_path.resolve())
    print("-", exported_hyp_path.resolve())
    print("-", discrete_time_claims_path.resolve())

print(f"[END] G11.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 154
from datetime import datetime as _dt
print(f"[START] G11.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

canonical_protocol_rows = [
    {
        "component": "unit_of_analysis",
        "status": "defined",
        "protocol_value": "enrollment",
        "notes": "All final benchmark comparisons are reported at the enrollment level."
    },
    {
        "component": "event_definition",
        "status": "defined",
        "protocol_value": "withdrawn_with_valid_date_unregistration",
        "notes": "Observed withdrawal event with a valid time stamp."
    },
    {
        "component": "official_horizons",
        "status": "defined",
        "protocol_value": "10,20,30",
        "notes": "Shared benchmark horizons used for Brier, IBS, and calibration reporting."
    },
    {
        "component": "primary_discrimination_metric",
        "status": "defined",
        "protocol_value": "time_dependent_concordance",
        "notes": "The canonical discrimination metric is time-dependent concordance rather than a static concordance proxy."
    },
    {
        "component": "censoring_treatment",
        "status": "defined",
        "protocol_value": "ipcw_brier_ibs_with_kaplan_meier_censoring_estimator",
        "notes": "Brier score and IBS use inverse-probability-of-censoring weighting with the Kaplan-Meier estimator for the censoring distribution."
    },
    {
        "component": "dynamic_vs_static_prediction_rule",
        "status": "defined",
        "protocol_value": "dynamic_weekly_vs_early_window_comparable",
        "notes": "The weekly discrete-time arms update predictions over person-period rows, whereas the comparable continuous-time arms use early-window enrollment representations; cross-family comparison happens only after shared enrollment-level horizon reporting."
    },
    {
        "component": "leakage_prevention_rule",
        "status": "defined",
        "protocol_value": "no_enrollment_identity_leakage",
        "notes": "The split is enforced at the enrollment level with no identity leakage between train and test."
    },
    {
        "component": "split_scope_boundary",
        "status": "defined",
        "protocol_value": "shared_curricular_context_not_context_disjoint",
        "notes": "The benchmark generalizes across enrollments under shared curricular context rather than under context-disjoint transportability conditions."
    },
    {
        "component": "primary_calibration_metric",
        "status": "defined",
        "protocol_value": "weighted_absolute_calibration_gap_by_horizon",
        "notes": "The main calibration criterion is the weighted absolute calibration gap at each manuscript horizon."
    },
    {
        "component": "expanded_calibration_strengthening",
        "status": "defined",
        "protocol_value": "intercept_and_slope_by_horizon",
        "notes": "Intercept and slope are retained as strengthening diagnostics rather than as the primary benchmark ranking criterion."
    },
    {
        "component": "bootstrap_inferential_role",
        "status": "defined",
        "protocol_value": "ranking_support_not_formal_hypothesis_testing",
        "notes": "Bootstrap is used to assess how stable the exported ranking appears, not to claim a formal hypothesis-testing result."
    },
    {
        "component": "ph_scope_boundary",
        "status": "defined",
        "protocol_value": "formal_classical_ph_audit_for_cox_anchor_only",
        "notes": "Formal classical PH auditing is available for the comparable Cox anchor, but not in an equivalent form for DeepSurv."
    },
]
protocol_df = pd.DataFrame(canonical_protocol_rows)
exported_protocol_path = g11_export_appendix(protocol_df, "table_paper_appendix_evaluation_protocol_audit.csv")

preproc_df = g11_load_table("table_paper_appendix_preprocessing_and_tuning_audit.csv", paper_scope="appendix")
if preproc_df is None:
    preproc_df = g11_load_table("table_appendix_preprocessing_and_tuning_audit_compact.csv")
    exported_preproc_path = g11_export_appendix(preproc_df, "table_paper_appendix_preprocessing_and_tuning_audit.csv") if preproc_df is not None else None
else:
    exported_preproc_path = G11_PAPER_APPENDIX_DIR / "table_paper_appendix_preprocessing_and_tuning_audit.csv"
bootstrap_compact_df = g11_load_table("table_appendix_bootstrap_uncertainty_compact.csv", paper_scope="appendix")
bootstrap_scope_df = g11_load_table("table_paper_appendix_bootstrap_inferential_scope_summary.csv", paper_scope="appendix")
if bootstrap_scope_df is None:
    bootstrap_scope_df = g11_load_table("table_appendix_bootstrap_inferential_scope_summary.csv")
    exported_bootstrap_scope_path = g11_export_appendix(bootstrap_scope_df, "table_paper_appendix_bootstrap_inferential_scope_summary.csv") if bootstrap_scope_df is not None else None
else:
    exported_bootstrap_scope_path = G11_PAPER_APPENDIX_DIR / "table_paper_appendix_bootstrap_inferential_scope_summary.csv"
split_df = g11_load_table("table_paper_appendix_split_context_audit.csv", paper_scope="appendix")
cox_summary_df = g11_load_table("table_paper_appendix_cox_ph_global_summary.csv", paper_scope="appendix")
cox_audit_df = g11_load_table("table_paper_appendix_cox_ph_audit.csv", paper_scope="appendix")
ph_scope_df = g11_load_table("table_paper_appendix_ph_scope_boundary.csv", paper_scope="appendix")
if ph_scope_df is None:
    ph_scope_df = g11_load_table("table_appendix_ph_scope_boundary.csv")
    exported_ph_scope_path = g11_export_appendix(ph_scope_df, "table_paper_appendix_ph_scope_boundary.csv") if ph_scope_df is not None else None
else:
    exported_ph_scope_path = G11_PAPER_APPENDIX_DIR / "table_paper_appendix_ph_scope_boundary.csv"

appendix_figure_path = G11_PAPER_APPENDIX_DIR / "fig_appendix_cox_ph_diagnostics.png"
if not appendix_figure_path.exists():
    appendix_figure_path = G11_OUTPUT_DIR / "figures" / "diagnostics" / "fig_cox_ph_diagnostics.png"

g11_display_heading("Evaluation protocol audit", level=3)
display(protocol_df)
component_col = g11_pick_column(protocol_df, ["component"])
protocol_value_col = g11_pick_column(protocol_df, ["protocol_value"])
protocol_focus_components = [
    "official_horizons",
    "primary_discrimination_metric",
    "censoring_treatment",
    "dynamic_vs_static_prediction_rule",
    "leakage_prevention_rule",
    "split_scope_boundary",
    "primary_calibration_metric",
    "expanded_calibration_strengthening",
    "bootstrap_inferential_role",
    "ph_scope_boundary",
]
protocol_lines = []
for component_name in protocol_focus_components:
    protocol_row = protocol_df.loc[protocol_df[component_col].astype(str) == component_name]
    if not protocol_row.empty:
        protocol_lines.append(
            f"{component_name} = {protocol_row.iloc[0][protocol_value_col]}."
        )
protocol_lines.append("This table is the manuscript contract layer for IPCW/Kaplan-Meier Brier-IBS scoring, time-dependent concordance, dynamic-weekly versus early-window comparability, anti-leakage split discipline, shared-context scope, and horizon-wise calibration.")
g11_display_bullets(protocol_lines)

g11_display_heading("Preprocessing and tuning audit", level=3)
display(preproc_df)
display(Markdown("**Audit interpretation**"))
display_name_col = g11_pick_column(preproc_df, ["display_name", "Model"], required=False)
n_candidates_col = g11_pick_column(preproc_df, ["n_tuning_candidates"], required=False)
validation_strategy_col = g11_pick_column(preproc_df, ["validation_strategy"], required=False)
selection_metric_col = g11_pick_column(preproc_df, ["selection_metric"], required=False)
early_stopping_col = g11_pick_column(preproc_df, ["early_stopping_used"], required=False)
if display_name_col is not None and n_candidates_col is not None:
    g11_display_bullets([
        "Tuning candidates by family: " + ", ".join(
            f"{row[display_name_col]}={int(row[n_candidates_col]) if pd.notna(row[n_candidates_col]) else 'N/A'}"
            for _, row in preproc_df.iterrows()
        ) + ".",
        "Validation strategy by family: " + ", ".join(
            f"{row[display_name_col]}={row[validation_strategy_col]}"
            for _, row in preproc_df.iterrows()
        ) + "." if validation_strategy_col is not None else "Validation strategy column is not present in the current export.",
        "Selection metric by family: " + ", ".join(
            f"{row[display_name_col]}={row[selection_metric_col]}"
            for _, row in preproc_df.iterrows()
        ) + "." if selection_metric_col is not None else "Selection metric column is not present in the current export.",
        "Early stopping usage: " + ", ".join(
            f"{row[display_name_col]}={row[early_stopping_col]}"
            for _, row in preproc_df.iterrows()
        ) + "." if early_stopping_col is not None else "Early stopping field is not present in the current export.",
        "This appendix table is the source-of-truth defense for preprocessing comparability and tuning discipline in the manuscript.",
    ])
else:
    g11_display_bullets(["The compact preprocessing audit is available and displayed, but its column names differ from the richer paper-facing export."])

g11_display_heading("Bootstrap uncertainty for the tuned benchmark hierarchy", level=3)
display(bootstrap_compact_df)
g11_display_heading("Bootstrap inferential scope", level=4)
display(bootstrap_scope_df)
if bootstrap_scope_df is not None:
    metric_label_col = g11_pick_column(bootstrap_scope_df, ["metric_label"])
    claim_status_col = g11_pick_column(bootstrap_scope_df, ["claim_status"])
    leading_model_col = g11_pick_column(bootstrap_scope_df, ["leading_model"])
    supported_col = g11_pick_column(bootstrap_scope_df, ["what_is_supported"])
    not_supported_col = g11_pick_column(bootstrap_scope_df, ["what_is_not_supported"])
    g11_display_heading("What bootstrap supports", level=5)
    g11_display_bullets([
        f"{row[metric_label_col]}: {row[supported_col]}"
        for _, row in bootstrap_scope_df.iterrows()
    ])
    g11_display_heading("What bootstrap does not support", level=5)
    g11_display_bullets([
        f"{row[metric_label_col]} ({row[claim_status_col]}, leader = {row[leading_model_col]}): {row[not_supported_col]}"
        for _, row in bootstrap_scope_df.iterrows()
    ])

g11_display_heading("Split and contextual overlap audit", level=3)
display(split_df)
train_col = g11_pick_column(split_df, ["Train"])
test_col = g11_pick_column(split_df, ["Test"])
total_col = g11_pick_column(split_df, ["Total"])
train_event_col = g11_pick_column(split_df, ["Train event rate"])
test_event_col = g11_pick_column(split_df, ["Test event rate"])
identity_leakage_col = g11_pick_column(split_df, ["Identity leakage"])
shared_modules_col = g11_pick_column(split_df, ["Shared modules"])
shared_presentations_col = g11_pick_column(split_df, ["Shared presentations"])
shared_module_presentations_col = g11_pick_column(split_df, ["Shared module-presentations"])
g11_display_bullets([
    f"The benchmark split covers {int(split_df.loc[0, total_col])} enrollments, with {int(split_df.loc[0, train_col])} in train and {int(split_df.loc[0, test_col])} in test.",
    f"Train and test event rates remain aligned at {split_df.loc[0, train_event_col]:.4f} and {split_df.loc[0, test_event_col]:.4f}.",
    f"Identity leakage is reported as {split_df.loc[0, identity_leakage_col]}, while shared curricular context remains complete at modules {split_df.loc[0, shared_modules_col]}, presentations {split_df.loc[0, shared_presentations_col]}, and module-presentations {split_df.loc[0, shared_module_presentations_col]}.",
    "This appendix table is therefore the formal basis for saying the split is leakage-free at the enrollment level but not context-disjoint.",
])

split_context_claims_df = pd.DataFrame([
    {
        "claim_section": "split_context",
        "claim_id": "split_counts",
        "source_table": "table_paper_appendix_split_context_audit.csv",
        "related_tex_label": "tab:appendix_split_context_audit",
        "metric": "enrollment_split_counts",
        "value_1": float(split_df.loc[0, train_col]),
        "value_2": float(split_df.loc[0, test_col]),
        "value_3": float(split_df.loc[0, total_col]),
        "claim_text": f"Split counts train={int(split_df.loc[0, train_col])}, test={int(split_df.loc[0, test_col])}, total={int(split_df.loc[0, total_col])}",
    },
    {
        "claim_section": "split_context",
        "claim_id": "event_rates",
        "source_table": "table_paper_appendix_split_context_audit.csv",
        "related_tex_label": "tab:appendix_split_context_audit",
        "metric": "event_rates",
        "value_1": float(split_df.loc[0, train_event_col]),
        "value_2": float(split_df.loc[0, test_event_col]),
        "value_3": pd.NA,
        "claim_text": f"Train/test event rates = {split_df.loc[0, train_event_col]:.4f} / {split_df.loc[0, test_event_col]:.4f}",
    },
])
split_context_claims_path = g11_export_appendix(split_context_claims_df, "table_paper_appendix_split_context_claims.csv")
g11_materialize_claim_table(split_context_claims_df, "table_paper_appendix_split_context_claims", "G11.3")

g11_display_heading("Proportional-hazards audit for the comparable Cox model", level=3)
display(cox_summary_df)
display(cox_audit_df)
g11_display_heading("PH scope boundary", level=4)
display(ph_scope_df)
g11_display_figure(appendix_figure_path, "Appendix PH figure preview")
cox_tested_col = g11_pick_column(cox_summary_df, ["n_covariates_tested", "Covariates tested"], required=False)
cox_flagged_count_col = g11_pick_column(cox_summary_df, ["n_covariates_flagged", "Flagged"], required=False)
cox_flagged_share_col = g11_pick_column(cox_summary_df, ["share_covariates_flagged"], required=False)
cox_global_col = g11_pick_column(cox_summary_df, ["global_classification", "Global interpretation"], required=False)
cox_flag_col = g11_pick_column(cox_audit_df, ["ph_flag_binary", "PH flag", "ph_flag", "flag"], required=False)
cox_covariate_col = g11_pick_column(cox_audit_df, ["covariate", "Covariate", "feature", "variable"], required=False)
if cox_flag_col is not None:
    flagged_df = cox_audit_df.loc[cox_audit_df[cox_flag_col].astype(str).str.lower() == "yes"].copy()
else:
    flagged_df = cox_audit_df.copy()

if cox_covariate_col is None and not flagged_df.empty:
    cox_covariate_col = flagged_df.columns[0]

ph_scope_lines = []
if isinstance(ph_scope_df, pd.DataFrame) and not ph_scope_df.empty:
    ph_scope_model_col = g11_pick_column(ph_scope_df, ["model", "Model"], required=False)
    ph_scope_exec_col = g11_pick_column(ph_scope_df, ["formal_ph_diagnostic_executed", "formal_diagnostic_executed"], required=False)
    ph_scope_coverage_col = g11_pick_column(ph_scope_df, ["coverage_status", "coverage"], required=False)
    if ph_scope_model_col is not None and ph_scope_exec_col is not None and ph_scope_coverage_col is not None:
        ph_scope_lines.append(
            "PH scope boundary by model: " + "; ".join(
                f"{row[ph_scope_model_col]} -> formal diagnostic executed={row[ph_scope_exec_col]}, coverage={row[ph_scope_coverage_col]}"
                for _, row in ph_scope_df.iterrows()
            ) + "."
        )

flagged_count_value = float(cox_summary_df.loc[0, cox_flagged_count_col]) if cox_flagged_count_col is not None else float(flagged_df.shape[0])
tested_count_value = float(cox_summary_df.loc[0, cox_tested_col]) if cox_tested_col is not None else pd.NA
flagged_share_value = float(cox_summary_df.loc[0, cox_flagged_share_col]) if cox_flagged_share_col is not None else (flagged_count_value / tested_count_value if pd.notna(tested_count_value) and tested_count_value else pd.NA)
global_classification_value = str(cox_summary_df.loc[0, cox_global_col]) if cox_global_col is not None else "not reported"

g11_display_bullets([
    f"The comparable Cox audit tested {int(tested_count_value)} covariates and flagged {int(flagged_count_value)}, for a flagged share of {flagged_share_value:.4f}." if pd.notna(tested_count_value) and pd.notna(flagged_share_value) else f"The comparable Cox audit flagged {int(flagged_count_value)} covariates in the current appendix-facing summary.",
    f"The current global PH classification is {global_classification_value}.",
    "Flagged covariates currently shown in the appendix audit: " + ", ".join(flagged_df[cox_covariate_col].astype(str).tolist()) + "." if cox_covariate_col is not None and not flagged_df.empty else "The appendix PH audit is available, but the flagged-covariate column differs from the richer canonical schema.",
    *ph_scope_lines,
    "This is the canonical place where the manuscript must separate formal PH scope boundary from a claim of total methodological failure.",
])

cox_ph_claims_df = pd.DataFrame([
    {
        "claim_section": "cox_ph",
        "claim_id": "cox_ph_flagged_share",
        "source_table": "table_paper_appendix_cox_ph_global_summary.csv",
        "related_tex_label": "tab:appendix_cox_ph_summary",
        "metric": "share_covariates_flagged",
        "value_1": flagged_share_value,
        "value_2": tested_count_value,
        "value_3": flagged_count_value,
        "claim_text": f"Comparable Cox PH audit flagged {int(flagged_count_value)} of {int(tested_count_value)} covariates ({flagged_share_value:.4f})" if pd.notna(tested_count_value) and pd.notna(flagged_share_value) else f"Comparable Cox PH audit flagged {int(flagged_count_value)} covariates in the current appendix-facing summary",
    },
    {
        "claim_section": "cox_ph",
        "claim_id": "cox_ph_global_classification",
        "source_table": "table_paper_appendix_cox_ph_global_summary.csv",
        "related_tex_label": "tab:appendix_cox_ph_summary",
        "metric": "global_classification",
        "value_1": pd.NA,
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": f"Global PH classification = {global_classification_value}",
    },
    {
        "claim_section": "cox_ph",
        "claim_id": "flagged_covariates",
        "source_table": "table_paper_appendix_cox_ph_audit.csv",
        "related_tex_label": "fig:appendix_cox_ph_diagnostics",
        "metric": "flagged_covariates",
        "value_1": float(flagged_df.shape[0]),
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": (
            "Flagged covariates: " + ", ".join(flagged_df[cox_covariate_col].astype(str).tolist())
            if cox_covariate_col is not None and not flagged_df.empty
            else "Flagged covariates listed in the available appendix audit export."
        ),
    },
])
cox_ph_claims_path = g11_export_appendix(cox_ph_claims_df, "table_paper_appendix_cox_ph_claims.csv")
g11_materialize_claim_table(cox_ph_claims_df, "table_paper_appendix_cox_ph_claims", "G11.3")

print("Exported or refreshed appendix assets:")
for exported_path in [exported_protocol_path, exported_preproc_path, exported_bootstrap_scope_path, exported_ph_scope_path, split_context_claims_path, cox_ph_claims_path, benchmark_claims_path, ablation_claims_path]:
    if exported_path is not None:
        print("-", exported_path.resolve())

print(f"[END] G11.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 155
from datetime import datetime as _dt
print(f"[START] G11.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

weighting_df = g11_load_table(
    "table_paper_appendix_weighting_sensitivity_summary.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_item6_loss_sensitivity",
)
comparable_window_df = g11_load_table(
    "table_paper_appendix_comparable_window_champions.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_comparable_window_champions",
)
comparable_stability_df = g11_load_table(
    "table_paper_appendix_comparable_window_leadership_stability.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_comparable_window_leadership_stability",
)
cross_arm_protocol_df = g11_load_table(
    "table_paper_appendix_cross_arm_parity_protocol.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_cross_arm_parity_protocol",
)
cross_arm_scope_df = g11_load_table(
    "table_paper_appendix_cross_arm_execution_scope.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_cross_arm_execution_scope",
)
cross_arm_champions_df = g11_load_table(
    "table_paper_appendix_cross_arm_window_champions.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_cross_arm_window_champions",
)
cross_arm_stability_df = g11_load_table(
    "table_paper_appendix_cross_arm_window_leadership_stability.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_cross_arm_window_leadership_stability",
)
cross_arm_decision_df = g11_load_table(
    "table_paper_appendix_cross_arm_decision_summary.csv",
    paper_scope="appendix",
    duckdb_table="table_5_16_cross_arm_decision_summary",
)

required_g114_frames = {
    "weighting_df": weighting_df,
    "comparable_window_df": comparable_window_df,
    "comparable_stability_df": comparable_stability_df,
    "cross_arm_protocol_df": cross_arm_protocol_df,
    "cross_arm_scope_df": cross_arm_scope_df,
    "cross_arm_champions_df": cross_arm_champions_df,
    "cross_arm_stability_df": cross_arm_stability_df,
    "cross_arm_decision_df": cross_arm_decision_df,
}
missing_g114_frames = [name for name, frame in required_g114_frames.items() if frame is None]
if missing_g114_frames:
    raise ValueError(f"G11.4 is missing required D5.16 paper-facing inputs: {missing_g114_frames}")

weighting_export_path = g11_export_appendix(weighting_df, "table_paper_appendix_weighting_sensitivity_summary.csv")
comparable_window_export_path = g11_export_appendix(comparable_window_df, "table_paper_appendix_comparable_window_champions.csv")
comparable_stability_export_path = g11_export_appendix(comparable_stability_df, "table_paper_appendix_comparable_window_leadership_stability.csv")
cross_arm_protocol_export_path = g11_export_appendix(cross_arm_protocol_df, "table_paper_appendix_cross_arm_parity_protocol.csv")
cross_arm_scope_export_path = g11_export_appendix(cross_arm_scope_df, "table_paper_appendix_cross_arm_execution_scope.csv")
cross_arm_decision_export_path = g11_export_appendix(cross_arm_decision_df, "table_paper_appendix_cross_arm_decision_summary.csv")
cross_arm_champions_export_path = g11_export_appendix(cross_arm_champions_df, "table_paper_appendix_cross_arm_window_champions.csv")
cross_arm_stability_export_path = g11_export_appendix(cross_arm_stability_df, "table_paper_appendix_cross_arm_window_leadership_stability.csv")

g11_display_heading("Dynamic-arm weighting sensitivity", level=3)
display(weighting_df)

canonical_weighting_df = weighting_df.loc[pd.to_numeric(weighting_df["canonical_window"], errors="coerce").fillna(False).astype(bool)].copy()
if canonical_weighting_df.empty:
    canonical_weighting_df = weighting_df.copy()
canonical_weighting_df["model_display"] = canonical_weighting_df["model_name"].map(normalize_model_names)
canonical_weighting_df = canonical_weighting_df.sort_values("model_display", key=lambda series: series.map(model_sort_key)).reset_index(drop=True)

weighting_lines = []
for _, row in canonical_weighting_df.iterrows():
    weighting_lines.append(
        f"At w={int(row['window_weeks'])}, {row['model_display']} keeps the official not-weighted variant because IBS moves from {float(row['ibs_not_weighted']):.4f} to {float(row['ibs_weighted']):.4f} when weighting is applied, a delta of {float(row['delta_ibs_weighted_minus_not_weighted']):.4f}."
    )
g11_display_bullets(weighting_lines + [
    "This layer is a weighting sensitivity only. It does not redefine the official benchmark roster and must remain separated from the cross-arm parity question.",
])

g11_display_heading("Comparable-arm window sensitivity", level=3)
display(comparable_window_df)
g11_display_heading("Comparable-arm leadership stability", level=4)
display(comparable_stability_df)

comparable_window_df["champion_display"] = comparable_window_df["champion_model_name"].map(normalize_model_names)
comparable_champion_sequence = ", ".join(comparable_window_df["champion_display"].astype(str).tolist())
comparable_unique_champions = comparable_window_df["champion_display"].dropna().unique().tolist()
canonical_window_row = comparable_window_df.loc[pd.to_numeric(comparable_window_df["window_weeks"], errors="coerce") == 4].iloc[0]

comparable_claims_df = pd.DataFrame([
    {
        "claim_section": "comparable_window_sensitivity",
        "claim_id": "window_champion_sequence",
        "source_table": "table_paper_appendix_comparable_window_champions.csv",
        "related_tex_label": "appendix:comparable_window_sensitivity",
        "metric": "window_champion_sequence",
        "value_1": float(len(comparable_unique_champions)),
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": f"Comparable-arm champion sequence across windows = {comparable_champion_sequence}",
    },
    {
        "claim_section": "comparable_window_sensitivity",
        "claim_id": "canonical_window_champion",
        "source_table": "table_paper_appendix_comparable_window_champions.csv",
        "related_tex_label": "appendix:comparable_window_sensitivity",
        "metric": "canonical_window_champion",
        "value_1": float(canonical_window_row["champion_ibs"]),
        "value_2": float(canonical_window_row["champion_c_index"]),
        "value_3": float(canonical_window_row["window_weeks"]),
        "claim_text": f"At the canonical window w=4, the comparable-arm champion is {canonical_window_row['champion_display']} with IBS {float(canonical_window_row['champion_ibs']):.4f} and TD concordance {float(canonical_window_row['champion_c_index']):.4f}",
    },
])
comparable_claims_path = g11_export_appendix(comparable_claims_df, "table_paper_appendix_comparable_window_claims.csv")
g11_materialize_claim_table(comparable_claims_df, "table_paper_appendix_comparable_window_claims", "G11.4")

g11_display_bullets([
    f"Across the tracked comparable windows, the champion sequence is {comparable_champion_sequence}.",
    f"The canonical window remains w=4 by contract, but the current comparable winner at w=4 is {canonical_window_row['champion_display']} and the same model is also the only champion observed across all tracked windows.",
    "This means the present data do not force a new official comparable window: window sensitivity and the editorial decision about w=4 must remain separated in the manuscript.",
])

g11_display_heading("Cross-arm parity protocol and execution scope", level=3)
display(cross_arm_protocol_df)
display(cross_arm_scope_df)
g11_display_heading("Cross-arm winners by window", level=4)
display(cross_arm_champions_df)
g11_display_heading("Cross-arm leadership stability", level=4)
display(cross_arm_stability_df)
g11_display_heading("Cross-arm decision summary", level=4)
display(cross_arm_decision_df)

cross_arm_champions_df["winning_model_display"] = cross_arm_champions_df["winning_model_name"].map(normalize_model_names)
cross_arm_arm_sequence = ", ".join(cross_arm_champions_df["winning_arm_name"].astype(str).tolist())
cross_arm_model_sequence = ", ".join(cross_arm_champions_df["winning_model_display"].astype(str).tolist())
cross_arm_narrative_row = cross_arm_decision_df.loc[
    cross_arm_decision_df["decision_key"].astype(str) == "cross_arm_changes_main_benchmark_narrative"
].iloc[0]

cross_arm_claims_df = pd.DataFrame([
    {
        "claim_section": "cross_arm_parity",
        "claim_id": "winning_arm_sequence",
        "source_table": "table_paper_appendix_cross_arm_window_champions.csv",
        "related_tex_label": "appendix:cross_arm_parity",
        "metric": "winning_arm_sequence",
        "value_1": float(cross_arm_champions_df.shape[0]),
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": f"Cross-arm winning arm sequence across windows = {cross_arm_arm_sequence}",
    },
    {
        "claim_section": "cross_arm_parity",
        "claim_id": "winning_model_sequence",
        "source_table": "table_paper_appendix_cross_arm_window_champions.csv",
        "related_tex_label": "appendix:cross_arm_parity",
        "metric": "winning_model_sequence",
        "value_1": float(cross_arm_champions_df.shape[0]),
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": f"Cross-arm winning model sequence across windows = {cross_arm_model_sequence}",
    },
    {
        "claim_section": "cross_arm_parity",
        "claim_id": "changes_main_benchmark_narrative",
        "source_table": "table_5_16_cross_arm_decision_summary",
        "related_tex_label": "appendix:cross_arm_parity",
        "metric": "changes_main_benchmark_narrative",
        "value_1": pd.NA,
        "value_2": pd.NA,
        "value_3": pd.NA,
        "claim_text": f"Cross-arm parity summary says the sensitivity-window answer changes the main benchmark narrative = {cross_arm_narrative_row['decision_value']}",
    },
])
cross_arm_claims_path = g11_export_appendix(cross_arm_claims_df, "table_paper_appendix_cross_arm_claims.csv")
g11_materialize_claim_table(cross_arm_claims_df, "table_paper_appendix_cross_arm_claims", "G11.4")

g11_display_bullets([
    f"Under the current equal-window cross-arm layer, the winning arm sequence is {cross_arm_arm_sequence} and the winning model sequence is {cross_arm_model_sequence}.",
    f"The D5.16 decision summary currently marks the answer to whether cross-arm parity changes the benchmark narrative as {cross_arm_narrative_row['decision_value']}.",
    "This answer must still be narrated as a separate sensitivity layer: official benchmark, window sensitivity, weighting sensitivity, and cross-arm parity remain distinct manuscript objects.",
])

print("Exported D5.16 manuscript-facing appendix tables:")
for exported_path in [
    weighting_export_path,
    comparable_window_export_path,
    comparable_stability_export_path,
    comparable_claims_path,
    cross_arm_protocol_export_path,
    cross_arm_scope_export_path,
    cross_arm_decision_export_path,
    cross_arm_champions_export_path,
    cross_arm_stability_export_path,
    cross_arm_claims_path,
]:
    print("-", exported_path.resolve())

print(f"[END] G11.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 156
from datetime import datetime as _dt
print(f"[START] G12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

shared_con = globals().get("con")
g11_con = globals().get("G11_CON")

if g11_con is not None and g11_con is not shared_con:
    try:
        g11_con.close()
        print("Closed G11 local read-only DuckDB connection.")
    except Exception as exc:
        print(f"Warning while closing G11 local connection: {exc}")
elif g11_con is shared_con and g11_con is not None:
    print("G11 local connection aliases the shared notebook connection; closing it once via the shared shutdown helper.")

globals()["G11_CON"] = None
if "shutdown_duckdb_connection_from_globals" not in globals():
    from util import shutdown_duckdb_connection_from_globals

shutdown_duckdb_connection_from_globals(globals())

print(f"[END] G12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 157

# %% Cell 158

# %% Cell 159
