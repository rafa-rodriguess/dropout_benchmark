from __future__ import annotations

"""
Ablation and stability audit module for benchmark stage F.

Methodological role:
- stage F is not a second benchmark and must not be used to reopen model
    selection across the full 14-model predictive roster
- the full benchmark comparison belongs to stages D and D16, whose role is to
    establish global ranking, sensitivity summaries, and final comparative scope
- stage F belongs to a later explanatory layer: it audits why selected tuned
    representatives behave as they do, how dependent they are on specific feature
    blocks, how stable their results remain under resampling, and where formal
    methodological boundaries must be stated in the manuscript

Scientific scope and retention logic:
- the ablation layer is intentionally selective rather than exhaustive
- only manuscript-facing representatives are retained for this stage because
    ablation is designed to answer a mechanistic question, not to give every weak
    benchmark model a second interpretive pass
- in the current benchmark state, the retained manuscript-facing subset is:
    cox_tuned, deepsurv_tuned, rsf_tuned, and mtlr_tuned
- the scientific logic is: broad selection first, selective interpretation
    second
- models that are clearly non-competitive, empirically weak, or methodologically
    redundant should not be expanded inside F unless a separate negative-control
    question is explicitly declared
- therefore, the retained subset is treated as an explanatory audit subset, not
    as a surrogate for the entire benchmark roster

Main questions answered by stage F:
- how much of tuned-model performance depends on structural covariates versus
    temporal or early-window behavioral information
- whether that dependence changes across representative model families
- whether the reported tuned-model ordering is stable enough under bootstrap to
    justify cautious manuscript language about directional advantage
- where assumption-sensitive models, especially Cox-type branches, require
    formal scope boundaries instead of unconditional claims

What stage F does:
- materializes ablation variants for the selected tuned representatives
- consolidates preprocessing and tuning provenance needed for manuscript audit
- rebuilds fixed tuned-model survival objects and quantifies uncertainty without
    retraining inside bootstrap iterations
- records inferential-scope and proportional-hazards boundary statements as
    auditable artifacts

What stage F does not do:
- it does not redefine the official benchmark winner
- it does not silently expand from the manuscript subset to the full benchmark
    roster
- it does not interpret bootstrap outputs as null-hypothesis tests or proof of
    universal rank invariance
- it does not treat assumption-limited comparable models as assumption-free just
    because they remain predictively useful

Input contract:
- benchmark_shared_config.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/benchmark_survival.duckdb
- canonical tuned benchmark tables, saved models, and ready datasets produced by
    upstream stages when the relevant F-stage branches are executed

Output contract:
- DuckDB audit tables materialized by stages F0 onward
- JSON metadata artifacts under outputs_benchmark_survival/metadata
- structured console summaries for ablation, preprocessing, uncertainty, and PH
    audit sections

Failure policy:
- missing runtime metadata or a missing DuckDB database raises immediately
- missing canonical upstream dependencies raise immediately in the corresponding
    F-stage section unless that section documents a narrower fallback policy
- this module does not silently widen its scope from the manuscript-facing audit
    subset to the full benchmark roster

Historical lineage note:
- the NOTEBOOK_NAME provenance label is retained because the pipeline catalog
    still records notebook-origin lineage for previously materialized artifacts
"""

# %% Cell 2
from datetime import datetime as _dt
print(f"[START] F0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F0 - Runtime / DuckDB bootstrap
# ==============================================================
from pathlib import Path
import atexit
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
import dropout_bench_v3_D_00_common as base

from util import shutdown_duckdb_connection_from_globals

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from IPython.display import display as _ipython_display
except ImportError:
    _ipython_display = None


def display(obj) -> None:
    if _ipython_display is not None:
        _ipython_display(obj)
        return
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string(index=False))
        return
    print(obj)

STAGE_PREFIX = "F"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_F_ablation_stability_refatorado_v30.ipynb"
CONFIG_TOML_PATH = PROJECT_ROOT / "benchmark_shared_config.toml"
RUN_METADATA_JSON_PATH = PROJECT_ROOT / "outputs_benchmark_survival" / "metadata" / "run_metadata.json"

if not CONFIG_TOML_PATH.exists():
    raise FileNotFoundError(f"Missing shared configuration TOML: {CONFIG_TOML_PATH}")
if not RUN_METADATA_JSON_PATH.exists():
    raise FileNotFoundError(f"Missing execution metadata JSON: {RUN_METADATA_JSON_PATH}")

with open(CONFIG_TOML_PATH, "rb") as f:
    SHARED_CONFIG = tomllib.load(f)
with open(RUN_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
    RUN_METADATA = json.load(f)

RUN_ID = str(RUN_METADATA["run_id"]).strip()

paths_cfg = SHARED_CONFIG.get("paths", {})
benchmark_cfg = SHARED_CONFIG.get("benchmark", {})


def _resolve_project_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


OUTPUT_DIR = _resolve_project_path(paths_cfg.get("output_dir", "outputs_benchmark_survival"))
TABLES_DIR = OUTPUT_DIR / paths_cfg.get("tables_subdir", "tables")
METADATA_DIR = OUTPUT_DIR / paths_cfg.get("metadata_subdir", "metadata")
DATA_OUTPUT_DIR = OUTPUT_DIR / paths_cfg.get("data_output_subdir", "data")
DUCKDB_PATH = OUTPUT_DIR / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")

RANDOM_SEED = int(benchmark_cfg.get("seed", 42))
TEST_SIZE = float(benchmark_cfg.get("test_size", 0.30))
EARLY_WINDOW_WEEKS = int(benchmark_cfg.get("early_window_weeks", 4))
MAIN_ENROLLMENT_WINDOW_WEEKS = int(benchmark_cfg.get("main_enrollment_window_weeks", 4))
CALIBRATION_BINS = int(benchmark_cfg.get("calibration_bins", 10))

for p in [OUTPUT_DIR, TABLES_DIR, METADATA_DIR, DATA_OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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

print("Runtime context ready.")
print("- SCRIPT_NAME   :", SCRIPT_NAME)
print("- STAGE_PREFIX  :", STAGE_PREFIX)
print("- LINEAGE LABEL :", NOTEBOOK_NAME)
print("- RUN_ID        :", RUN_ID)
print("- DUCKDB_PATH   :", DUCKDB_PATH)

print(f"[END] F0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 4


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def print_duckdb_table(con, table_name: str, title: str | None = None, limit: int = 20) -> None:
    title = title or table_name
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    preview = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
    print(f"\n{title}")
    print(f"Table: {table_name}")
    print(f"Row count: {n_rows}")
    print(preview.to_string(index=False) if not preview.empty else "[empty table]")

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

def load_duckdb_table_or_raise(table_name: str) -> pd.DataFrame:
    available = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if table_name not in available:
        raise FileNotFoundError(f"Required DuckDB table not found: {table_name}")
    return con.execute(f"SELECT * FROM {table_name}").fetchdf()

def load_duckdb_table_optional(table_name: str):
    available = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if table_name not in available:
        return None
    return con.execute(f"SELECT * FROM {table_name}").fetchdf()

def require_duckdb_tables(table_names: list[str], stage_id: str) -> None:
    available = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    missing = [table_name for table_name in table_names if table_name not in available]
    if missing:
        raise FileNotFoundError(
            f"{stage_id}: missing required DuckDB table(s): " + ", ".join(missing)
        )

def persist_duckdb_artifact(df: pd.DataFrame, table_name: str, stage_id: str) -> None:
    materialize_dataframe(con, df, table_name, stage_id)

def infer_table_name_from_pathlike(pathlike) -> str:
    return Path(pathlike).stem

ensure_pipeline_table_catalog(con)
required_runtime = ["NOTEBOOK_NAME", "RUN_ID", "con", "save_json", "materialize_dataframe", "register_duckdb_table", "print_duckdb_table", "infer_table_name_from_pathlike", "load_duckdb_table_or_raise", "load_duckdb_table_optional", "require_duckdb_tables", "persist_duckdb_artifact", "shutdown_duckdb_connection_from_globals"]
missing_runtime = [name for name in required_runtime if name not in globals()]
if missing_runtime:
    raise NameError("F0.1 runtime contract failed. Missing required object(s): " + ", ".join(missing_runtime))
print("Runtime contract validated.")
CALIBRATION_BINS = int(benchmark_cfg.get("calibration_bins", 10))

print(f"[END] F0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 7
from datetime import datetime as _dt
print(f"[START] F1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F1 — Define Ablation Study Design and Protocol
# --------------------------------------------------------------
# Purpose:
#   Define the ablation study configuration, feature-block logic,
#   evaluation protocol, and documentation layer for the benchmark.
#
# Methodological note:
#   This step does not train any model. It only formalizes the
#   ablation study that will be executed in subsequent steps.
#
#   The canonical benchmark roster now spans 14 predictive models
#   (D5.2-D5.15) plus one contract-only stage (D5.1), as frozen by
#   D5.16.
#
#   Stage F does not reopen that full benchmark. Its ablation
#   layer remains an explicit manuscript-facing audit subset built on
#   four tuned representatives, chosen to cover the two central paper
#   arms with one transparent and one flexible model each.
#
#   Models included in the manuscript-facing audit subset:
#     - cox_tuned
#     - deepsurv_tuned
#     - rsf_tuned
#     - mtlr_tuned
#
#   Main goal:
#     quantify how much performance depends on different feature
#     blocks, especially:
#       - static covariates
#       - early-window behavior summaries
#
#   Scope note:
#     The benchmark itself now spans the full D5.2-D5.15 roster, but
#     the ablation layer remains intentionally selective so the paper
#     can compare feature-block dependence without creating a second
#     14-model ablation benchmark.
#
# Inputs expected from previous setup stages:
#   - OUTPUT_DIR
#   - TABLES_DIR
#   - METADATA_DIR
#   - save_json
#   - HORIZONS_WEEKS
#
# Outputs:
#   - ablation_config.json
#   - table_ablation_model_registry.csv
#   - table_ablation_feature_blocks.csv
#   - table_ablation_scenarios.csv
# ==============================================================

print("\n" + "=" * 70)
print("F1 — Define Ablation Study Design and Protocol")
print("=" * 70)
print("Methodological note: this step defines the ablation study only.")
print("No model is trained here.")

# ------------------------------
# 1) Basic checks
# ------------------------------

# Resolve canonical benchmark horizons from upstream DuckDB artifacts.
if "HORIZONS_WEEKS" not in globals():
    available_tables_f1 = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if "table_benchmark_support_reference" in available_tables_f1:
        _support_ref_f1 = load_duckdb_table_or_raise("table_benchmark_support_reference")
        HORIZONS_WEEKS = (
            pd.to_numeric(_support_ref_f1["horizon_week"], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
    else:
        HORIZONS_WEEKS = [10, 20, 30]

required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "save_json", "HORIZONS_WEEKS"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

import pandas as pd

CANONICAL_MODEL_INVENTORY = load_duckdb_table_or_raise("table_5_16_model_inventory")
PREDICTIVE_MODEL_INVENTORY = CANONICAL_MODEL_INVENTORY[
    CANONICAL_MODEL_INVENTORY["family_group"] != "contract"
].copy()

if int(PREDICTIVE_MODEL_INVENTORY.shape[0]) != 14:
    raise ValueError(
        "F1 expected 14 predictive roster entries in table_5_16_model_inventory. "
        f"Found {int(PREDICTIVE_MODEL_INVENTORY.shape[0])}."
    )

MANUSCRIPT_AUDIT_MODEL_IDS = [
    "poisson_pexp_tuned",
    "linear_tuned",
    "gb_weekly_tuned",
    "neural_tuned",
    "cox_tuned",
    "deepsurv_tuned",
    "rsf_tuned",
    "mtlr_tuned",
]

MANUSCRIPT_AUDIT_SUBSET_BY_MODEL_ID = base.get_manuscript_audit_subset_by_model_id()

# ------------------------------
# 2) Define tuned models included in ablation
# ------------------------------
ABLATION_MODEL_REGISTRY = [
    # --- Dynamic arm ---
    {
        "stage_id": "D5.6",
        "model_id": "poisson_pexp_tuned",
        "canonical_model_name": "poisson_piecewise_exponential",
        "display_name": "Poisson Piecewise-Exponential (Tuned)",
        "family": "discrete_time_poisson",
        "operational_family_group": "dynamic_weekly",
        "paper_methodological_arm": "dynamic_weekly_person_period",
        "input_representation": "weekly_person_period",
        "training_contract": "dynamic_weekly_person_period",
        "base_data_level": "person_period",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Best IBS in the dynamic arm; GLM parametric paradigm; coefficient-interpretable.",
        "uses_dynamic_temporal_features": True,
        "uses_static_features": True,
        "uses_early_window_features": False,
        "ablation_positioning_note": "Tuned representative of the Poisson piecewise-exponential family.",
    },
    {
        "stage_id": "D5.2",
        "model_id": "linear_tuned",
        "canonical_model_name": "linear_discrete_time_hazard",
        "display_name": "Linear Discrete-Time Hazard (Tuned)",
        "family": "discrete_time_linear",
        "operational_family_group": "dynamic_weekly",
        "paper_methodological_arm": "dynamic_weekly_person_period",
        "input_representation": "weekly_person_period",
        "training_contract": "dynamic_weekly_person_period",
        "base_data_level": "person_period",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Classical linear logistic baseline for the dynamic arm; maximum interpretability.",
        "uses_dynamic_temporal_features": True,
        "uses_static_features": True,
        "uses_early_window_features": False,
        "ablation_positioning_note": "Tuned representative of the linear discrete-time hazard family.",
    },
    {
        "stage_id": "D5.7",
        "model_id": "gb_weekly_tuned",
        "canonical_model_name": "gb_weekly_hazard",
        "display_name": "GB Weekly Hazard (Tuned)",
        "family": "discrete_time_gb",
        "operational_family_group": "dynamic_weekly",
        "paper_methodological_arm": "dynamic_weekly_person_period",
        "input_representation": "weekly_person_period",
        "training_contract": "dynamic_weekly_person_period",
        "base_data_level": "person_period",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Gradient-boosted tree paradigm representative for the dynamic arm.",
        "uses_dynamic_temporal_features": True,
        "uses_static_features": True,
        "uses_early_window_features": False,
        "ablation_positioning_note": "Tuned representative of the GB weekly hazard family.",
    },
    {
        "stage_id": "D5.3",
        "model_id": "neural_tuned",
        "canonical_model_name": "neural_discrete_time_survival",
        "display_name": "Neural Discrete-Time Survival (Tuned)",
        "family": "discrete_time_neural",
        "operational_family_group": "dynamic_neural",
        "paper_methodological_arm": "dynamic_weekly_person_period",
        "input_representation": "weekly_person_period",
        "training_contract": "dynamic_weekly_person_period",
        "base_data_level": "person_period",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Only neural representative for the dynamic arm.",
        "uses_dynamic_temporal_features": True,
        "uses_static_features": True,
        "uses_early_window_features": False,
        "ablation_positioning_note": "Tuned representative of the neural discrete-time survival family.",
    },
    # --- Comparable arm ---
    {
        "stage_id": "D5.4",
        "model_id": "cox_tuned",
        "canonical_model_name": "cox_comparable",
        "display_name": "Cox Comparable (Tuned)",
        "family": "continuous_time_cox",
        "operational_family_group": "comparable_continuous_time",
        "paper_methodological_arm": "comparable_continuous_time_early_window",
        "input_representation": "early_window_enrollment",
        "training_contract": "comparable_continuous_time_early_window",
        "base_data_level": "enrollment",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Interpretable comparable-arm baseline retained as the reference continuous-time representative.",
        "uses_dynamic_temporal_features": False,
        "uses_static_features": True,
        "uses_early_window_features": True,
        "ablation_positioning_note": "Tuned representative of the Cox comparable family.",
    },
    {
        "stage_id": "D5.5",
        "model_id": "deepsurv_tuned",
        "canonical_model_name": "deepsurv",
        "display_name": "DeepSurv (Tuned)",
        "family": "continuous_time_deepsurv",
        "operational_family_group": "comparable_neural",
        "paper_methodological_arm": "comparable_continuous_time_early_window",
        "input_representation": "early_window_enrollment",
        "training_contract": "comparable_continuous_time_early_window",
        "base_data_level": "enrollment",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Flexible comparable-arm representative retained to contrast with Cox under the same early-window contract.",
        "uses_dynamic_temporal_features": False,
        "uses_static_features": True,
        "uses_early_window_features": True,
        "ablation_positioning_note": "Tuned representative of the DeepSurv family.",
    },
    {
        "stage_id": "D5.9",
        "model_id": "rsf_tuned",
        "canonical_model_name": "random_survival_forest",
        "display_name": "Random Survival Forest (Tuned)",
        "family": "tree_survival_rsf",
        "operational_family_group": "comparable_tree_survival",
        "paper_methodological_arm": "comparable_continuous_time_early_window",
        "input_representation": "early_window_enrollment",
        "training_contract": "comparable_continuous_time_early_window",
        "base_data_level": "enrollment",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Best-performing comparable-arm tree-survival representative retained as the principal nonparametric benchmark.",
        "uses_dynamic_temporal_features": False,
        "uses_static_features": True,
        "uses_early_window_features": True,
        "ablation_positioning_note": "Tuned representative of the comparable random survival forest family.",
    },
    {
        "stage_id": "D5.14",
        "model_id": "mtlr_tuned",
        "canonical_model_name": "neural_mtlr",
        "display_name": "Neural-MTLR (Tuned)",
        "family": "continuous_time_mtlr",
        "operational_family_group": "comparable_neural",
        "paper_methodological_arm": "comparable_continuous_time_early_window",
        "input_representation": "early_window_enrollment",
        "training_contract": "comparable_continuous_time_early_window",
        "base_data_level": "enrollment",
        "selection_scope": "manuscript_audit_subset",
        "selection_rationale": "Competitive comparable-arm neural discrete-duration representative retained to contrast with DeepSurv and the tree-survival winner.",
        "uses_dynamic_temporal_features": False,
        "uses_static_features": True,
        "uses_early_window_features": True,
        "ablation_positioning_note": "Tuned representative of the Neural-MTLR family.",
    },
]

ablation_model_registry_df = pd.DataFrame(ABLATION_MODEL_REGISTRY)

# ------------------------------
# 3) Define feature blocks
# ------------------------------
ABLATION_FEATURE_BLOCKS = [
    {
        "block_id": "static_structural",
        "block_label": "Static structural covariates",
        "applies_to_data_level": "enrollment",
        "conceptual_role": "student background and enrollment-level structure",
        "examples": "gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability",
    },
    {
        "block_id": "early_window_behavior",
        "block_label": "Early-window behavior summaries",
        "applies_to_data_level": "enrollment",
        "conceptual_role": "compressed early-course activity profile",
        "examples": "clicks_first_4_weeks, active_weeks_first_4, mean_clicks_first_4_weeks",
    },
    {
        "block_id": "dynamic_temporal_behavioral",
        "block_label": "Weekly temporal-behavioral features",
        "applies_to_data_level": "person_period",
        "conceptual_role": "week-by-week real-time behavioral signal",
        "examples": "total_clicks_week, active_this_week, n_vle_rows_week, n_distinct_sites_week, cum_clicks_until_t, recency, streak",
    },
    {
        "block_id": "discrete_time_index",
        "block_label": "Discrete time index (week)",
        "applies_to_data_level": "person_period",
        "conceptual_role": "explicit week index encoding time-varying hazard baseline",
        "examples": "week",
    },
]

ablation_feature_blocks_df = pd.DataFrame(ABLATION_FEATURE_BLOCKS)

# ------------------------------
# 4) Define ablation scenarios
# ------------------------------
# Strategy:
#   For each family, compare:
#   - full tuned model
#   - remove structural/static block
#   - remove temporal/early-window block
#   - only structural/static block
#   - only temporal/early-window block
#
# Note:
#   The exact executable datasets will be materialized later.
ABLATION_SCENARIOS = [
    {
        "scenario_id": "full_features",
        "scenario_label": "Full tuned feature set",
        "scenario_type": "reference",
        "interpretation_goal": "Reference tuned benchmark for each family.",
    },
    {
        "scenario_id": "drop_static_structural",
        "scenario_label": "Drop static structural covariates",
        "scenario_type": "leave_one_block_out",
        "interpretation_goal": "Estimate how much performance depends on background/structural covariates.",
    },
    {
        "scenario_id": "drop_temporal_signal",
        "scenario_label": "Drop early-window behavior block",
        "scenario_type": "leave_one_block_out",
        "interpretation_goal": "Estimate how much performance depends on compressed early-window behavioral information.",
    },
    {
        "scenario_id": "only_static_structural",
        "scenario_label": "Only static structural covariates",
        "scenario_type": "single_block_only",
        "interpretation_goal": "Assess how far structural covariates alone can go.",
    },
    {
        "scenario_id": "only_temporal_signal",
        "scenario_label": "Only early-window behavior block",
        "scenario_type": "single_block_only",
        "interpretation_goal": "Assess how far early-window behavioral summaries alone can go.",
    },
]

ablation_scenarios_df = pd.DataFrame(ABLATION_SCENARIOS)

# ------------------------------
# 5) Protocol definition
# ------------------------------
ABLATION_PROTOCOL = {
    "ablation_scope": "manuscript_audit_subset",
    "included_models": [item["model_id"] for item in ABLATION_MODEL_REGISTRY],
    "canonical_benchmark_context": {
        "total_stage_count": int(CANONICAL_MODEL_INVENTORY.shape[0]),
        "predictive_roster_size": 14,
        "contract_stage_count": int((CANONICAL_MODEL_INVENTORY["family_group"] == "contract").sum()),
        "contract_stage_excluded_from_model_counts": True,
        "source_inventory_table": "table_5_16_model_inventory",
        "source_primary_summary_table": "table_5_16_model_primary_summary",
    },
    "subset_context": {
        "subset_model_count": int(len(MANUSCRIPT_AUDIT_MODEL_IDS)),
        "scope_decision": "selective_manuscript_audit_subset",
        "rationale": (
            "The main benchmark roster is larger, but ablation remains deliberately selective so the paper can compare structural versus early-window dependence across the retained comparable-arm representatives that remain scientifically interpretable after D16."
        ),
    },
    "main_question": (
        "How much of model performance is driven by structural covariates versus early-window behavioral summaries?"
    ),
    "secondary_question": (
        "Does the relative importance of structural versus early-window information vary across the retained comparable model families?"
    ),
    "evaluation_protocol": {
        "reuse_existing_train_test_split": True,
        "reuse_existing_horizons": [int(h) for h in HORIZONS_WEEKS],
        "reuse_existing_primary_metrics": [
            "ibs",
            "c_index",
            "brier_at_horizon",
            "calibration_at_horizon",
            "risk_auc_at_horizon",
        ],
        "primary_comparison_logic": (
            "Compare each ablation scenario against the full tuned version within the same model family."
        ),
    },
    "interpretation_rules": {
        "large_drop_after_removal": "The removed block is important for that family.",
        "small_drop_after_removal": "The removed block contributes little incremental signal.",
        "strong_only_block_performance": "That block alone carries substantial predictive information.",
        "cross_family_difference": (
            "Different model families may exploit the same information blocks differently."
        ),
    },
    "paper_positioning_note": (
        "The ablation study is intended to explain performance sources after the benchmark ranking is already established, and it should be read as a selective interpretive layer rather than as a re-ranking of the full 14-model predictive roster."
    ),
}

# ------------------------------
# 6) Save outputs
# ------------------------------
model_registry_table = "table_ablation_model_registry"
feature_blocks_table = "table_ablation_feature_blocks"
scenarios_table = "table_ablation_scenarios"
config_path = METADATA_DIR / "ablation_config.json"

persist_duckdb_artifact(ablation_model_registry_df, model_registry_table, "F1")
persist_duckdb_artifact(ablation_feature_blocks_df, feature_blocks_table, "F1")
persist_duckdb_artifact(ablation_scenarios_df, scenarios_table, "F1")
save_json(ABLATION_PROTOCOL, config_path)

# ------------------------------
# 7) Output for feedback
# ------------------------------
print("\nAblation model registry:")
display(ablation_model_registry_df)

print("\nAblation feature blocks:")
display(ablation_feature_blocks_df)

print("\nAblation scenarios:")
display(ablation_scenarios_df)

print("\nAblation protocol summary:")
display(pd.DataFrame([{
    "ablation_scope": ABLATION_PROTOCOL["ablation_scope"],
    "included_models": ", ".join(ABLATION_PROTOCOL["included_models"]),
    "canonical_predictive_roster_size": ABLATION_PROTOCOL["canonical_benchmark_context"]["predictive_roster_size"],
    "subset_model_count": ABLATION_PROTOCOL["subset_context"]["subset_model_count"],
    "reuse_existing_train_test_split": ABLATION_PROTOCOL["evaluation_protocol"]["reuse_existing_train_test_split"],
    "reuse_existing_horizons": ", ".join(str(x) for x in ABLATION_PROTOCOL["evaluation_protocol"]["reuse_existing_horizons"]),
    "main_question": ABLATION_PROTOCOL["main_question"],
    "secondary_question": ABLATION_PROTOCOL["secondary_question"],
}]))

print("\nSaved:")
print("- DuckDB table:", model_registry_table)
print("- DuckDB table:", feature_blocks_table)
print("- DuckDB table:", scenarios_table)
print("-", config_path.resolve())

print(f"[END] F1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 10
from datetime import datetime as _dt
print(f"[START] F2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F2 — Materialize Ablation Variants (Revised v2)
# --------------------------------------------------------------
# Purpose:
#   Materialize ablation-ready dataset variants for the tuned
#   benchmark models, without training them yet.
#
# Methodological note:
#   In the current retained subset, all selected models operate on
#   the comparable early-window enrollment contract.
# ==============================================================

print("\n" + "=" * 70)
print("F2 — Materialize Ablation Variants (Revised v2)")
print("=" * 70)
print("Methodological note: this step materializes ablation-ready")
print("feature subsets only. No model is trained here.")
print("Revised behavior: ready datasets are loaded from DuckDB.")
print("Current retained subset: comparable early-window models only.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

from pathlib import Path
import pandas as pd

# ------------------------------
# 2) Helper to load dataset
# ------------------------------
def load_ready_dataset(base_name: str) -> pd.DataFrame:
    return load_duckdb_table_or_raise(base_name)


# ------------------------------
# 3) Define feature blocks
# ------------------------------
STATIC_STRUCTURAL_FEATURES = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
]

EARLY_WINDOW_FEATURES = [
    "clicks_first_4_weeks",
    "active_weeks_first_4",
    "mean_clicks_first_4_weeks",
]

TEMPORAL_BEHAVIORAL_FEATURES = [
    "total_clicks_week",
    "active_this_week",
    "n_vle_rows_week",
    "n_distinct_sites_week",
    "cum_clicks_until_t",
    "recency",
    "streak",
]

DISCRETE_TIME_INDEX = [
    "week",
]

# ------------------------------
# 4) Define auxiliary / target columns
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

TARGET_DISCRETE = ["event_t"]

AUX_ENROLLMENT = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "duration",
    "duration_raw",
    "used_zero_week_fallback_for_censoring",
    "split",
    "time_for_split",
    "time_bucket",
    "event_time_bucket_label",
]

TARGET_ENROLLMENT = ["event"]

# ------------------------------
# 5) Scenario map by family
# ------------------------------
SCENARIO_MAP = {
    "full_features": {
        "enrollment": STATIC_STRUCTURAL_FEATURES + EARLY_WINDOW_FEATURES,
        "person_period": STATIC_STRUCTURAL_FEATURES + TEMPORAL_BEHAVIORAL_FEATURES + DISCRETE_TIME_INDEX,
    },
    "drop_static_structural": {
        "enrollment": EARLY_WINDOW_FEATURES,
        "person_period": TEMPORAL_BEHAVIORAL_FEATURES + DISCRETE_TIME_INDEX,
    },
    "drop_temporal_signal": {
        "enrollment": STATIC_STRUCTURAL_FEATURES,
        "person_period": STATIC_STRUCTURAL_FEATURES + DISCRETE_TIME_INDEX,
    },
    "only_static_structural": {
        "enrollment": STATIC_STRUCTURAL_FEATURES,
        "person_period": STATIC_STRUCTURAL_FEATURES + DISCRETE_TIME_INDEX,
    },
    "only_temporal_signal": {
        "enrollment": EARLY_WINDOW_FEATURES,
        "person_period": TEMPORAL_BEHAVIORAL_FEATURES + DISCRETE_TIME_INDEX,
    },
}

SCENARIO_LABELS = {
    "full_features": "Full tuned feature set",
    "drop_static_structural": "Drop static structural covariates",
    "drop_temporal_signal": "Drop temporal signal block",
    "only_static_structural": "Only static structural covariates",
    "only_temporal_signal": "Only temporal signal block",
}

# ------------------------------
# 6) Dataset registry loaded from DuckDB
# ------------------------------
DATASET_REGISTRY = [
    # --- Dynamic arm: person-period (pp_linear_hazard_* for Poisson/Linear/GB) ---
    {
        "model_id": "poisson_pexp_tuned",
        "family": "discrete_time_poisson",
        "data_level": "person_period",
        "train_df": load_ready_dataset("pp_linear_hazard_ready_train"),
        "test_df": load_ready_dataset("pp_linear_hazard_ready_test"),
        "aux_cols": AUX_DISCRETE,
        "target_cols": TARGET_DISCRETE,
    },
    {
        "model_id": "linear_tuned",
        "family": "discrete_time_linear",
        "data_level": "person_period",
        "train_df": load_ready_dataset("pp_linear_hazard_ready_train"),
        "test_df": load_ready_dataset("pp_linear_hazard_ready_test"),
        "aux_cols": AUX_DISCRETE,
        "target_cols": TARGET_DISCRETE,
    },
    {
        "model_id": "gb_weekly_tuned",
        "family": "discrete_time_gb",
        "data_level": "person_period",
        "train_df": load_ready_dataset("pp_linear_hazard_ready_train"),
        "test_df": load_ready_dataset("pp_linear_hazard_ready_test"),
        "aux_cols": AUX_DISCRETE,
        "target_cols": TARGET_DISCRETE,
    },
    # --- Dynamic arm: person-period (pp_neural_hazard_* for Neural) ---
    {
        "model_id": "neural_tuned",
        "family": "discrete_time_neural",
        "data_level": "person_period",
        "train_df": load_ready_dataset("pp_neural_hazard_context_ready_train"),
        "test_df": load_ready_dataset("pp_neural_hazard_context_ready_test"),
        "aux_cols": AUX_DISCRETE,
        "target_cols": TARGET_DISCRETE,
    },
    # --- Comparable arm: enrollment-level ---
    {
        "model_id": "cox_tuned",
        "family": "continuous_time_cox",
        "data_level": "enrollment",
        "train_df": load_ready_dataset("enrollment_cox_ready_train"),
        "test_df": load_ready_dataset("enrollment_cox_ready_test"),
        "aux_cols": AUX_ENROLLMENT,
        "target_cols": TARGET_ENROLLMENT,
    },
    {
        "model_id": "deepsurv_tuned",
        "family": "continuous_time_deepsurv",
        "data_level": "enrollment",
        "train_df": load_ready_dataset("enrollment_deepsurv_ready_train"),
        "test_df": load_ready_dataset("enrollment_deepsurv_ready_test"),
        "aux_cols": AUX_ENROLLMENT,
        "target_cols": TARGET_ENROLLMENT,
    },
    {
        "model_id": "rsf_tuned",
        "family": "tree_survival_rsf",
        "data_level": "enrollment",
        "train_df": load_ready_dataset("enrollment_cox_ready_train"),
        "test_df": load_ready_dataset("enrollment_cox_ready_test"),
        "aux_cols": AUX_ENROLLMENT,
        "target_cols": TARGET_ENROLLMENT,
    },
    {
        "model_id": "mtlr_tuned",
        "family": "continuous_time_mtlr",
        "data_level": "enrollment",
        "train_df": load_ready_dataset("enrollment_cox_ready_train"),
        "test_df": load_ready_dataset("enrollment_cox_ready_test"),
        "aux_cols": AUX_ENROLLMENT,
        "target_cols": TARGET_ENROLLMENT,
    },
]

# ------------------------------
# 7) Materialize variants
# ------------------------------
variant_registry_rows = []
variant_feature_manifest_rows = []

for item in DATASET_REGISTRY:
    model_id = item["model_id"]
    family = item["family"]
    data_level = item["data_level"]
    train_df = item["train_df"]
    test_df = item["test_df"]
    aux_cols = [c for c in item["aux_cols"] if c in train_df.columns]
    target_cols = [c for c in item["target_cols"] if c in train_df.columns]

    for scenario_id, scenario_def in SCENARIO_MAP.items():
        feature_cols = [c for c in scenario_def[data_level] if c in train_df.columns]

        ordered_cols = aux_cols + target_cols + feature_cols
        ordered_cols = [c for c in ordered_cols if c in train_df.columns]

        train_variant = train_df[ordered_cols].copy()
        test_variant = test_df[ordered_cols].copy()

        variant_base_name = f"{model_id}__{scenario_id}"
        train_name = f"{variant_base_name}__train"
        test_name = f"{variant_base_name}__test"

        persist_duckdb_artifact(train_variant, train_name, "F2")
        persist_duckdb_artifact(test_variant, test_name, "F2")

        variant_registry_rows.append({
            "model_id": model_id,
            "family": family,
            "data_level": data_level,
            "scenario_id": scenario_id,
            "scenario_label": SCENARIO_LABELS[scenario_id],
            "train_table_name": train_name,
            "test_table_name": test_name,
            "storage_contract": "duckdb_only",
            "n_train_rows": int(train_variant.shape[0]),
            "n_test_rows": int(test_variant.shape[0]),
            "n_columns_total": int(len(ordered_cols)),
            "n_aux_columns": int(len(aux_cols)),
            "n_target_columns": int(len(target_cols)),
            "n_feature_columns": int(len(feature_cols)),
        })

        for col in ordered_cols:
            if col in aux_cols:
                role = "auxiliary"
            elif col in target_cols:
                role = "target"
            else:
                role = "feature"

            if col in STATIC_STRUCTURAL_FEATURES:
                block = "static_structural"
            elif col in EARLY_WINDOW_FEATURES:
                block = "early_window_behavior"
            elif col in TEMPORAL_BEHAVIORAL_FEATURES:
                block = "dynamic_temporal_behavioral"
            elif col in DISCRETE_TIME_INDEX:
                block = "discrete_time_index"
            else:
                block = "aux_or_target"

            variant_feature_manifest_rows.append({
                "model_id": model_id,
                "family": family,
                "data_level": data_level,
                "scenario_id": scenario_id,
                "scenario_label": SCENARIO_LABELS[scenario_id],
                "column_name": col,
                "role": role,
                "feature_block": block,
            })

variant_registry_df = pd.DataFrame(variant_registry_rows)
variant_feature_manifest_df = pd.DataFrame(variant_feature_manifest_rows)

# ------------------------------
# 8) Save registry artifacts
# ------------------------------
variant_registry_table = "table_ablation_variant_registry"
variant_feature_manifest_table = "table_ablation_variant_feature_manifest"
variant_registry_json_path = METADATA_DIR / "ablation_variant_registry.json"

persist_duckdb_artifact(variant_registry_df, variant_registry_table, "F2")
persist_duckdb_artifact(variant_feature_manifest_df, variant_feature_manifest_table, "F2")

save_json(
    {
        "n_variants_materialized": int(variant_registry_df.shape[0]),
        "models_included": sorted(variant_registry_df["model_id"].unique().tolist()),
        "scenarios_included": sorted(variant_registry_df["scenario_id"].unique().tolist()),
        "data_levels": sorted(variant_registry_df["data_level"].unique().tolist()),
        "discrete_time_week_always_retained": False,
        "storage_contract": "duckdb_only",
    },
    variant_registry_json_path,
)

# ------------------------------
# 9) Output for feedback
# ------------------------------
print("\nAblation variant registry:")
display(variant_registry_df)

print("\nAblation variant feature manifest (first rows):")
display(variant_feature_manifest_df.head(60))

print("\nSaved:")
print("- DuckDB table:", variant_registry_table)
print("- DuckDB table:", variant_feature_manifest_table)
print("-", variant_registry_json_path.resolve())

print(f"[END] F2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 12
from datetime import datetime as _dt
print(f"[START] F3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F3 — Dynamic Arm Ablation: Poisson, Linear, GB, Neural
# --------------------------------------------------------------
# Purpose:
#   Train and evaluate discrete-time person-period models
#   on ablation feature subsets (D1–D4 dynamic arm).
#
# Models trained per scenario:
#   - poisson_pexp_tuned  (PoissonRegressor, sklearn)
#   - linear_tuned        (LogisticRegression pipeline, sklearn)
#   - gb_weekly_tuned     (GradientBoostingClassifier pipeline, sklearn)
#   - neural_tuned        (LogisticHazard, pycox)
#
# Evaluation: EvalSurv (pycox) — IBS, C-index, Brier@10/20/30,
#             calibration@10/20/30, AUC@10/20/30
# ==============================================================

print("\n" + "=" * 70)
print("F3 — Dynamic Arm Ablation (Poisson / Linear / GB / Neural)")
print("=" * 70)

# ------------------------------
# 1) Basic checks
# ------------------------------
if "CALIBRATION_BINS" not in globals():
    CALIBRATION_BINS = int(SHARED_CONFIG.get("benchmark", {}).get("calibration_bins", 10))

required_names = ["METADATA_DIR", "save_json", "HORIZONS_WEEKS", "RANDOM_SEED", "OUTPUT_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torchtuples as tt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from pycox.evaluation import EvalSurv
    from pycox.models import LogisticHazard
    PYCOX_AVAILABLE = True
except Exception:
    PYCOX_AVAILABLE = False

if not PYCOX_AVAILABLE:
    raise ImportError("pycox is required for F3.")

MODEL_DIR = OUTPUT_DIR / "models"

# ------------------------------
# 2) Helpers
# ------------------------------

DISCRETE_ARM_MODEL_IDS = ["poisson_pexp_tuned", "linear_tuned", "gb_weekly_tuned", "neural_tuned"]

CATEGORICAL_FEATURES_PP = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
]

AUX_PP = [
    "enrollment_id", "id_student", "code_module", "code_presentation",
    "event_observed", "t_event_week", "t_final_week",
    "used_zero_week_fallback_for_censoring", "split",
    "time_for_split", "time_bucket", "event_time_bucket_label",
]

TARGET_PP = "event_t"


def _read_json_f3(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_feature_cols_pp(df: pd.DataFrame) -> list:
    excluded = set(AUX_PP) | {TARGET_PP}
    return [c for c in df.columns if c not in excluded]


def _build_preprocessor_pp(feature_cols: list):
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES_PP]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    numeric_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def _build_truth_pp(df: pd.DataFrame) -> pd.DataFrame:
    """Enrollment-level truth from person-period DataFrame."""
    truth = (
        df.groupby("enrollment_id", as_index=False)
        .agg(
            event=("event_observed", "max"),
            t_event_week=("t_event_week", "first"),
            t_final_week=("t_final_week", "first"),
        )
        .sort_values("enrollment_id")
        .reset_index(drop=True)
    )
    truth["duration"] = truth.apply(
        lambda r: int(r["t_event_week"]) if pd.notna(r["t_event_week"]) else int(r["t_final_week"]),
        axis=1,
    ).astype(int)
    truth["event"] = truth["event"].astype(int)
    return truth[["enrollment_id", "event", "duration"]]


def _reconstruct_survival(test_df: pd.DataFrame, hazard_probs: np.ndarray, max_week: int) -> pd.DataFrame:
    """Build week × enrollment survival DataFrame from per-row hazard predictions."""
    df = test_df[["enrollment_id", "week"]].copy()
    # Replace NaN/Inf produced by overflow with a neutral hazard before clipping
    safe_hazards = np.nan_to_num(hazard_probs, nan=0.0, posinf=1.0, neginf=0.0)
    df["survival_factor"] = 1.0 - np.clip(safe_hazards, 1e-8, 1.0 - 1e-8)
    df = df.sort_values(["enrollment_id", "week"])
    df["cumulative_survival"] = df.groupby("enrollment_id")["survival_factor"].cumprod()
    pivot = df.pivot(index="week", columns="enrollment_id", values="cumulative_survival")
    # Fill any missing weeks by forward-filling, then backward-filling
    week_grid = pd.RangeIndex(start=1, stop=max_week + 1, step=1)
    pivot = pivot.reindex(week_grid).ffill().bfill().clip(lower=0.0, upper=1.0)
    pivot.columns.name = "enrollment_id"
    return pivot


def _evaluate_pp_survival(surv_df: pd.DataFrame, truth_test: pd.DataFrame, horizons: list) -> dict:
    """Same evaluation logic as F4 (enrollment-level), reused for person-period arm."""
    durations = truth_test["duration"].astype(int).to_numpy()
    events = truth_test["event"].astype(int).to_numpy()

    eval_surv = EvalSurv(surv=surv_df, durations=durations, events=events, censor_surv="km")

    primary_rows = []
    try:
        c_index = float(eval_surv.concordance_td())
        c_note = "pycox_concordance_td"
    except Exception as exc:
        c_index = np.nan
        c_note = f"failed: {exc}"

    try:
        ibs_grid = np.arange(1, int(max(horizons)) + 1, dtype=int)
        ibs_value = float(eval_surv.integrated_brier_score(ibs_grid))
        ibs_note = "pycox_integrated_brier_score"
    except Exception as exc:
        ibs_value = np.nan
        ibs_note = f"failed: {exc}"

    primary_rows.append({"metric_name": "ibs", "metric_value": ibs_value, "notes": ibs_note})
    primary_rows.append({"metric_name": "c_index", "metric_value": c_index, "notes": c_note})
    primary_df = pd.DataFrame(primary_rows)

    try:
        brier_h = eval_surv.brier_score(np.array(horizons, dtype=int))
        brier_df = pd.DataFrame({
            "horizon_week": list(brier_h.index.astype(int)),
            "metric_name": ["brier_at_horizon"] * len(brier_h.index),
            "metric_value": list(brier_h.values.astype(float)),
            "notes": ["pycox_brier_score"] * len(brier_h.index),
        })
    except Exception as exc:
        brier_df = pd.DataFrame({
            "horizon_week": horizons,
            "metric_name": ["brier_at_horizon"] * len(horizons),
            "metric_value": [np.nan] * len(horizons),
            "notes": [f"failed: {exc}"] * len(horizons),
        })

    calibration_rows, support_rows, pred_vs_obs_rows, risk_auc_rows = [], [], [], []

    def _get_surv_at_h(surv_frame, h):
        idx = np.asarray(surv_frame.index, dtype=float)
        pos = np.searchsorted(idx, float(h), side="right") - 1
        if pos < 0:
            return pd.Series(np.ones(surv_frame.shape[1]), index=surv_frame.columns)
        return surv_frame.iloc[pos]

    for h in horizons:
        pred_surv_h = _get_surv_at_h(surv_df, h)
        pred_risk_h = 1.0 - pred_surv_h

        ev = truth_test.copy()
        ev["pred_surv"] = ev["enrollment_id"].map(pred_surv_h.to_dict())
        ev["pred_risk"] = ev["enrollment_id"].map(pred_risk_h.to_dict())
        ev["evaluable"] = (
            ((ev["event"] == 1) & (ev["duration"] <= h)) | (ev["duration"] >= h)
        ).astype(int)
        ev = ev[ev["evaluable"] == 1].copy()
        ev["obs_event"] = ((ev["event"] == 1) & (ev["duration"] <= h)).astype(int)
        ev["obs_surv"] = 1 - ev["obs_event"]

        support_rows.append({
            "horizon_week": h,
            "n_evaluable_enrollments": int(ev.shape[0]),
            "n_events_by_horizon": int(ev["obs_event"].sum()),
            "event_rate_by_horizon": float(ev["obs_event"].mean()) if ev.shape[0] > 0 else np.nan,
        })

        _risk_vals = ev["pred_risk"].dropna()
        _obs_vals = ev.loc[ev["pred_risk"].notna(), "obs_event"]
        if _obs_vals.nunique() >= 2 and len(_risk_vals) > 0:
            try:
                rauc = roc_auc_score(_obs_vals, _risk_vals)
            except Exception:
                rauc = np.nan
        else:
            rauc = np.nan
        risk_auc_rows.append({
            "horizon_week": h,
            "metric_name": "risk_auc_at_horizon",
            "metric_value": float(rauc) if pd.notna(rauc) else np.nan,
            "notes": "roc_auc_on_evaluable_subset",
        })

        pred_vs_obs_rows.append({
            "horizon_week": h,
            "n_evaluable_enrollments": int(ev.shape[0]),
            "mean_predicted_survival": float(ev["pred_surv"].mean()) if ev.shape[0] > 0 else np.nan,
            "mean_observed_survival": float(ev["obs_surv"].mean()) if ev.shape[0] > 0 else np.nan,
            "abs_gap": float(abs(ev["pred_surv"].mean() - ev["obs_surv"].mean())) if ev.shape[0] > 0 else np.nan,
        })

        ev_calib = ev.dropna(subset=["pred_risk", "pred_surv"]).copy()
        if ev_calib.shape[0] > 0:
            ev = ev_calib
            ranked = ev["pred_risk"].rank(method="first")
            n_bins = int(min(CALIBRATION_BINS, max(1, ev.shape[0])))
            ev["calib_bin"] = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
            calib_tab = (
                ev.groupby("calib_bin")
                .agg(n=("enrollment_id", "count"),
                     mean_pred=("pred_risk", "mean"),
                     obs_rate=("obs_event", "mean"))
                .reset_index()
            )
            calib_tab["gap"] = (calib_tab["mean_pred"] - calib_tab["obs_rate"]).abs()
            calibration_rows.append({
                "horizon_week": h,
                "metric_name": "calibration_at_horizon",
                "metric_value": float(np.average(calib_tab["gap"], weights=calib_tab["n"])),
                "notes": "Weighted absolute calibration gap across bins",
            })

    return {
        "primary": primary_df,
        "brier": brier_df,
        "calibration": pd.DataFrame(calibration_rows),
        "secondary": pd.DataFrame(risk_auc_rows),
        "support": pd.DataFrame(support_rows),
        "pred_vs_obs": pd.DataFrame(pred_vs_obs_rows),
    }


def _load_neural_config(window: int = 4) -> dict:
    cfg_path = METADATA_DIR / f"neural_not_weighted_tuned_model_config_w{window}.json"
    return _read_json_f3(cfg_path)["best_candidate"]


# ------------------------------
# 3) Main training/evaluation loop
# ------------------------------
DYNAMIC_SCENARIOS = ["full_features", "drop_static_structural", "drop_temporal_signal",
                     "only_static_structural", "only_temporal_signal"]

results_primary, results_brier, results_calibration = [], [], []
results_secondary, results_support, results_pred_vs_obs = [], [], []
results_training_audit = []

ablation_registry_pp = ablation_registry = load_duckdb_table_or_raise("table_ablation_variant_registry")
ablation_registry_pp = ablation_registry_pp[
    ablation_registry["model_id"].isin(DISCRETE_ARM_MODEL_IDS)
].copy()

neural_best = _load_neural_config(window=4)

for _, reg_row in ablation_registry_pp.iterrows():
    model_id  = reg_row["model_id"]
    scenario_id = reg_row["scenario_id"]
    scenario_label = reg_row["scenario_label"]

    train_df = load_duckdb_table_or_raise(reg_row["train_table_name"])
    test_df  = load_duckdb_table_or_raise(reg_row["test_table_name"])

    feature_cols = _get_feature_cols_pp(train_df)
    preprocessor, num_cols, cat_cols = _build_preprocessor_pp(feature_cols)

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test  = preprocessor.transform(test_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray().astype(np.float32)
        X_test  = X_test.toarray().astype(np.float32)
    else:
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test  = np.asarray(X_test,  dtype=np.float32)

    y_train = train_df[TARGET_PP].to_numpy().astype(int)
    y_test  = test_df[TARGET_PP].to_numpy().astype(int)

    truth_test = _build_truth_pp(test_df)
    max_week = int(truth_test["duration"].max())

    # ------ fit ------
    if model_id == "poisson_pexp_tuned":
        model = PoissonRegressor(alpha=1e-3, max_iter=300)
        model.fit(X_train, y_train)
        hazard_raw = model.predict(X_test)
        hazard_probs = np.clip(hazard_raw, 1e-8, 1.0 - 1e-8)

    elif model_id == "linear_tuned":
        model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        model.fit(X_train, y_train)
        hazard_probs = model.predict_proba(X_test)[:, 1]

    elif model_id == "gb_weekly_tuned":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        hazard_probs = model.predict_proba(X_test)[:, 1]

    elif model_id == "neural_tuned":
        # Person-period ablation: each row is a binary hazard label (event_t).
        # LogisticHazard expects enrollment-level (duration, event) pairs, so we
        # use sklearn MLPClassifier for row-level binary classification instead.
        from sklearn.neural_network import MLPClassifier
        _hidden = tuple(int(h) for h in neural_best["hidden_dims"])
        model = MLPClassifier(
            hidden_layer_sizes=_hidden,
            activation="relu",
            max_iter=int(neural_best.get("best_epoch", 20)) * 5,
            learning_rate_init=float(neural_best["learning_rate"]),
            batch_size=int(neural_best.get("batch_size", 1024)),
            random_state=RANDOM_SEED,
            early_stopping=False,
        )
        model.fit(X_train, y_train)
        hazard_probs = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"F3: unsupported model_id={model_id}")

    # ------ reconstruct survival ------
    surv_df = _reconstruct_survival(test_df, hazard_probs, max_week)

    # ------ evaluate ------
    eval_out = _evaluate_pp_survival(surv_df, truth_test, HORIZONS_WEEKS)

    for key, lst in [("primary", results_primary), ("brier", results_brier),
                     ("calibration", results_calibration), ("secondary", results_secondary),
                     ("support", results_support), ("pred_vs_obs", results_pred_vs_obs)]:
        sub = eval_out[key].copy()
        sub["model_id"] = model_id
        sub["scenario_id"] = scenario_id
        sub["scenario_label"] = scenario_label
        lst.append(sub)

    results_training_audit.append({
        "model_id": model_id,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "target_col_used": TARGET_PP,
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_feature_columns_raw": int(len(feature_cols)),
        "n_numeric_features_raw": int(len(num_cols)),
        "n_categorical_features_raw": int(len(cat_cols)),
        "n_features_after_transform": int(X_train.shape[1]),
        "week_retained": "week" in feature_cols,
    })

# ------------------------------
# 4) Consolidate results
# ------------------------------
ablation_primary_df   = pd.concat(results_primary,    ignore_index=True)
ablation_brier_df     = pd.concat(results_brier,      ignore_index=True)
ablation_calib_df     = pd.concat(results_calibration, ignore_index=True)
ablation_secondary_df = pd.concat(results_secondary,   ignore_index=True)
ablation_support_df   = pd.concat(results_support,     ignore_index=True)
ablation_pvso_df      = pd.concat(results_pred_vs_obs, ignore_index=True)
ablation_audit_df     = pd.DataFrame(results_training_audit)

leaderboard_rows = []
for (mid, sid, slbl), g in ablation_primary_df.groupby(["model_id", "scenario_id", "scenario_label"]):
    row = {
        "model_id": mid,
        "scenario_id": sid,
        "scenario_label": slbl,
        "ibs": float(g.loc[g["metric_name"] == "ibs", "metric_value"].iloc[0]),
        "c_index": float(g.loc[g["metric_name"] == "c_index", "metric_value"].iloc[0]),
    }
    for h in HORIZONS_WEEKS:
        def _get(df, mid=mid, sid=sid, col="metric_value", h=h):
            mask = (df["model_id"] == mid) & (df["scenario_id"] == sid) & (df["horizon_week"] == h)
            vals = df.loc[mask, col]
            return float(vals.iloc[0]) if not vals.empty else np.nan

        row[f"brier_h{h}"] = _get(ablation_brier_df)
        row[f"calibration_h{h}"] = _get(ablation_calib_df)
        auc_mask = (
            (ablation_secondary_df["model_id"] == mid) &
            (ablation_secondary_df["scenario_id"] == sid) &
            (ablation_secondary_df["horizon_week"] == h) &
            (ablation_secondary_df["metric_name"] == "risk_auc_at_horizon")
        )
        auc_vals = ablation_secondary_df.loc[auc_mask, "metric_value"]
        row[f"risk_auc_h{h}"] = float(auc_vals.iloc[0]) if not auc_vals.empty else np.nan
    leaderboard_rows.append(row)

ablation_leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
    by=["model_id", "ibs", "c_index"], ascending=[True, True, False]
).reset_index(drop=True)

delta_rows = []
for mid in sorted(ablation_leaderboard_df["model_id"].unique()):
    ref = ablation_leaderboard_df[
        (ablation_leaderboard_df["model_id"] == mid) &
        (ablation_leaderboard_df["scenario_id"] == "full_features")
    ].iloc[0]
    for _, r in ablation_leaderboard_df[ablation_leaderboard_df["model_id"] == mid].iterrows():
        delta = {
            "model_id": mid,
            "scenario_id": r["scenario_id"],
            "scenario_label": r["scenario_label"],
            "delta_ibs_vs_full": float(r["ibs"] - ref["ibs"]),
            "delta_c_index_vs_full": float(r["c_index"] - ref["c_index"]),
        }
        for h in HORIZONS_WEEKS:
            delta[f"delta_brier_h{h}_vs_full"]       = float(r[f"brier_h{h}"] - ref[f"brier_h{h}"])
            delta[f"delta_calibration_h{h}_vs_full"] = float(r[f"calibration_h{h}"] - ref[f"calibration_h{h}"])
            delta[f"delta_risk_auc_h{h}_vs_full"]    = float(r[f"risk_auc_h{h}"] - ref[f"risk_auc_h{h}"])
        delta_rows.append(delta)

ablation_delta_df = pd.DataFrame(delta_rows).sort_values(by=["model_id", "scenario_id"]).reset_index(drop=True)

# ------------------------------
# 5) Persist to DuckDB
# ------------------------------
primary_table       = "table_ablation_discrete_primary_metrics"
brier_table         = "table_ablation_discrete_brier_by_horizon"
calibration_table   = "table_ablation_discrete_calibration_by_horizon"
secondary_table     = "table_ablation_discrete_secondary_metrics"
support_table       = "table_ablation_discrete_support_by_horizon"
pred_vs_obs_table   = "table_ablation_discrete_predicted_vs_observed_survival"
audit_table         = "table_ablation_discrete_training_audit"
leaderboard_table   = "table_ablation_discrete_leaderboard"
delta_table         = "table_ablation_discrete_delta_vs_full"
config_path         = METADATA_DIR / "ablation_discrete_run_summary.json"

persist_duckdb_artifact(ablation_primary_df,   primary_table,     "F3")
persist_duckdb_artifact(ablation_brier_df,      brier_table,       "F3")
persist_duckdb_artifact(ablation_calib_df,      calibration_table, "F3")
persist_duckdb_artifact(ablation_secondary_df,  secondary_table,   "F3")
persist_duckdb_artifact(ablation_support_df,    support_table,     "F3")
persist_duckdb_artifact(ablation_pvso_df,       pred_vs_obs_table, "F3")
persist_duckdb_artifact(ablation_audit_df,      audit_table,       "F3")
persist_duckdb_artifact(ablation_leaderboard_df, leaderboard_table, "F3")
persist_duckdb_artifact(ablation_delta_df,      delta_table,       "F3")

save_json(
    {
        "models_run": sorted(ablation_leaderboard_df["model_id"].unique().tolist()),
        "scenarios_run": sorted(ablation_leaderboard_df["scenario_id"].unique().tolist()),
        "horizons": [int(h) for h in HORIZONS_WEEKS],
        "n_total_runs": int(ablation_leaderboard_df.shape[0]),
        "placeholder_reason": None,
    },
    config_path,
)

print("\nF3 dynamic arm ablation complete.")
print("Models:", sorted(ablation_leaderboard_df["model_id"].unique().tolist()))
print("Scenarios:", sorted(ablation_leaderboard_df["scenario_id"].unique().tolist()))
display(ablation_leaderboard_df)
display(ablation_delta_df)

print(f"[END] F3.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 28
from datetime import datetime as _dt
print(f"[START] F4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F4 — Train and Evaluate Ablation Variants for Retained Comparable Models
# --------------------------------------------------------------
# Purpose:
#   Train and evaluate the ablation variants for the tuned
#   continuous-time model families:
#     - cox_tuned
#     - deepsurv_tuned
#     - rsf_tuned
#     - mtlr_tuned
#
# Methodological note:
#   This step reuses the benchmark split and the benchmark
#   evaluation protocol. The objective is to measure how much
#   performance changes when static or early-window feature
#   blocks are removed or isolated.
# ==============================================================

print("\n" + "=" * 70)
print("F4 — Train and Evaluate Ablation Variants for Retained Comparable Models")
print("=" * 70)
print("Methodological note: this step trains and evaluates ablation")
print("variants for the retained comparable-model subset only.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = [
    "OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR",
    "HORIZONS_WEEKS", "CALIBRATION_BINS", "RANDOM_SEED", "save_json"
]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy
import torch
import torchtuples as tt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

try:
    from pycox.evaluation import EvalSurv
    from pycox.models import CoxPH, MTLR
    PYCOX_AVAILABLE = True
except Exception:
    PYCOX_AVAILABLE = False

if not LIFELINES_AVAILABLE:
    raise ImportError("lifelines is required for P30.")

if not PYCOX_AVAILABLE:
    raise ImportError("pycox is required for P30.")

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
# 3) Registry / paths
# ------------------------------
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ablation_registry = load_duckdb_table_or_raise("table_ablation_variant_registry")

TARGET_MODELS = ["cox_tuned", "deepsurv_tuned", "rsf_tuned", "mtlr_tuned"]

ablation_registry_ct = ablation_registry[
    ablation_registry["model_id"].isin(TARGET_MODELS)
].copy()

DEEPSURV_CONFIG_PATH = METADATA_DIR / "deepsurv_tuned_model_config.json"
RSF_CONFIG_PATH = METADATA_DIR / "rsf_tuned_model_config.json"
MTLR_CONFIG_PATH = METADATA_DIR / "mtlr_tuned_model_config.json"

# ------------------------------
# 4) Column definitions
# ------------------------------
AUX_ENROLLMENT = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "duration",
    "duration_raw",
    "used_zero_week_fallback_for_censoring",
    "split",
    "time_for_split",
    "time_bucket",
    "event_time_bucket_label",
]

TARGET_COL = "event"
DURATION_COL = "duration"

CATEGORICAL_FEATURES = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

print(f"[END] F4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 30
from datetime import datetime as _dt
print(f"[START] F4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 5) Helpers
# ------------------------------
def load_variant(table_name: str) -> pd.DataFrame:
    return load_duckdb_table_or_raise(table_name)

def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required configuration file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print(f"[END] F4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 32
from datetime import datetime as _dt
print(f"[START] F4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_feature_columns(df: pd.DataFrame):
    excluded = set(AUX_ENROLLMENT + [TARGET_COL])
    return [c for c in df.columns if c not in excluded]

print(f"[END] F4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 34
from datetime import datetime as _dt
print(f"[START] F4.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def split_feature_types(feature_cols):
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols

print(f"[END] F4.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 36
from datetime import datetime as _dt
print(f"[START] F4.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_preprocessor(feature_cols):
    numeric_cols, categorical_cols = split_feature_types(feature_cols)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols

print(f"[END] F4.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 38
from datetime import datetime as _dt
print(f"[START] F4.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_surv_at_horizon(surv_frame: pd.DataFrame, h: int) -> pd.Series:
    idx = np.asarray(surv_frame.index, dtype=float)
    pos = np.searchsorted(idx, float(h), side="right") - 1
    if pos < 0:
        return pd.Series(np.ones(surv_frame.shape[1]), index=surv_frame.columns)
    return surv_frame.iloc[pos]


def build_surv_structured_array(durations: np.ndarray, events: np.ndarray):
    return Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))


def build_mtlr_network(input_dim: int, candidate: dict, out_features: int) -> tt.practical.MLPVanilla:
    return tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=list(candidate["hidden_dims"]),
        out_features=out_features,
        batch_norm=False,
        dropout=float(candidate["dropout"]),
    )


def normalize_mtlr_survival_frame(raw_survival_df: pd.DataFrame, enrollment_ids: list, max_week: int) -> pd.DataFrame:
    raw = raw_survival_df.copy()
    raw.columns = list(enrollment_ids)
    raw.columns.name = "enrollment_id"
    raw.index = pd.Index(pd.to_numeric(raw.index, errors="raise").astype(float), name="raw_time")
    raw = raw.sort_index()
    raw_values = raw.to_numpy(dtype=float)
    week_grid = np.arange(1, int(max_week) + 1, dtype=float)
    cut_times = raw.index.to_numpy(dtype=float)
    positions = np.searchsorted(cut_times, week_grid, side="right") - 1
    survival_matrix = np.ones((week_grid.shape[0], raw_values.shape[1]), dtype=float)
    valid_mask = positions >= 0
    if valid_mask.any():
        survival_matrix[valid_mask] = raw_values[positions[valid_mask]]
    survival_wide_df = pd.DataFrame(survival_matrix, index=week_grid.astype(int), columns=enrollment_ids)
    survival_wide_df.index.name = "week"
    return survival_wide_df.clip(lower=0.0, upper=1.0)

print(f"[END] F4.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 40
from datetime import datetime as _dt
print(f"[START] F4.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def evaluate_continuous_survival(surv_df: pd.DataFrame, truth_test: pd.DataFrame, horizons: list[int]):
    durations_test = truth_test["duration"].astype(int).to_numpy()
    events_test = truth_test["event"].astype(int).to_numpy()

    eval_surv = EvalSurv(
        surv=surv_df,
        durations=durations_test,
        events=events_test,
        censor_surv="km",
    )

    primary_rows = []
    try:
        c_index = float(eval_surv.concordance_td())
        c_index_note = "pycox_concordance_td"
    except Exception as e:
        c_index = np.nan
        c_index_note = f"failed: {str(e)}"

    try:
        max_requested_horizon = int(max(horizons))
        ibs_grid = np.arange(1, max_requested_horizon + 1, dtype=int)
        ibs_value = float(eval_surv.integrated_brier_score(ibs_grid))
        ibs_note = "pycox_integrated_brier_score"
    except Exception as e:
        ibs_value = np.nan
        ibs_note = f"failed: {str(e)}"

    primary_rows.append({"metric_name": "ibs", "metric_value": ibs_value, "notes": ibs_note})
    primary_rows.append({"metric_name": "c_index", "metric_value": c_index, "notes": c_index_note})
    primary_df = pd.DataFrame(primary_rows)

    try:
        brier_h = eval_surv.brier_score(np.array(horizons, dtype=int))
        brier_df = pd.DataFrame({
            "horizon_week": list(brier_h.index.astype(int)),
            "metric_name": ["brier_at_horizon"] * len(brier_h.index),
            "metric_value": list(brier_h.values.astype(float)),
            "notes": ["pycox_brier_score"] * len(brier_h.index),
        })
    except Exception as e:
        brier_df = pd.DataFrame({
            "horizon_week": horizons,
            "metric_name": ["brier_at_horizon"] * len(horizons),
            "metric_value": [np.nan] * len(horizons),
            "notes": [f"failed: {str(e)}"] * len(horizons),
        })

    support_rows = []
    calibration_rows = []
    pred_vs_obs_rows = []
    risk_auc_rows = []

    for h in horizons:
        pred_surv_h = get_surv_at_horizon(surv_df, h)
        pred_risk_h = 1.0 - pred_surv_h

        eval_df = truth_test.copy()
        eval_df["pred_survival_h"] = eval_df["enrollment_id"].map(pred_surv_h.to_dict())
        eval_df["pred_risk_h"] = eval_df["enrollment_id"].map(pred_risk_h.to_dict())

        eval_df["is_evaluable_at_h"] = (
            ((eval_df["event"] == 1) & (eval_df["duration"] <= h)) |
            (eval_df["duration"] >= h)
        ).astype(int)

        eval_df = eval_df[eval_df["is_evaluable_at_h"] == 1].copy()
        eval_df["observed_event_by_h"] = ((eval_df["event"] == 1) & (eval_df["duration"] <= h)).astype(int)
        eval_df["observed_survival_by_h"] = 1 - eval_df["observed_event_by_h"]

        support_rows.append({
            "horizon_week": h,
            "n_evaluable_enrollments": int(eval_df.shape[0]),
            "n_events_by_horizon": int(eval_df["observed_event_by_h"].sum()),
            "event_rate_by_horizon": float(eval_df["observed_event_by_h"].mean()) if eval_df.shape[0] > 0 else np.nan,
        })

        if eval_df["observed_event_by_h"].nunique() >= 2:
            risk_auc = roc_auc_score(eval_df["observed_event_by_h"], eval_df["pred_risk_h"])
        else:
            risk_auc = np.nan

        risk_auc_rows.append({
            "horizon_week": h,
            "metric_name": "risk_auc_at_horizon",
            "metric_value": float(risk_auc) if pd.notna(risk_auc) else np.nan,
            "notes": "roc_auc_on_evaluable_subset",
        })

        pred_vs_obs_rows.append({
            "horizon_week": h,
            "n_evaluable_enrollments": int(eval_df.shape[0]),
            "mean_predicted_survival": float(eval_df["pred_survival_h"].mean()) if eval_df.shape[0] > 0 else np.nan,
            "mean_observed_survival": float(eval_df["observed_survival_by_h"].mean()) if eval_df.shape[0] > 0 else np.nan,
            "abs_gap": float(abs(eval_df["pred_survival_h"].mean() - eval_df["observed_survival_by_h"].mean())) if eval_df.shape[0] > 0 else np.nan,
        })

        if eval_df.shape[0] > 0:
            ranked = eval_df["pred_risk_h"].rank(method="first")
            n_bins_eff = int(min(CALIBRATION_BINS, max(1, eval_df.shape[0])))
            eval_df["calibration_bin"] = pd.qcut(
                ranked,
                q=n_bins_eff,
                labels=False,
                duplicates="drop"
            )

            calib_tab = (
                eval_df.groupby("calibration_bin")
                .agg(
                    n=("enrollment_id", "count"),
                    mean_predicted_risk=("pred_risk_h", "mean"),
                    observed_event_rate=("observed_event_by_h", "mean"),
                )
                .reset_index()
            )
            calib_tab["horizon_week"] = h
            calib_tab["abs_calibration_gap"] = (
                calib_tab["mean_predicted_risk"] - calib_tab["observed_event_rate"]
            ).abs()

            calibration_rows.append({
                "horizon_week": h,
                "metric_name": "calibration_at_horizon",
                "metric_value": float(np.average(calib_tab["abs_calibration_gap"], weights=calib_tab["n"])),
                "notes": "Weighted absolute calibration gap across bins",
            })

    return {
        "primary": primary_df,
        "brier": pd.DataFrame(brier_df),
        "calibration": pd.DataFrame(calibration_rows),
        "secondary": pd.DataFrame(risk_auc_rows),
        "support": pd.DataFrame(support_rows),
        "pred_vs_obs": pd.DataFrame(pred_vs_obs_rows),
    }

print(f"[END] F4.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 42
from datetime import datetime as _dt
print(f"[START] F4.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 6) Train/evaluate ablation variants
# ------------------------------
results_primary = []
results_brier = []
results_calibration = []
results_secondary = []
results_support = []
results_pred_vs_obs = []
results_training_audit = []

deepsurv_cfg = read_json(DEEPSURV_CONFIG_PATH)
rsf_cfg = read_json(RSF_CONFIG_PATH)
mtlr_cfg = read_json(MTLR_CONFIG_PATH)
deepsurv_best = deepsurv_cfg["best_candidate"]
rsf_best = rsf_cfg["best_candidate"]["params"]
mtlr_best = mtlr_cfg["best_candidate"]

for _, reg_row in ablation_registry_ct.iterrows():
    model_id = reg_row["model_id"]
    scenario_id = reg_row["scenario_id"]
    scenario_label = reg_row["scenario_label"]

    train_df = load_variant(reg_row["train_table_name"])
    test_df = load_variant(reg_row["test_table_name"])

    feature_cols = get_feature_columns(train_df)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(feature_cols)

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray().astype(np.float32)
        X_test_dense = X_test.toarray().astype(np.float32)
    else:
        X_train_dense = np.asarray(X_train).astype(np.float32)
        X_test_dense = np.asarray(X_test).astype(np.float32)

    y_train_event = train_df[TARGET_COL].to_numpy().astype(np.int32)
    y_test_event = test_df[TARGET_COL].to_numpy().astype(np.int32)
    y_train_duration = train_df[DURATION_COL].to_numpy().astype(np.float32)
    y_test_duration = test_df[DURATION_COL].to_numpy().astype(np.float32)

    # ---------- model fit ----------
    if model_id == "cox_tuned":
        feature_names = [f"x{i}" for i in range(X_train_dense.shape[1])]
        train_fit_df = pd.DataFrame(X_train_dense, columns=feature_names)
        train_fit_df["duration"] = y_train_duration
        train_fit_df["event"] = y_train_event

        # train-only zero-variance filter
        stds = train_fit_df[feature_names].std(axis=0, ddof=0)
        keep_feature_names = stds[stds > 0].index.tolist()

        train_fit_df = train_fit_df[keep_feature_names + ["duration", "event"]].copy()
        test_fit_df = pd.DataFrame(X_test_dense, columns=feature_names)[keep_feature_names].copy()

        model = CoxPHFitter(
            penalizer=0.001,
            l1_ratio=0.0,
        )
        model.fit(
            train_fit_df,
            duration_col="duration",
            event_col="event",
            show_progress=False,
        )

        surv_df = model.predict_survival_function(
            test_fit_df,
            times=np.arange(0, int(np.max(y_test_duration)) + 1, dtype=int)
        ).copy()

    elif model_id == "deepsurv_tuned":
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        net = tt.practical.MLPVanilla(
            in_features=X_train_dense.shape[1],
            num_nodes=list(deepsurv_best["hidden_dims"]),
            out_features=1,
            batch_norm=True,
            dropout=float(deepsurv_best["dropout"]),
            output_bias=False,
        )

        model = CoxPH(net, tt.optim.AdamW)
        model.optimizer.set_lr(float(deepsurv_best["learning_rate"]))

        _ = model.fit(
            X_train_dense,
            (y_train_duration, y_train_event),
            batch_size=int(deepsurv_best.get("batch_size", 256)),
            epochs=int(deepsurv_best.get("best_epoch", 55)),
            verbose=False,
        )

        _ = model.compute_baseline_hazards()
        surv_df = model.predict_surv_df(X_test_dense)

    elif model_id == "rsf_tuned":
        y_train_struct = build_surv_structured_array(y_train_duration, y_train_event)
        model = RandomSurvivalForest(
            n_estimators=int(rsf_best["n_estimators"]),
            min_samples_leaf=int(rsf_best["min_samples_leaf"]),
            max_depth=None if rsf_best["max_depth"] is None else int(rsf_best["max_depth"]),
            max_features=rsf_best["max_features"],
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_dense, y_train_struct)

        times = np.arange(0, int(np.max(y_test_duration)) + 1, dtype=int)
        surv_functions = model.predict_survival_function(X_test_dense)
        surv_matrix = np.vstack([fn(times) for fn in surv_functions]).T
        surv_df = pd.DataFrame(surv_matrix, index=times)

    elif model_id == "mtlr_tuned":
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        train_durations = np.maximum(y_train_duration.astype(np.float32), 1.0)
        train_events = y_train_event.astype(np.int64)
        labtrans = MTLR.label_transform(int(mtlr_best["params"]["num_durations"]))
        y_train_mtlr = labtrans.fit_transform(train_durations, train_events)

        net = build_mtlr_network(
            input_dim=X_train_dense.shape[1],
            candidate=mtlr_best["params"],
            out_features=int(labtrans.out_features),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MTLR(
            net,
            tt.optim.Adam(
                lr=float(mtlr_best["params"]["learning_rate"]),
                weight_decay=float(mtlr_best["params"]["weight_decay"]),
            ),
            duration_index=labtrans.cuts,
            device=device,
        )
        _ = model.fit(
            X_train_dense,
            y_train_mtlr,
            batch_size=int(mtlr_best["params"]["batch_size"]),
            epochs=int(mtlr_best.get("best_epoch", 12)),
            verbose=False,
        )

        raw_surv_df = model.predict_surv_df(X_test_dense)
        surv_df = normalize_mtlr_survival_frame(
            raw_surv_df,
            test_df["enrollment_id"].tolist(),
            max_week=int(max(np.max(y_test_duration), max(HORIZONS_WEEKS))),
        )

    else:
        raise ValueError(f"Unsupported model_id in P30: {model_id}")

    surv_df.columns = test_df["enrollment_id"].tolist()
    surv_df.columns.name = "enrollment_id"
    surv_df.index = surv_df.index.astype(int)

    truth_test = test_df[["enrollment_id", "event", "duration"]].copy()

    eval_outputs = evaluate_continuous_survival(surv_df, truth_test, HORIZONS_WEEKS)

    primary_df = eval_outputs["primary"].copy()
    primary_df["model_id"] = model_id
    primary_df["scenario_id"] = scenario_id
    primary_df["scenario_label"] = scenario_label
    results_primary.append(primary_df)

    brier_df = eval_outputs["brier"].copy()
    brier_df["model_id"] = model_id
    brier_df["scenario_id"] = scenario_id
    brier_df["scenario_label"] = scenario_label
    results_brier.append(brier_df)

    calibration_df = eval_outputs["calibration"].copy()
    calibration_df["model_id"] = model_id
    calibration_df["scenario_id"] = scenario_id
    calibration_df["scenario_label"] = scenario_label
    results_calibration.append(calibration_df)

    secondary_df = eval_outputs["secondary"].copy()
    secondary_df["model_id"] = model_id
    secondary_df["scenario_id"] = scenario_id
    secondary_df["scenario_label"] = scenario_label
    results_secondary.append(secondary_df)

    support_df = eval_outputs["support"].copy()
    support_df["model_id"] = model_id
    support_df["scenario_id"] = scenario_id
    support_df["scenario_label"] = scenario_label
    results_support.append(support_df)

    pred_vs_obs_df = eval_outputs["pred_vs_obs"].copy()
    pred_vs_obs_df["model_id"] = model_id
    pred_vs_obs_df["scenario_id"] = scenario_id
    pred_vs_obs_df["scenario_label"] = scenario_label
    results_pred_vs_obs.append(pred_vs_obs_df)

    results_training_audit.append({
        "model_id": model_id,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "n_train_rows": int(train_df.shape[0]),
        "n_test_rows": int(test_df.shape[0]),
        "n_feature_columns_raw": int(len(feature_cols)),
        "n_numeric_features_raw": int(len(numeric_cols)),
        "n_categorical_features_raw": int(len(categorical_cols)),
        "n_features_after_transform": int(X_train_dense.shape[1]),
    })

# ------------------------------
# 7) Consolidate outputs
# ------------------------------
ablation_primary_df = pd.concat(results_primary, ignore_index=True)
ablation_brier_df = pd.concat(results_brier, ignore_index=True)
ablation_calibration_df = pd.concat(results_calibration, ignore_index=True)
ablation_secondary_df = pd.concat(results_secondary, ignore_index=True)
ablation_support_df = pd.concat(results_support, ignore_index=True)
ablation_pred_vs_obs_df = pd.concat(results_pred_vs_obs, ignore_index=True)
ablation_training_audit_df = pd.DataFrame(results_training_audit)

ablation_leaderboard_rows = []
for (model_id, scenario_id, scenario_label), g in ablation_primary_df.groupby(["model_id", "scenario_id", "scenario_label"]):
    row = {
        "model_id": model_id,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "ibs": float(g.loc[g["metric_name"] == "ibs", "metric_value"].iloc[0]),
        "c_index": float(g.loc[g["metric_name"] == "c_index", "metric_value"].iloc[0]),
    }
    for h in HORIZONS_WEEKS:
        row[f"brier_h{h}"] = float(
            ablation_brier_df[
                (ablation_brier_df["model_id"] == model_id) &
                (ablation_brier_df["scenario_id"] == scenario_id) &
                (ablation_brier_df["horizon_week"] == h)
            ]["metric_value"].iloc[0]
        )
        row[f"calibration_h{h}"] = float(
            ablation_calibration_df[
                (ablation_calibration_df["model_id"] == model_id) &
                (ablation_calibration_df["scenario_id"] == scenario_id) &
                (ablation_calibration_df["horizon_week"] == h)
            ]["metric_value"].iloc[0]
        )
        row[f"risk_auc_h{h}"] = float(
            ablation_secondary_df[
                (ablation_secondary_df["model_id"] == model_id) &
                (ablation_secondary_df["scenario_id"] == scenario_id) &
                (ablation_secondary_df["horizon_week"] == h) &
                (ablation_secondary_df["metric_name"] == "risk_auc_at_horizon")
            ]["metric_value"].iloc[0]
        )
    ablation_leaderboard_rows.append(row)

ablation_leaderboard_df = pd.DataFrame(ablation_leaderboard_rows).sort_values(
    by=["model_id", "ibs", "c_index"],
    ascending=[True, True, False]
).reset_index(drop=True)

delta_rows = []
for model_id in sorted(ablation_leaderboard_df["model_id"].unique()):
    ref = ablation_leaderboard_df[
        (ablation_leaderboard_df["model_id"] == model_id) &
        (ablation_leaderboard_df["scenario_id"] == "full_features")
    ].iloc[0]

    sub_df = ablation_leaderboard_df[ablation_leaderboard_df["model_id"] == model_id].copy()
    for _, r in sub_df.iterrows():
        delta = {
            "model_id": model_id,
            "scenario_id": r["scenario_id"],
            "scenario_label": r["scenario_label"],
            "delta_ibs_vs_full": float(r["ibs"] - ref["ibs"]),
            "delta_c_index_vs_full": float(r["c_index"] - ref["c_index"]),
        }
        for h in HORIZONS_WEEKS:
            delta[f"delta_brier_h{h}_vs_full"] = float(r[f"brier_h{h}"] - ref[f"brier_h{h}"])
            delta[f"delta_calibration_h{h}_vs_full"] = float(r[f"calibration_h{h}"] - ref[f"calibration_h{h}"])
            delta[f"delta_risk_auc_h{h}_vs_full"] = float(r[f"risk_auc_h{h}"] - ref[f"risk_auc_h{h}"])
        delta_rows.append(delta)

ablation_delta_vs_full_df = pd.DataFrame(delta_rows).sort_values(
    by=["model_id", "scenario_id"]
).reset_index(drop=True)

# ------------------------------
# 8) Save artifacts
# ------------------------------
primary_table = "table_ablation_continuous_primary_metrics"
brier_table = "table_ablation_continuous_brier_by_horizon"
calibration_table = "table_ablation_continuous_calibration_by_horizon"
secondary_table = "table_ablation_continuous_secondary_metrics"
support_table = "table_ablation_continuous_support_by_horizon"
pred_vs_obs_table = "table_ablation_continuous_predicted_vs_observed_survival"
audit_table = "table_ablation_continuous_training_audit"
leaderboard_table = "table_ablation_continuous_leaderboard"
delta_table = "table_ablation_continuous_delta_vs_full"
config_path = METADATA_DIR / "ablation_continuous_run_summary.json"

persist_duckdb_artifact(ablation_primary_df, primary_table, "F4")
persist_duckdb_artifact(ablation_brier_df, brier_table, "F4")
persist_duckdb_artifact(ablation_calibration_df, calibration_table, "F4")
persist_duckdb_artifact(ablation_secondary_df, secondary_table, "F4")
persist_duckdb_artifact(ablation_support_df, support_table, "F4")
persist_duckdb_artifact(ablation_pred_vs_obs_df, pred_vs_obs_table, "F4")
persist_duckdb_artifact(ablation_training_audit_df, audit_table, "F4")
persist_duckdb_artifact(ablation_leaderboard_df, leaderboard_table, "F4")
persist_duckdb_artifact(ablation_delta_vs_full_df, delta_table, "F4")

save_json(
    {
        "models_run": sorted(ablation_leaderboard_df["model_id"].unique().tolist()),
        "scenarios_run": sorted(ablation_leaderboard_df["scenario_id"].unique().tolist()),
        "horizons": [int(h) for h in HORIZONS_WEEKS],
        "n_total_runs": int(ablation_leaderboard_df.shape[0]),
        "benchmark_scope": "manuscript_audit_subset",
        "canonical_predictive_roster_size": 14,
    },
    config_path,
)

# ------------------------------
# 9) Output for feedback
# ------------------------------
print("\nAblation continuous training audit:")
display(ablation_training_audit_df)

print("\nAblation continuous leaderboard:")
display(ablation_leaderboard_df)

print("\nAblation continuous delta vs full:")
display(ablation_delta_vs_full_df)

print("\nSaved:")
print("- DuckDB table:", primary_table)
print("- DuckDB table:", brier_table)
print("- DuckDB table:", calibration_table)
print("- DuckDB table:", secondary_table)
print("- DuckDB table:", support_table)
print("- DuckDB table:", pred_vs_obs_table)
print("- DuckDB table:", audit_table)
print("- DuckDB table:", leaderboard_table)
print("- DuckDB table:", delta_table)
print("-", config_path.resolve())

print(f"[END] F4.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 44
from datetime import datetime as _dt
print(f"[START] F5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F5 — Consolidate Ablation Results Across the Manuscript Audit Subset
# --------------------------------------------------------------
# Purpose:
#   Consolidate the ablation results from:
#     - discrete-time tuned families (F3)
#     - continuous-time tuned families (F4)
#
# Methodological note:
#   This step does not train any model. It only consolidates the
#   ablation outputs already generated in previous steps.
#
# Main goals:
#   - create a unified ablation leaderboard
#   - compare deltas vs full_features across families
#   - prepare paper-friendly summary tables
# ==============================================================

print("\n" + "=" * 70)
print("F5 — Consolidate Ablation Results Across the Manuscript Audit Subset")
print("=" * 70)
print("Methodological note: this step consolidates ablation outputs only.")
print("No model is trained here.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["TABLES_DIR", "METADATA_DIR", "save_json", "HORIZONS_WEEKS"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------
# 2) Required files
# ------------------------------
required_tables_f5 = [
    "table_ablation_discrete_leaderboard",
    "table_ablation_discrete_delta_vs_full",
    "table_ablation_continuous_leaderboard",
    "table_ablation_continuous_delta_vs_full",
    "table_ablation_model_registry",
]
require_duckdb_tables(required_tables_f5, "F5")

# ------------------------------
# 3) Load inputs
# ------------------------------
ablation_discrete_leaderboard = load_duckdb_table_or_raise("table_ablation_discrete_leaderboard")
ablation_discrete_delta = load_duckdb_table_or_raise("table_ablation_discrete_delta_vs_full")

ablation_continuous_leaderboard = load_duckdb_table_or_raise("table_ablation_continuous_leaderboard")
ablation_continuous_delta = load_duckdb_table_or_raise("table_ablation_continuous_delta_vs_full")
ablation_model_registry = load_duckdb_table_or_raise("table_ablation_model_registry")

# ------------------------------
# 4) Consolidate leaderboards
# ------------------------------
ablation_leaderboard_all = pd.concat(
    [ablation_discrete_leaderboard, ablation_continuous_leaderboard],
    ignore_index=True
)

family_map = dict(zip(ablation_model_registry["model_id"], ablation_model_registry["family"]))
display_name_map = dict(zip(ablation_model_registry["model_id"], ablation_model_registry["display_name"]))
arm_map = dict(zip(ablation_model_registry["model_id"], ablation_model_registry["paper_methodological_arm"]))

ablation_leaderboard_all["family"] = ablation_leaderboard_all["model_id"].map(family_map)
ablation_leaderboard_all["display_name"] = ablation_leaderboard_all["model_id"].map(display_name_map)
ablation_leaderboard_all["paper_methodological_arm"] = ablation_leaderboard_all["model_id"].map(arm_map)

ablation_leaderboard_all = ablation_leaderboard_all[
    [
        "model_id", "display_name", "family", "paper_methodological_arm", "scenario_id", "scenario_label",
        "ibs", "c_index",
        "brier_h10", "calibration_h10", "risk_auc_h10",
        "brier_h20", "calibration_h20", "risk_auc_h20",
        "brier_h30", "calibration_h30", "risk_auc_h30",
    ]
].copy()

ablation_leaderboard_all = ablation_leaderboard_all.sort_values(
    by=["model_id", "ibs", "c_index"],
    ascending=[True, True, False]
).reset_index(drop=True)

# ------------------------------
# 5) Consolidate deltas
# ------------------------------
ablation_delta_all = pd.concat(
    [ablation_discrete_delta, ablation_continuous_delta],
    ignore_index=True
)

ablation_delta_all["family"] = ablation_delta_all["model_id"].map(family_map)
ablation_delta_all["display_name"] = ablation_delta_all["model_id"].map(display_name_map)
ablation_delta_all["paper_methodological_arm"] = ablation_delta_all["model_id"].map(arm_map)

ablation_delta_all = ablation_delta_all[
    [
        "model_id", "display_name", "family", "paper_methodological_arm", "scenario_id", "scenario_label",
        "delta_ibs_vs_full", "delta_c_index_vs_full",
        "delta_brier_h10_vs_full", "delta_calibration_h10_vs_full", "delta_risk_auc_h10_vs_full",
        "delta_brier_h20_vs_full", "delta_calibration_h20_vs_full", "delta_risk_auc_h20_vs_full",
        "delta_brier_h30_vs_full", "delta_calibration_h30_vs_full", "delta_risk_auc_h30_vs_full",
    ]
].copy()

ablation_delta_all = ablation_delta_all.sort_values(
    by=["model_id", "scenario_id"]
).reset_index(drop=True)

# ------------------------------
# 6) Paper-friendly summary:
#    compare scenario effects per model
# ------------------------------
summary_rows = []

for model_id in sorted(ablation_delta_all["model_id"].unique()):
    sub = ablation_delta_all[ablation_delta_all["model_id"] == model_id].copy()

    def row_for(scenario):
        r = sub[sub["scenario_id"] == scenario]
        return r.iloc[0] if not r.empty else None

    full_r = row_for("full_features")
    drop_static_r = row_for("drop_static_structural")
    drop_temporal_r = row_for("drop_temporal_signal")

    summary_rows.append({
        "model_id": model_id,
        "display_name": display_name_map.get(model_id, model_id),
        "family": family_map.get(model_id, "unknown"),
        "paper_methodological_arm": arm_map.get(model_id, "unknown"),
        "delta_ibs_drop_static": np.nan if drop_static_r is None else float(drop_static_r["delta_ibs_vs_full"]),
        "delta_ibs_drop_temporal": np.nan if drop_temporal_r is None else float(drop_temporal_r["delta_ibs_vs_full"]),
        "delta_c_index_drop_static": np.nan if drop_static_r is None else float(drop_static_r["delta_c_index_vs_full"]),
        "delta_c_index_drop_temporal": np.nan if drop_temporal_r is None else float(drop_temporal_r["delta_c_index_vs_full"]),
        "delta_risk_auc_h10_drop_static": np.nan if drop_static_r is None else float(drop_static_r["delta_risk_auc_h10_vs_full"]),
        "delta_risk_auc_h10_drop_temporal": np.nan if drop_temporal_r is None else float(drop_temporal_r["delta_risk_auc_h10_vs_full"]),
        "delta_risk_auc_h20_drop_static": np.nan if drop_static_r is None else float(drop_static_r["delta_risk_auc_h20_vs_full"]),
        "delta_risk_auc_h20_drop_temporal": np.nan if drop_temporal_r is None else float(drop_temporal_r["delta_risk_auc_h20_vs_full"]),
        "delta_risk_auc_h30_drop_static": np.nan if drop_static_r is None else float(drop_static_r["delta_risk_auc_h30_vs_full"]),
        "delta_risk_auc_h30_drop_temporal": np.nan if drop_temporal_r is None else float(drop_temporal_r["delta_risk_auc_h30_vs_full"]),
    })

ablation_summary_by_model = pd.DataFrame(summary_rows)

# dominance ratio / contrast helper
ablation_summary_by_model["ibs_temporal_vs_static_ratio"] = (
    ablation_summary_by_model["delta_ibs_drop_temporal"] /
    ablation_summary_by_model["delta_ibs_drop_static"].replace(0, np.nan)
)

ablation_summary_by_model["abs_cindex_temporal_vs_static_ratio"] = (
    ablation_summary_by_model["delta_c_index_drop_temporal"].abs() /
    ablation_summary_by_model["delta_c_index_drop_static"].abs().replace(0, np.nan)
)

# ------------------------------
# 7) Cross-family scenario comparison
# ------------------------------
scenario_comparison_rows = []

SCENARIOS_KEEP = [
    "full_features",
    "drop_static_structural",
    "drop_temporal_signal",
]

for scenario_id in SCENARIOS_KEEP:
    sub = ablation_leaderboard_all[ablation_leaderboard_all["scenario_id"] == scenario_id].copy()
    sub = sub.sort_values(by=["ibs", "c_index"], ascending=[True, False]).reset_index(drop=True)
    sub["rank_ibs"] = sub["ibs"].rank(method="min", ascending=True)
    sub["rank_c_index"] = sub["c_index"].rank(method="min", ascending=False)
    scenario_comparison_rows.append(sub)

ablation_scenario_comparison = pd.concat(scenario_comparison_rows, ignore_index=True)

# ------------------------------
# 8) Save outputs
# ------------------------------
leaderboard_all_table = "table_ablation_leaderboard_manuscript_subset"
delta_all_table = "table_ablation_delta_manuscript_subset"
summary_by_model_table = "table_ablation_summary_by_model"
scenario_comparison_table = "table_ablation_scenario_comparison"
config_path = METADATA_DIR / "ablation_consolidated_summary.json"

persist_duckdb_artifact(ablation_leaderboard_all, leaderboard_all_table, "F5")
persist_duckdb_artifact(ablation_delta_all, delta_all_table, "F5")
persist_duckdb_artifact(ablation_summary_by_model, summary_by_model_table, "F5")
persist_duckdb_artifact(ablation_scenario_comparison, scenario_comparison_table, "F5")

save_json(
    {
        "n_models": int(ablation_leaderboard_all["model_id"].nunique()),
        "n_rows_leaderboard": int(ablation_leaderboard_all.shape[0]),
        "n_rows_delta": int(ablation_delta_all.shape[0]),
        "scenarios_included": sorted(ablation_leaderboard_all["scenario_id"].unique().tolist()),
        "horizons": [int(h) for h in HORIZONS_WEEKS],
        "benchmark_scope": "manuscript_audit_subset",
        "canonical_predictive_roster_size": 14,
    },
    config_path,
)

# ------------------------------
# 9) Output for feedback
# ------------------------------
print("\nAblation leaderboard — manuscript audit subset:")
display(ablation_leaderboard_all)

print("\nAblation delta — manuscript audit subset:")
display(ablation_delta_all)

print("\nAblation summary by model:")
display(ablation_summary_by_model)

print("\nAblation scenario comparison:")
display(ablation_scenario_comparison)

print("\nSaved:")
print("- DuckDB table:", leaderboard_all_table)
print("- DuckDB table:", delta_all_table)
print("- DuckDB table:", summary_by_model_table)
print("- DuckDB table:", scenario_comparison_table)
print("-", config_path.resolve())

print(f"[END] F5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F5.CONV — Cross-Paradigm Ablation Convergence Analysis (§5.7)
# --------------------------------------------------------------
# Diagnoses whether the temporal–signal dominance finding is
# consistent across all eight model paradigms in the audit subset.
# Writes table_ablation_convergence_paradigm to DuckDB and CSV.
# ==============================================================

conv_abl = ablation_summary_by_model[
    [
        "model_id", "display_name", "family", "paper_methodological_arm",
        "delta_ibs_drop_temporal", "delta_ibs_drop_static",
        "ibs_temporal_vs_static_ratio",
        "delta_c_index_drop_temporal", "delta_c_index_drop_static",
    ]
].copy()

# Temporal dominates when REMOVING temporal features hurts MORE than removing static
conv_abl["temporal_dominates_ibs"]     = conv_abl["delta_ibs_drop_temporal"] > conv_abl["delta_ibs_drop_static"]
conv_abl["temporal_dominates_cindex"]  = conv_abl["delta_c_index_drop_temporal"].abs() > conv_abl["delta_c_index_drop_static"].abs()
conv_abl["convergence_verdict"] = conv_abl["temporal_dominates_ibs"].map(
    {True: "TEMPORALLY_DOMINANT", False: "MIXED_OR_STATIC_DOMINANT"}
)
conv_abl["ibs_ratio_vs_threshold_2x"]  = conv_abl["ibs_temporal_vs_static_ratio"] >= 2.0
conv_abl["paradigm_group"] = conv_abl["family"].str.split("_").str[:2].str.join("_")

n_models_total   = int(conv_abl.shape[0])
n_temporal_ibs   = int(conv_abl["temporal_dominates_ibs"].sum())
n_temporal_ci    = int(conv_abl["temporal_dominates_cindex"].sum())
pct_temporal_ibs = round(100.0 * n_temporal_ibs / max(n_models_total, 1), 1)

conv_meta = {
    "n_models_in_subset":       n_models_total,
    "n_temporal_dominant_ibs":  n_temporal_ibs,
    "pct_temporal_dominant_ibs": pct_temporal_ibs,
    "n_temporal_dominant_cindex": n_temporal_ci,
    "unanimous_ibs": n_temporal_ibs == n_models_total,
    "unanimous_cindex": n_temporal_ci == n_models_total,
    "paradigms_represented": sorted(conv_abl["paradigm_group"].unique().tolist()),
}

convergence_paradigm_table = "table_ablation_convergence_paradigm"
convergence_meta_path      = METADATA_DIR / "ablation_convergence_paradigm.json"
convergence_csv_path       = TABLES_DIR / "table_ablation_convergence_paradigm.csv"

persist_duckdb_artifact(conv_abl, convergence_paradigm_table, "F5.CONV")
conv_abl.to_csv(convergence_csv_path, index=False)
save_json(conv_meta, convergence_meta_path)

print("\nF5.CONV — Cross-paradigm ablation convergence:")
print(f"  Models in subset       : {n_models_total}")
print(f"  Temporal dominant (IBS): {n_temporal_ibs}/{n_models_total} ({pct_temporal_ibs}%)")
print(f"  Unanimous (IBS)        : {conv_meta['unanimous_ibs']}")
display(conv_abl[["model_id", "paper_methodological_arm", "delta_ibs_drop_temporal",
                   "delta_ibs_drop_static", "ibs_temporal_vs_static_ratio", "convergence_verdict"]])
print("Saved:", convergence_paradigm_table, "→", convergence_csv_path.resolve())

# %% Cell 46
from datetime import datetime as _dt
print(f"[START] F6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F6 — Consolidated Preprocessing and Tuning Audit
# --------------------------------------------------------------
# Purpose:
#   Consolidate preprocessing and tuning documentation across the
#   selected tuned manuscript-facing audit subset into a single audit
#   table suitable for editorial freezing and appendix use.
#
# Methodological note:
#   This step does not retrain any model.
#   It reuses previously exported preprocessing summaries,
#   preprocessing configs, tuning configs, and tuning-result tables.
#
# Scope:
#   Manuscript-facing audit subset only:
#   - cox_tuned
#   - deepsurv_tuned
#   - rsf_tuned
#   - mtlr_tuned
#
# Outputs:
#   - table_preprocessing_and_tuning_audit.csv
#   - preprocessing_and_tuning_audit_summary.json
# ==============================================================

print("\n" + "=" * 70)
print("F6 — Consolidated Preprocessing and Tuning Audit")
print("=" * 70)
print("Methodological note: this step consolidates preprocessing and")
print("tuning documentation across the selected manuscript-facing audit subset.")
print("No model is retrained and no benchmark metric is recomputed here.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

from pathlib import Path
import json
import numpy as np
import pandas as pd

print(f"[END] F6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 48
from datetime import datetime as _dt
print(f"[START] F6.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 2) Helper functions
# ------------------------------
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print(f"[END] F6.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 50
from datetime import datetime as _dt
print(f"[START] F6.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path

print(f"[END] F6.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 52
from datetime import datetime as _dt
print(f"[START] F6.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def safe_first(df: pd.DataFrame, col: str, default=np.nan):
    if col in df.columns and df.shape[0] > 0:
        return df.iloc[0][col]
    return default

print(f"[END] F6.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 54
from datetime import datetime as _dt
print(f"[START] F6.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def stringify_search_space(value):
    if value is None:
        return "not_reported"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}={v}")
        return "; ".join(parts)
    return str(value)

print(f"[END] F6.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 56
from datetime import datetime as _dt
print(f"[START] F6.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def stringify_list_field(value):
    if value is None:
        return "not_reported"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)

print(f"[END] F6.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 58
from datetime import datetime as _dt
print(f"[START] F6.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_optional_table(table_name: str) -> pd.DataFrame:
    df = load_duckdb_table_optional(table_name)
    if df is None:
        return pd.DataFrame()
    return df


# ------------------------------
# 3) Resolve inputs
# ------------------------------
# preprocessing summaries from DuckDB / configs from files
cox_preproc_config_path = METADATA_DIR / "cox_preprocessing_config.json"
deepsurv_preproc_config_path = METADATA_DIR / "deepsurv_preprocessing_config.json"
rsf_preproc_config_path = METADATA_DIR / "rsf_preprocessing_config.json"
mtlr_preproc_config_path = METADATA_DIR / "mtlr_preprocessing_config.json"

# tuning results from DuckDB / configs from files
cox_tuning_config_path = METADATA_DIR / "cox_tuned_model_config.json"
deepsurv_tuning_config_path = METADATA_DIR / "deepsurv_tuned_model_config.json"
rsf_tuning_config_path = METADATA_DIR / "rsf_tuned_model_config.json"
mtlr_tuning_config_path = METADATA_DIR / "mtlr_tuned_model_config.json"

# ------------------------------
# 4) Load artifacts
# ------------------------------
cox_preproc_summary = load_optional_table("table_cox_preprocessing_summary")
deepsurv_preproc_summary = load_optional_table("table_deepsurv_preprocessing_summary")
rsf_preproc_summary = load_optional_table("table_rsf_preprocessing_summary")
mtlr_preproc_summary = load_optional_table("table_mtlr_preprocessing_summary")

cox_preproc_config = read_json(cox_preproc_config_path)
deepsurv_preproc_config = read_json(deepsurv_preproc_config_path)
rsf_preproc_config = read_json(rsf_preproc_config_path)
mtlr_preproc_config = read_json(mtlr_preproc_config_path)

cox_tuning_results = load_optional_table("table_cox_tuning_results")
deepsurv_tuning_results = load_optional_table("table_deepsurv_tuning_results")
rsf_tuning_results = load_optional_table("table_rsf_tuning_results")
mtlr_tuning_results = load_optional_table("table_mtlr_tuning_results")

cox_tuning_config = read_json(cox_tuning_config_path)
deepsurv_tuning_config = read_json(deepsurv_tuning_config_path)
rsf_tuning_config = read_json(rsf_tuning_config_path)
mtlr_tuning_config = read_json(mtlr_tuning_config_path)

# ------------------------------
# 5) Build consolidated audit rows
# ------------------------------
audit_rows = []

# ---- cox tuned
audit_rows.append({
    "model_id": "cox_tuned",
    "display_name": "Cox Comparable (Tuned)",
    "family": "continuous_time_cox",
    "data_level": "enrollment_early_window",
    "raw_input_design": "static covariates + early-window summaries (4 weeks)",
    "numeric_imputation": safe_first(cox_preproc_summary, "numeric_imputation"),
    "categorical_imputation": safe_first(cox_preproc_summary, "categorical_imputation"),
    "categorical_encoding": safe_first(cox_preproc_summary, "categorical_encoding"),
    "numeric_scaling": safe_first(cox_preproc_summary, "numeric_scaling"),
    "fit_on_train_only": "yes",
    "class_imbalance_handling": "none",
    "n_input_features_raw": safe_first(cox_preproc_summary, "n_input_features_raw"),
    "n_features_after_transform": safe_first(cox_preproc_summary, "n_features_after_transform"),
    "validation_unit": "enrollment",
    "validation_strategy": "train/validation split on enrollment_id with event stratification when possible",
    "validation_fraction": 0.20,
    "selection_metric": cox_tuning_config.get("selection_metric", "val_neg_partial_log_likelihood"),
    "search_space": stringify_search_space({
        "penalizer_grid": cox_tuning_config.get("penalizer_grid"),
        "l1_ratio_grid": cox_tuning_config.get("l1_ratio_grid"),
    }),
    "best_candidate_summary": stringify_search_space(cox_tuning_config.get("best_candidate")),
    "preprocessing_note": cox_preproc_config.get("cox_positioning_note", "not_reported"),
    "tuning_note": cox_tuning_config.get("benchmark_positioning_note", "not_reported"),
})

# ---- deepsurv tuned
audit_rows.append({
    "model_id": "deepsurv_tuned",
    "display_name": "DeepSurv (Tuned)",
    "family": "continuous_time_deepsurv",
    "data_level": "enrollment_early_window",
    "raw_input_design": "static covariates + early-window summaries (4 weeks)",
    "numeric_imputation": safe_first(deepsurv_preproc_summary, "numeric_imputation"),
    "categorical_imputation": safe_first(deepsurv_preproc_summary, "categorical_imputation"),
    "categorical_encoding": safe_first(deepsurv_preproc_summary, "categorical_encoding"),
    "numeric_scaling": safe_first(deepsurv_preproc_summary, "numeric_scaling"),
    "fit_on_train_only": "yes",
    "class_imbalance_handling": "none",
    "n_input_features_raw": safe_first(deepsurv_preproc_summary, "n_input_features_raw"),
    "n_features_after_transform": safe_first(deepsurv_preproc_summary, "n_features_after_transform"),
    "validation_unit": "row_within_training_set",
    "validation_strategy": "random internal validation fraction on training rows",
    "validation_fraction": deepsurv_tuning_config.get("validation_fraction", 0.20),
    "selection_metric": deepsurv_tuning_config.get("selection_metric", "best_val_loss"),
    "search_space": stringify_search_space({
        "hidden_dims_grid": deepsurv_tuning_config.get("hidden_dims_grid"),
        "dropout_grid": deepsurv_tuning_config.get("dropout_grid"),
        "learning_rate_grid": deepsurv_tuning_config.get("learning_rate_grid"),
        "weight_decay_grid": deepsurv_tuning_config.get("weight_decay_grid"),
        "batch_norm": deepsurv_tuning_config.get("batch_norm"),
        "epochs": deepsurv_tuning_config.get("epochs"),
    }),
    "n_tuning_candidates": int(deepsurv_tuning_results.shape[0]),
    "early_stopping_used": "yes",
    "early_stopping_patience": deepsurv_tuning_config.get("patience", "not_reported"),
    "complexity_control": "network architecture grid + dropout + weight decay + early stopping + best epoch refit",
    "best_candidate_summary": stringify_search_space(deepsurv_tuning_config.get("best_candidate")),
    "preprocessing_note": deepsurv_preproc_config.get("comparability_note", "not_reported"),
    "tuning_note": deepsurv_tuning_config.get("benchmark_positioning_note", "not_reported"),
})

# ---- rsf tuned
audit_rows.append({
    "model_id": "rsf_tuned",
    "display_name": "Random Survival Forest (Tuned)",
    "family": "continuous_time_tree_ensemble",
    "data_level": "enrollment_early_window",
    "raw_input_design": "static covariates + early-window summaries (4 weeks)",
    "numeric_imputation": safe_first(rsf_preproc_summary, "numeric_imputation"),
    "categorical_imputation": safe_first(rsf_preproc_summary, "categorical_imputation"),
    "categorical_encoding": safe_first(rsf_preproc_summary, "categorical_encoding"),
    "numeric_scaling": safe_first(rsf_preproc_summary, "numeric_scaling"),
    "fit_on_train_only": "yes",
    "class_imbalance_handling": "none",
    "n_input_features_raw": safe_first(rsf_preproc_summary, "n_input_features_raw"),
    "n_features_after_transform": safe_first(rsf_preproc_summary, "n_features_after_transform"),
    "validation_unit": "enrollment",
    "validation_strategy": "train/validation split on enrollment_id with event stratification when possible",
    "validation_fraction": 0.20,
    "selection_metric": rsf_tuning_config.get("selection_metric", "val_ibs"),
    "search_space": stringify_search_space(rsf_tuning_config.get("search_space", rsf_tuning_config.get("best_candidate", {}).get("params"))),
    "n_tuning_candidates": int(rsf_tuning_results.shape[0]),
    "early_stopping_used": "no",
    "early_stopping_patience": "not_applicable",
    "complexity_control": "forest size + leaf size + depth + feature subsampling",
    "best_candidate_summary": stringify_search_space(rsf_tuning_config.get("best_candidate")),
    "preprocessing_note": rsf_preproc_config.get("comparability_note", "not_reported"),
    "tuning_note": rsf_tuning_config.get("benchmark_positioning_note", "not_reported"),
})

# ---- mtlr tuned
audit_rows.append({
    "model_id": "mtlr_tuned",
    "display_name": "Neural-MTLR (Tuned)",
    "family": "continuous_time_neural_mtlr",
    "data_level": "enrollment_early_window",
    "raw_input_design": "static covariates + early-window summaries (4 weeks)",
    "numeric_imputation": safe_first(mtlr_preproc_summary, "numeric_imputation"),
    "categorical_imputation": safe_first(mtlr_preproc_summary, "categorical_imputation"),
    "categorical_encoding": safe_first(mtlr_preproc_summary, "categorical_encoding"),
    "numeric_scaling": safe_first(mtlr_preproc_summary, "numeric_scaling"),
    "fit_on_train_only": "yes",
    "class_imbalance_handling": "none",
    "n_input_features_raw": safe_first(mtlr_preproc_summary, "n_input_features_raw"),
    "n_features_after_transform": safe_first(mtlr_preproc_summary, "n_features_after_transform"),
    "validation_unit": "enrollment",
    "validation_strategy": "train/validation split on enrollment_id with deterministic survival-time discretization",
    "validation_fraction": 0.10,
    "selection_metric": mtlr_tuning_config.get("selection_metric", "val_ibs"),
    "search_space": stringify_search_space(mtlr_tuning_config.get("search_space", mtlr_tuning_config.get("best_candidate", {}).get("params"))),
    "n_tuning_candidates": int(mtlr_tuning_results.shape[0]),
    "early_stopping_used": "yes",
    "early_stopping_patience": mtlr_tuning_config.get("early_stopping_patience", "not_reported"),
    "complexity_control": "discretization grid + network architecture + dropout + weight decay + early stopping",
    "best_candidate_summary": stringify_search_space(mtlr_tuning_config.get("best_candidate")),
    "preprocessing_note": mtlr_preproc_config.get("comparability_note", "not_reported"),
    "tuning_note": mtlr_tuning_config.get("benchmark_positioning_note", "not_reported"),
})

preprocessing_tuning_audit_df = pd.DataFrame(audit_rows)

# ------------------------------
# 6) Appendix-oriented compact version
# ------------------------------
appendix_audit_df = preprocessing_tuning_audit_df[
    [
        "display_name",
        "family",
        "data_level",
        "numeric_imputation",
        "categorical_imputation",
        "categorical_encoding",
        "numeric_scaling",
        "fit_on_train_only",
        "class_imbalance_handling",
        "validation_unit",
        "validation_strategy",
        "validation_fraction",
        "selection_metric",
        "n_tuning_candidates",
        "early_stopping_used",
        "early_stopping_patience",
        "complexity_control",
    ]
].copy()

appendix_audit_df = appendix_audit_df.rename(columns={
    "display_name": "model",
    "data_level": "input_level",
    "numeric_imputation": "num_impute",
    "categorical_imputation": "cat_impute",
    "categorical_encoding": "cat_encoding",
    "numeric_scaling": "num_scaling",
    "fit_on_train_only": "fit_train_only",
    "class_imbalance_handling": "imbalance",
    "validation_unit": "val_unit",
    "validation_strategy": "val_strategy",
    "validation_fraction": "val_frac",
    "selection_metric": "select_metric",
    "n_tuning_candidates": "n_candidates",
    "early_stopping_used": "early_stop",
    "early_stopping_patience": "patience",
    "complexity_control": "complexity_control",
})

# ------------------------------
# 7) Save outputs
# ------------------------------
audit_table = "table_preprocessing_and_tuning_audit"
appendix_audit_table = "table_appendix_preprocessing_and_tuning_audit_compact"
summary_json_path = METADATA_DIR / "preprocessing_and_tuning_audit_summary.json"

persist_duckdb_artifact(preprocessing_tuning_audit_df, audit_table, "F6")
persist_duckdb_artifact(appendix_audit_df, appendix_audit_table, "F6")

save_json(
    {
        "n_models": int(preprocessing_tuning_audit_df.shape[0]),
        "model_ids": preprocessing_tuning_audit_df["model_id"].tolist(),
        "appendix_ready_table": appendix_audit_table,
        "full_audit_table": audit_table,
        "benchmark_scope": "manuscript_audit_subset",
        "canonical_predictive_roster_size": 14,
        "notes": [
            "All preprocessing transformations were fit on training data only.",
            "Cox and RSF tuning do not use early stopping.",
            "DeepSurv and Neural-MTLR use neural-network training controls and tuned stopping horizons.",
            "This audit is intentionally restricted to the retained manuscript-facing comparable subset.",
        ],
    },
    summary_json_path,
)

# 8) Output for feedback
# ------------------------------
print("\nFull preprocessing and tuning audit:")
display(preprocessing_tuning_audit_df)

print("\nAppendix-oriented compact audit:")
display(appendix_audit_df)

print("\nSaved:")
print("- DuckDB table:", audit_table)
print("- DuckDB table:", appendix_audit_table)
print("-", summary_json_path.resolve())

print(f"[END] F6.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 60
from datetime import datetime as _dt
print(f"[START] F7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F7 — Bootstrap Uncertainty Audit for Tuned Benchmark Models (Revised v4)
# --------------------------------------------------------------
# Purpose:
#   Quantify uncertainty for the selected tuned manuscript-facing
#   audit subset using enrollment-level bootstrap resampling on the
#   held-out test set.
#
# Methodological note:
#   This step does NOT retrain models during each bootstrap draw.
#   It rebuilds the tuned models' test-set survival predictions once,
#   then performs enrollment-level resampling on the held-out test
#   enrollments to estimate uncertainty in:
#     - IBS
#     - time-dependent concordance
#     - Brier@10, Brier@20, Brier@30
#
#   It also computes rank-stability summaries to assess whether the
#   subset ordering is robust enough to support manuscript-facing
#   wording about relative stability inside this selected audit scope.
#
# Important correction in v4:
#   Bootstrap samples are materialized with synthetic unique ids
#   (boot_{iter}_{j}) so that surv_df columns remain unique and
#   perfectly aligned with the corresponding truth table.
# ==============================================================

print("\n" + "=" * 70)
print("F7 — Bootstrap Uncertainty Audit for Tuned Benchmark Models (Revised v4)")
print("=" * 70)
print("Methodological note: this step rebuilds tuned-model survival predictions")
print("on the held-out test set and performs enrollment-level bootstrap")
print("resampling to quantify uncertainty in the selected manuscript audit subset.")
print("No model is retrained inside bootstrap iterations.")

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "HORIZONS_WEEKS", "RANDOM_SEED"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

import pandas as pd
import scipy
import torch
import torch.nn as nn
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.util import Surv

try:
    from pycox.evaluation import EvalSurv
    from pycox.models import CoxPH, MTLR
    PYCOX_AVAILABLE = True
except Exception:
    PYCOX_AVAILABLE = False

try:
    import torchtuples as tt
    TORCHTUPLES_AVAILABLE = True
except Exception:
    TORCHTUPLES_AVAILABLE = False

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

if not PYCOX_AVAILABLE:
    raise ImportError("pycox is required for P31.6.")
if not TORCHTUPLES_AVAILABLE:
    raise ImportError("torchtuples is required for P31.6.")
if not LIFELINES_AVAILABLE:
    raise ImportError("lifelines is required for P31.6.")

# ------------------------------
# 2) Compatibility patch for SciPy / PyCox
# ------------------------------
SCIPY_SIMPS_PATCHED = False
SCIPY_SIMPS_NOTE = "not_needed"
try:
    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        def _simps_compat(y, x=None, dx=1.0, axis=-1, even=None):
            return scipy.integrate.simpson(y, x=x, dx=dx, axis=axis)
        scipy.integrate.simps = _simps_compat
        SCIPY_SIMPS_PATCHED = True
        SCIPY_SIMPS_NOTE = "patched_simps_to_simpson"
except Exception as e:
    SCIPY_SIMPS_NOTE = f"patch_failed: {str(e)}"

# ------------------------------
# 3) Config
# ------------------------------
BOOTSTRAP_CONFIG = {
    "n_bootstrap": 200,
    "ci_alpha": 0.05,
    "metrics": ["ibs", "c_index_td"] + [f"brier_h{h}" for h in HORIZONS_WEEKS],
    "max_horizon_for_ibs": int(max(HORIZONS_WEEKS)),
    "resampling_unit": "enrollment",
    "random_seed": int(RANDOM_SEED),
    "note": (
        "Enrollment-level bootstrap on the held-out test set using fixed tuned-model predictions."
    ),
    "inferential_scope": (
        "Quantifies held-out test-set sampling variability conditional on fixed tuned models and fixed survival predictions."
    ),
    "model_retraining_policy": "no_refit_within_bootstrap_iterations",
    "supported_use": (
        "Use for uncertainty intervals and cautious directional rank-stability language inside the selected manuscript audit subset."
    ),
    "unsupported_use": (
        "Do not use as a retraining-based generalization claim, null-hypothesis test, or proof of universally fixed hierarchy."
    ),
    "alternative_refit_status": "not_materialized_in_current_stage_due_compute_cost",
    "scipy_simps_note": SCIPY_SIMPS_NOTE,
    "sanity_tolerance_abs": 0.02,
}

N_BOOT = BOOTSTRAP_CONFIG["n_bootstrap"]
ALPHA = BOOTSTRAP_CONFIG["ci_alpha"]
LOW_Q = 100 * (ALPHA / 2)
HIGH_Q = 100 * (1 - ALPHA / 2)
SANITY_TOL = BOOTSTRAP_CONFIG["sanity_tolerance_abs"]

# ------------------------------
# 4) Paths
# ------------------------------
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"
CANONICAL_WINDOW_WEEKS = int(MAIN_ENROLLMENT_WINDOW_WEEKS)

COX_TRAIN_TABLE = "enrollment_cox_ready_train"
COX_TEST_TABLE = "enrollment_cox_ready_test"
COX_MODEL_PATH = MODELS_DIR / "cox_early_window_tuned.joblib"
COX_PREPROC_PATH = MODELS_DIR / "cox_preprocessor.joblib"
COX_STABILITY_PATH = "table_cox_tuned_stability_notes"
COX_PRIMARY_METRICS_PATH = "table_cox_tuned_primary_metrics"
COX_BRIER_PATH = "table_cox_tuned_brier_by_horizon"

DEEPSURV_TRAIN_TABLE = "enrollment_deepsurv_ready_train"
DEEPSURV_TEST_TABLE = "enrollment_deepsurv_ready_test"
DEEPSURV_MODEL_PATH = MODELS_DIR / "deepsurv_tuned.pt"
DEEPSURV_PREPROC_PATH = MODELS_DIR / "deepsurv_preprocessor.joblib"
DEEPSURV_CONFIG_PATH = METADATA_DIR / "deepsurv_tuned_model_config.json"
DEEPSURV_PRIMARY_METRICS_PATH = "table_deepsurv_tuned_primary_metrics"
DEEPSURV_BRIER_PATH = "table_deepsurv_tuned_brier_by_horizon"

RSF_TRAIN_TABLE = "enrollment_cox_ready_train"
RSF_TEST_TABLE = "enrollment_cox_ready_test"
RSF_MODEL_PATH = MODELS_DIR / "rsf_tuned.joblib"
RSF_PREPROC_PATH = MODELS_DIR / "rsf_preprocessor.joblib"
RSF_PRIMARY_METRICS_PATH = "table_rsf_tuned_primary_metrics"
RSF_BRIER_PATH = "table_rsf_tuned_brier_by_horizon"

MTLR_TRAIN_TABLE = "enrollment_cox_ready_train"
MTLR_TEST_TABLE = "enrollment_cox_ready_test"
MTLR_MODEL_PATH = MODELS_DIR / "mtlr_tuned.pt"
MTLR_PREPROC_PATH = MODELS_DIR / "mtlr_preprocessor.joblib"
MTLR_CONFIG_PATH = METADATA_DIR / "mtlr_tuned_model_config.json"
MTLR_PRIMARY_METRICS_PATH = "table_mtlr_tuned_primary_metrics"
MTLR_BRIER_PATH = "table_mtlr_tuned_brier_by_horizon"

available_tables_f7 = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
required_tables = [
    COX_TRAIN_TABLE, COX_TEST_TABLE,
    DEEPSURV_TRAIN_TABLE, DEEPSURV_TEST_TABLE,
    RSF_TRAIN_TABLE, RSF_TEST_TABLE,
    MTLR_TRAIN_TABLE, MTLR_TEST_TABLE,
]
missing_tables = [t for t in required_tables if t not in available_tables_f7]
if missing_tables:
    raise FileNotFoundError("Missing required DuckDB tables for P31.6:\n- " + "\n- ".join(missing_tables))

required_table_artifacts = [
    COX_STABILITY_PATH, COX_PRIMARY_METRICS_PATH, COX_BRIER_PATH,
    DEEPSURV_PRIMARY_METRICS_PATH, DEEPSURV_BRIER_PATH,
    RSF_PRIMARY_METRICS_PATH, RSF_BRIER_PATH,
    MTLR_PRIMARY_METRICS_PATH, MTLR_BRIER_PATH,
]
missing_table_artifacts = [table_name for table_name in required_table_artifacts if table_name not in available_tables_f7]
if missing_table_artifacts:
    raise FileNotFoundError("Missing required DuckDB artifacts for P31.6:\n- " + "\n- ".join(missing_table_artifacts))

required_paths = [
    COX_MODEL_PATH, COX_PREPROC_PATH,
    DEEPSURV_MODEL_PATH, DEEPSURV_PREPROC_PATH, DEEPSURV_CONFIG_PATH,
    RSF_MODEL_PATH, RSF_PREPROC_PATH,
    MTLR_MODEL_PATH, MTLR_PREPROC_PATH, MTLR_CONFIG_PATH,
]
missing_paths = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    raise FileNotFoundError("Missing required files for P31.6:\n- " + "\n- ".join(missing_paths))

# ------------------------------
# 5) Helpers
# ------------------------------
AUX_ENROLLMENT = [
    "enrollment_id", "id_student", "code_module", "code_presentation",
    "duration", "duration_raw", "used_zero_week_fallback_for_censoring",
    "split", "time_for_split", "time_bucket", "event_time_bucket_label"
]
TARGET_ENROLLMENT = ["event"]

AUX_DISCRETE = [
    "enrollment_id", "id_student", "code_module", "code_presentation",
    "event_observed", "t_event_week", "t_final_week",
    "used_zero_week_fallback_for_censoring", "split",
    "time_for_split", "time_bucket", "event_time_bucket_label"
]
TARGET_DISCRETE = ["event_t"]

print(f"[END] F7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 62
from datetime import datetime as _dt
print(f"[START] F7.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_csv(path_or_table) -> pd.DataFrame:
    table_name = path_or_table if isinstance(path_or_table, str) else infer_table_name_from_pathlike(path_or_table)
    return load_duckdb_table_or_raise(table_name)

print(f"[END] F7.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 64
from datetime import datetime as _dt
print(f"[START] F7.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print(f"[END] F7.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 66
from datetime import datetime as _dt
print(f"[START] F7.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def safe_save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

print(f"[END] F7.3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 68
from datetime import datetime as _dt
print(f"[START] F7.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def get_feature_cols(df: pd.DataFrame, aux_cols: list[str], target_cols: list[str]) -> list[str]:
    excluded = set(aux_cols + target_cols)
    return [c for c in df.columns if c not in excluded]

print(f"[END] F7.4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 70
from datetime import datetime as _dt
print(f"[START] F7.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_truth_from_discrete(df: pd.DataFrame) -> pd.DataFrame:
    event_col = "event_observed" if "event_observed" in df.columns else "event_t"
    duration_col = "time_for_split" if "time_for_split" in df.columns else "week"
    truth = (
        df.groupby("enrollment_id", as_index=False)
        .agg(
            event=(event_col, "max"),
            duration=(duration_col, "max"),
        )
    )

def build_truth_from_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    event_col = "event" if "event" in df.columns else "event_observed" if "event_observed" in df.columns else None
    duration_col = "duration" if "duration" in df.columns else "t_final_week" if "t_final_week" in df.columns else None
    if event_col is None or duration_col is None:
        raise KeyError("Enrollment-level truth builder requires event/duration columns compatible with the tuned benchmark inputs.")
    truth = df[["enrollment_id", event_col, duration_col]].copy()
    truth = truth.rename(columns={event_col: "event", duration_col: "duration"})
    truth["event"] = pd.to_numeric(truth["event"], errors="coerce").fillna(0).astype(int)
    truth["duration"] = pd.to_numeric(truth["duration"], errors="coerce").astype(int)
    truth = truth.drop_duplicates(subset=["enrollment_id"]).reset_index(drop=True)
    return truth


def build_surv_from_prediction_table(test_predictions_table: str) -> pd.DataFrame:
    pred_df = load_csv(test_predictions_table).copy()
    pred_df = pred_df.sort_values(["horizon_week", "enrollment_id"]).copy()
    surv_df = (
        pred_df[["enrollment_id", "horizon_week", "pred_survival_h"]]
        .drop_duplicates(subset=["enrollment_id", "horizon_week"])
        .pivot(index="horizon_week", columns="enrollment_id", values="pred_survival_h")
        .sort_index()
    )
    max_week = int(pd.to_numeric(pred_df["horizon_week"], errors="coerce").max())
    full_week_index = pd.Index(np.arange(1, max_week + 1), name="horizon_week")
    surv_df = surv_df.reindex(full_week_index).ffill().fillna(1.0).clip(lower=1e-8, upper=1.0)
    surv_df.columns.name = "enrollment_id"
    return surv_df


def normalize_mtlr_survival_frame(raw_survival_df: pd.DataFrame, enrollment_ids: list[str], max_week: int) -> pd.DataFrame:
    raw = raw_survival_df.copy()
    raw.columns = list(enrollment_ids)
    raw.columns.name = "enrollment_id"
    raw.index = pd.Index(pd.to_numeric(raw.index, errors="raise").astype(float), name="raw_time")
    raw = raw.sort_index()
    raw_values = raw.to_numpy(dtype=float)
    week_grid = np.arange(1, int(max_week) + 1, dtype=float)
    cut_times = raw.index.to_numpy(dtype=float)
    positions = np.searchsorted(cut_times, week_grid, side="right") - 1
    survival_matrix = np.ones((week_grid.shape[0], raw_values.shape[1]), dtype=float)
    valid_mask = positions >= 0
    if valid_mask.any():
        survival_matrix[valid_mask] = raw_values[positions[valid_mask]]
    survival_wide_df = pd.DataFrame(survival_matrix, index=week_grid.astype(int), columns=enrollment_ids)
    survival_wide_df.index.name = "week"
    return survival_wide_df.clip(lower=1e-8, upper=1.0)

def build_sksurv_structured_array(truth_df: pd.DataFrame):
    return Surv.from_arrays(
        event=pd.to_numeric(truth_df["event"], errors="coerce").fillna(0).astype(bool).to_numpy(),
        time=pd.to_numeric(truth_df["duration"], errors="coerce").astype(float).to_numpy(),
    )

print(f"[END] F7.5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 72
from datetime import datetime as _dt
print(f"[START] F7.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def align_truth_to_surv_df(truth_df: pd.DataFrame, surv_df: pd.DataFrame) -> pd.DataFrame:
    truth_idx = truth_df.set_index("enrollment_id")
    missing = [eid for eid in surv_df.columns.tolist() if eid not in truth_idx.index]
    if missing:
        raise KeyError(f"Missing enrollment_ids in truth_df alignment: {missing[:10]}")
    aligned = truth_idx.loc[list(surv_df.columns)].reset_index()
    return aligned

print(f"[END] F7.6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 74
from datetime import datetime as _dt
print(f"[START] F7.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def eval_surv_metrics_from_surv_df(
    surv_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    horizons: list[int],
    train_truth_df: pd.DataFrame | None = None,
) -> dict:
    truth_aligned = align_truth_to_surv_df(truth_df, surv_df)

    durations = truth_aligned["duration"].astype(int).to_numpy()
    events = truth_aligned["event"].astype(int).to_numpy()

    ev = EvalSurv(
        surv=surv_df,
        durations=durations,
        events=events,
        censor_surv="km",
    )

    out = {}

    try:
        out["c_index_td"] = float(ev.concordance_td())
    except Exception:
        out["c_index_td"] = np.nan

    for h in horizons:
        out[f"brier_h{int(h)}"] = np.nan

    if train_truth_df is not None:
        try:
            y_train_surv = build_sksurv_structured_array(train_truth_df)
            y_test_surv = build_sksurv_structured_array(truth_aligned)
            max_supported_brier_week = int(np.max(durations)) - 1
            if max_supported_brier_week >= 1:
                brier_time_grid = np.arange(1, min(int(max(horizons)), max_supported_brier_week) + 1, dtype=int)
                survival_estimate_matrix = surv_df.loc[brier_time_grid, truth_aligned["enrollment_id"].tolist()].to_numpy(dtype=float).T
                brier_times, brier_scores = sksurv_brier_score(
                    y_train_surv,
                    y_test_surv,
                    survival_estimate_matrix,
                    brier_time_grid.astype(float),
                )
                if len(brier_times) < 2:
                    out["ibs"] = float(brier_scores.astype(float)[0])
                else:
                    out["ibs"] = float(
                        np.trapezoid(brier_scores.astype(float), brier_times.astype(float))
                        / (float(brier_times[-1]) - float(brier_times[0]))
                    )
                brier_lookup = dict(zip(brier_times.astype(int), brier_scores.astype(float)))
                for h in horizons:
                    if int(h) in brier_lookup:
                        out[f"brier_h{int(h)}"] = float(brier_lookup[int(h)])
            else:
                out["ibs"] = np.nan
        except Exception:
            out["ibs"] = np.nan
    else:
        try:
            out["ibs"] = float(ev.integrated_brier_score(np.arange(1, int(max(horizons)) + 1)))
        except Exception:
            out["ibs"] = np.nan

        try:
            brier_h = ev.brier_score(np.array(horizons, dtype=int))
            for h, v in zip(brier_h.index.astype(int), brier_h.values.astype(float)):
                out[f"brier_h{int(h)}"] = float(v)
        except Exception:
            pass

    return out

print(f"[END] F7.7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 76
from datetime import datetime as _dt
print(f"[START] F7.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def bootstrap_eval_surv_metrics(
    surv_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    model_label: str,
    horizons: list[int],
    n_boot: int,
    rng: np.random.Generator,
    train_truth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    truth_aligned = align_truth_to_surv_df(truth_df, surv_df)
    enrollment_ids = truth_aligned["enrollment_id"].tolist()
    n = len(enrollment_ids)
    rows = []

    # base lookup once
    truth_lookup = truth_aligned.set_index("enrollment_id")
    surv_lookup = surv_df.copy()

    for b in range(1, n_boot + 1):
        sampled_ids = rng.choice(enrollment_ids, size=n, replace=True).tolist()
        boot_ids = [f"boot_{b}_{j}" for j in range(n)]

        # bootstrap truth with unique synthetic ids
        truth_sample = truth_lookup.loc[sampled_ids].reset_index()
        truth_sample["enrollment_id"] = boot_ids

        # bootstrap survival with same unique synthetic ids
        surv_sample = surv_lookup.loc[:, sampled_ids].copy()
        surv_sample.columns = boot_ids
        surv_sample.columns.name = "enrollment_id"

        metrics = eval_surv_metrics_from_surv_df(
            surv_sample,
            truth_sample,
            horizons,
            train_truth_df=train_truth_df,
        )

        row = {"model": model_label, "bootstrap_iter": b}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)

print(f"[END] F7.8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 78
from datetime import datetime as _dt
print(f"[START] F7.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def summarize_bootstrap(boot_df: pd.DataFrame, point_row: dict, metric_cols: list[str], model_label: str) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        vals = boot_df[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vals) == 0:
            rows.append({
                "model": model_label,
                "metric": metric,
                "point_estimate": point_row.get(metric, np.nan),
                "bootstrap_mean": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_successful_bootstrap": 0,
            })
        else:
            rows.append({
                "model": model_label,
                "metric": metric,
                "point_estimate": point_row.get(metric, np.nan),
                "bootstrap_mean": float(np.mean(vals)),
                "ci_lower": float(np.percentile(vals, LOW_Q)),
                "ci_upper": float(np.percentile(vals, HIGH_Q)),
                "n_successful_bootstrap": int(len(vals)),
            })
    return pd.DataFrame(rows)

print(f"[END] F7.9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 80
from datetime import datetime as _dt
print(f"[START] F7.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_rank_stability(boot_long: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    rows = []
    metric_df = boot_long[["model", "bootstrap_iter", metric]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for boot_iter, g in metric_df.groupby("bootstrap_iter"):
        g = g.copy()
        g["rank"] = g[metric].rank(method="min", ascending=not higher_is_better)
        for _, r in g.iterrows():
            rows.append({
                "bootstrap_iter": int(boot_iter),
                "model": r["model"],
                "metric": metric,
                "rank": int(r["rank"]),
            })
    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return pd.DataFrame(columns=["model", "metric", "rank", "frequency", "proportion"])

    out = (
        rank_df.groupby(["model", "metric", "rank"])
        .size()
        .reset_index(name="frequency")
    )
    total_per_model_metric = out.groupby(["model", "metric"])["frequency"].transform("sum")
    out["proportion"] = out["frequency"] / total_per_model_metric
    return out.sort_values(["metric", "rank", "model"]).reset_index(drop=True)

print(f"[END] F7.10 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 82
from datetime import datetime as _dt
print(f"[START] F7.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def build_pairwise_win_probability(boot_long: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    models = sorted(boot_long["model"].unique().tolist())
    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            tmp1 = boot_long[boot_long["model"] == m1][["bootstrap_iter", metric]].rename(columns={metric: "v1"})
            tmp2 = boot_long[boot_long["model"] == m2][["bootstrap_iter", metric]].rename(columns={metric: "v2"})
            merged = tmp1.merge(tmp2, on="bootstrap_iter", how="inner")
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
            if merged.empty:
                prob_m1_better = np.nan
                prob_m2_better = np.nan
            else:
                if higher_is_better:
                    prob_m1_better = float((merged["v1"] > merged["v2"]).mean())
                    prob_m2_better = float((merged["v2"] > merged["v1"]).mean())
                else:
                    prob_m1_better = float((merged["v1"] < merged["v2"]).mean())
                    prob_m2_better = float((merged["v2"] < merged["v1"]).mean())
            rows.append({
                "metric": metric,
                "model_a": m1,
                "model_b": m2,
                "prob_a_better_than_b": prob_m1_better,
                "prob_b_better_than_a": prob_m2_better,
                "n_bootstrap_pairs": int(merged.shape[0]) if not merged.empty else 0,
            })
    return pd.DataFrame(rows)

print(f"[END] F7.11 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 84
from datetime import datetime as _dt
print(f"[START] F7.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_frozen_metrics(primary_path: Path, brier_path: Path, model_name: str) -> dict:
    primary = load_duckdb_table_or_raise(infer_table_name_from_pathlike(primary_path))
    brier = load_duckdb_table_or_raise(infer_table_name_from_pathlike(brier_path))

    out = {"model": model_name}
    if "metric_name" in primary.columns:
        ibs_row = primary[primary["metric_name"].isin(["ibs", "IBS"])]
        cidx_row = primary[primary["metric_name"].isin(["c_index", "c_index_td", "C-index"])]
        out["ibs"] = float(ibs_row["metric_value"].iloc[0]) if ibs_row.shape[0] > 0 else np.nan
        out["c_index_td"] = float(cidx_row["metric_value"].iloc[0]) if cidx_row.shape[0] > 0 else np.nan
    else:
        out["ibs"] = np.nan
        out["c_index_td"] = np.nan

    if "horizon_week" in brier.columns:
        brier = brier.copy()
        brier["horizon_week"] = pd.to_numeric(brier["horizon_week"], errors="coerce")
        value_col = "metric_value" if "metric_value" in brier.columns else "brier_value" if "brier_value" in brier.columns else None
        for h in HORIZONS_WEEKS:
            row = brier[brier["horizon_week"] == h]
            out[f"brier_h{h}"] = float(pd.to_numeric(row[value_col], errors="coerce").iloc[0]) if row.shape[0] > 0 and value_col else np.nan
    else:
        for h in HORIZONS_WEEKS:
            out[f"brier_h{h}"] = np.nan

    return out

print(f"[END] F7.12 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 86
from datetime import datetime as _dt
print(f"[START] F7.13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# 6) Model builders / predictors
# ------------------------------
class TunedHazardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

print(f"[END] F7.13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 88
from datetime import datetime as _dt
print(f"[START] F7.14 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def predict_proba_in_batches_torch(model, X_np: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[start:start+batch_size].astype(np.float32)).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(p)
    return np.concatenate(probs, axis=0)

print(f"[END] F7.14 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 90
from datetime import datetime as _dt
print(f"[START] F7.15 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def rebuild_linear_surv():
    train_df = load_csv(LINEAR_TRAIN_TABLE).copy()
    test_pred_df = load_csv(LINEAR_TEST_PREDICTIONS_TABLE).copy()
    test_pred_df = test_pred_df.sort_values(["enrollment_id", "week"]).copy()

    surv_wide = (
        test_pred_df[["enrollment_id", "week", "pred_survival"]]
        .drop_duplicates(subset=["enrollment_id", "week"])
        .pivot(index="week", columns="enrollment_id", values="pred_survival")
        .sort_index()
    )
    max_week_test = int(pd.to_numeric(test_pred_df["week"], errors="coerce").max())
    full_week_index = pd.Index(np.arange(0, max_week_test + 1), name="week")
    surv_wide = surv_wide.reindex(full_week_index).ffill().clip(lower=1e-8, upper=1.0)

    truth_df = build_truth_from_discrete(test_pred_df)
    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_wide.columns)].copy()

    train_truth_df = build_truth_from_discrete(train_df)

    return surv_wide, truth_df, train_truth_df

print(f"[END] F7.15 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 92
from datetime import datetime as _dt
print(f"[START] F7.16 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def rebuild_neural_surv():
    train_df = load_csv(NEURAL_TRAIN_TABLE).copy()
    test_pred_df = load_csv(NEURAL_TEST_PREDICTIONS_TABLE).copy()
    test_pred_df = test_pred_df.sort_values(["enrollment_id", "week"]).copy()

    truth_df = build_truth_from_discrete(test_pred_df)
    max_duration_test = int(pd.to_numeric(truth_df["duration"], errors="coerce").max())
    eval_time_grid_full = np.arange(0, max_duration_test + 1, dtype=int)

    surv_wide = (
        test_pred_df[["enrollment_id", "week", "pred_survival"]]
        .drop_duplicates(subset=["enrollment_id", "week"])
        .pivot(index="week", columns="enrollment_id", values="pred_survival")
        .sort_index()
    )
    surv_wide = surv_wide.reindex(eval_time_grid_full).ffill().fillna(1.0).clip(lower=1e-8, upper=1.0)
    surv_wide.columns.name = "enrollment_id"

    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_wide.columns)].copy()

    train_truth_df = build_truth_from_discrete(train_df)

    return surv_wide, truth_df, train_truth_df

print(f"[END] F7.16 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 94
from datetime import datetime as _dt
print(f"[START] F7.17 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def rebuild_cox_surv():
    train_df = load_csv(COX_TRAIN_TABLE)
    test_df = load_csv(COX_TEST_TABLE)
    model = joblib.load(COX_MODEL_PATH)
    preproc = joblib.load(COX_PREPROC_PATH)

    feature_cols = get_feature_cols(test_df, AUX_ENROLLMENT, TARGET_ENROLLMENT)
    X_test = preproc.transform(test_df[feature_cols])
    feature_names_out = list(preproc.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test, columns=feature_names_out)

    times = np.arange(0, int(pd.to_numeric(test_df["duration"], errors="coerce").max()) + 1, dtype=int)
    pred_surv = model.predict_survival_function(X_test_df, times=times)

    surv_df = pred_surv.copy()
    surv_df.columns = test_df["enrollment_id"].tolist()
    surv_df.columns.name = "enrollment_id"
    surv_df.index = surv_df.index.astype(int)
    surv_df = surv_df.clip(lower=1e-8, upper=1.0)

    truth_df = build_truth_from_enrollment(test_df)
    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_df.columns)].copy()

    train_truth_df = build_truth_from_enrollment(train_df)

    return surv_df, truth_df, train_truth_df

print(f"[END] F7.17 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 96
from datetime import datetime as _dt
print(f"[START] F7.18 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

def rebuild_deepsurv_surv():
    train_df = load_csv(DEEPSURV_TRAIN_TABLE)
    test_df = load_csv(DEEPSURV_TEST_TABLE)
    preproc = joblib.load(DEEPSURV_PREPROC_PATH)
    cfg = read_json(DEEPSURV_CONFIG_PATH)
    best = cfg["best_candidate"]

    feature_cols = get_feature_cols(test_df, AUX_ENROLLMENT, TARGET_ENROLLMENT)

    X_train = preproc.transform(train_df[feature_cols]).astype(np.float32)
    X_test = preproc.transform(test_df[feature_cols]).astype(np.float32)

    y_train = (
        pd.to_numeric(train_df["duration"], errors="coerce").astype(np.float32).to_numpy(),
        pd.to_numeric(train_df["event"], errors="coerce").astype(np.float32).to_numpy(),
    )

    net = tt.practical.MLPVanilla(
        in_features=X_train.shape[1],
        num_nodes=best["hidden_dims"],
        out_features=1,
        batch_norm=True,
        dropout=float(best["dropout"]),
        output_bias=False,
    )

    model = CoxPH(net, tt.optim.AdamW)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(DEEPSURV_MODEL_PATH, map_location=device)
    model.net.load_state_dict(state_dict)
    model.compute_baseline_hazards(input=X_train, target=y_train)

    surv_df = model.predict_surv_df(X_test)
    surv_df.columns = test_df["enrollment_id"].tolist()
    surv_df.columns.name = "enrollment_id"
    surv_df.index = surv_df.index.astype(float)
    surv_df = surv_df.clip(lower=1e-8, upper=1.0)

    truth_df = build_truth_from_enrollment(test_df)
    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_df.columns)].copy()

    train_truth_df = build_truth_from_enrollment(train_df)

    return surv_df, truth_df, train_truth_df

print(f"[END] F7.18 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 98
from datetime import datetime as _dt
print(f"[START] F7.19 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")


def rebuild_rsf_surv():
    train_df = load_csv(RSF_TRAIN_TABLE)
    test_df = load_csv(RSF_TEST_TABLE)
    model = joblib.load(RSF_MODEL_PATH)
    preproc = joblib.load(RSF_PREPROC_PATH)

    feature_cols = get_feature_cols(test_df, AUX_ENROLLMENT, TARGET_ENROLLMENT)
    X_test = preproc.transform(test_df[feature_cols])
    times = np.arange(0, int(pd.to_numeric(test_df["duration"], errors="coerce").max()) + 1, dtype=int)
    surv_functions = model.predict_survival_function(X_test)
    surv_matrix = np.vstack([fn(times) for fn in surv_functions]).T
    surv_df = pd.DataFrame(surv_matrix, index=times, columns=test_df["enrollment_id"].tolist()).clip(lower=1e-8, upper=1.0)
    surv_df.columns.name = "enrollment_id"

    truth_df = build_truth_from_enrollment(test_df)
    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_df.columns)].copy()
    train_truth_df = build_truth_from_enrollment(train_df)
    return surv_df, truth_df, train_truth_df


def rebuild_mtlr_surv():
    train_df = load_csv(MTLR_TRAIN_TABLE)
    test_df = load_csv(MTLR_TEST_TABLE)
    preproc = joblib.load(MTLR_PREPROC_PATH)
    cfg = read_json(MTLR_CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_cols = get_feature_cols(test_df, AUX_ENROLLMENT, TARGET_ENROLLMENT)
    X_test = np.asarray(preproc.transform(test_df[feature_cols]), dtype=np.float32)

    net = torch.load(MTLR_MODEL_PATH, map_location=device, weights_only=False)
    model = MTLR(
        net,
        tt.optim.Adam,
        duration_index=np.asarray(cfg["duration_index"], dtype=float),
        device=device,
    )
    raw_surv_df = model.predict_surv_df(X_test)
    surv_df = normalize_mtlr_survival_frame(
        raw_surv_df,
        test_df["enrollment_id"].astype(str).tolist(),
        max_week=int(max(float(pd.to_numeric(test_df["duration"], errors="coerce").max()), float(max(HORIZONS_WEEKS)))),
    )

    truth_df = build_truth_from_enrollment(test_df)
    truth_df = truth_df[truth_df["enrollment_id"].isin(surv_df.columns)].copy()
    train_truth_df = build_truth_from_enrollment(train_df)
    return surv_df, truth_df, train_truth_df

# ------------------------------
# 7) Rebuild tuned-model survival predictions
# ------------------------------
print("\nRebuilding tuned-model survival objects on the held-out test set...")

model_surv_objects = {}

surv_cox, truth_cox, train_truth_cox = rebuild_cox_surv()
model_surv_objects["Cox Comparable (Tuned)"] = {
    "surv_df": surv_cox,
    "truth_df": truth_cox,
    "train_truth_df": train_truth_cox,
    "frozen_metrics": load_frozen_metrics(COX_PRIMARY_METRICS_PATH, COX_BRIER_PATH, "Cox Comparable (Tuned)"),
}

surv_deepsurv, truth_deepsurv, train_truth_deepsurv = rebuild_deepsurv_surv()
model_surv_objects["DeepSurv (Tuned)"] = {
    "surv_df": surv_deepsurv,
    "truth_df": truth_deepsurv,
    "train_truth_df": train_truth_deepsurv,
    "frozen_metrics": load_frozen_metrics(DEEPSURV_PRIMARY_METRICS_PATH, DEEPSURV_BRIER_PATH, "DeepSurv (Tuned)"),
}

surv_rsf, truth_rsf, train_truth_rsf = rebuild_rsf_surv()
model_surv_objects["Random Survival Forest (Tuned)"] = {
    "surv_df": surv_rsf,
    "truth_df": truth_rsf,
    "train_truth_df": train_truth_rsf,
    "frozen_metrics": load_frozen_metrics(RSF_PRIMARY_METRICS_PATH, RSF_BRIER_PATH, "Random Survival Forest (Tuned)"),
}

surv_mtlr, truth_mtlr, train_truth_mtlr = rebuild_mtlr_surv()
model_surv_objects["Neural-MTLR (Tuned)"] = {
    "surv_df": surv_mtlr,
    "truth_df": truth_mtlr,
    "train_truth_df": train_truth_mtlr,
    "frozen_metrics": load_frozen_metrics(MTLR_PRIMARY_METRICS_PATH, MTLR_BRIER_PATH, "Neural-MTLR (Tuned)"),
}

print("Done.")

# ------------------------------
# 8) Point estimates from rebuilt survival objects
# ------------------------------
point_rows = []
for model_name, obj in model_surv_objects.items():
    metrics = eval_surv_metrics_from_surv_df(
        obj["surv_df"],
        obj["truth_df"],
        HORIZONS_WEEKS,
        train_truth_df=obj.get("train_truth_df"),
    )
    row = {"model": model_name}
    row.update(metrics)
    point_rows.append(row)

point_estimates_df = pd.DataFrame(point_rows)

print("\nPoint estimates recomputed from rebuilt survival objects:")
display(point_estimates_df)

# ------------------------------
# 9) Sanity audit against frozen metrics
# ------------------------------
frozen_rows = []
for model_name, obj in model_surv_objects.items():
    frozen_rows.append(obj["frozen_metrics"])
frozen_estimates_df = pd.DataFrame(frozen_rows)

sanity_rows = []
metric_cols = ["ibs", "c_index_td"] + [f"brier_h{h}" for h in HORIZONS_WEEKS]
for _, point_row in point_estimates_df.iterrows():
    model_name = point_row["model"]
    frozen_row = frozen_estimates_df[frozen_estimates_df["model"] == model_name].iloc[0]
    for metric in metric_cols:
        rebuilt_val = point_row.get(metric, np.nan)
        frozen_val = frozen_row.get(metric, np.nan)
        shared_missing = bool(pd.isna(rebuilt_val) and pd.isna(frozen_val))
        abs_diff = abs(rebuilt_val - frozen_val) if pd.notna(rebuilt_val) and pd.notna(frozen_val) else np.nan
        sanity_rows.append({
            "model": model_name,
            "metric": metric,
            "rebuilt_value": rebuilt_val,
            "frozen_value": frozen_val,
            "abs_diff": abs_diff,
            "within_tolerance": True if shared_missing else (bool(abs_diff <= SANITY_TOL) if pd.notna(abs_diff) else False),
        })

sanity_audit_df = pd.DataFrame(sanity_rows)

print("\nSanity audit against frozen tuned-model metrics:")
display(sanity_audit_df)

bad_rows = sanity_audit_df[sanity_audit_df["within_tolerance"] == False].copy()

sanity_status = "pass"
sanity_severity = "none"
if bad_rows.shape[0] > 0:
    n_bad = int(bad_rows.shape[0])
    n_models_bad = int(bad_rows["model"].nunique()) if "model" in bad_rows.columns else n_bad
    n_metrics_bad = int(bad_rows["metric"].nunique()) if "metric" in bad_rows.columns else n_bad

    if n_bad == 1 and n_models_bad == 1 and n_metrics_bad == 1:
        sanity_status = "warning_single_metric_mismatch"
        sanity_severity = "warning"
        print("\nSanity warning: exactly one rebuilt-vs-frozen metric mismatch was detected.")
        print("Proceeding to bootstrap and preserving the mismatch in the audit outputs.")
    else:
        sanity_status = "fail_multiple_mismatches"
        sanity_severity = "fatal"
        raise RuntimeError(
            "Rebuilt survival objects failed sanity comparison against frozen metrics with "
            f"{n_bad} mismatched rows across {n_models_bad} model(s). "
            "Do not proceed to bootstrap until the alignment/rebuild issue is resolved."
        )

sanity_metadata_path = METADATA_DIR / "metadata_bootstrap_rebuild_sanity.json"
save_json(
    {
        "sanity_status": sanity_status,
        "sanity_severity": sanity_severity,
        "n_mismatched_rows": int(bad_rows.shape[0]),
        "n_total_rows": int(sanity_audit_df.shape[0]),
    },
    sanity_metadata_path,
)

# ------------------------------
# 10) Bootstrap uncertainty
# ------------------------------
rng = np.random.default_rng(BOOTSTRAP_CONFIG["random_seed"])

bootstrap_all = []
for model_name, obj in model_surv_objects.items():
    print(f"Bootstrapping: {model_name}")
    boot_df = bootstrap_eval_surv_metrics(
        surv_df=obj["surv_df"],
        truth_df=obj["truth_df"],
        model_label=model_name,
        horizons=HORIZONS_WEEKS,
        n_boot=N_BOOT,
        rng=rng,
        train_truth_df=obj.get("train_truth_df"),
    )
    bootstrap_all.append(boot_df)

bootstrap_long_df = pd.concat(bootstrap_all, ignore_index=True)

# ------------------------------
# 11) Summaries
# ------------------------------
summary_frames = []
for _, point_row in point_estimates_df.iterrows():
    model_name = point_row["model"]
    boot_df = bootstrap_long_df[bootstrap_long_df["model"] == model_name].copy()
    tmp = summarize_bootstrap(
        boot_df=boot_df,
        point_row=point_row.to_dict(),
        metric_cols=metric_cols,
        model_label=model_name,
    )
    summary_frames.append(tmp)

bootstrap_summary_df = pd.concat(summary_frames, ignore_index=True)

appendix_compact_df = bootstrap_summary_df.copy()
appendix_compact_df["ci_95"] = appendix_compact_df.apply(
    lambda r: (
        f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        if pd.notna(r["ci_lower"]) and pd.notna(r["ci_upper"])
        else "NA"
    ),
    axis=1,
)
appendix_compact_df = appendix_compact_df[[
    "model", "metric", "point_estimate", "ci_95", "n_successful_bootstrap"
]].copy()

# ------------------------------
# 12) Rank stability / pairwise stability
# ------------------------------
rank_ibs_df = build_rank_stability(bootstrap_long_df, metric="ibs", higher_is_better=False)
rank_cidx_df = build_rank_stability(bootstrap_long_df, metric="c_index_td", higher_is_better=True)
rank_stability_df = pd.concat([rank_ibs_df, rank_cidx_df], ignore_index=True)

pairwise_ibs_df = build_pairwise_win_probability(bootstrap_long_df, metric="ibs", higher_is_better=False)
pairwise_cidx_df = build_pairwise_win_probability(bootstrap_long_df, metric="c_index_td", higher_is_better=True)
pairwise_stability_df = pd.concat([pairwise_ibs_df, pairwise_cidx_df], ignore_index=True)

# ------------------------------
# 13) Save artifacts
# ------------------------------
point_estimates_path = TABLES_DIR / "table_bootstrap_point_estimates_tuned_models.csv"
bootstrap_long_path = TABLES_DIR / "table_bootstrap_metrics_long_tuned_models.csv"
bootstrap_summary_path = TABLES_DIR / "table_bootstrap_uncertainty_summary_tuned_models.csv"
appendix_compact_path = TABLES_DIR / "table_appendix_bootstrap_uncertainty_compact.csv"
rank_stability_path = TABLES_DIR / "table_bootstrap_rank_stability_tuned_models.csv"
pairwise_stability_path = TABLES_DIR / "table_bootstrap_pairwise_stability_tuned_models.csv"
sanity_audit_path = TABLES_DIR / "table_bootstrap_rebuild_sanity_audit.csv"
config_path = METADATA_DIR / "bootstrap_uncertainty_audit_config.json"

point_estimates_df.to_csv(point_estimates_path, index=False)
materialize_dataframe(con, point_estimates_df, infer_table_name_from_pathlike(point_estimates_path), "F7")
bootstrap_long_df.to_csv(bootstrap_long_path, index=False)
materialize_dataframe(con, bootstrap_long_df, infer_table_name_from_pathlike(bootstrap_long_path), "F7")
bootstrap_summary_df.to_csv(bootstrap_summary_path, index=False)
materialize_dataframe(con, bootstrap_summary_df, infer_table_name_from_pathlike(bootstrap_summary_path), "F7")
appendix_compact_df.to_csv(appendix_compact_path, index=False)
materialize_dataframe(con, appendix_compact_df, infer_table_name_from_pathlike(appendix_compact_path), "F7")
rank_stability_df.to_csv(rank_stability_path, index=False)
materialize_dataframe(con, rank_stability_df, infer_table_name_from_pathlike(rank_stability_path), "F7")
pairwise_stability_df.to_csv(pairwise_stability_path, index=False)
materialize_dataframe(con, pairwise_stability_df, infer_table_name_from_pathlike(pairwise_stability_path), "F7")
sanity_audit_df.to_csv(sanity_audit_path, index=False)
materialize_dataframe(con, sanity_audit_df, infer_table_name_from_pathlike(sanity_audit_path), "F7")

safe_save_json(
    {
        **BOOTSTRAP_CONFIG,
        "selected_models": TARGET_MODELS,
        "selected_model_display_names": [
            MANUSCRIPT_AUDIT_SUBSET_BY_MODEL_ID[model_id]["display_name"]
            for model_id in TARGET_MODELS
        ],
    },
    config_path,
)

# ------------------------------
# 14) Output for feedback
# ------------------------------
display(bootstrap_summary_df)

print("\nAppendix-oriented compact bootstrap summary:")
display(appendix_compact_df)

print("\nRank stability:")
display(rank_stability_df)

print("\nPairwise stability:")
display(pairwise_stability_df)

print("\nSaved:")
print("-", point_estimates_path.resolve())
print("-", bootstrap_long_path.resolve())
print("-", bootstrap_summary_path.resolve())
print("-", appendix_compact_path.resolve())
print("-", rank_stability_path.resolve())
print("-", pairwise_stability_path.resolve())
print("-", sanity_audit_path.resolve())
print("-", config_path.resolve())

print(f"[END] F7.19 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 100
from datetime import datetime as _dt
print(f"[START] F8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# F8 — Proportional Hazards Audit for the Comparable Cox Model
# (Revised v2)
# --------------------------------------------------------------
# Purpose:
#   Perform a formal proportional-hazards (PH) diagnostic for the
#   tuned comparable Cox model and export paper-ready audit artifacts.
# ==============================================================

print("\n" + "=" * 70)
print("F8 — Proportional Hazards Audit for the Comparable Cox Model (Revised v2)")
print("=" * 70)

# ------------------------------
# 1) Basic checks
# ------------------------------
required_names = ["OUTPUT_DIR", "TABLES_DIR", "METADATA_DIR", "save_json"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
    )

import warnings
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

if not LIFELINES_AVAILABLE:
    raise ImportError("lifelines is required for P31.7.")

# ------------------------------
# 2) Resolve raw/base Cox train dataframe
# ------------------------------
cox_train_base_df = None

candidate_df_names = [
    "train_df_cox",
    "cox_train_df",
    "enrollment_cox_ready_train",
    "df_cox_train",
]

for name in candidate_df_names:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, pd.DataFrame) and obj.shape[0] > 0:
            cox_train_base_df = obj.copy()
            print(f"Resolved Cox base dataframe from in-memory object: {name}")
            break

if cox_train_base_df is None and "con" in globals():
    candidate_queries = [
        "SELECT * FROM enrollment_cox_ready_train",
        "SELECT * FROM cox_ready_train",
        "SELECT * FROM train_df_cox",
    ]
    for q in candidate_queries:
        try:
            tmp = con.execute(q).fetchdf()
            if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 0:
                cox_train_base_df = tmp.copy()
                print(f"Resolved Cox base dataframe from SQL query: {q}")
                break
        except Exception:
            pass

if cox_train_base_df is None:
    raise NameError(
        "Could not resolve a Cox training dataframe "
        "(train_df_cox / cox_train_df / enrollment_cox_ready_train)."
    )

# ------------------------------
# 3) Resolve duration and event columns
# ------------------------------
duration_col = None
event_col = None

duration_candidates = ["duration", "time", "time_to_event", "T"]
event_candidates = ["event", "event_observed", "E"]

for c in duration_candidates:
    if c in cox_train_base_df.columns:
        duration_col = c
        break

for c in event_candidates:
    if c in cox_train_base_df.columns:
        event_col = c
        break

if duration_col is None or event_col is None:
    raise ValueError(
        f"Could not identify duration/event columns. "
        f"duration_col={duration_col}, event_col={event_col}"
    )

# ------------------------------
# 4) Try to resolve already-transformed Cox matrix
# ------------------------------
X_cox_train = None
cox_feature_names = None

X_candidates = [
    "X_train_cox",
    "X_cox_train",
    "X_train_comparable_cox",
    "X_train_continuous_time_cox",
]
feature_name_candidates = [
    "cox_feature_names_out",
    "feature_names_out_cox",
    "cox_selected_features",
    "cox_model_feature_cols",
    "cox_covariate_columns",
]

for name in X_candidates:
    if name in globals():
        X_candidate = globals()[name]
        try:
            if hasattr(X_candidate, "shape") and X_candidate.shape[0] == cox_train_base_df.shape[0]:
                X_cox_train = X_candidate
                print(f"Resolved transformed Cox matrix from: {name}")
                break
        except Exception:
            pass

for name in feature_name_candidates:
    if name in globals():
        f_candidate = globals()[name]
        if isinstance(f_candidate, (list, tuple, np.ndarray)) and len(f_candidate) > 0:
            cox_feature_names = [str(x) for x in f_candidate]
            print(f"Resolved Cox feature names from: {name}")
            break

# ------------------------------
# 5) If needed, resolve Cox preprocessor and transform
# ------------------------------
cox_preprocessor_obj = None
preprocessor_candidates = [
    "cox_preprocessor",
    "preprocessor_cox",
    "comparable_cox_preprocessor",
]

for name in preprocessor_candidates:
    if name in globals():
        cox_preprocessor_obj = globals()[name]
        print(f"Resolved Cox preprocessor from: {name}")
        break

if cox_preprocessor_obj is None and "COX_PREPROC_PATH" in globals():
    try:
        import joblib
        if Path(COX_PREPROC_PATH).exists():
            cox_preprocessor_obj = joblib.load(COX_PREPROC_PATH)
            print(f"Resolved Cox preprocessor from file: {COX_PREPROC_PATH}")
    except Exception as e:
        print(f"Warning: could not load Cox preprocessor from file: {e}")

exclude_cols = {
    duration_col, event_col,
    "enrollment_id", "id_student", "code_module", "code_presentation",
    "student_id", "module_presentation_id"
}

if X_cox_train is None:
    if cox_preprocessor_obj is None:
        raise ValueError(
            "Could not resolve a transformed Cox matrix and no Cox preprocessor "
            "was found to rebuild it."
        )

    raw_feature_cols = [c for c in cox_train_base_df.columns if c not in exclude_cols]
    if len(raw_feature_cols) == 0:
        raise ValueError("No raw feature columns available to rebuild Cox transformed matrix.")

    X_raw = cox_train_base_df[raw_feature_cols].copy()
    X_cox_train = cox_preprocessor_obj.transform(X_raw)
    print("Rebuilt transformed Cox matrix using the resolved Cox preprocessor.")

    if cox_feature_names is None:
        if hasattr(cox_preprocessor_obj, "get_feature_names_out"):
            cox_feature_names = [str(x) for x in cox_preprocessor_obj.get_feature_names_out()]
            print("Recovered transformed feature names from preprocessor.get_feature_names_out().")

# ------------------------------
# 6) Build numeric Cox audit dataframe
# ------------------------------
if hasattr(X_cox_train, "toarray"):
    X_cox_dense = X_cox_train.toarray()
else:
    X_cox_dense = np.asarray(X_cox_train)

if X_cox_dense.ndim != 2:
    raise ValueError(f"Transformed Cox matrix must be 2D. Got shape={X_cox_dense.shape}")

n_rows, n_cols = X_cox_dense.shape

if cox_feature_names is None:
    cox_feature_names = [f"x_{i}" for i in range(n_cols)]
elif len(cox_feature_names) != n_cols:
    print(
        f"Warning: feature-name length mismatch "
        f"(len={len(cox_feature_names)} vs n_cols={n_cols}). "
        f"Falling back to generic names."
    )
    cox_feature_names = [f"x_{i}" for i in range(n_cols)]

cox_numeric_df = pd.DataFrame(X_cox_dense, columns=cox_feature_names, index=cox_train_base_df.index)

cox_model_df = pd.concat(
    [
        cox_train_base_df[[duration_col, event_col]].copy(),
        cox_numeric_df
    ],
    axis=1
)

cox_model_df = cox_model_df.replace([np.inf, -np.inf], np.nan)

n_before = cox_model_df.shape[0]
cox_model_df = cox_model_df.dropna(axis=0).copy()
n_after = cox_model_df.shape[0]

if n_after == 0:
    raise ValueError("After dropping NaN/Inf rows, no rows remain for PH audit.")

cox_model_df[duration_col] = pd.to_numeric(cox_model_df[duration_col], errors="raise")
cox_model_df[event_col] = pd.to_numeric(cox_model_df[event_col], errors="raise").astype(int)

covariate_cols = [c for c in cox_model_df.columns if c not in [duration_col, event_col]]

# drop constant columns
non_constant_covariates = []
dropped_constant_covariates = []

for c in covariate_cols:
    if cox_model_df[c].nunique(dropna=True) <= 1:
        dropped_constant_covariates.append(c)
    else:
        non_constant_covariates.append(c)

covariate_cols = non_constant_covariates
cox_model_df = cox_model_df[[duration_col, event_col] + covariate_cols].copy()

if len(covariate_cols) == 0:
    raise ValueError("No non-constant numeric Cox covariates remained for PH audit.")

print(f"Rows before cleaning: {n_before:,}")
print(f"Rows after cleaning:  {n_after:,}")
print(f"Covariates tested:    {len(covariate_cols):,}")
if dropped_constant_covariates:
    print(f"Dropped constant covariates: {len(dropped_constant_covariates)}")

# ------------------------------
# 7) Resolve tuned Cox penalization settings if available
# ------------------------------
penalizer_value = 0.001
l1_ratio_value = 0.0

candidate_config_objs = [
    "COX_TUNING_CONFIG",
    "cox_tuning_config",
    "cox_best_config",
    "cox_model_config",
]

for name in candidate_config_objs:
    if name in globals():
        cfg = globals()[name]
        try:
            if isinstance(cfg, dict):
                best = cfg.get("best_candidate", cfg)
                if "penalizer" in best:
                    penalizer_value = float(best["penalizer"])
                if "l1_ratio" in best:
                    l1_ratio_value = float(best["l1_ratio"])
        except Exception:
            pass

print(f"Audit Cox refit with penalizer={penalizer_value}, l1_ratio={l1_ratio_value}")

# ------------------------------
# 8) Fit audit Cox model
# ------------------------------
cox_audit_model = CoxPHFitter(
    penalizer=penalizer_value,
    l1_ratio=l1_ratio_value,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cox_audit_model.fit(
        cox_model_df,
        duration_col=duration_col,
        event_col=event_col,
        show_progress=False
    )

# ------------------------------
# 9) Formal PH test
# ------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ph_test_rank = proportional_hazard_test(
        cox_audit_model,
        cox_model_df,
        time_transform="rank"
    )

ph_summary = ph_test_rank.summary.copy().reset_index()
if "index" in ph_summary.columns:
    ph_summary = ph_summary.rename(columns={"index": "covariate"})

rename_map = {}
for col in ph_summary.columns:
    cl = col.lower()
    if cl in ["p", "p_value"]:
        rename_map[col] = "p_value"
    elif cl in ["test_statistic", "test statistic", "chi2"]:
        rename_map[col] = "test_statistic"
ph_summary = ph_summary.rename(columns=rename_map)

if "p_value" not in ph_summary.columns:
    p_candidates = [c for c in ph_summary.columns if "p" in c.lower()]
    if p_candidates:
        ph_summary = ph_summary.rename(columns={p_candidates[0]: "p_value"})

if "test_statistic" not in ph_summary.columns:
    stat_candidates = [c for c in ph_summary.columns if "test" in c.lower() or "chi" in c.lower()]
    if stat_candidates:
        ph_summary = ph_summary.rename(columns={stat_candidates[0]: "test_statistic"})

ph_summary["p_value"] = pd.to_numeric(ph_summary["p_value"], errors="coerce")
ph_summary["test_statistic"] = pd.to_numeric(ph_summary["test_statistic"], errors="coerce")

alpha = 0.05
ph_summary["ph_flag"] = np.where(
    ph_summary["p_value"] < alpha,
    "possible_violation",
    "no_material_evidence"
)
ph_summary["ph_flag_binary"] = np.where(ph_summary["p_value"] < alpha, "yes", "no")

ph_audit_df = (
    ph_summary[["covariate", "test_statistic", "p_value", "ph_flag", "ph_flag_binary"]]
    .sort_values(["p_value", "covariate"], ascending=[True, True])
    .reset_index(drop=True)
)

# ------------------------------
# 10) Global summary
# ------------------------------
n_tested = int(ph_audit_df.shape[0])
n_flagged = int((ph_audit_df["p_value"] < alpha).sum())
top_flagged = ph_audit_df.loc[ph_audit_df["p_value"] < alpha, "covariate"].head(5).tolist()

if n_flagged == 0:
    global_classification = "A_no_material_evidence"
    global_interpretation = (
        "No material evidence of proportional-hazards violation was detected "
        "for the comparable Cox benchmark at the chosen threshold."
    )
elif n_flagged <= max(2, int(0.15 * n_tested)):
    global_classification = "B_localized_departures"
    global_interpretation = (
        "The comparable Cox benchmark showed some localized departures from "
        "proportional hazards, but not a pattern severe enough to prevent its "
        "use as the methodological anchor of the comparable continuous-time family."
    )
else:
    global_classification = "C_broad_departure"
    global_interpretation = (
        "The comparable Cox benchmark showed broad evidence of proportional-hazards "
        "departure and should therefore be interpreted as an approximate comparable "
        "benchmark rather than a fully assumption-clean Cox specification."
    )

global_summary_df = pd.DataFrame([{
    "model": "Cox Comparable (Tuned)",
    "duration_col": duration_col,
    "event_col": event_col,
    "n_rows_used": int(cox_model_df.shape[0]),
    "n_covariates_tested": n_tested,
    "alpha": alpha,
    "n_covariates_flagged": n_flagged,
    "share_covariates_flagged": float(n_flagged / n_tested) if n_tested > 0 else np.nan,
    "top_flagged_covariates": "; ".join(top_flagged) if top_flagged else "none",
    "global_classification": global_classification,
    "global_interpretation": global_interpretation,
    "audit_note": (
        "Formal proportional-hazards audit for the comparable Cox branch. "
        "DeepSurv shares the Cox-type ranking structure but was not subjected "
        "to an identical classical PH diagnostic."
    ),
}])

# ------------------------------
# 11) Optional figure
# ------------------------------
figure_created = False
figure_note = "not_created"

figures_dir = OUTPUT_DIR / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

diagnostic_fig_dir = figures_dir / "diagnostics"
diagnostic_fig_dir.mkdir(parents=True, exist_ok=True)

fig_path_png = diagnostic_fig_dir / "fig_cox_ph_diagnostics.png"
fig_path_pdf = diagnostic_fig_dir / "fig_cox_ph_diagnostics.pdf"

if MATPLOTLIB_AVAILABLE:
    try:
        plot_df = ph_audit_df.nsmallest(min(8, ph_audit_df.shape[0]), "p_value").copy()
        plot_df = plot_df.sort_values("p_value", ascending=False)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(plot_df["covariate"], plot_df["p_value"])
        ax.axvline(alpha, linestyle="--")
        ax.set_xlabel("PH test p-value")
        ax.set_ylabel("Covariate")
        ax.set_title("Comparable Cox PH diagnostic (smallest p-values)")
        plt.tight_layout()

        fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig_path_pdf, bbox_inches="tight")
        plt.close(fig)

        figure_created = True
        figure_note = "top_smallest_p_values_barh"
    except Exception as e:
        figure_note = f"figure_failed: {str(e)}"

# ------------------------------
# 12) Save artifacts
# ------------------------------
audit_path_main = TABLES_DIR / "table_appendix_cox_ph_audit.csv"
global_summary_path_main = TABLES_DIR / "table_appendix_cox_ph_global_summary.csv"

metadata_path = METADATA_DIR / "cox_ph_audit_summary.json"

ph_audit_df.to_csv(audit_path_main, index=False)
materialize_dataframe(con, ph_audit_df, infer_table_name_from_pathlike(audit_path_main), "F8")
global_summary_df.to_csv(global_summary_path_main, index=False)
materialize_dataframe(con, global_summary_df, infer_table_name_from_pathlike(global_summary_path_main), "F8")

metadata_payload = {
    "step_id": "P31.7",
    "model": "Cox Comparable (Tuned)",
    "duration_col": duration_col,
    "event_col": event_col,
    "n_rows_before_cleaning": int(n_before),
    "n_rows_after_cleaning": int(n_after),
    "n_rows_used": int(cox_model_df.shape[0]),
    "n_covariates_tested": n_tested,
    "n_covariates_flagged": n_flagged,
    "alpha": alpha,
    "global_classification": global_classification,
    "global_interpretation": global_interpretation,
    "top_flagged_covariates": top_flagged,
    "dropped_constant_covariates": dropped_constant_covariates,
    "penalizer": penalizer_value,
    "l1_ratio": l1_ratio_value,
    "figure_created": figure_created,
    "figure_note": figure_note,
    "deep_surv_note": (
        "DeepSurv shares the Cox-type ranking structure but was not subjected "
        "to an identical classical proportional-hazards diagnostic."
    ),
}
save_json(metadata_payload, metadata_path)

# ------------------------------
# 13) Output
# ------------------------------
print("\nGlobal PH audit summary:")
display(global_summary_df)

print("\nPH audit by covariate:")
display(ph_audit_df)

if figure_created:
    print("\nPH diagnostic figure created:")
    print("-", fig_path_png.resolve())
    print("-", fig_path_pdf.resolve())
else:
    print("\nPH diagnostic figure not created.")
    print("Reason:", figure_note)

print("\nSaved artifacts:")
print("-", audit_path_main.resolve())
print("-", global_summary_path_main.resolve())
print("-", metadata_path.resolve())

print(f"[END] F8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 102
from datetime import datetime as _dt
print(f"[START] F9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

from pathlib import Path
import json
import pandas as pd
import numpy as np
from util import shutdown_duckdb_connection_from_globals

PROJECT_ROOT = globals().get("PROJECT_ROOT", Path.cwd())
OUTPUT_DIR = globals().get("OUTPUT_DIR", PROJECT_ROOT / "outputs_benchmark_survival")
TABLES_DIR = globals().get("TABLES_DIR", OUTPUT_DIR / "tables")
METADATA_DIR = globals().get("METADATA_DIR", OUTPUT_DIR / "metadata")

for p in [OUTPUT_DIR, TABLES_DIR, METADATA_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)

def _load_df_from_memory_or_csv(global_name: str, filename: str) -> pd.DataFrame:
    obj = globals().get(global_name)
    if isinstance(obj, pd.DataFrame) and obj.shape[0] > 0:
        return obj.copy()
    path = TABLES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact for F9: {path}")
    return pd.read_csv(path)

def _save_json_payload(payload: dict, path: Path) -> None:
    if "save_json" in globals() and callable(save_json):
        save_json(payload, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

def _materialize_if_possible(df: pd.DataFrame, path: Path, stage_id: str) -> None:
    if "con" not in globals():
        return
    if "materialize_dataframe" not in globals() or "infer_table_name_from_pathlike" not in globals():
        return
    try:
        materialize_dataframe(con, df, infer_table_name_from_pathlike(path), stage_id)
    except Exception as exc:
        print(f"Warning: DuckDB materialization skipped for {path.name}: {exc}")

def _pairwise_prob_leader_better(pairwise_df: pd.DataFrame, metric: str, leader: str, challenger: str) -> float:
    direct = pairwise_df[
        (pairwise_df["metric"] == metric)
        & (pairwise_df["model_a"] == leader)
        & (pairwise_df["model_b"] == challenger)
    ]
    if not direct.empty:
        return float(direct.iloc[0]["prob_a_better_than_b"])

    reverse = pairwise_df[
        (pairwise_df["metric"] == metric)
        & (pairwise_df["model_a"] == challenger)
        & (pairwise_df["model_b"] == leader)
    ]
    if not reverse.empty:
        return float(reverse.iloc[0]["prob_b_better_than_a"])

    return np.nan

bootstrap_summary_df = _load_df_from_memory_or_csv(
    "bootstrap_summary_df",
    "table_bootstrap_uncertainty_summary_tuned_models.csv",
)
rank_stability_df = _load_df_from_memory_or_csv(
    "rank_stability_df",
    "table_bootstrap_rank_stability_tuned_models.csv",
)
pairwise_stability_df = _load_df_from_memory_or_csv(
    "pairwise_stability_df",
    "table_bootstrap_pairwise_stability_tuned_models.csv",
)
cox_ph_global_summary_df = _load_df_from_memory_or_csv(
    "global_summary_df",
    "table_appendix_cox_ph_global_summary.csv",
)

metric_labels = {
    "ibs": "Integrated Brier Score",
    "c_index_td": "Time-dependent concordance",
}
direction_labels = {
    "ibs": "lower_is_better",
    "c_index_td": "higher_is_better",
}

inferential_rows = []
for metric in ["ibs", "c_index_td"]:
    rank_metric_df = rank_stability_df[rank_stability_df["metric"] == metric].copy()
    rank1_df = rank_metric_df[rank_metric_df["rank"] == 1].sort_values("proportion", ascending=False).reset_index(drop=True)
    if rank1_df.empty:
        raise ValueError(f"No rank-stability data available for metric={metric}")

    leader_model = str(rank1_df.loc[0, "model"])
    leader_rank1_share = float(rank1_df.loc[0, "proportion"])
    runner_up_model = str(rank1_df.loc[1, "model"]) if rank1_df.shape[0] > 1 else "none"
    runner_up_rank1_share = float(rank1_df.loc[1, "proportion"]) if rank1_df.shape[0] > 1 else 0.0

    leader_summary = bootstrap_summary_df[
        (bootstrap_summary_df["model"] == leader_model)
        & (bootstrap_summary_df["metric"] == metric)
    ].copy()
    if leader_summary.empty:
        raise ValueError(f"Missing bootstrap summary row for leader={leader_model}, metric={metric}")

    leader_point = float(leader_summary.iloc[0]["point_estimate"])
    leader_ci_lower = float(leader_summary.iloc[0]["ci_lower"])
    leader_ci_upper = float(leader_summary.iloc[0]["ci_upper"])
    n_bootstrap = int(leader_summary.iloc[0]["n_successful_bootstrap"])

    challengers = [m for m in rank1_df["model"].tolist() if m != leader_model]
    challenger_probs = [
        _pairwise_prob_leader_better(pairwise_stability_df, metric, leader_model, challenger)
        for challenger in challengers
    ]
    challenger_probs = [p for p in challenger_probs if pd.notna(p)]
    weakest_pairwise_prob = min(challenger_probs) if challenger_probs else np.nan

    if leader_rank1_share >= 0.95 and (pd.isna(weakest_pairwise_prob) or weakest_pairwise_prob >= 0.95):
        claim_status = "strong_bootstrap_leadership_support"
        supported_claim = (
            f"Bootstrap strongly supports {leader_model} as the most stable leader for {metric_labels[metric]} "
            "on the held-out benchmark test set."
        )
    elif leader_rank1_share >= 0.75 and (pd.isna(weakest_pairwise_prob) or weakest_pairwise_prob >= 0.75):
        claim_status = "largely_stable_bootstrap_leadership"
        supported_claim = (
            f"Bootstrap supports a largely stable, but not absolute, leadership pattern for {metric_labels[metric]}, led by {leader_model}."
        )
    else:
        claim_status = "directional_signal_only"
        supported_claim = (
            f"Bootstrap preserves directional signal for {metric_labels[metric]}, but the ranking is not stable enough for a strong hierarchy claim."
        )

    unsupported_claim = (
        "These artifacts do not constitute a formal null-hypothesis test, do not prove a universally fixed hierarchy across samples, "
        "and should not be narrated as strict superiority without qualification."
    )

    inferential_rows.append({
        "metric": metric,
        "metric_label": metric_labels[metric],
        "direction": direction_labels[metric],
        "benchmark_scope": "manuscript_audit_subset",
        "bootstrap_design": "enrollment_level_resampling_on_fixed_test_predictions",
        "inferential_scope": BOOTSTRAP_CONFIG["inferential_scope"],
        "model_retraining_policy": BOOTSTRAP_CONFIG["model_retraining_policy"],
        "alternative_refit_status": BOOTSTRAP_CONFIG["alternative_refit_status"],
        "leading_model": leader_model,
        "leading_model_point_estimate": leader_point,
        "leading_model_ci_lower": leader_ci_lower,
        "leading_model_ci_upper": leader_ci_upper,
        "leading_model_rank1_share": leader_rank1_share,
        "runner_up_model": runner_up_model,
        "runner_up_rank1_share": runner_up_rank1_share,
        "weakest_pairwise_win_probability_among_challengers": weakest_pairwise_prob,
        "n_successful_bootstrap": n_bootstrap,
        "claim_status": claim_status,
        "what_is_supported": supported_claim,
        "what_is_not_supported": unsupported_claim,
        "recommended_manuscript_use": BOOTSTRAP_CONFIG["supported_use"],
        "recommended_manuscript_non_use": BOOTSTRAP_CONFIG["unsupported_use"],
    })

bootstrap_inferential_scope_df = pd.DataFrame(inferential_rows)

cox_row = cox_ph_global_summary_df.iloc[0].to_dict()
ph_scope_boundary_df = pd.DataFrame([
    {
        "model": "Cox Comparable (Tuned)",
        "model_family_role": "continuous_time_anchor",
        "formal_ph_diagnostic_executed": "yes",
        "formal_test_basis": "lifelines proportional_hazard_test with rank time transform",
        "coverage_status": str(cox_row.get("global_classification", "unknown")),
        "summary_evidence": str(cox_row.get("global_interpretation", "")),
        "manuscript_claim_boundary": (
            "This stage supports a formal PH audit only for the comparable Cox anchor; any departures should be narrated with the audit classification rather than as assumption-free validity."
        ),
    },
    {
        "model": "DeepSurv (Tuned)",
        "model_family_role": "continuous_time_neural_cox_type_model",
        "formal_ph_diagnostic_executed": "no",
        "formal_test_basis": "no identical classical PH diagnostic was executed in stage F",
        "coverage_status": "incomplete_formal_ph_coverage",
        "summary_evidence": (
            "DeepSurv is covered here by predictive metrics, calibration, and bootstrap uncertainty artifacts, but not by a classical PH diagnostic equivalent to the Cox audit."
        ),
        "manuscript_claim_boundary": (
            "DeepSurv can be benchmarked on predictive performance, but this stage does not claim a formal PH pass equivalent to the comparable Cox branch."
        ),
    },
])

bootstrap_scope_path = TABLES_DIR / "table_appendix_bootstrap_inferential_scope_summary.csv"
ph_scope_path = TABLES_DIR / "table_appendix_ph_scope_boundary.csv"
bootstrap_scope_metadata_path = METADATA_DIR / "metadata_bootstrap_inferential_scope_summary.json"
ph_scope_metadata_path = METADATA_DIR / "metadata_ph_scope_boundary.json"

bootstrap_inferential_scope_df.to_csv(bootstrap_scope_path, index=False)
ph_scope_boundary_df.to_csv(ph_scope_path, index=False)
_materialize_if_possible(bootstrap_inferential_scope_df, bootstrap_scope_path, "F9")
_materialize_if_possible(ph_scope_boundary_df, ph_scope_path, "F9")

_save_json_payload(
    {
        "step_id": "F9_bootstrap_scope",
        "source_tables": [
            "table_bootstrap_uncertainty_summary_tuned_models.csv",
            "table_bootstrap_rank_stability_tuned_models.csv",
            "table_bootstrap_pairwise_stability_tuned_models.csv",
        ],
        "n_metrics_summarized": int(bootstrap_inferential_scope_df.shape[0]),
        "metrics": bootstrap_inferential_scope_df["metric"].tolist(),
    },
    bootstrap_scope_metadata_path,
)
_save_json_payload(
    {
        "step_id": "F9_ph_scope_boundary",
        "source_tables": ["table_appendix_cox_ph_global_summary.csv"],
        "models_covered": ph_scope_boundary_df["model"].tolist(),
        "formal_ph_coverage_models": ph_scope_boundary_df.loc[
            ph_scope_boundary_df["formal_ph_diagnostic_executed"] == "yes",
            "model",
        ].tolist(),
        "non_equivalent_coverage_models": ph_scope_boundary_df.loc[
            ph_scope_boundary_df["formal_ph_diagnostic_executed"] == "no",
            "model",
        ].tolist(),
    },
    ph_scope_metadata_path,
)

print("\nBootstrap inferential-scope summary:")
display(bootstrap_inferential_scope_df)

print("\nPH scope-boundary summary:")
display(ph_scope_boundary_df)

print("\nSaved artifacts:")
print("-", bootstrap_scope_path.resolve())
print("-", ph_scope_path.resolve())
print("-", bootstrap_scope_metadata_path.resolve())
print("-", ph_scope_metadata_path.resolve())

shutdown_duckdb_connection_from_globals(globals())

print(f"[END] F9 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 103
