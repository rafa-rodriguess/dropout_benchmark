from __future__ import annotations

"""
Post-hoc audit module for benchmark stage E.

Purpose:
- materialize the post-hoc audit stack used to assess benchmark comparability,
    calibration evidence, sensitivity coverage, and horizon-choice stability
- preserve audit continuity when selected upstream benchmark artifacts are not
    available by materializing schema-stable placeholders where the stage design
    explicitly allows that behavior

Input contract:
- benchmark_shared_config.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/benchmark_survival.duckdb
- canonical upstream benchmark tables produced by stages D, F, and G when those
    audit branches are available

Output contract:
- DuckDB audit tables materialized by stages E0 through E7
- JSON metadata artifacts under outputs_benchmark_survival/metadata
- structured console summaries for each audit stage

Failure policy:
- missing runtime metadata or a missing DuckDB database raises immediately
- canonical upstream dependencies raise immediately when the corresponding audit
    stage is strict
- stages designed to tolerate absent upstream evidence must record the missing
    dependency explicitly and emit schema-stable placeholder outputs rather than
    failing silently

Historical lineage note:
- the NOTEBOOK_NAME provenance label is retained because the pipeline catalog
    still records notebook-origin lineage for previously materialized artifacts
"""

# %% Cell 2
from datetime import datetime as _dt
print(f"[START] E0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E0 - Runtime / DuckDB bootstrap
# ==============================================================
from pathlib import Path
import duckdb
import atexit
import json
import pandas as pd
import numpy as np
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

STAGE_PREFIX = "E"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
NOTEBOOK_NAME = "dropout_bench_v3_E_posthoc_audits.ipynb"
CONFIG_TOML_PATH = PROJECT_ROOT / "benchmark_shared_config.toml"
RUN_METADATA_JSON_PATH = PROJECT_ROOT / "outputs_benchmark_survival" / "metadata" / "run_metadata.json"

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
        },
    }

if not RUN_METADATA_JSON_PATH.exists():
    raise FileNotFoundError(f"Missing execution metadata JSON: {RUN_METADATA_JSON_PATH}")
with open(RUN_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
    RUN_METADATA = json.load(f)
RUN_ID = str(RUN_METADATA["run_id"]).strip()

paths_cfg = SHARED_CONFIG.get("paths", {})
SHARED_BENCHMARK_CONFIG = SHARED_CONFIG.get("benchmark", {})

def _resolve_project_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else PROJECT_ROOT / p

DATA_DIR = _resolve_project_path(paths_cfg.get("data_dir", "content"))
OUTPUT_DIR = _resolve_project_path(paths_cfg.get("output_dir", "outputs_benchmark_survival"))
TABLES_DIR = OUTPUT_DIR / paths_cfg.get("tables_subdir", "tables")
METADATA_DIR = OUTPUT_DIR / paths_cfg.get("metadata_subdir", "metadata")
DATA_OUTPUT_DIR = OUTPUT_DIR / paths_cfg.get("data_output_subdir", "data")
DUCKDB_PATH = OUTPUT_DIR / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")

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

print("Runtime context ready.")
print("- SCRIPT_NAME   :", SCRIPT_NAME)
print("- STAGE_PREFIX  :", STAGE_PREFIX)
print("- LINEAGE LABEL :", NOTEBOOK_NAME)
print("- RUN_ID       :", RUN_ID)
print("- DUCKDB_PATH  :", DUCKDB_PATH)

print(f"[END] E0 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 4
from datetime import datetime as _dt
print(f"[START] E0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

import json
from util import shutdown_duckdb_connection_from_globals

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


def empty_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


BENCHMARK_HORIZON_MODEL_SPECS = base.get_benchmark_horizon_model_specs()

BENCHMARK_HORIZON_MODEL_SPEC_BY_KEY = {
    spec["model_key"]: spec for spec in BENCHMARK_HORIZON_MODEL_SPECS
}


def benchmark_family_from_group(family_group: str) -> str:
    return base.benchmark_family_from_group(family_group)


def load_horizon_metric_values(table_name: str, metric_name: str) -> dict[int, float]:
    df = load_duckdb_table_optional(table_name)
    if df is None or df.empty:
        return {}
    working_df = df.copy()
    if "metric_name" in working_df.columns:
        working_df = working_df.loc[
            working_df["metric_name"].astype(str).str.lower().eq(metric_name.lower())
        ].copy()
    if working_df.empty or "horizon_week" not in working_df.columns or "metric_value" not in working_df.columns:
        return {}
    return {
        int(horizon_week): float(metric_value)
        for horizon_week, metric_value in working_df[["horizon_week", "metric_value"]].itertuples(index=False)
        if pd.notna(horizon_week) and pd.notna(metric_value)
    }


def ensure_benchmark_horizon_tables() -> None:
    required_tables = {
        "table_benchmark_risk_auc_by_horizon_wide",
        "table_benchmark_brier_by_horizon_wide",
        "table_benchmark_calibration_by_horizon_wide",
        "table_benchmark_support_reference",
    }
    available = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
    if required_tables.issubset(available):
        return

    primary_df = load_duckdb_table_optional("table_5_16_model_primary_summary")
    if primary_df is None or primary_df.empty:
        return

    benchmark_rows = []
    for _, row in primary_df.iterrows():
        model_key = str(row.get("model_name", "")).strip()
        primary_table = str(row.get("primary_metrics_table", "")).strip()
        if model_key not in BENCHMARK_HORIZON_MODEL_SPEC_BY_KEY or not primary_table.endswith("_primary_metrics"):
            continue

        spec = BENCHMARK_HORIZON_MODEL_SPEC_BY_KEY[model_key]
        table_prefix = primary_table[: -len("_primary_metrics")]
        secondary_table = f"{table_prefix}_secondary_metrics"
        brier_table = f"{table_prefix}_brier_by_horizon"
        calibration_table = f"{table_prefix}_calibration_summary"
        support_table = f"{table_prefix}_support_by_horizon"

        risk_auc_by_horizon = load_horizon_metric_values(secondary_table, "risk_auc_at_horizon")
        brier_by_horizon = load_horizon_metric_values(brier_table, "brier_at_horizon")
        calibration_by_horizon = load_horizon_metric_values(calibration_table, "calibration_at_horizon")
        support_df = load_duckdb_table_optional(support_table)
        support_by_horizon = {}
        if support_df is not None and not support_df.empty and "horizon_week" in support_df.columns:
            for support_row in support_df.to_dict("records"):
                horizon_week = pd.to_numeric(pd.Series([support_row.get("horizon_week")]), errors="coerce").iloc[0]
                if pd.isna(horizon_week):
                    continue
                support_by_horizon[int(horizon_week)] = {
                    "n_evaluable_enrollments": float(pd.to_numeric(pd.Series([support_row.get("n_evaluable_enrollments")]), errors="coerce").iloc[0]) if pd.notna(support_row.get("n_evaluable_enrollments")) else np.nan,
                    "n_events_by_horizon": float(pd.to_numeric(pd.Series([support_row.get("n_events_by_horizon")]), errors="coerce").iloc[0]) if pd.notna(support_row.get("n_events_by_horizon")) else np.nan,
                    "event_rate_by_horizon": float(pd.to_numeric(pd.Series([support_row.get("event_rate_by_horizon")]), errors="coerce").iloc[0]) if pd.notna(support_row.get("event_rate_by_horizon")) else np.nan,
                }

        horizon_set = sorted(set(risk_auc_by_horizon) | set(brier_by_horizon) | set(calibration_by_horizon) | set(support_by_horizon))
        for horizon_week in horizon_set:
            support_info = support_by_horizon.get(horizon_week, {})
            benchmark_rows.append({
                "model_key": model_key,
                "display_name": spec["display_name"],
                "family": benchmark_family_from_group(spec["family_group"]),
                "family_name": benchmark_family_from_group(spec["family_group"]),
                "horizon_week": int(horizon_week),
                "risk_auc_value": risk_auc_by_horizon.get(horizon_week, np.nan),
                "brier_value": brier_by_horizon.get(horizon_week, np.nan),
                "calibration_gap": calibration_by_horizon.get(horizon_week, np.nan),
                "n_evaluable_enrollments": support_info.get("n_evaluable_enrollments", np.nan),
                "n_events_by_horizon": support_info.get("n_events_by_horizon", np.nan),
                "event_rate_by_horizon": support_info.get("event_rate_by_horizon", np.nan),
            })

    if not benchmark_rows:
        return

    leaderboard_df = pd.DataFrame(benchmark_rows).sort_values(
        ["family", "display_name", "horizon_week"], kind="mergesort"
    ).reset_index(drop=True)

    def _pivot_metric(source_df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        out = (
            source_df.pivot_table(
                index=["model_key", "display_name", "family"],
                columns="horizon_week",
                values=value_col,
                aggfunc="first",
            )
            .reset_index()
        )
        out.columns = [f"{prefix}h{int(col)}" if isinstance(col, (int, float)) else col for col in out.columns]
        out["family_name"] = out["family"]
        return out

    risk_auc_wide_df = _pivot_metric(leaderboard_df, "risk_auc_value", "risk_auc_")
    brier_wide_df = _pivot_metric(leaderboard_df, "brier_value", "brier_")
    calibration_wide_df = _pivot_metric(leaderboard_df, "calibration_gap", "calibration_h")

    support_reference_df = (
        leaderboard_df[["horizon_week", "n_evaluable_enrollments", "n_events_by_horizon", "event_rate_by_horizon"]]
        .sort_values(["horizon_week", "n_evaluable_enrollments"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(subset=["horizon_week"], keep="first")
        .reset_index(drop=True)
    )

    materialize_dataframe(con, leaderboard_df, "table_benchmark_leaderboard_by_horizon", "E1.2")
    materialize_dataframe(con, risk_auc_wide_df, "table_benchmark_risk_auc_by_horizon_wide", "E1.2")
    materialize_dataframe(con, brier_wide_df, "table_benchmark_brier_by_horizon_wide", "E1.2")
    materialize_dataframe(con, calibration_wide_df, "table_benchmark_calibration_by_horizon_wide", "E1.2")
    materialize_dataframe(con, support_reference_df, "table_benchmark_support_reference", "E1.2")

ensure_pipeline_table_catalog(con)
required_runtime = ["NOTEBOOK_NAME", "RUN_ID", "con", "save_json", "materialize_dataframe", "register_duckdb_table", "print_duckdb_table", "load_duckdb_table_or_raise", "load_duckdb_table_optional", "shutdown_duckdb_connection_from_globals"]
missing_runtime = [name for name in required_runtime if name not in globals()]
if missing_runtime:
    raise NameError("E0.1 runtime contract failed. Missing required object(s): " + ", ".join(missing_runtime))
print("Runtime contract validated.")

print(f"[END] E0.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 6
from datetime import datetime as _dt
print(f"[START] E0.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E0.2 — Dependency Map for Late-Stage Audit Sections
# --------------------------------------------------------------
# Purpose:
#   Make late-stage dependencies explicit before this stage module is
#   rerun selectively after editorial cleanup.
#
# Methodological note:
#   This stage documents execution contracts only.
#   It does not train models and does not recompute metrics.
# ==============================================================

print("\n" + "=" * 70)
print("E0.2 — Dependency Map for Late-Stage Audit Sections")
print("=" * 70)
print("Methodological note: this cell documents dependencies only.")

required_names = ["TABLES_DIR", "METADATA_DIR"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Please run prior setup stages first."
    )

import pandas as pd

dependency_rows = [
    {
        "section_id": "E1",
        "title": "Benchmark comparability audit",
        "depends_on": "canonical tuned benchmark tables",
        "primary_outputs": "table_p10_0_benchmark_comparability_*",
        "paper_facing": False,
        "notes": "Standalone audit retained as optional post-hoc check."
    },
    {
        "section_id": "E2",
        "title": "Calibration artifact audit",
        "depends_on": "per-model calibration tables + benchmark-wide calibration tables",
        "primary_outputs": "table_p11_0_calibration_*",
        "paper_facing": True,
        "notes": "Feeds calibration strengthening and final paper freeze."
    },
    {
        "section_id": "E3",
        "title": "Calibration slope/intercept audit",
        "depends_on": "E2 calibration artifacts or legacy reliability bins",
        "primary_outputs": "table_p11_1_calibration_slope_intercept_*",
        "paper_facing": True,
        "notes": "Optional strengthening layer merged into calibration summaries when available."
    },
    {
        "section_id": "E4",
        "title": "Unified calibration strengthening summary",
        "depends_on": "E2 + E3",
        "primary_outputs": "table_p11_2_calibration_strengthening_*",
        "paper_facing": True,
        "notes": "Strengthens reviewer-facing calibration interpretation."
    },
    {
        "section_id": "E5",
        "title": "Sensitivity artifact audit",
        "depends_on": "sensitivity outputs if that branch is executed",
        "primary_outputs": "table_p12_0_sensitivity_*",
        "paper_facing": False,
        "notes": "Kept as optional post-hoc robustness layer."
    },
    {
        "section_id": "E6",
        "title": "Horizon-choice stress test",
        "depends_on": "E5",
        "primary_outputs": "table_p12_3_horizon_choice_*",
        "paper_facing": False,
        "notes": "Optional; not required by the manuscript freeze."
    },
    {
        "section_id": "E7",
        "title": "Unified sensitivity summary",
        "depends_on": "E5 + E6",
        "primary_outputs": "table_p12_4_sensitivity_*",
        "paper_facing": False,
        "notes": "Optional; retained for robustness documentation only."
    },
    {
        "section_id": "F1",
        "title": "Define ablation protocol",
        "depends_on": "benchmark-ready tuned datasets",
        "primary_outputs": "table_ablation_model_registry.csv and companions",
        "paper_facing": True,
        "notes": "Foundational registry for all downstream ablation stages."
    },
    {
        "section_id": "F2",
        "title": "Materialize ablation variants",
        "depends_on": "F1 + tuned ready datasets",
        "primary_outputs": "table_ablation_variant_registry.csv",
        "paper_facing": True,
        "notes": "Feeds both discrete-time and continuous-time ablation training stages."
    },
    {
        "section_id": "F3",
        "title": "Discrete-time tuned ablation runs",
        "depends_on": "F2",
        "primary_outputs": "table_ablation_discrete_*",
        "paper_facing": True,
        "notes": "Feeds F5 consolidated ablation outputs."
    },
    {
        "section_id": "F4",
        "title": "Continuous-time tuned ablation runs",
        "depends_on": "F2",
        "primary_outputs": "table_ablation_continuous_*",
        "paper_facing": True,
        "notes": "Feeds F5 consolidated ablation outputs."
    },
    {
        "section_id": "F5",
        "title": "Consolidate ablation results",
        "depends_on": "F3 + F4",
        "primary_outputs": "table_ablation_summary_by_model.csv",
        "paper_facing": True,
        "notes": "Canonical source for the paper ablation table and figure."
    },
    {
        "section_id": "F6",
        "title": "Preprocessing and tuning audit",
        "depends_on": "tuning result tables + preprocessing summaries",
        "primary_outputs": "table_preprocessing_and_tuning_audit.csv",
        "paper_facing": True,
        "notes": "Feeds appendix preprocessing/tuning table."
    },
    {
        "section_id": "F7",
        "title": "Bootstrap uncertainty audit",
        "depends_on": "saved tuned models + test-ready datasets",
        "primary_outputs": "table_appendix_bootstrap_uncertainty_compact.csv",
        "paper_facing": True,
        "notes": "Feeds appendix uncertainty table."
    },
    {
        "section_id": "F8",
        "title": "Comparable Cox PH audit",
        "depends_on": "cox tuned model + enrollment_cox_ready_train",
        "primary_outputs": "table_appendix_cox_ph_global_summary.csv + fig_appendix_cox_ph_diagnostics.png",
        "paper_facing": True,
        "notes": "Feeds appendix PH table and figure."
    },
    {
        "section_id": "G1",
        "title": "Define explainability protocol",
        "depends_on": "tuned benchmark families already frozen",
        "primary_outputs": "table_explainability_model_registry.csv",
        "paper_facing": True,
        "notes": "Registry cell for the full explainability layer."
    },
    {
        "section_id": "G2",
        "title": "Linear tuned explainability",
        "depends_on": "linear tuned model + preprocessor",
        "primary_outputs": "table_linear_explainability_*",
        "paper_facing": True,
        "notes": "Feeds G6 consolidated explainability outputs."
    },
    {
        "section_id": "G3",
        "title": "Neural tuned explainability",
        "depends_on": "neural tuned ready data + preprocessor",
        "primary_outputs": "table_neural_explainability_*",
        "paper_facing": True,
        "notes": "Feeds G6 consolidated explainability outputs."
    },
    {
        "section_id": "G4",
        "title": "Cox tuned explainability",
        "depends_on": "cox tuned model + preprocessor",
        "primary_outputs": "table_cox_explainability_*",
        "paper_facing": True,
        "notes": "Feeds G6 consolidated explainability outputs."
    },
    {
        "section_id": "G5",
        "title": "DeepSurv tuned explainability",
        "depends_on": "deepsurv tuned ready data + preprocessor",
        "primary_outputs": "table_deepsurv_explainability_*",
        "paper_facing": True,
        "notes": "Feeds G6 consolidated explainability outputs."
    },
    {
        "section_id": "G6",
        "title": "Consolidate explainability",
        "depends_on": "G2 + G3 + G4 + G5",
        "primary_outputs": "table_explainability_summary_by_model.csv + table_explainability_all_blocks_normalized.csv",
        "paper_facing": True,
        "notes": "Canonical source for the paper explainability table and figure."
    },
    {
        "section_id": "G7",
        "title": "Freeze curated paper artifacts",
        "depends_on": "benchmark leaderboard + E2/E3/E4 + F5/F6/F7/F8 + G6",
        "primary_outputs": "outputs_benchmark_survival/paper_main and paper_appendix",
        "paper_facing": True,
        "notes": "Curates only the CSV and PNG artifacts cited by the manuscript."
    },
    {
        "section_id": "G8",
        "title": "Display curated paper figures",
        "depends_on": "G7",
        "primary_outputs": "runtime displays only",
        "paper_facing": True,
        "notes": "Visual QA for exported PNG assets."
    },
    {
        "section_id": "G9",
        "title": "Preview curated paper evidence",
        "depends_on": "G7",
        "primary_outputs": "runtime prints and displays only",
        "paper_facing": True,
        "notes": "Runtime-side synthesis from curated paper artifacts only."
    },
]

dependency_map_df = pd.DataFrame(dependency_rows)
dependency_map_path = TABLES_DIR / "table_late_stage_dependency_map.csv"
materialize_dataframe(con, dependency_map_df, "table_late_stage_dependency_map", "E0")

paper_path_df = dependency_map_df[dependency_map_df["paper_facing"]].copy()

print("\nLate-stage dependency map:")
display(dependency_map_df)

print("\nPaper-facing execution path:")
display(paper_path_df[["section_id", "title", "depends_on", "primary_outputs"]])

print("\nCurated late-stage policy:")
print("- Auxiliary comparable discrete arm sections were removed from the stage workflow.")
print("- Narrative interpretation comments were reduced so conclusions come from executed outputs.")
print("- The curated manuscript path now terminates at outputs_benchmark_survival/paper_main and outputs_benchmark_survival/paper_appendix.")

print("\nSaved:")
print("-", dependency_map_path.resolve())

print(f"[END] E0.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 9
from datetime import datetime as _dt
print(f"[START] E1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# E1 — Benchmark Comparability Audit
# ============================================================
# Purpose:
#   Audit structural comparability across benchmark model families.
#   This step does not retrain anything. It creates an auditable map
#   of which inputs/tables/representations are used by each family.
#
# Main outputs:
#   - table_p10_0_benchmark_comparability_audit.csv
#   - table_p10_0_benchmark_comparability_summary.csv
#   - metadata_p10_0_benchmark_comparability_audit.json
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 70)
print("E1 — Benchmark Comparability Audit")
print("=" * 70)

if "con" not in globals():
    raise NameError("DuckDB connection 'con' not found. Run the setup stages first.")

OUT_BASE = Path("outputs_benchmark_survival")
OUT_TABLES = OUT_BASE / "tables"
OUT_METADATA = OUT_BASE / "metadata"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_METADATA.mkdir(parents=True, exist_ok=True)

inventory_df = load_duckdb_table_optional("table_5_16_model_inventory")

def _paper_arm_from_family_group(family_group: str) -> str:
    return base.paper_arm_from_family_group(family_group)


def _representation_from_family_group(family_group: str) -> str:
    return base.representation_from_family_group(family_group)


dynamic_operational_groups = {"dynamic_weekly", "dynamic_neural"}
comparable_operational_groups = {
    "comparable_continuous_time",
    "comparable_tree_survival",
    "comparable_parametric",
    "comparable_neural",
}


canonical_table_map = {}
if inventory_df is not None and not inventory_df.empty:
    for _, row in inventory_df.iterrows():
        mapped_tables = []
        primary_table = row.get("primary_metrics_table", pd.NA)
        if pd.notna(primary_table):
            mapped_tables.append(str(primary_table))
        required_tables_json = row.get("required_tables_json", "[]")
        try:
            mapped_tables.extend(json.loads(required_tables_json))
        except Exception:
            pass

        for mapped_table in mapped_tables:
            mapped_table = str(mapped_table)
            if not mapped_table or mapped_table == "None":
                continue
            canonical_table_map.setdefault(mapped_table, {
                "stage_id": str(row.get("stage_id", "")),
                "model_name": str(row.get("model_name", "")),
                "family_group": str(row.get("family_group", "")),
                "paper_arm": _paper_arm_from_family_group(str(row.get("family_group", ""))),
                "representation_type": _representation_from_family_group(str(row.get("family_group", ""))),
            })

available_tables = sorted(
    con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist()
)

# ------------------------------------------------------------
# 1) Candidate benchmark-related tables
# ------------------------------------------------------------
candidate_rows = []

for tbl in available_tables:
    lname = tbl.lower()

    is_relevant = any(token in lname for token in [
        "pp_", "person_period", "hazard", "survival",
        "cox", "deepsurv", "discrete", "continuous",
        "neural", "linear", "input", "ready", "split"
    ])

    if not is_relevant:
        continue

    try:
        cols_df = con.execute(f"PRAGMA table_info('{tbl}')").fetchdf()
        cols = cols_df["name"].astype(str).tolist()
        n_rows = con.execute(f"SELECT COUNT(*) AS n FROM {tbl}").fetchdf()["n"].iloc[0]

        colset = set(cols)
        has_week = "week" in colset
        has_id_student = "id_student" in colset
        has_module = "code_module" in colset
        has_presentation = "code_presentation" in colset
        has_event = "event_observed" in colset or "event" in colset
        has_t_final = "t_final_week" in colset
        has_t_event = "t_event_week" in colset

        canonical_match = canonical_table_map.get(tbl)

        if canonical_match is not None:
            temporal_representation = canonical_match["representation_type"]
            benchmark_family = canonical_match["paper_arm"]
            detected_from = "canonical_stage_catalog"
            canonical_stage_id = canonical_match["stage_id"]
            canonical_model_name = canonical_match["model_name"]
            canonical_family_group = canonical_match["family_group"]
        else:
            # Fallback heuristics are only used for auxiliary tables that are
            # not explicitly enumerated in the canonical D5.16 roster.
            if has_week and n_rows > 50000:
                temporal_representation = "dynamic_weekly"
            elif (not has_week) and has_t_final:
                temporal_representation = "enrollment_level_survival"
            elif (not has_week) and ("week" not in colset):
                temporal_representation = "static_or_summary_level"
            else:
                temporal_representation = "other"

            if "cox" in lname:
                benchmark_family = "continuous_time"
            elif "deepsurv" in lname:
                benchmark_family = "continuous_time_neural"
            elif "linear_hazard" in lname or "neural_hazard" in lname or "hazard" in lname:
                benchmark_family = "discrete_time"
            elif "person_period" in lname or "pp_" in lname:
                benchmark_family = "person_period_upstream"
            elif "survival" in lname:
                benchmark_family = "enrollment_survival_upstream"
            else:
                benchmark_family = "other"
            detected_from = "fallback_name_and_schema_heuristics"
            canonical_stage_id = pd.NA
            canonical_model_name = pd.NA
            canonical_family_group = pd.NA

        candidate_rows.append({
            "table_name": tbl,
            "n_rows": int(n_rows),
            "n_columns": int(len(cols)),
            "has_id_student": has_id_student,
            "has_code_module": has_module,
            "has_code_presentation": has_presentation,
            "has_week": has_week,
            "has_event_signal": has_event,
            "has_t_event_week": has_t_event,
            "has_t_final_week": has_t_final,
            "temporal_representation_detected": temporal_representation,
            "benchmark_family_detected": benchmark_family,
            "detected_from": detected_from,
            "canonical_stage_id": canonical_stage_id,
            "canonical_model_name": canonical_model_name,
            "canonical_family_group": canonical_family_group,
            "columns_preview": ", ".join(cols[:20]),
        })

    except Exception as e:
        candidate_rows.append({
            "table_name": tbl,
            "n_rows": np.nan,
            "n_columns": np.nan,
            "has_id_student": np.nan,
            "has_code_module": np.nan,
            "has_code_presentation": np.nan,
            "has_week": np.nan,
            "has_event_signal": np.nan,
            "has_t_event_week": np.nan,
            "has_t_final_week": np.nan,
            "temporal_representation_detected": "error",
            "benchmark_family_detected": "error",
            "columns_preview": f"ERROR: {str(e)}",
        })

table_candidates = pd.DataFrame(candidate_rows)

if table_candidates.empty:
    raise RuntimeError(
        "No benchmark-related tables were detected. "
        "Run the upstream benchmark stages before P10.0."
    )

# ------------------------------------------------------------
# 2) Roster-driven structural audit layer
# ------------------------------------------------------------
# This layer derives the comparability map from the canonical
# D5.16 roster so the audit covers the full current benchmark
# instead of a manually curated subset.

if inventory_df is None or inventory_df.empty:
    raise FileNotFoundError(
        "Required canonical roster table not found: table_5_16_model_inventory. "
        "Run D5.16 before executing E1."
    )


def _default_input_table(model_name: str, family_group: str) -> str:
    model_name = str(model_name)
    family_group = str(family_group)
    if model_name == "runtime_contract_materialization":
        return "table_5_1_runtime_contract / table_5_1_modeling_contract"
    if model_name == "linear_discrete_time_hazard":
        return "pp_linear_hazard_ready_*"
    if model_name == "neural_discrete_time_hazard":
        return "pp_neural_hazard_ready_*"
    if family_group in dynamic_operational_groups:
        return "weekly dynamic person-period treatment tables"
    if model_name == "deepsurv":
        return "enrollment_deepsurv_ready_*"
    if family_group in comparable_operational_groups:
        return "enrollment_cox_ready_*"
    return "catalog-driven benchmark artifacts"


def _profile_from_inventory_row(row: pd.Series) -> dict:
    stage_id = str(row["stage_id"])
    stage_order = int(row["stage_order"])
    model_name = str(row["model_name"])
    family_group = str(row["family_group"])
    artifact_status = str(row.get("artifact_status", "unknown"))

    if family_group == "contract":
        paper_arm = "contract_stage"
        representation_type = "contract_materialization"
        input_representation = "runtime_contract_tables"
        training_contract = "runtime_contract_materialization"
        update_regime = "not_a_model"
        benchmark_role = "contract_stage"
        comparability_status = "not_applicable"
        rationale = (
            "D5.1 materializes runtime and modeling contracts only. "
            "It belongs to the benchmark roster but is not a predictive model."
        )
    elif family_group in dynamic_operational_groups:
        paper_arm = "dynamic_weekly_person_period"
        representation_type = "dynamic_weekly"
        input_representation = "weekly_person_period"
        training_contract = "dynamic_weekly_person_period"
        update_regime = "weekly_updating"
        benchmark_role = "main_benchmark_model"
        comparability_status = "within_arm_comparable_dynamic_weekly"
        rationale = (
            "This model belongs to the dynamic weekly paper arm. "
            "Operational subgroup granularity inside the dynamic arm does not change the shared weekly person-period contract."
        )
    elif family_group in comparable_operational_groups:
        paper_arm = "comparable_continuous_time_early_window"
        representation_type = "early_window_summary_or_enrollment_level"
        input_representation = "early_window_enrollment_summary"
        training_contract = "static_after_early_window"
        update_regime = "static_after_early_window"
        benchmark_role = "main_benchmark_model"
        comparability_status = "within_arm_comparable_early_window"
        rationale = (
            "This model belongs to the comparable continuous-time paper arm. "
            "Operational subgroup granularity inside the comparable arm does not change the shared early-window enrollment contract."
        )
    else:
        paper_arm = "other"
        representation_type = "other"
        input_representation = "other"
        training_contract = "other"
        update_regime = "other"
        benchmark_role = "other"
        comparability_status = "requires_manual_review"
        rationale = "Unrecognized roster entry; requires manual review."

    if paper_arm == "dynamic_weekly_person_period":
        cross_arm_note = "Structurally asymmetric versus the early-window comparable arm."
    elif paper_arm == "comparable_continuous_time_early_window":
        cross_arm_note = "Structurally asymmetric versus the weekly person-period dynamic arm."
    else:
        cross_arm_note = "Not part of cross-arm predictive comparison."

    return {
        "stage_id": stage_id,
        "stage_order": stage_order,
        "model_or_family_label": model_name,
        "family": paper_arm,
        "operational_family_group": family_group,
        "likely_input_table": _default_input_table(model_name, family_group),
        "representation_type": representation_type,
        "input_representation": input_representation,
        "training_contract": training_contract,
        "update_regime": update_regime,
        "benchmark_role": benchmark_role,
        "comparability_status": comparability_status,
        "artifact_status": artifact_status,
        "cross_arm_note": cross_arm_note,
        "rationale": rationale,
    }


audit_rows = [
    _profile_from_inventory_row(row)
    for _, row in inventory_df.sort_values("stage_order", kind="mergesort").iterrows()
]
table_p10_0_benchmark_comparability_audit = pd.DataFrame(audit_rows)

# ------------------------------------------------------------
# 3) Family-level summary
# ------------------------------------------------------------
operational_group_count = int(
    table_p10_0_benchmark_comparability_audit.loc[
        table_p10_0_benchmark_comparability_audit["benchmark_role"] == "main_benchmark_model",
        "operational_family_group",
    ].nunique()
)
predictive_mask = table_p10_0_benchmark_comparability_audit["benchmark_role"] == "main_benchmark_model"

overall_findings = [
    {
        "finding": "n_total_roster_entries",
        "value": int(table_p10_0_benchmark_comparability_audit.shape[0]),
    },
    {
        "finding": "n_predictive_roster_entries",
        "value": int(predictive_mask.sum()),
    },
    {
        "finding": "n_contract_stage_entries",
        "value": int((table_p10_0_benchmark_comparability_audit["benchmark_role"] == "contract_stage").sum()),
    },
    {
        "finding": "n_dynamic_weekly_person_period_models",
        "value": int((table_p10_0_benchmark_comparability_audit["family"] == "dynamic_weekly_person_period").sum()),
    },
    {
        "finding": "n_comparable_continuous_time_models",
        "value": int((table_p10_0_benchmark_comparability_audit["family"] == "comparable_continuous_time_early_window").sum()),
    },
    {
        "finding": "n_predictive_operational_subgroups",
        "value": operational_group_count,
    },
    {
        "finding": "benchmark_comparability_diagnosis",
        "value": (
            "The predictive benchmark remains a two-arm design: a dynamic weekly person-period arm "
            "and a comparable continuous-time early-window arm. The finer `family_group` labels are "
            "operational registry metadata and should not be read as additional methodological families."
        ),
    },
    {
        "finding": "comparability_interpretation_rule",
        "value": (
            "Within-arm comparisons are contract-comparable. Cross-arm comparisons remain benchmark-valid "
            "but should be narrated as harmonized comparisons across two permitted representations rather than "
            "as pure architecture-only contests."
        ),
    },
]

table_p10_0_benchmark_comparability_summary = pd.DataFrame(overall_findings)

# ------------------------------------------------------------
# 4) Save outputs
# ------------------------------------------------------------
path_candidates = OUT_TABLES / "table_p10_0_detected_benchmark_related_tables.csv"
path_audit = OUT_TABLES / "table_p10_0_benchmark_comparability_audit.csv"
path_summary = OUT_TABLES / "table_p10_0_benchmark_comparability_summary.csv"
path_metadata = OUT_METADATA / "metadata_p10_0_benchmark_comparability_audit.json"

materialize_dataframe(con, table_candidates, "table_p10_0_detected_benchmark_related_tables", "E1")
materialize_dataframe(con, table_p10_0_benchmark_comparability_audit, "table_p10_0_benchmark_comparability_audit", "E1")
materialize_dataframe(con, table_p10_0_benchmark_comparability_summary, "table_p10_0_benchmark_comparability_summary", "E1")

metadata = {
    "step": "P10.0",
    "title": "Benchmark Comparability Audit",
    "purpose": "Audit structural comparability across the current canonical benchmark roster.",
    "main_diagnosis": (
        "The current benchmark contains one contract stage plus fourteen predictive models. "
        "Those predictive models are organized under two paper-facing methodological arms while preserving "
        "finer operational subgroups for registry and freeze purposes."
    ),
    "output_tables": [
        path_candidates.as_posix(),
        path_audit.as_posix(),
        path_summary.as_posix(),
    ],
    "canonical_roster_table": "table_5_16_model_inventory",
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# ------------------------------------------------------------
# 5) Display
# ------------------------------------------------------------
print("\nDetected benchmark-related tables:")
display(table_candidates)

print("\nComparability audit:")
display(table_p10_0_benchmark_comparability_audit)

print("\nComparability summary:")
display(table_p10_0_benchmark_comparability_summary)

print("\nSaved:")
print("-", path_candidates.resolve())
print("-", path_audit.resolve())
print("-", path_summary.resolve())
print("-", path_metadata.resolve())

print(f"[END] E1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 11
from datetime import datetime as _dt
print(f"[START] E1.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E1.1 — Integrate Comparable Discrete-Time Minimal Arm into
#         Benchmark Results
# ==============================================================
# Purpose:
#   Consolidate benchmark results for the comparable arm,
#   incorporating the newly evaluated
#   discrete_time_comparable_minimal branch when that upstream
#   artifact is actually materialized.
#
# Main outputs:
#   - table_p10_5_comparable_arm_benchmark_results
#   - table_p10_5_comparable_arm_best_by_variant
#   - table_p10_5_benchmark_result_inventory
#   - metadata_p10_5_comparable_arm_integration.json
# ==============================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 70)
print("E1.1 — Integrate Comparable Discrete-Time Minimal Arm into Benchmark Results")
print("=" * 70)

required_names = ["con", "load_duckdb_table_or_raise", "load_duckdb_table_optional", "materialize_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

OUT_BASE = Path("outputs_benchmark_survival")
OUT_TABLES = OUT_BASE / "tables"
OUT_METADATA = OUT_BASE / "metadata"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_METADATA.mkdir(parents=True, exist_ok=True)

available_tables = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())

# --------------------------------------------------------------
# 1) Load currently available result artifacts
# --------------------------------------------------------------
source_table = "table_p10_4_comparable_discrete_metrics_by_horizon"

# Optional existing comparable-arm result sources
candidate_existing_tables = [
    "table_enrollment_model_metrics",
    "table_benchmark_enrollment_model_metrics",
    "table_main_benchmark_metrics",
    "table_model_metrics",
    "table_benchmark_results",
]

existing_loaded = []
for table_name in candidate_existing_tables:
    df = load_duckdb_table_optional(table_name)
    if df is not None:
        df = df.copy()
        df["source_table"] = table_name
        existing_loaded.append(df)

inventory_rows = []
inventory_rows.append({
    "artifact_name": source_table,
    "artifact_exists": source_table in available_tables,
    "artifact_role": "new_comparable_discrete_result",
    "n_rows": int(len(load_duckdb_table_or_raise(source_table))) if source_table in available_tables else np.nan,
})

for table_name in candidate_existing_tables:
    df = load_duckdb_table_optional(table_name)
    inventory_rows.append({
        "artifact_name": table_name,
        "artifact_exists": df is not None,
        "artifact_role": "optional_existing_benchmark_result_source",
        "n_rows": int(len(df)) if df is not None else np.nan,
    })

table_p10_5_benchmark_result_inventory = pd.DataFrame(inventory_rows)

# --------------------------------------------------------------
# 2) Integrate if and only if the upstream discrete result exists
# --------------------------------------------------------------
if source_table in available_tables:
    new_discrete_df = load_duckdb_table_or_raise(source_table)
    disc = new_discrete_df.copy()

    disc_norm = pd.DataFrame({
        "benchmark_variant": disc["benchmark_variant"],
        "benchmark_arm": disc["benchmark_arm"],
        "model_family": disc["benchmark_variant"],
        "model_label": "Comparable discrete-time minimal arm",
        "representation_type": disc["representation_type"],
        "update_regime": disc["update_regime"],
        "model_class": disc["model_class"],
        "horizon": pd.to_numeric(disc["horizon"], errors="coerce"),
        "metric_primary_name": "auc_enrollment_score_by_horizon",
        "metric_primary_value": pd.to_numeric(disc["auc_enrollment_score_by_horizon"], errors="coerce"),
        "metric_ap_name": "average_precision_by_horizon",
        "metric_ap_value": pd.to_numeric(disc["average_precision_by_horizon"], errors="coerce"),
        "metric_brier_name": "brier_by_horizon_on_risk",
        "metric_brier_value": pd.to_numeric(disc["brier_by_horizon_on_risk"], errors="coerce"),
        "n_test_enrollments_total": pd.to_numeric(disc["n_test_enrollments_total"], errors="coerce"),
        "n_test_enrollments_with_support": pd.to_numeric(disc["n_test_enrollments_with_support"], errors="coerce"),
        "support_fraction": pd.to_numeric(disc["support_fraction"], errors="coerce"),
        "positive_rate_by_horizon_supported": pd.to_numeric(disc["positive_rate_by_horizon_supported"], errors="coerce"),
        "feature_set_used": disc["feature_set_used"].astype(str),
        "result_status": "materialized_now",
        "source_note": "P10.4 comparable discrete-time minimal arm",
    })

    table_p10_5_comparable_arm_benchmark_results = disc_norm.copy()

    best_idx = (
        table_p10_5_comparable_arm_benchmark_results
        .groupby("benchmark_variant")["metric_primary_value"]
        .idxmax()
        .dropna()
        .astype(int)
        .tolist()
    )

    table_p10_5_comparable_arm_best_by_variant = (
        table_p10_5_comparable_arm_benchmark_results
        .loc[best_idx]
        .sort_values(["metric_primary_value", "horizon"], ascending=[False, True])
        .reset_index(drop=True)
    )

    integration_status = "applied"
    status_note = "Upstream comparable discrete-time result was available and integrated."
else:
    table_p10_5_comparable_arm_benchmark_results = pd.DataFrame([{
        "benchmark_variant": "discrete_time_comparable_minimal",
        "benchmark_arm": "comparable_arm",
        "model_family": "discrete_time_comparable_minimal",
        "model_label": "Comparable discrete-time minimal arm",
        "representation_type": pd.NA,
        "update_regime": pd.NA,
        "model_class": pd.NA,
        "horizon": pd.NA,
        "metric_primary_name": "auc_enrollment_score_by_horizon",
        "metric_primary_value": pd.NA,
        "metric_ap_name": "average_precision_by_horizon",
        "metric_ap_value": pd.NA,
        "metric_brier_name": "brier_by_horizon_on_risk",
        "metric_brier_value": pd.NA,
        "n_test_enrollments_total": pd.NA,
        "n_test_enrollments_with_support": pd.NA,
        "support_fraction": pd.NA,
        "positive_rate_by_horizon_supported": pd.NA,
        "feature_set_used": pd.NA,
        "result_status": "upstream_not_materialized",
        "source_note": "P10.4 comparable discrete-time minimal arm table not found in DuckDB.",
    }])

    table_p10_5_comparable_arm_best_by_variant = pd.DataFrame([{
        "benchmark_variant": "discrete_time_comparable_minimal",
        "benchmark_arm": "comparable_arm",
        "model_family": "discrete_time_comparable_minimal",
        "model_label": "Comparable discrete-time minimal arm",
        "result_status": "upstream_not_materialized",
        "selection_note": "No best-by-variant row can be selected because the upstream result table does not exist.",
    }])

    integration_status = "na_upstream_missing"
    status_note = "Upstream comparable discrete-time result was not materialized, so integration is marked N/A rather than failing artificially."

# --------------------------------------------------------------
# 3) Persist outputs
# --------------------------------------------------------------
materialize_dataframe(con, table_p10_5_comparable_arm_benchmark_results, "table_p10_5_comparable_arm_benchmark_results", "E1")
materialize_dataframe(con, table_p10_5_comparable_arm_best_by_variant, "table_p10_5_comparable_arm_best_by_variant", "E1")
materialize_dataframe(con, table_p10_5_benchmark_result_inventory, "table_p10_5_benchmark_result_inventory", "E1")

path_metadata = OUT_METADATA / "metadata_p10_5_comparable_arm_integration.json"
metadata = {
    "step": "E1.1",
    "title": "Integrate Comparable Discrete-Time Minimal Arm into Benchmark Results",
    "source_table_expected": source_table,
    "integration_status": integration_status,
    "notes": [
        status_note,
        "This step does not fabricate upstream results.",
        "If the upstream result table is absent, the step records N/A explicitly and continues."
    ],
    "output_tables": [
        "table_p10_5_comparable_arm_benchmark_results",
        "table_p10_5_comparable_arm_best_by_variant",
        "table_p10_5_benchmark_result_inventory",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nBenchmark result inventory:")
display(table_p10_5_benchmark_result_inventory)

print("\nComparable-arm benchmark results:")
display(table_p10_5_comparable_arm_benchmark_results)

print("\nBest result by variant:")
display(table_p10_5_comparable_arm_best_by_variant)

print("\nSaved:")
print("- table_p10_5_comparable_arm_benchmark_results")
print("- table_p10_5_comparable_arm_best_by_variant")
print("- table_p10_5_benchmark_result_inventory")
print("-", path_metadata.resolve())

print(f"[END] E1.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 13
from datetime import datetime as _dt
print(f"[START] E2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E2 — Survival Calibration Audit Inventory
# ==============================================================
# Purpose:
#   Inventory the canonical calibration-related artifacts that now
#   live in DuckDB and summarize whether the calibration audit stack
#   is materially available for downstream posthoc analysis.

import json
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 70)
print("E2 — Survival Calibration Audit Inventory")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "materialize_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

ensure_benchmark_horizon_tables()

available_tables = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())

required_artifacts = [
    ("table_benchmark_brier_by_horizon_wide", "benchmark_brier_wide"),
    ("table_benchmark_calibration_by_horizon_wide", "benchmark_calibration_wide"),
    ("table_benchmark_support_reference", "benchmark_support_reference"),
    ("table_p26_5_all_tuned_calibration_audit", "all_tuned_calibration_audit"),
    ("table_p26_5_reliability_bins_all_tuned_models", "all_tuned_reliability_bins"),
    ("table_p11_1_calibration_slope_intercept_by_horizon", "slope_intercept_by_horizon"),
]

inventory_rows = []
for table_name, artifact_role in required_artifacts:
    exists = table_name in available_tables
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]) if exists else np.nan
    inventory_rows.append({
        "artifact_name": table_name,
        "artifact_role": artifact_role,
        "artifact_type": "table",
        "available": bool(exists),
        "n_rows": n_rows,
    })

table_p11_0_calibration_artifact_inventory = pd.DataFrame(inventory_rows).sort_values(
    ["artifact_role", "artifact_name"], kind="mergesort"
).reset_index(drop=True)

summary_rows = []
for table_name, artifact_role in required_artifacts:
    row = table_p11_0_calibration_artifact_inventory.loc[
        table_p11_0_calibration_artifact_inventory["artifact_name"] == table_name
    ].iloc[0]
    summary_rows.append({
        "component_name": artifact_role,
        "artifact_name": table_name,
        "available": bool(row["available"]),
        "n_rows": row["n_rows"],
    })

table_p11_0_calibration_audit_summary = pd.DataFrame(summary_rows).sort_values(
    ["component_name"], kind="mergesort"
).reset_index(drop=True)

table_p11_0_calibration_missing_components = (
    table_p11_0_calibration_audit_summary.loc[
        ~table_p11_0_calibration_audit_summary["available"]
    ].reset_index(drop=True)
)

materialize_dataframe(
    con,
    table_p11_0_calibration_artifact_inventory,
    "table_p11_0_calibration_artifact_inventory",
    "E2",
)
materialize_dataframe(
    con,
    table_p11_0_calibration_audit_summary,
    "table_p11_0_calibration_audit_summary",
    "E2",
)
materialize_dataframe(
    con,
    table_p11_0_calibration_missing_components,
    "table_p11_0_calibration_missing_components",
    "E2",
)

path_metadata = OUT_METADATA / "metadata_p11_0_calibration_audit_inventory.json"
metadata = {
    "step": "E2",
    "title": "Survival Calibration Audit Inventory",
    "audit_status": "computed_from_duckdb_catalog",
    "notes": [
        "E2 now inventories calibration-related artifacts directly from DuckDB.",
        "Availability flags reflect the actual presence of canonical upstream tables."
    ],
    "output_tables": [
        "table_p11_0_calibration_artifact_inventory",
        "table_p11_0_calibration_audit_summary",
        "table_p11_0_calibration_missing_components",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nCalibration artifact inventory:")
display(table_p11_0_calibration_artifact_inventory)
print("\nCalibration audit summary:")
display(table_p11_0_calibration_audit_summary)
print("\nMissing components:")
display(table_p11_0_calibration_missing_components)

print(f"[END] E2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 17
from datetime import datetime as _dt
print(f"[START] E3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E3 — Horizon-wise Calibration Slope and Intercept Audit
# ==============================================================
# Purpose:
#   Re-materialize and summarize the canonical slope/intercept audit
#   exported upstream by the D-stage pipeline when it exists. If the canonical
#   upstream table is absent, this stage records the missing dependency
#   and materializes schema-stable placeholder outputs.

import json
from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("E3 — Horizon-wise Calibration Slope and Intercept Audit")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_optional", "materialize_dataframe", "empty_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

required_cols = [
    "model_id",
    "model",
    "family",
    "horizon_week",
    "calibration_intercept_logit",
    "calibration_slope_logit",
    "n_bins_used",
]
missing_upstream_tables = []
slope_df = load_duckdb_table_optional("table_p11_1_calibration_slope_intercept_by_horizon")
if slope_df is None:
    missing_upstream_tables.append("table_p11_1_calibration_slope_intercept_by_horizon")
    table_p11_1_calibration_slope_intercept_by_horizon = empty_dataframe(required_cols)
    table_p11_1_calibration_slope_intercept_summary = empty_dataframe([
        "model_id",
        "model",
        "family",
        "n_horizons",
        "n_horizons_with_nonmissing_slope",
        "n_horizons_with_nonmissing_intercept",
        "mean_abs_intercept",
        "mean_abs_slope_distance_from_1",
        "mean_n_bins_used",
    ])
else:
    slope_df = slope_df.copy()
    missing_cols = [c for c in required_cols if c not in slope_df.columns]
    if missing_cols:
        raise KeyError(
            "Canonical slope/intercept table is missing required columns: "
            + ", ".join(missing_cols)
        )

    slope_df["horizon_week"] = pd.to_numeric(slope_df["horizon_week"], errors="coerce")
    slope_df["calibration_intercept_logit"] = pd.to_numeric(slope_df["calibration_intercept_logit"], errors="coerce")
    slope_df["calibration_slope_logit"] = pd.to_numeric(slope_df["calibration_slope_logit"], errors="coerce")
    slope_df["n_bins_used"] = pd.to_numeric(slope_df["n_bins_used"], errors="coerce")

    table_p11_1_calibration_slope_intercept_by_horizon = (
        slope_df[required_cols]
        .sort_values(["family", "model", "horizon_week"], kind="mergesort")
        .reset_index(drop=True)
    )

    table_p11_1_calibration_slope_intercept_summary = (
        table_p11_1_calibration_slope_intercept_by_horizon
        .groupby(["model_id", "model", "family"], dropna=False)
        .agg(
            n_horizons=("horizon_week", "nunique"),
            n_horizons_with_nonmissing_slope=("calibration_slope_logit", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            n_horizons_with_nonmissing_intercept=("calibration_intercept_logit", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_abs_intercept=("calibration_intercept_logit", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            mean_abs_slope_distance_from_1=("calibration_slope_logit", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce") - 1)))),
            mean_n_bins_used=("n_bins_used", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        )
        .reset_index()
        .sort_values(["family", "model"], kind="mergesort")
        .reset_index(drop=True)
    )

materialize_dataframe(
    con,
    table_p11_1_calibration_slope_intercept_by_horizon,
    "table_p11_1_calibration_slope_intercept_by_horizon",
    "E3",
)
materialize_dataframe(
    con,
    table_p11_1_calibration_slope_intercept_summary,
    "table_p11_1_calibration_slope_intercept_summary",
    "E3",
)

path_metadata = OUT_METADATA / "metadata_p11_1_calibration_slope_intercept_audit.json"
metadata = {
    "step": "E3",
    "title": "Horizon-wise Calibration Slope and Intercept Audit",
    "audit_status": "computed_from_canonical_upstream_table" if not missing_upstream_tables else "missing_required_upstream_table",
    "notes": [
        "E3 now consumes the canonical slope/intercept table materialized by the D-stage pipeline.",
        "When the upstream table is absent, E3 materializes empty placeholders and records the missing dependency instead of aborting the full audit run."
    ],
    "missing_upstream_tables": missing_upstream_tables,
    "output_tables": [
        "table_p11_1_calibration_slope_intercept_by_horizon",
        "table_p11_1_calibration_slope_intercept_summary",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nE3 completed.")
print("Saved:")
print("- table_p11_1_calibration_slope_intercept_by_horizon")
print("- table_p11_1_calibration_slope_intercept_summary")
print("-", path_metadata.resolve())

print(f"[END] E3 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 24
from datetime import datetime as _dt
print(f"[START] E4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E4 — Build Unified Calibration Strengthening Table
# ==============================================================
# Purpose:
#   Merge the canonical calibration audit with the canonical
#   slope/intercept audit when both upstream tables exist. If one or
#   more canonical upstream tables are absent, this stage records the
#   missing dependencies and materializes schema-stable placeholders.

import json
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 70)
print("E4 — Build Unified Calibration Strengthening Table")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_optional", "materialize_dataframe", "empty_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

audit_df = load_duckdb_table_optional("table_p26_5_all_tuned_calibration_audit")
slope_df = load_duckdb_table_optional("table_p11_1_calibration_slope_intercept_by_horizon")
missing_upstream_tables = []
if audit_df is None:
    missing_upstream_tables.append("table_p26_5_all_tuned_calibration_audit")
if slope_df is None:
    missing_upstream_tables.append("table_p11_1_calibration_slope_intercept_by_horizon")

required_audit_cols = [
    "model_id",
    "model",
    "family",
    "horizon_week",
    "event_rate_at_horizon",
    "weighted_absolute_calibration_gap_by_horizon",
    "n_events_by_horizon",
    "n_evaluable_enrollments",
]
required_slope_cols = [
    "model_id",
    "model",
    "family",
    "horizon_week",
    "calibration_intercept_logit",
    "calibration_slope_logit",
    "n_bins_used",
]
if missing_upstream_tables:
    table_p11_2_calibration_strengthening_by_horizon = empty_dataframe([
        "model_id",
        "model",
        "family",
        "horizon_week",
        "event_rate_at_horizon",
        "weighted_absolute_calibration_gap_by_horizon",
        "n_events_by_horizon",
        "n_evaluable_enrollments",
        "calibration_intercept_logit",
        "calibration_slope_logit",
        "n_bins_used",
        "abs_intercept_strength",
        "abs_slope_distance_from_1",
    ])
    table_p11_2_calibration_strengthening_summary = empty_dataframe([
        "model_id",
        "model",
        "family",
        "n_horizons",
        "mean_event_rate",
        "mean_abs_gap",
        "mean_abs_intercept",
        "mean_abs_slope_distance_from_1",
        "mean_n_bins_used",
    ])
else:
    missing_audit = [c for c in required_audit_cols if c not in audit_df.columns]
    if missing_audit:
        raise KeyError(
            "Canonical calibration audit table is missing required columns: "
            + ", ".join(missing_audit)
        )

    missing_slope = [c for c in required_slope_cols if c not in slope_df.columns]
    if missing_slope:
        raise KeyError(
            "Canonical slope/intercept table is missing required columns: "
            + ", ".join(missing_slope)
        )

    for col in ["horizon_week", "event_rate_at_horizon", "weighted_absolute_calibration_gap_by_horizon", "n_events_by_horizon", "n_evaluable_enrollments"]:
        audit_df[col] = pd.to_numeric(audit_df[col], errors="coerce")
    for col in ["horizon_week", "calibration_intercept_logit", "calibration_slope_logit", "n_bins_used"]:
        slope_df[col] = pd.to_numeric(slope_df[col], errors="coerce")

    merged = audit_df[
        [
            "model_id",
            "model",
            "family",
            "horizon_week",
            "event_rate_at_horizon",
            "weighted_absolute_calibration_gap_by_horizon",
            "n_events_by_horizon",
            "n_evaluable_enrollments",
        ]
    ].merge(
        slope_df[
            [
                "model_id",
                "model",
                "family",
                "horizon_week",
                "calibration_intercept_logit",
                "calibration_slope_logit",
                "n_bins_used",
            ]
        ],
        on=["model_id", "model", "family", "horizon_week"],
        how="left",
        validate="many_to_one",
    )

    merged["abs_intercept_strength"] = merged["calibration_intercept_logit"].abs()
    merged["abs_slope_distance_from_1"] = (merged["calibration_slope_logit"] - 1).abs()

    table_p11_2_calibration_strengthening_by_horizon = (
        merged.sort_values(["family", "model", "horizon_week"], kind="mergesort")
        .reset_index(drop=True)
    )

    table_p11_2_calibration_strengthening_summary = (
        table_p11_2_calibration_strengthening_by_horizon
        .groupby(["model_id", "model", "family"], dropna=False)
        .agg(
            n_horizons=("horizon_week", "nunique"),
            mean_event_rate=("event_rate_at_horizon", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_abs_gap=("weighted_absolute_calibration_gap_by_horizon", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_abs_intercept=("abs_intercept_strength", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_abs_slope_distance_from_1=("abs_slope_distance_from_1", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_n_bins_used=("n_bins_used", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        )
        .reset_index()
        .sort_values(["family", "model"], kind="mergesort")
        .reset_index(drop=True)
    )

materialize_dataframe(
    con,
    table_p11_2_calibration_strengthening_by_horizon,
    "table_p11_2_calibration_strengthening_by_horizon",
    "E4",
)
materialize_dataframe(
    con,
    table_p11_2_calibration_strengthening_summary,
    "table_p11_2_calibration_strengthening_summary",
    "E4",
)

path_metadata = OUT_METADATA / "metadata_p11_2_calibration_strengthening_audit.json"
metadata = {
    "step": "E4",
    "title": "Build Unified Calibration Strengthening Table",
    "audit_status": "computed_from_canonical_upstream_tables" if not missing_upstream_tables else "missing_required_upstream_tables",
    "notes": [
        "E4 now consumes the canonical calibration-audit and slope/intercept tables materialized by the D-stage pipeline.",
        "When upstream calibration tables are absent, E4 materializes empty placeholders and records the missing dependencies instead of aborting the full audit run."
    ],
    "missing_upstream_tables": missing_upstream_tables,
    "output_tables": [
        "table_p11_2_calibration_strengthening_by_horizon",
        "table_p11_2_calibration_strengthening_summary",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nE4 completed.")
print("Saved:")
print("- table_p11_2_calibration_strengthening_by_horizon")
print("- table_p11_2_calibration_strengthening_summary")
print("-", path_metadata.resolve())

print(f"[END] E4 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 26
from datetime import datetime as _dt
print(f"[START] E4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E4.1 — Comparable-Arm Risk Stratification Stability Audit
# ==============================================================
# Purpose:
#   Evaluate whether the tuned comparable continuous-time models
#   preserve ordered risk stratification across benchmark horizons.
#
# Main outputs:
#   - table_p11_3_comparable_arm_risk_stratification_by_horizon
#   - table_p11_3_comparable_arm_risk_stratification_summary
#   - metadata_p11_3_comparable_arm_risk_stratification.json
# ==============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 70)
print("E4.1 — Comparable-Arm Risk Stratification Stability Audit")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_or_raise", "load_duckdb_table_optional", "materialize_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

model_specs = [
    {
        "table_name": "table_cox_tuned_test_predictions",
        "model_id": "cox_tuned",
        "model": "Cox Comparable (Tuned)",
        "family": "continuous_time_cox",
    },
    {
        "table_name": "table_deepsurv_tuned_test_predictions",
        "model_id": "deepsurv_tuned",
        "model": "DeepSurv (Tuned)",
        "family": "continuous_time_deepsurv",
    },
]

required_cols = ["enrollment_id", "event", "duration", "horizon_week", "pred_risk_h"]
strat_rows = []
summary_rows = []

for spec in model_specs:
    pred_df = load_duckdb_table_or_raise(spec["table_name"]).copy()
    missing_cols = [c for c in required_cols if c not in pred_df.columns]
    if missing_cols:
        raise KeyError(
            f"{spec['table_name']} is missing required columns: " + ", ".join(missing_cols)
        )

    pred_df = pred_df[required_cols].copy()
    pred_df["event"] = pd.to_numeric(pred_df["event"], errors="coerce")
    pred_df["duration"] = pd.to_numeric(pred_df["duration"], errors="coerce")
    pred_df["horizon_week"] = pd.to_numeric(pred_df["horizon_week"], errors="coerce")
    pred_df["pred_risk_h"] = pd.to_numeric(pred_df["pred_risk_h"], errors="coerce")
    pred_df = pred_df.dropna(subset=["duration", "horizon_week", "pred_risk_h"]).copy()

    pred_df["is_evaluable"] = ((pred_df["event"] == 1) | (pred_df["duration"] >= pred_df["horizon_week"]))
    pred_df["event_by_horizon"] = ((pred_df["event"] == 1) & (pred_df["duration"] <= pred_df["horizon_week"]))
    pred_df = pred_df.loc[pred_df["is_evaluable"]].copy()

    for horizon_week, g in pred_df.groupby("horizon_week", dropna=False):
        g = g.copy().sort_values(["pred_risk_h", "enrollment_id"], kind="mergesort")
        if len(g) < 12 or g["pred_risk_h"].nunique(dropna=True) < 3:
            continue

        n_strata = 5 if len(g) >= 50 else 4 if len(g) >= 24 else 3
        risk_rank = g["pred_risk_h"].rank(method="first")
        g["risk_stratum"] = pd.qcut(risk_rank, q=n_strata, labels=False, duplicates="drop")
        g["risk_stratum"] = pd.to_numeric(g["risk_stratum"], errors="coerce") + 1

        strata_df = (
            g.groupby("risk_stratum", dropna=False)
            .agg(
                n_in_stratum=("enrollment_id", "size"),
                mean_predicted_risk=("pred_risk_h", "mean"),
                observed_event_rate=("event_by_horizon", "mean"),
            )
            .reset_index()
            .sort_values("risk_stratum", kind="mergesort")
            .reset_index(drop=True)
        )
        strata_df["horizon_week"] = horizon_week
        strata_df["model_id"] = spec["model_id"]
        strata_df["model"] = spec["model"]
        strata_df["family"] = spec["family"]

        strat_rows.extend(strata_df.to_dict("records"))

        obs_series = pd.to_numeric(strata_df["observed_event_rate"], errors="coerce")
        pred_series = pd.to_numeric(strata_df["mean_predicted_risk"], errors="coerce")
        stratum_series = pd.to_numeric(strata_df["risk_stratum"], errors="coerce")

        summary_rows.append({
            "model_id": spec["model_id"],
            "model": spec["model"],
            "family": spec["family"],
            "horizon_week": float(horizon_week),
            "n_evaluable_enrollments": int(len(g)),
            "n_strata": int(len(strata_df)),
            "spearman_stratum_vs_observed_event_rate": float(stratum_series.corr(obs_series, method="spearman")),
            "spearman_stratum_vs_predicted_risk": float(stratum_series.corr(pred_series, method="spearman")),
            "top_minus_bottom_event_rate": float(obs_series.iloc[-1] - obs_series.iloc[0]),
            "top_minus_bottom_predicted_risk": float(pred_series.iloc[-1] - pred_series.iloc[0]),
            "is_observed_event_rate_monotone": bool(np.all(np.diff(obs_series.to_numpy()) >= -1e-12)),
            "is_predicted_risk_monotone": bool(np.all(np.diff(pred_series.to_numpy()) >= -1e-12)),
        })

if not strat_rows:
    raise ValueError("E4.1 could not build any comparable-arm risk strata from the canonical prediction tables.")

table_p11_3_comparable_arm_risk_stratification_by_horizon = (
    pd.DataFrame(strat_rows)
    .sort_values(["family", "model", "horizon_week", "risk_stratum"], kind="mergesort")
    .reset_index(drop=True)
)

table_p11_3_comparable_arm_risk_stratification_summary = (
    pd.DataFrame(summary_rows)
    .groupby(["model_id", "model", "family"], dropna=False)
    .agg(
        n_horizons=("horizon_week", "nunique"),
        mean_spearman_stratum_vs_observed_event_rate=("spearman_stratum_vs_observed_event_rate", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        mean_spearman_stratum_vs_predicted_risk=("spearman_stratum_vs_predicted_risk", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        mean_top_minus_bottom_event_rate=("top_minus_bottom_event_rate", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        mean_top_minus_bottom_predicted_risk=("top_minus_bottom_predicted_risk", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        n_monotone_observed_horizons=("is_observed_event_rate_monotone", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        n_monotone_predicted_horizons=("is_predicted_risk_monotone", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    )
    .reset_index()
    .sort_values(["family", "model"], kind="mergesort")
    .reset_index(drop=True)
)

materialize_dataframe(
    con,
    table_p11_3_comparable_arm_risk_stratification_by_horizon,
    "table_p11_3_comparable_arm_risk_stratification_by_horizon",
    "E4.1",
)
materialize_dataframe(
    con,
    table_p11_3_comparable_arm_risk_stratification_summary,
    "table_p11_3_comparable_arm_risk_stratification_summary",
    "E4.1",
)

path_metadata = OUT_METADATA / "metadata_p11_3_comparable_arm_risk_stratification.json"
metadata = {
    "step": "E4.1",
    "title": "Comparable-Arm Risk Stratification Stability Audit",
    "audit_status": "computed_from_canonical_prediction_tables",
    "notes": [
        "E4.1 uses the canonical tuned test-prediction tables exported upstream by the D-stage pipeline.",
        "Observed horizon event rates are computed only on evaluable enrollments at each horizon.",
        "This audit complements the classical Cox PH check by testing ordered empirical risk separation in the comparable arm."
    ],
    "output_tables": [
        "table_p11_3_comparable_arm_risk_stratification_by_horizon",
        "table_p11_3_comparable_arm_risk_stratification_summary",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nE4.1 completed.")
print("Saved:")
print("- table_p11_3_comparable_arm_risk_stratification_by_horizon")
print("- table_p11_3_comparable_arm_risk_stratification_summary")
print("-", path_metadata.resolve())

print(f"[END] E4.1 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 28
from datetime import datetime as _dt
print(f"[START] E4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E4.2 — Recalibration Opportunity Signal Audit
# ==============================================================
# Purpose:
#   Summarize where the canonical calibration evidence suggests that
#   a future validation-based recalibration experiment would be most
#   informative for the tuned benchmark models.
#
# Main outputs:
#   - table_p11_4_recalibration_opportunity_by_horizon
#   - table_p11_4_recalibration_opportunity_summary
#   - metadata_p11_4_recalibration_opportunity.json
# ==============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 70)
print("E4.2 — Recalibration Opportunity Signal Audit")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_optional", "materialize_dataframe", "empty_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

audit_df = load_duckdb_table_optional("table_p26_5_all_tuned_calibration_audit")
slope_df = load_duckdb_table_optional("table_p11_1_calibration_slope_intercept_by_horizon")
missing_upstream_tables = []
if audit_df is None:
    missing_upstream_tables.append("table_p26_5_all_tuned_calibration_audit")
if slope_df is None:
    missing_upstream_tables.append("table_p11_1_calibration_slope_intercept_by_horizon")

required_audit_cols = [
    "model_id",
    "model",
    "family",
    "horizon_week",
    "weighted_absolute_calibration_gap_by_horizon",
    "n_evaluable_enrollments",
    "n_events_by_horizon",
    "event_rate_at_horizon",
]
required_slope_cols = [
    "model_id",
    "model",
    "family",
    "horizon_week",
    "calibration_intercept_logit",
    "calibration_slope_logit",
    "n_bins_used",
]
if missing_upstream_tables:
    table_p11_4_recalibration_opportunity_by_horizon = empty_dataframe([
        "model_id",
        "model",
        "family",
        "horizon_week",
        "weighted_absolute_calibration_gap_by_horizon",
        "n_evaluable_enrollments",
        "n_events_by_horizon",
        "event_rate_at_horizon",
        "calibration_intercept_logit",
        "calibration_slope_logit",
        "n_bins_used",
        "abs_intercept_logit",
        "abs_slope_distance_from_1",
        "recalibration_priority",
        "recalibration_priority_rank",
        "recommended_next_step",
    ])
    table_p11_4_recalibration_opportunity_summary = empty_dataframe([
        "model_id",
        "model",
        "family",
        "n_horizons",
        "mean_abs_gap",
        "mean_abs_intercept",
        "mean_abs_slope_distance_from_1",
        "n_high_priority_horizons",
        "n_medium_priority_horizons",
        "n_low_priority_horizons",
        "max_priority",
        "recommended_next_step",
    ])
else:
    missing_audit_cols = [c for c in required_audit_cols if c not in audit_df.columns]
    missing_slope_cols = [c for c in required_slope_cols if c not in slope_df.columns]
    if missing_audit_cols:
        raise KeyError(
            "Canonical calibration audit is missing required columns: "
            + ", ".join(missing_audit_cols)
        )
    if missing_slope_cols:
        raise KeyError(
            "Canonical slope/intercept audit is missing required columns: "
            + ", ".join(missing_slope_cols)
        )

    for col in ["horizon_week", "weighted_absolute_calibration_gap_by_horizon", "n_evaluable_enrollments", "n_events_by_horizon", "event_rate_at_horizon"]:
        audit_df[col] = pd.to_numeric(audit_df[col], errors="coerce")
    for col in ["horizon_week", "calibration_intercept_logit", "calibration_slope_logit", "n_bins_used"]:
        slope_df[col] = pd.to_numeric(slope_df[col], errors="coerce")

    merged = audit_df[required_audit_cols].merge(
        slope_df[required_slope_cols],
        on=["model_id", "model", "family", "horizon_week"],
        how="left",
        validate="many_to_one",
    )
    merged["abs_intercept_logit"] = merged["calibration_intercept_logit"].abs()
    merged["abs_slope_distance_from_1"] = (merged["calibration_slope_logit"] - 1).abs()


def classify_recalibration_priority(row: pd.Series) -> str:
    gap = float(row.get("weighted_absolute_calibration_gap_by_horizon", np.nan))
    abs_intercept = float(row.get("abs_intercept_logit", np.nan))
    abs_slope_dist = float(row.get("abs_slope_distance_from_1", np.nan))

    if (
        (pd.notna(gap) and gap >= 0.08)
        or (pd.notna(abs_intercept) and abs_intercept >= 0.35)
        or (pd.notna(abs_slope_dist) and abs_slope_dist >= 0.35)
    ):
        return "high_priority"
    if (
        (pd.notna(gap) and gap >= 0.04)
        or (pd.notna(abs_intercept) and abs_intercept >= 0.20)
        or (pd.notna(abs_slope_dist) and abs_slope_dist >= 0.20)
    ):
        return "medium_priority"
    return "low_priority"


if not missing_upstream_tables:
    priority_rank = {"high_priority": 3, "medium_priority": 2, "low_priority": 1}
    merged["recalibration_priority"] = merged.apply(classify_recalibration_priority, axis=1)
    merged["recalibration_priority_rank"] = merged["recalibration_priority"].map(priority_rank).fillna(0)
    merged["recommended_next_step"] = merged["recalibration_priority"].map({
        "high_priority": "run_validation_based_recalibration",
        "medium_priority": "consider_validation_based_recalibration",
        "low_priority": "monitor_only",
    })

    table_p11_4_recalibration_opportunity_by_horizon = (
        merged.sort_values(["family", "model", "horizon_week"], kind="mergesort")
        .reset_index(drop=True)
    )

    summary_df = (
        table_p11_4_recalibration_opportunity_by_horizon
        .groupby(["model_id", "model", "family"], dropna=False)
        .agg(
            n_horizons=("horizon_week", "nunique"),
            mean_abs_gap=("weighted_absolute_calibration_gap_by_horizon", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_abs_intercept=("abs_intercept_logit", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            mean_abs_slope_distance_from_1=("abs_slope_distance_from_1", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            n_high_priority_horizons=("recalibration_priority", lambda s: int((pd.Series(s) == "high_priority").sum())),
            n_medium_priority_horizons=("recalibration_priority", lambda s: int((pd.Series(s) == "medium_priority").sum())),
            n_low_priority_horizons=("recalibration_priority", lambda s: int((pd.Series(s) == "low_priority").sum())),
            max_priority_rank=("recalibration_priority_rank", lambda s: int(pd.to_numeric(s, errors="coerce").max())),
        )
        .reset_index()
    )

    priority_label = {3: "high_priority", 2: "medium_priority", 1: "low_priority", 0: "unknown"}
    summary_df["max_priority"] = summary_df["max_priority_rank"].map(priority_label)
    summary_df["recommended_next_step"] = summary_df["max_priority"].map({
        "high_priority": "run_validation_based_recalibration",
        "medium_priority": "consider_validation_based_recalibration",
        "low_priority": "monitor_only",
        "unknown": "monitor_only",
    })

    table_p11_4_recalibration_opportunity_summary = (
        summary_df
        .drop(columns=["max_priority_rank"])
        .sort_values(["family", "model"], kind="mergesort")
        .reset_index(drop=True)
    )

materialize_dataframe(
    con,
    table_p11_4_recalibration_opportunity_by_horizon,
    "table_p11_4_recalibration_opportunity_by_horizon",
    "E4.2",
)
materialize_dataframe(
    con,
    table_p11_4_recalibration_opportunity_summary,
    "table_p11_4_recalibration_opportunity_summary",
    "E4.2",
)

path_metadata = OUT_METADATA / "metadata_p11_4_recalibration_opportunity.json"
metadata = {
    "step": "E4.2",
    "title": "Recalibration Opportunity Signal Audit",
    "audit_status": "computed_from_canonical_calibration_tables" if not missing_upstream_tables else "missing_required_upstream_tables",
    "notes": [
        "E4.2 does not fit recalibrated models on the test set.",
        "Instead, it summarizes where the canonical horizon-wise calibration evidence suggests that a validation-based recalibration experiment would be most informative.",
        "The classification thresholds are editorially conservative and intended for manuscript synthesis, not for deployment decisions.",
        "When canonical calibration inputs are absent, E4.2 materializes empty placeholders and records the missing dependencies instead of aborting the full audit run."
    ],
    "missing_upstream_tables": missing_upstream_tables,
    "output_tables": [
        "table_p11_4_recalibration_opportunity_by_horizon",
        "table_p11_4_recalibration_opportunity_summary",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nE4.2 completed.")
print("Saved:")
print("- table_p11_4_recalibration_opportunity_by_horizon")
print("- table_p11_4_recalibration_opportunity_summary")
print("-", path_metadata.resolve())

print(f"[END] E4.2 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 36
from datetime import datetime as _dt
print(f"[START] E5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E5 — Sensitivity Design Audit
# ==============================================================
# Purpose:
#   Audit current sensitivity-design coverage in the stage workflow
#   and identify what is already present vs. what is still
#   missing for Group D.
#
# Main outputs:
#   - table_p12_0_sensitivity_artifact_inventory.csv
#   - table_p12_0_sensitivity_design_summary.csv
#   - table_p12_0_sensitivity_missing_components.csv
#   - metadata_p12_0_sensitivity_design_audit.json
# ==============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 70)
print("E5 — Sensitivity Design Audit")
print("=" * 70)

required_names = ["con", "materialize_dataframe", "load_duckdb_table_optional", "empty_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

OUT_BASE = Path("outputs_benchmark_survival")
OUT_TABLES = OUT_BASE / "tables"
OUT_METADATA = OUT_BASE / "metadata"
OUT_FIGURES = OUT_BASE / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_METADATA.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# 1) Inventory potentially relevant sensitivity artifacts
# --------------------------------------------------------------
candidate_files = []
available_tables_for_inventory = sorted(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())
discovery_tokens = ["p12_", "sensitivity", "window_sensitivity", "horizon_choice"]
excluded_discovery_prefixes = ("table_p12_0_", "table_p12_4_", "metadata_p12_0_", "metadata_p12_4_")
for tbl in available_tables_for_inventory:
    name_l = tbl.lower()
    if name_l.startswith(excluded_discovery_prefixes):
        continue
    if any(tok in name_l for tok in discovery_tokens):
        candidate_files.append({"artifact_name": tbl, "artifact_path": tbl, "artifact_type": "table", "exists": True, "size_bytes": 0.0})

tokens = discovery_tokens

for folder, kind in [
    (OUT_TABLES, "table"),
    (OUT_FIGURES, "figure"),
    (OUT_METADATA, "metadata"),
]:
    if folder.exists():
        for p in sorted(folder.iterdir()):
            name_l = p.name.lower()
            if name_l.startswith(excluded_discovery_prefixes):
                continue
            if any(tok in name_l for tok in tokens):
                candidate_files.append({
                    "artifact_name": p.name,
                    "artifact_path": str(p.resolve()),
                    "artifact_type": kind,
                    "exists": True,
                    "size_bytes": p.stat().st_size if p.exists() else np.nan,
                })

table_p12_0_sensitivity_artifact_inventory = pd.DataFrame(candidate_files)

if table_p12_0_sensitivity_artifact_inventory.empty:
    table_p12_0_sensitivity_artifact_inventory = pd.DataFrame([{
        "artifact_name": "(none found)",
        "artifact_path": "",
        "artifact_type": "none",
        "exists": False,
        "size_bytes": 0.0,
    }])

table_names = set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())

for col in ["artifact_role", "component_group", "available", "n_rows"]:
    if col not in table_p12_0_sensitivity_artifact_inventory.columns:
        table_p12_0_sensitivity_artifact_inventory[col] = pd.NA

expected_artifacts = [
    {
        "artifact_name": "table_p12_1_window_sensitivity_design",
        "artifact_type": "table",
        "artifact_role": "window_design_spec",
        "component_group": "window_sensitivity",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_1_window_sensitivity_created_columns",
        "artifact_type": "table",
        "artifact_role": "window_design_columns",
        "component_group": "window_sensitivity",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_2_window_sensitivity_comparison",
        "artifact_type": "table",
        "artifact_role": "window_comparison_table",
        "component_group": "window_sensitivity",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_2_window_sensitivity_summary",
        "artifact_type": "table",
        "artifact_role": "window_summary_table",
        "component_group": "window_sensitivity",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_2_window_sensitivity_registry",
        "artifact_type": "table",
        "artifact_role": "window_registry_table",
        "component_group": "window_sensitivity",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_benchmark_risk_auc_by_horizon_wide",
        "artifact_type": "table",
        "artifact_role": "horizon_auc_upstream",
        "component_group": "horizon_choice_upstream",
        "missing_status": "missing_upstream",
    },
    {
        "artifact_name": "table_benchmark_brier_by_horizon_wide",
        "artifact_type": "table",
        "artifact_role": "horizon_brier_upstream",
        "component_group": "horizon_choice_upstream",
        "missing_status": "missing_upstream",
    },
    {
        "artifact_name": "table_benchmark_calibration_by_horizon_wide",
        "artifact_type": "table",
        "artifact_role": "horizon_calibration_upstream",
        "component_group": "horizon_choice_upstream",
        "missing_status": "missing_upstream",
    },
    {
        "artifact_name": "table_p12_3_horizon_choice_stress_test",
        "artifact_type": "table",
        "artifact_role": "horizon_stress_test_table",
        "component_group": "horizon_choice_audit",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_3_horizon_choice_summary",
        "artifact_type": "table",
        "artifact_role": "horizon_summary_table",
        "component_group": "horizon_choice_audit",
        "missing_status": "missing",
    },
    {
        "artifact_name": "table_p12_3_horizon_choice_registry",
        "artifact_type": "table",
        "artifact_role": "horizon_registry_table",
        "component_group": "horizon_choice_audit",
        "missing_status": "missing",
    },
]

inventory_rows = []
for spec in expected_artifacts:
    artifact_name = spec["artifact_name"]
    artifact_type = spec["artifact_type"]
    exists = artifact_name in table_names if artifact_type == "table" else False
    n_rows = float(con.execute(f"SELECT COUNT(*) FROM {artifact_name}").fetchone()[0]) if exists and artifact_type == "table" else 0.0
    inventory_rows.append({
        "artifact_name": artifact_name,
        "artifact_path": artifact_name if exists and artifact_type == "table" else "",
        "artifact_type": artifact_type,
        "artifact_role": spec["artifact_role"],
        "component_group": spec["component_group"],
        "exists": bool(exists),
        "available": bool(exists),
        "size_bytes": 0.0,
        "n_rows": n_rows,
        "materialized_units": n_rows if exists else 0.0,
        "expected_missing_status": spec["missing_status"],
    })

discovered_artifacts = table_p12_0_sensitivity_artifact_inventory.copy()
discovered_artifacts["artifact_role"] = discovered_artifacts["artifact_role"].fillna("discovered_supporting_artifact")
discovered_artifacts["component_group"] = discovered_artifacts["component_group"].fillna("discovered_supporting_artifact")
discovered_artifacts["available"] = discovered_artifacts["exists"].fillna(False).astype(bool)
discovered_artifacts["n_rows"] = discovered_artifacts.apply(
    lambda row: float(con.execute(f"SELECT COUNT(*) FROM {row['artifact_name']}").fetchone()[0])
    if str(row.get("artifact_type", "")) == "table" and bool(row.get("exists", False)) and str(row.get("artifact_name", "")) in table_names
    else 1.0 if bool(row.get("exists", False))
    else 0.0,
    axis=1,
)
discovered_artifacts["materialized_units"] = discovered_artifacts["n_rows"].fillna(0.0)
discovered_artifacts["expected_missing_status"] = discovered_artifacts.get("expected_missing_status", pd.Series(index=discovered_artifacts.index, dtype=object)).fillna("not_expected")

inventory_df = pd.DataFrame(inventory_rows)
if not inventory_df.empty:
    known_artifacts = set(inventory_df["artifact_name"].astype(str).tolist())
    discovered_artifacts = discovered_artifacts.loc[
        ~discovered_artifacts["artifact_name"].astype(str).isin(known_artifacts)
    ].copy()

table_p12_0_sensitivity_artifact_inventory = pd.concat(
    [inventory_df, discovered_artifacts],
    ignore_index=True,
    sort=False,
)

def _artifact_status(row: pd.Series) -> str:
    exists = bool(row.get("exists", False) or row.get("available", False))
    n_rows = pd.to_numeric(pd.Series([row.get("n_rows", np.nan)]), errors="coerce").iloc[0]
    if exists and (pd.isna(n_rows) or n_rows > 0):
        return "present"
    if exists and n_rows == 0:
        return "empty"
    expected_missing_status = str(row.get("expected_missing_status", "missing"))
    return expected_missing_status


summary_rows = []
for component_group, group_df in table_p12_0_sensitivity_artifact_inventory.groupby("component_group", dropna=False):
    group_df = group_df.copy()
    statuses = group_df.apply(_artifact_status, axis=1)
    if (statuses == "missing_upstream").any():
        status = "missing_upstream"
    elif (statuses == "missing").any():
        status = "partial" if (statuses == "present").any() or (statuses == "empty").any() else "missing"
    elif (statuses == "empty").any():
        status = "empty"
    else:
        status = "present"

    summary_rows.append({
        "component_name": str(component_group),
        "status": status,
        "n_materialized_units": float(group_df["exists"].fillna(False).astype(bool).sum()),
        "key_signal_1": f"expected_artifacts={int(len(group_df))}",
        "key_signal_2": f"available_artifacts={int(group_df['exists'].fillna(False).astype(bool).sum())}",
    })

table_p12_0_sensitivity_design_summary = (
    pd.DataFrame(summary_rows)
    .sort_values(["component_name"], kind="mergesort")
    .reset_index(drop=True)
)

table_p12_0_sensitivity_missing_components = (
    table_p12_0_sensitivity_artifact_inventory.assign(status=lambda df: df.apply(_artifact_status, axis=1))
    .loc[lambda df: df["status"] != "present"]
    [[
        "component_group",
        "artifact_role",
        "artifact_name",
        "status",
        "exists",
        "n_rows",
    ]]
    .sort_values(["component_group", "artifact_role", "artifact_name"], kind="mergesort")
    .reset_index(drop=True)
)

materialize_dataframe(
    con,
    table_p12_0_sensitivity_artifact_inventory,
    "table_p12_0_sensitivity_artifact_inventory",
    "E5",
)
materialize_dataframe(
    con,
    table_p12_0_sensitivity_design_summary,
    "table_p12_0_sensitivity_design_summary",
    "E5",
)
materialize_dataframe(
    con,
    table_p12_0_sensitivity_missing_components,
    "table_p12_0_sensitivity_missing_components",
    "E5",
)

path_metadata = OUT_METADATA / "metadata_p12_0_sensitivity_design_audit.json"
metadata = {
    "step": "E5",
    "title": "Sensitivity Design Audit",
    "audit_status": "computed_from_duckdb_catalog",
    "notes": [
        "E5 inventories expected sensitivity artifacts and distinguishes present, empty, missing, and missing_upstream states.",
        "The horizon-choice upstream benchmark-wide tables are tracked explicitly so downstream summaries do not mistake placeholder outputs for completed evidence."
    ],
    "output_tables": [
        "table_p12_0_sensitivity_artifact_inventory",
        "table_p12_0_sensitivity_design_summary",
        "table_p12_0_sensitivity_missing_components",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nSensitivity artifact inventory:")
display(table_p12_0_sensitivity_artifact_inventory)
print("\nSensitivity design summary:")
display(table_p12_0_sensitivity_design_summary)
print("\nSensitivity missing components:")
display(table_p12_0_sensitivity_missing_components)
print("\nSaved:")
print("- table_p12_0_sensitivity_artifact_inventory")
print("- table_p12_0_sensitivity_design_summary")
print("- table_p12_0_sensitivity_missing_components")
print("-", path_metadata.resolve())

print(f"[END] E5 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 41
from datetime import datetime as _dt
print(f"[START] E6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E6 — Horizon-Choice Stress-Test Framing
# ==============================================================
# Purpose:
#   Reshape the canonical benchmark-wide horizon tables exported by
#   the D-stage pipeline and summarize how benchmark rankings vary by horizon.
#   If the benchmark-wide upstream tables are absent, this stage records
#   the missing dependencies and materializes schema-stable placeholders.

import json
from pathlib import Path
import numpy as np
import pandas as pd

print("\n" + "=" * 70)
print("E6 — Horizon-Choice Stress-Test Framing")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_optional", "materialize_dataframe", "empty_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

ensure_benchmark_horizon_tables()

auc_df = load_duckdb_table_optional("table_benchmark_risk_auc_by_horizon_wide")
brier_df = load_duckdb_table_optional("table_benchmark_brier_by_horizon_wide")
calib_df = load_duckdb_table_optional("table_benchmark_calibration_by_horizon_wide")
registry_df = load_duckdb_table_optional("table_benchmark_model_registry")
missing_upstream_tables = []
if auc_df is None:
    missing_upstream_tables.append("table_benchmark_risk_auc_by_horizon_wide")
if brier_df is None:
    missing_upstream_tables.append("table_benchmark_brier_by_horizon_wide")
if calib_df is None:
    missing_upstream_tables.append("table_benchmark_calibration_by_horizon_wide")

id_cols = ["model_key", "display_name", "family"]
if not missing_upstream_tables:
    for df_name, df in [("auc_df", auc_df), ("brier_df", brier_df), ("calib_df", calib_df)]:
        missing_id = [c for c in id_cols if c not in df.columns]
        if missing_id:
            raise KeyError(f"{df_name} is missing required identifier columns: " + ", ".join(missing_id))

def _wide_to_long(df: pd.DataFrame, value_name: str, accepted_prefixes: list[str]) -> pd.DataFrame:
    value_cols = []
    for col in df.columns:
        col_s = str(col)
        if any(col_s.startswith(pref) for pref in accepted_prefixes):
            value_cols.append(col)
    if not value_cols:
        raise KeyError(
            f"No horizon columns found for {value_name}. Expected prefixes: " + ", ".join(accepted_prefixes)
        )
    out = df[id_cols + value_cols].melt(id_vars=id_cols, var_name="metric_horizon", value_name=value_name)
    out["horizon_week"] = out["metric_horizon"].astype(str).str.extract(r"h+(\d+)", expand=False)
    out["horizon_week"] = pd.to_numeric(out["horizon_week"], errors="coerce")
    return out.drop(columns=["metric_horizon"])

if missing_upstream_tables:
    table_p12_3_horizon_choice_stress_test = empty_dataframe([
        "model_key",
        "display_name",
        "family",
        "horizon_week",
        "auc_value",
        "brier_value",
        "calibration_gap_value",
    ])
    table_p12_3_horizon_choice_summary = empty_dataframe([
        "model_key",
        "display_name",
        "family",
        "n_horizons",
        "auc_range",
        "brier_range",
        "calibration_gap_range",
    ])
else:
    auc_long = _wide_to_long(auc_df, "auc_value", ["risk_auc_h"])
    brier_long = _wide_to_long(brier_df, "brier_value", ["brier_h"])
    calib_long = _wide_to_long(calib_df, "calibration_gap_value", ["calibration_h"])

    table_p12_3_horizon_choice_stress_test = (
        auc_long.merge(brier_long, on=id_cols + ["horizon_week"], how="outer")
        .merge(calib_long, on=id_cols + ["horizon_week"], how="outer")
        .sort_values(["family", "display_name", "horizon_week"], kind="mergesort")
        .reset_index(drop=True)
    )

    summary_rows = []
    for (model_key, display_name, family), g in table_p12_3_horizon_choice_stress_test.groupby(id_cols, dropna=False):
        summary_rows.append({
            "model_key": model_key,
            "display_name": display_name,
            "family": family,
            "n_horizons": int(pd.to_numeric(g["horizon_week"], errors="coerce").nunique()),
            "auc_range": float(pd.to_numeric(g["auc_value"], errors="coerce").max() - pd.to_numeric(g["auc_value"], errors="coerce").min()),
            "brier_range": float(pd.to_numeric(g["brier_value"], errors="coerce").max() - pd.to_numeric(g["brier_value"], errors="coerce").min()),
            "calibration_gap_range": float(pd.to_numeric(g["calibration_gap_value"], errors="coerce").max() - pd.to_numeric(g["calibration_gap_value"], errors="coerce").min()),
        })
    table_p12_3_horizon_choice_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["family", "display_name"], kind="mergesort")
        .reset_index(drop=True)
    )

registry_rows = [
    {"artifact_name": "table_benchmark_risk_auc_by_horizon_wide", "artifact_exists": auc_df is not None, "artifact_role": "required_benchmark_horizon_table"},
    {"artifact_name": "table_benchmark_brier_by_horizon_wide", "artifact_exists": brier_df is not None, "artifact_role": "required_benchmark_horizon_table"},
    {"artifact_name": "table_benchmark_calibration_by_horizon_wide", "artifact_exists": calib_df is not None, "artifact_role": "required_benchmark_horizon_table"},
]
if registry_df is not None:
    registry_rows.append({"artifact_name": "table_benchmark_model_registry", "artifact_exists": True, "artifact_role": "optional_model_registry"})
else:
    registry_rows.append({"artifact_name": "table_benchmark_model_registry", "artifact_exists": False, "artifact_role": "optional_model_registry"})
table_p12_3_horizon_choice_registry = pd.DataFrame(registry_rows)

materialize_dataframe(con, table_p12_3_horizon_choice_stress_test, "table_p12_3_horizon_choice_stress_test", "E6")
materialize_dataframe(con, table_p12_3_horizon_choice_summary, "table_p12_3_horizon_choice_summary", "E6")
materialize_dataframe(con, table_p12_3_horizon_choice_registry, "table_p12_3_horizon_choice_registry", "E6")

path_metadata = OUT_METADATA / "metadata_p12_3_horizon_choice_stress_test.json"
metadata = {
    "step": "E6",
    "title": "Horizon-Choice Stress-Test Framing",
    "audit_status": "computed_from_canonical_upstream_tables" if not missing_upstream_tables else "missing_required_upstream_tables",
    "notes": [
        "E6 consumes the canonical benchmark-wide horizon tables from the D-stage pipeline.",
        "When one or more benchmark-wide horizon tables are absent, E6 materializes empty placeholders and records the missing dependencies instead of aborting the full audit run."
    ],
    "missing_upstream_tables": missing_upstream_tables,
    "output_tables": [
        "table_p12_3_horizon_choice_stress_test",
        "table_p12_3_horizon_choice_summary",
        "table_p12_3_horizon_choice_registry",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nE6 completed.")
print("Saved:")
print("- table_p12_3_horizon_choice_stress_test")
print("- table_p12_3_horizon_choice_summary")
print("- table_p12_3_horizon_choice_registry")
print("-", path_metadata.resolve())

print(f"[END] E6 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 43
from datetime import datetime as _dt
print(f"[START] E7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==============================================================
# E7 — Unified Sensitivity Summary
# ==============================================================
# Purpose:
#   Consolidate the sensitivity-related outputs from E5 and E6 into
#   a unified summary and component registry, preserving explicit
#   component names from the design-audit inventory.

import json
from pathlib import Path
import numpy as np
import pandas as pd

print("\n" + "=" * 70)
print("E7 — Unified Sensitivity Summary")
print("=" * 70)

OUT_BASE = Path("outputs_benchmark_survival")
OUT_METADATA = OUT_BASE / "metadata"
OUT_METADATA.mkdir(parents=True, exist_ok=True)

required_names = ["con", "load_duckdb_table_or_raise", "load_duckdb_table_optional", "materialize_dataframe"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise NameError(
        "Missing required objects from previous setup stages: "
        + ", ".join(missing_names)
        + ". Run prior setup stages first."
    )

sens_inventory_df = load_duckdb_table_optional("table_p12_0_sensitivity_artifact_inventory")
if sens_inventory_df is None and "table_p12_0_sensitivity_artifact_inventory" in globals():
    sens_inventory_df = table_p12_0_sensitivity_artifact_inventory
if sens_inventory_df is None:
    raise FileNotFoundError("Required DuckDB table not found: table_p12_0_sensitivity_artifact_inventory")
sens_inventory_df = sens_inventory_df.copy()

sens_summary_df = load_duckdb_table_optional("table_p12_0_sensitivity_design_summary")
if sens_summary_df is None and "table_p12_0_sensitivity_design_summary" in globals():
    sens_summary_df = table_p12_0_sensitivity_design_summary
if sens_summary_df is None:
    sens_summary_df = pd.DataFrame()
sens_summary_df = sens_summary_df.copy()

horizon_summary_df = load_duckdb_table_or_raise("table_p12_3_horizon_choice_summary").copy()
horizon_stress_df = load_duckdb_table_or_raise("table_p12_3_horizon_choice_stress_test").copy()
horizon_registry_df = load_duckdb_table_optional("table_p12_3_horizon_choice_registry")
non_horizon_inventory_df = sens_inventory_df.loc[
    ~sens_inventory_df["component_group"].astype(str).str.startswith("horizon_choice")
].copy()
non_horizon_summary_df = sens_summary_df.loc[
    ~sens_summary_df.get("component_name", pd.Series(dtype=object)).astype(str).str.startswith("horizon_choice")
].copy() if not sens_summary_df.empty and "component_name" in sens_summary_df.columns else sens_summary_df.copy()


def _combine_component_status(statuses: pd.Series) -> str:
    statuses = pd.Series(statuses).dropna().astype(str)
    if statuses.empty:
        return "missing"
    ordered_statuses = statuses.tolist()
    if any(status == "missing_upstream" for status in ordered_statuses):
        return "missing_upstream"
    if any(status == "missing" for status in ordered_statuses):
        return "partial" if any(status in {"present", "empty"} for status in ordered_statuses) else "missing"
    if any(status == "empty" for status in ordered_statuses):
        return "empty"
    return "present"

# Design-audit components: preserve explicit names from the inventory
design_component_rows = []
for _, row in sens_inventory_df.iterrows():
    artifact_name = str(row.get("artifact_name", "unknown_artifact"))
    artifact_role = row.get("artifact_role", pd.NA)
    component_name = artifact_role if pd.notna(artifact_role) else artifact_name
    exists_flag = bool(row.get("exists", False) or row.get("available", False))
    n_rows = pd.to_numeric(pd.Series([row.get("n_rows", np.nan)]), errors="coerce").iloc[0]
    if exists_flag and (pd.isna(n_rows) or n_rows > 0):
        row_status = "present"
    elif exists_flag and n_rows == 0:
        row_status = "empty"
    else:
        row_status = str(row.get("expected_missing_status", "missing"))
    design_component_rows.append({
        "component_group": "design_audit",
        "component_name": str(component_name),
        "status": row_status,
        "n_materialized_units": row.get("materialized_units", row.get("n_rows", 0.0)),
        "key_signal_1": artifact_name,
        "key_signal_2": row.get("artifact_type", pd.NA),
    })

# If the sensitivity design summary has higher-level rows, keep them too
if "component_name" in sens_summary_df.columns:
    for _, row in sens_summary_df.iterrows():
        design_component_rows.append({
            "component_group": "design_audit_summary",
            "component_name": str(row.get("component_name", "design_summary_component")),
            "status": str(row.get("status", "present")),
            "n_materialized_units": row.get("n_materialized_units", np.nan),
            "key_signal_1": row.get("key_signal_1", pd.NA),
            "key_signal_2": row.get("key_signal_2", pd.NA),
        })

horizon_component_rows = []
required_horizon_upstream_ok = True
if horizon_registry_df is not None and not horizon_registry_df.empty:
    required_mask = horizon_registry_df["artifact_role"].astype(str) == "required_benchmark_horizon_table"
    if required_mask.any():
        required_horizon_upstream_ok = bool(horizon_registry_df.loc[required_mask, "artifact_exists"].fillna(False).astype(bool).all())

if not required_horizon_upstream_ok:
    horizon_component_rows.append({
        "component_group": "horizon_choice",
        "component_name": "benchmark_horizon_choice",
        "status": "missing_upstream",
        "n_materialized_units": 0,
        "key_signal_1": f"stress_rows={len(horizon_stress_df)}",
        "key_signal_2": f"summary_rows={len(horizon_summary_df)}",
    })
elif horizon_summary_df.empty:
    horizon_component_rows.append({
        "component_group": "horizon_choice",
        "component_name": "benchmark_horizon_choice",
        "status": "empty",
        "n_materialized_units": 0,
        "key_signal_1": f"stress_rows={len(horizon_stress_df)}",
        "key_signal_2": f"summary_rows={len(horizon_summary_df)}",
    })
else:
    for _, row in horizon_summary_df.iterrows():
        horizon_component_rows.append({
            "component_group": "horizon_choice",
            "component_name": str(row.get("display_name", "unknown_model")),
            "status": "present",
            "n_materialized_units": row.get("n_horizons", np.nan),
            "key_signal_1": f"auc_range={row.get('auc_range', pd.NA)}",
            "key_signal_2": f"calibration_gap_range={row.get('calibration_gap_range', pd.NA)}",
        })

table_p12_4_sensitivity_unified_summary = (
    pd.DataFrame(design_component_rows + horizon_component_rows)
    .sort_values(["component_group", "component_name"], kind="mergesort")
    .reset_index(drop=True)
)

table_p12_4_sensitivity_component_registry = pd.DataFrame([
    {
        "component_name": "design_audit",
        "status": _combine_component_status(non_horizon_summary_df.get("status", pd.Series(dtype=object))),
        "n_materialized_units": int(len(non_horizon_inventory_df)),
        "key_signal_1": f"inventory_rows={len(non_horizon_inventory_df)}",
        "key_signal_2": f"summary_rows={len(non_horizon_summary_df)}",
    },
    {
        "component_name": "horizon_choice",
        "status": _combine_component_status(pd.Series([row["status"] for row in horizon_component_rows], dtype=object)),
        "n_materialized_units": int(len(horizon_summary_df)),
        "key_signal_1": f"stress_rows={len(horizon_stress_df)}",
        "key_signal_2": f"summary_rows={len(horizon_summary_df)}",
    },
])

table_p12_4_sensitivity_family_summary = (
    horizon_summary_df.groupby("family", dropna=False)
    .agg(
        n_models=("display_name", "nunique"),
        mean_auc_range=("auc_range", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        mean_brier_range=("brier_range", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
        mean_calibration_gap_range=("calibration_gap_range", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
    )
    .reset_index()
    .sort_values(["family"], kind="mergesort")
    .reset_index(drop=True)
)

materialize_dataframe(
    con,
    table_p12_4_sensitivity_unified_summary,
    "table_p12_4_sensitivity_unified_summary",
    "E7",
)
materialize_dataframe(
    con,
    table_p12_4_sensitivity_component_registry,
    "table_p12_4_sensitivity_component_registry",
    "E7",
)
materialize_dataframe(
    con,
    table_p12_4_sensitivity_family_summary,
    "table_p12_4_sensitivity_family_summary",
    "E7",
)

path_metadata = OUT_METADATA / "metadata_p12_4_sensitivity_unified_summary.json"
metadata = {
    "step": "E7",
    "title": "Unified Sensitivity Summary",
    "audit_status": _combine_component_status(table_p12_4_sensitivity_component_registry["status"]),
    "notes": [
        "E7 consolidates explicit design-audit components from the E5 inventory.",
        "The horizon-choice block is populated from the canonical E6 tables when the benchmark-wide upstream inputs exist.",
        "Aggregate component status now distinguishes present, empty, partial, and missing_upstream states."
    ],
    "output_tables": [
        "table_p12_4_sensitivity_unified_summary",
        "table_p12_4_sensitivity_component_registry",
        "table_p12_4_sensitivity_family_summary",
    ],
}
with open(path_metadata, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nUnified sensitivity summary:")
display(table_p12_4_sensitivity_unified_summary)
print("\nComponent registry:")
display(table_p12_4_sensitivity_component_registry)
print("\nFamily summary:")
display(table_p12_4_sensitivity_family_summary)

print(f"[END] E7 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 45
from datetime import datetime as _dt
print(f"[START] E8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

shutdown_duckdb_connection_from_globals(globals())

print(f"[END] E8 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% Cell 46
