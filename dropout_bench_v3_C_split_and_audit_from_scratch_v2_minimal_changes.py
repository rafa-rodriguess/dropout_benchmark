from __future__ import annotations

"""
Production split-and-audit module for benchmark stage C.

Purpose:
- build the canonical paper-aligned enrollment split from DuckDB-only inputs
- build the approved context-held-out split design
- propagate both split contracts onto the stage B modeling tables
- materialize the model-ready train/test DuckDB tables consumed downstream

Input contract:
- benchmark_shared_config.toml
- DuckDB table enrollment_survival_ready
- DuckDB table enrollment_model_table
- DuckDB table pp_linear_hazard_input
- DuckDB table pp_neural_hazard_input
- DuckDB table enrollment_cox_input_configured
- DuckDB table enrollment_deepsurv_input_configured

Output contract:
- DuckDB tables table_c1_event_time_audit
- DuckDB tables table_enrollment_split_assignment, table_enrollment_split_summary,
  table_enrollment_split_bucket_summary, table_enrollment_split_leakage_check,
  table_context_overlap_summary, table_c2_2_compact_audit
- DuckDB tables table_context_group_split_design, table_context_group_split_summary,
  enrollment_survival_ready_context_split, table_context_enrollment_split_assignment,
  table_c2_5_linkage_audit, table_c2_5_split_comparison_audit,
  table_c2_5_context_group_audit
- DuckDB tables pp_linear_hazard_input_split, pp_neural_hazard_input_split,
  enrollment_cox_input_configured_split, enrollment_deepsurv_input_configured_split,
  table_c3_split_propagation_audit, pp_linear_hazard_ready_train,
  pp_linear_hazard_ready_test, pp_neural_hazard_ready_train,
  pp_neural_hazard_ready_test, enrollment_cox_ready_train,
  enrollment_cox_ready_test, enrollment_deepsurv_ready_train,
  enrollment_deepsurv_ready_test, table_c3_ready_dataset_audit
- DuckDB tables pp_linear_hazard_input_context_split,
  pp_neural_hazard_input_context_split,
  enrollment_cox_input_configured_context_split,
  enrollment_deepsurv_input_configured_context_split,
  table_c3_1_context_split_propagation_audit,
  pp_linear_hazard_context_ready_train, pp_linear_hazard_context_ready_test,
  pp_neural_hazard_context_ready_train, pp_neural_hazard_context_ready_test,
  enrollment_cox_context_ready_train, enrollment_cox_context_ready_test,
  enrollment_deepsurv_context_ready_train, enrollment_deepsurv_context_ready_test,
  table_c3_1_context_ready_dataset_audit

Failure policy:
- missing DuckDB source tables or required columns raise immediately
- invalid split strata or incomplete split propagation raise immediately
- missing approved context groups raise immediately
- no CSV fallback, notebook fallback, or permissive split fallback is permitted
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


STAGE_PREFIX = "C"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREVIEW_ROWS = 20
REQUIRED_SURVIVAL_READY_COLUMNS = [
    "id_student",
    "code_module",
    "code_presentation",
    "final_result",
    "event_observed",
    "date_unregistration",
    "is_withdrawn_without_valid_unregistration",
    "t_event_week",
    "t_last_obs_week",
    "t_final_week",
]
REQUIRED_ENROLLMENT_MODEL_COLUMNS = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "event",
    "duration",
]
REQUIRED_MODELING_INPUT_TABLES = [
    "pp_linear_hazard_input",
    "pp_neural_hazard_input",
    "enrollment_cox_input_configured",
    "enrollment_deepsurv_input_configured",
]
REQUIRED_SPLIT_JOIN_COLUMNS = ["id_student", "code_module", "code_presentation"]
APPROVED_CONTEXT_TEST_GROUPS = (
    "AAA||2014J",
    "BBB||2013J",
    "CCC||2014B",
    "DDD||2013J",
    "EEE||2013J",
    "FFF||2014B",
    "GGG||2014J",
)
CONTEXT_SPLIT_NAME = "context_heldout_v1"


@dataclass
class PipelineContext:
    project_root: Path
    script_name: str
    config: dict[str, Any]
    config_toml_path: Path
    output_dir: Path
    tables_dir: Path
    metadata_dir: Path
    duckdb_path: Path
    run_id: str
    run_metadata_path: Path
    benchmark_config: dict[str, Any]
    con: Any


def stage_label(number: str) -> str:
    return f"{STAGE_PREFIX}{number}"


def log_stage_start(number: str, title: str) -> None:
    print(f"[START] {stage_label(number)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# ==============================================================")
    print(f"# {stage_label(number)} - {title}")
    print("# ==============================================================")


def log_stage_end(number: str) -> None:
    print(f"[END] {stage_label(number)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_artifact(label: str, location: str) -> None:
    print(f"ARTIFACT | {label} | {location}")


def get_sensitivity_windows(ctx: PipelineContext) -> list[int]:
    from dropout_bench_v3_D_00_common import resolve_early_window_sensitivity_weeks

    return resolve_early_window_sensitivity_weeks(ctx.benchmark_config)


def available_tables(con: Any) -> set[str]:
    return set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())


def get_table_columns(con: Any, table_name: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]


def require_tables(con: Any, required_tables: list[str], stage_id: str) -> None:
    actual_tables = available_tables(con)
    missing_tables = [table_name for table_name in required_tables if table_name not in actual_tables]
    if missing_tables:
        raise RuntimeError(
            f"{stage_label(stage_id)} requires DuckDB table(s): {', '.join(missing_tables)}"
        )


def require_columns(con: Any, table_name: str, required_columns: list[str]) -> None:
    actual_columns = set(get_table_columns(con, table_name))
    missing_columns = sorted(set(required_columns) - actual_columns)
    if missing_columns:
        raise KeyError(f"{table_name} is missing required columns: {', '.join(missing_columns)}")


def fetch_required_dataframe(ctx: PipelineContext, table_name: str, required_columns: list[str]) -> pd.DataFrame:
    require_tables(ctx.con, [table_name], stage_id="0")
    require_columns(ctx.con, table_name, required_columns)
    return ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()


def materialize_dataframe_table(
    ctx: PipelineContext,
    df: pd.DataFrame,
    table_name: str,
    stage_id: str,
    label: str,
) -> None:
    from util import print_duckdb_table, refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_c_materialize_df__"
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
        notebook_name=ctx.script_name,
        cell_name=stage_label(stage_id),
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_duckdb_table(ctx.con, table_name, title=label, limit=PREVIEW_ROWS)
    print_artifact(label, f"duckdb://{table_name}")


def materialize_select_table(
    ctx: PipelineContext,
    select_sql: str,
    table_name: str,
    stage_id: str,
    label: str,
) -> None:
    from util import print_duckdb_table, refresh_pipeline_catalog_schema_view, register_duckdb_table

    ctx.con.execute(f"DROP TABLE IF EXISTS {table_name}")
    ctx.con.execute(f"CREATE TABLE {table_name} AS {select_sql}")
    register_duckdb_table(
        con=ctx.con,
        table_name=table_name,
        notebook_name=ctx.script_name,
        cell_name=stage_label(stage_id),
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_duckdb_table(ctx.con, table_name, title=label, limit=PREVIEW_ROWS)
    print_artifact(label, f"duckdb://{table_name}")


def build_enrollment_id(df: pd.DataFrame) -> pd.Series:
    required_columns = ["id_student", "code_module", "code_presentation"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "build_enrollment_id requires columns: "
            + ", ".join(required_columns)
            + ". Missing: "
            + ", ".join(missing_columns)
        )
    return (
        df["id_student"].astype(str)
        + "|"
        + df["code_module"].astype(str)
        + "|"
        + df["code_presentation"].astype(str)
    )


def enrollment_join_condition(left_alias: str, right_alias: str) -> str:
    return f"""
    CAST({left_alias}.id_student AS BIGINT) = CAST({right_alias}.id_student AS BIGINT)
    AND CAST({left_alias}.code_module AS VARCHAR) = CAST({right_alias}.code_module AS VARCHAR)
    AND CAST({left_alias}.code_presentation AS VARCHAR) = CAST({right_alias}.code_presentation AS VARCHAR)
    """


def normalize_enrollment_backbone(enrollment_df: pd.DataFrame) -> pd.DataFrame:
    work = enrollment_df.copy()
    if "enrollment_id" not in work.columns:
        work["enrollment_id"] = build_enrollment_id(work)
    work["enrollment_id"] = work["enrollment_id"].astype(str)
    work["event"] = pd.to_numeric(work["event"], errors="raise").astype(int)
    work["duration"] = pd.to_numeric(work["duration"], errors="raise").astype(int)
    if work["duration"].lt(0).any():
        raise ValueError("enrollment_model_table contains negative duration values.")
    duplicate_count = int(work["enrollment_id"].duplicated().sum())
    if duplicate_count != 0:
        raise ValueError(
            f"enrollment_model_table must be unique by enrollment_id. Duplicate rows: {duplicate_count}"
        )
    return (
        work.sort_values(
            ["enrollment_id", "id_student", "code_module", "code_presentation"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_event_duration_buckets(
    duration_series: pd.Series,
    event_series: pd.Series,
    n_buckets: int,
) -> pd.Series:
    if n_buckets < 2:
        raise ValueError(f"temporal_buckets_q must be at least 2. Received {n_buckets}.")
    duration = pd.to_numeric(duration_series, errors="raise").astype(int)
    event = pd.to_numeric(event_series, errors="raise").astype(int)
    if not event.isin([0, 1]).all():
        raise ValueError("Main split event labels must be binary 0/1.")
    event_mask = event.eq(1)
    if int(event_mask.sum()) < n_buckets:
        raise ValueError(
            "Not enough observed events to build the configured duration buckets for stratification."
        )

    bucket_labels = [f"bucket_{index}" for index in range(1, n_buckets + 1)]
    output = pd.Series("not_applicable_for_event_0", index=duration.index, dtype="object")

    event_duration = duration.loc[event_mask].sort_index()
    event_ranks = event_duration.rank(method="first", pct=True)
    event_buckets = pd.cut(
        event_ranks,
        bins=np.linspace(0.0, 1.0, num=n_buckets + 1),
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )
    if event_buckets.isna().any():
        raise ValueError("Observed-event duration bucketing produced null assignments.")

    output.loc[event_mask] = event_buckets.astype(str)
    return output.astype(str)


def validate_stratification_labels(strata_labels: pd.Series, test_size: float) -> None:
    counts = strata_labels.astype(str).value_counts().sort_index()
    invalid_counts = counts.loc[counts < 2]
    if not invalid_counts.empty:
        raise ValueError(
            "Main split stratification produced classes with fewer than 2 enrollments: "
            + ", ".join(f"{label}={count}" for label, count in invalid_counts.items())
        )
    expected_test_size = int(round(float(len(strata_labels)) * float(test_size)))
    if expected_test_size <= 0:
        raise ValueError("Configured test_size does not allocate any test enrollments.")
    if expected_test_size < len(counts):
        raise ValueError(
            "Configured test_size is too small for the number of stratification classes. "
            f"test_rows={expected_test_size}, classes={len(counts)}"
        )


def summarize_split(enrollment_df: pd.DataFrame) -> pd.DataFrame:
    return (
        enrollment_df.groupby("split", dropna=False)
        .agg(
            n_enrollments=("enrollment_id", "nunique"),
            n_events=("event", "sum"),
            event_rate=("event", "mean"),
            min_duration=("duration", "min"),
            median_duration=("duration", "median"),
            max_duration=("duration", "max"),
        )
        .reset_index()
        .sort_values("split")
        .reset_index(drop=True)
    )


def summarize_split_by_bucket(enrollment_df: pd.DataFrame) -> pd.DataFrame:
    return (
        enrollment_df.groupby(["split", "event", "duration_bucket"], dropna=False)
        .agg(n_enrollments=("enrollment_id", "nunique"))
        .reset_index()
        .sort_values(["split", "event", "duration_bucket"])
        .reset_index(drop=True)
    )


def build_main_split_assignment(ctx: PipelineContext) -> pd.DataFrame:
    enrollment_df = fetch_required_dataframe(
        ctx,
        table_name="enrollment_model_table",
        required_columns=REQUIRED_ENROLLMENT_MODEL_COLUMNS,
    )
    enrollment_df = normalize_enrollment_backbone(enrollment_df)

    test_size = float(ctx.benchmark_config.get("test_size", 0.3))
    random_state = int(ctx.benchmark_config.get("seed", 42))
    n_duration_buckets = int(ctx.benchmark_config.get("temporal_buckets_q", 4))

    enrollment_df["duration_bucket"] = build_event_duration_buckets(
        duration_series=enrollment_df["duration"],
        event_series=enrollment_df["event"],
        n_buckets=n_duration_buckets,
    )
    enrollment_df["strata_label"] = (
        "event_"
        + enrollment_df["event"].astype(int).astype(str)
        + "__"
        + enrollment_df["duration_bucket"].astype(str)
    )
    validate_stratification_labels(enrollment_df["strata_label"], test_size=test_size)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dummy_features = np.zeros(shape=(len(enrollment_df), 1))
    dummy_labels = enrollment_df["strata_label"].astype(str).to_numpy()
    train_index, test_index = next(splitter.split(dummy_features, dummy_labels))

    enrollment_df["split"] = "train"
    enrollment_df.loc[enrollment_df.index[test_index], "split"] = "test"

    output_columns = [
        "enrollment_id",
        "id_student",
        "code_module",
        "code_presentation",
        "event",
        "duration",
        "duration_bucket",
        "strata_label",
        "split",
    ]
    return enrollment_df[output_columns].copy()


def build_context_group_inventory(ctx: PipelineContext) -> pd.DataFrame:
    survival_df = fetch_required_dataframe(
        ctx,
        table_name="enrollment_survival_ready",
        required_columns=REQUIRED_SURVIVAL_READY_COLUMNS,
    )
    survival_df["group_id"] = (
        survival_df["code_module"].astype(str) + "||" + survival_df["code_presentation"].astype(str)
    )

    group_inventory = (
        survival_df.groupby(["code_module", "code_presentation", "group_id"], dropna=False)
        .agg(
            n_enrollments=("id_student", "size"),
            n_unique_students=("id_student", "nunique"),
            n_events=("event_observed", "sum"),
            event_rate=("event_observed", "mean"),
        )
        .reset_index()
        .sort_values(["code_module", "code_presentation"])
        .reset_index(drop=True)
    )

    available_group_ids = set(group_inventory["group_id"].astype(str).tolist())
    missing_approved_groups = sorted(set(APPROVED_CONTEXT_TEST_GROUPS) - available_group_ids)
    if missing_approved_groups:
        raise RuntimeError(
            "Approved context-held-out groups are missing from enrollment_survival_ready: "
            + ", ".join(missing_approved_groups)
        )

    group_inventory["split_role"] = np.where(
        group_inventory["group_id"].isin(APPROVED_CONTEXT_TEST_GROUPS),
        "test",
        "train",
    )
    group_inventory["is_test_group"] = group_inventory["split_role"].eq("test").astype(int)
    return group_inventory


def build_ready_table_audit(
    ctx: PipelineContext,
    ready_tables: list[str],
    split_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name in ready_tables:
        columns = get_table_columns(ctx.con, table_name)
        row_count = int(ctx.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        distinct_enrollment_count = int(
            ctx.con.execute(f"SELECT COUNT(DISTINCT enrollment_id) FROM {table_name}").fetchone()[0]
        )
        split_values_present = ctx.con.execute(
            f"SELECT STRING_AGG(DISTINCT {split_column}, ', ' ORDER BY {split_column}) FROM {table_name}"
        ).fetchone()[0]

        target_column = None
        target_sum = None
        if "event_t" in columns:
            target_column = "event_t"
            target_sum = float(
                ctx.con.execute(f"SELECT COALESCE(SUM(event_t), 0) FROM {table_name}").fetchone()[0]
            )
        elif "event" in columns:
            target_column = "event"
            target_sum = float(
                ctx.con.execute(f"SELECT COALESCE(SUM(event), 0) FROM {table_name}").fetchone()[0]
            )

        rows.append(
            {
                "table_name": table_name,
                "n_rows": row_count,
                "n_distinct_enrollments": distinct_enrollment_count,
                "n_columns": len(columns),
                f"{split_column}_values_present": split_values_present,
                "target_column": target_column,
                "target_sum": target_sum,
            }
        )
    return pd.DataFrame(rows)


def initialize_context() -> PipelineContext:
    from util import (
        ensure_pipeline_catalog,
        ensure_run_metadata,
        load_shared_config,
        open_duckdb_connection,
        resolve_project_path,
    )

    log_stage_start("0", "Initialize deterministic stage C runtime")

    config, config_toml_path = load_shared_config(PROJECT_ROOT)
    paths_config = config.get("paths", {})
    benchmark_config = config.get("benchmark", {})

    output_dir = resolve_project_path(PROJECT_ROOT, paths_config.get("output_dir", "outputs_benchmark_survival"))
    tables_dir = output_dir / paths_config.get("tables_subdir", "tables")
    metadata_dir = output_dir / paths_config.get("metadata_subdir", "metadata")
    duckdb_path = output_dir / paths_config.get("duckdb_filename", "benchmark_survival.duckdb")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_metadata, run_metadata_path = ensure_run_metadata(metadata_dir, SCRIPT_NAME)
    run_id = str(run_metadata.get("run_id", "")).strip()
    if not run_id:
        raise KeyError("run_metadata.json does not contain a valid run_id.")

    con = open_duckdb_connection(duckdb_path)
    ensure_pipeline_catalog(con)

    ctx = PipelineContext(
        project_root=PROJECT_ROOT,
        script_name=SCRIPT_NAME,
        config=config,
        config_toml_path=config_toml_path,
        output_dir=output_dir,
        tables_dir=tables_dir,
        metadata_dir=metadata_dir,
        duckdb_path=duckdb_path,
        run_id=run_id,
        run_metadata_path=run_metadata_path,
        benchmark_config=benchmark_config,
        con=con,
    )

    print(f"- SCRIPT_NAME: {ctx.script_name}")
    print(f"- RUN_ID     : {ctx.run_id}")
    print(f"- DUCKDB_PATH: {ctx.duckdb_path}")
    print_artifact("shared_config", str(ctx.config_toml_path))
    print_artifact("run_metadata", str(ctx.run_metadata_path))
    log_stage_end("0")
    return ctx


def stage_c1_event_time_audit(ctx: PipelineContext) -> None:
    log_stage_start("1", "Audit stage A survival timing contract")

    survival_df = fetch_required_dataframe(
        ctx,
        table_name="enrollment_survival_ready",
        required_columns=REQUIRED_SURVIVAL_READY_COLUMNS,
    )

    table_c1_event_time_audit = pd.DataFrame(
        [
            {
                "n_total_enrollments": int(len(survival_df)),
                "n_unique_enrollments": int(
                    (
                        survival_df["id_student"].astype(str)
                        + "||"
                        + survival_df["code_module"].astype(str)
                        + "||"
                        + survival_df["code_presentation"].astype(str)
                    ).nunique()
                ),
                "n_event_observed": int(
                    pd.to_numeric(survival_df["event_observed"], errors="raise").astype(int).sum()
                ),
                "event_rate": float(
                    pd.to_numeric(survival_df["event_observed"], errors="raise").astype(int).mean()
                ),
                "n_withdrawn_without_valid_unregistration": int(
                    pd.to_numeric(
                        survival_df["is_withdrawn_without_valid_unregistration"],
                        errors="raise",
                    ).astype(int).sum()
                ),
                "max_t_event_week": float(pd.to_numeric(survival_df["t_event_week"], errors="raise").max()),
                "max_t_last_obs_week": float(
                    pd.to_numeric(survival_df["t_last_obs_week"], errors="raise").max()
                ),
                "max_t_final_week": float(pd.to_numeric(survival_df["t_final_week"], errors="raise").max()),
            }
        ]
    )
    materialize_dataframe_table(
        ctx,
        df=table_c1_event_time_audit,
        table_name="table_c1_event_time_audit",
        stage_id="1",
        label="C1 output: stage A event-time audit",
    )
    log_stage_end("1")


def stage_c2_main_split(ctx: PipelineContext) -> None:
    log_stage_start("2", "Build the canonical paper-aligned main split")

    split_assignment = build_main_split_assignment(ctx)
    table_split_summary = summarize_split(split_assignment)
    table_split_bucket_summary = summarize_split_by_bucket(split_assignment)

    train_ids = set(split_assignment.loc[split_assignment["split"] == "train", "enrollment_id"])
    test_ids = set(split_assignment.loc[split_assignment["split"] == "test", "enrollment_id"])
    overlap_ids = train_ids.intersection(test_ids)
    table_leakage_check = pd.DataFrame(
        [
            {
                "n_train_enrollments": len(train_ids),
                "n_test_enrollments": len(test_ids),
                "n_enrollments_checked": len(train_ids.union(test_ids)),
                "n_enrollments_with_leakage": len(overlap_ids),
                "identity_leakage_detected": bool(overlap_ids),
            }
        ]
    )
    if bool(table_leakage_check.loc[0, "identity_leakage_detected"]):
        raise RuntimeError("Main split identity leakage detected in enrollment assignments.")

    materialize_dataframe_table(
        ctx,
        df=split_assignment,
        table_name="table_enrollment_split_assignment",
        stage_id="2",
        label="C2 output: canonical enrollment split assignment",
    )
    materialize_dataframe_table(
        ctx,
        df=table_split_summary,
        table_name="table_enrollment_split_summary",
        stage_id="2",
        label="C2 output: enrollment split summary",
    )
    materialize_dataframe_table(
        ctx,
        df=table_split_bucket_summary,
        table_name="table_enrollment_split_bucket_summary",
        stage_id="2",
        label="C2 output: enrollment split bucket summary",
    )
    materialize_dataframe_table(
        ctx,
        df=table_leakage_check,
        table_name="table_enrollment_split_leakage_check",
        stage_id="2",
        label="C2 output: enrollment split leakage check",
    )
    log_stage_end("2")


def stage_c2_1_context_overlap(ctx: PipelineContext) -> None:
    log_stage_start("2.1", "Audit context overlap in the main split")

    split_df = ctx.con.execute("SELECT * FROM table_enrollment_split_assignment").fetchdf()
    split_df["module_presentation"] = (
        split_df["code_module"].astype(str) + "||" + split_df["code_presentation"].astype(str)
    )

    train_df = split_df.loc[split_df["split"] == "train"].copy()
    test_df = split_df.loc[split_df["split"] == "test"].copy()
    table_context_overlap_summary = pd.DataFrame(
        [
            {
                "context_level": "code_module",
                "n_train_unique": int(train_df["code_module"].nunique()),
                "n_test_unique": int(test_df["code_module"].nunique()),
                "n_shared": int(
                    len(set(train_df["code_module"].astype(str)).intersection(set(test_df["code_module"].astype(str))))
                ),
            },
            {
                "context_level": "code_presentation",
                "n_train_unique": int(train_df["code_presentation"].nunique()),
                "n_test_unique": int(test_df["code_presentation"].nunique()),
                "n_shared": int(
                    len(
                        set(train_df["code_presentation"].astype(str)).intersection(
                            set(test_df["code_presentation"].astype(str))
                        )
                    )
                ),
            },
            {
                "context_level": "module_presentation",
                "n_train_unique": int(train_df["module_presentation"].nunique()),
                "n_test_unique": int(test_df["module_presentation"].nunique()),
                "n_shared": int(
                    len(
                        set(train_df["module_presentation"].astype(str)).intersection(
                            set(test_df["module_presentation"].astype(str))
                        )
                    )
                ),
            },
        ]
    )
    materialize_dataframe_table(
        ctx,
        df=table_context_overlap_summary,
        table_name="table_context_overlap_summary",
        stage_id="2.1",
        label="C2.1 output: main split context overlap summary",
    )
    log_stage_end("2.1")


def stage_c2_2_compact_audit(ctx: PipelineContext) -> None:
    log_stage_start("2.2", "Materialize the compact audit for the main split")

    split_df = ctx.con.execute("SELECT * FROM table_enrollment_split_assignment").fetchdf()
    train_df = split_df.loc[split_df["split"] == "train"].copy()
    test_df = split_df.loc[split_df["split"] == "test"].copy()

    table_c2_2_compact_audit = pd.DataFrame(
        [
            {
                "split_unit": "enrollment",
                "stratification": "event_status + event_duration_bucket_or_censor_marker",
                "n_total_enrollments": int(split_df["enrollment_id"].nunique()),
                "n_train_enrollments": int(train_df["enrollment_id"].nunique()),
                "n_test_enrollments": int(test_df["enrollment_id"].nunique()),
                "train_event_rate": float(pd.to_numeric(train_df["event"], errors="raise").astype(int).mean()),
                "test_event_rate": float(pd.to_numeric(test_df["event"], errors="raise").astype(int).mean()),
            }
        ]
    )
    materialize_dataframe_table(
        ctx,
        df=table_c2_2_compact_audit,
        table_name="table_c2_2_compact_audit",
        stage_id="2.2",
        label="C2.2 output: compact main split audit",
    )
    log_stage_end("2.2")


def stage_c2_3_context_design(ctx: PipelineContext) -> None:
    log_stage_start("2.3", "Build the approved context-held-out group design")

    group_inventory = build_context_group_inventory(ctx)
    table_context_group_split_summary = pd.DataFrame(
        [
            {
                "n_total_groups": int(len(group_inventory)),
                "n_test_groups": int(group_inventory["is_test_group"].sum()),
                "n_train_groups": int(group_inventory["split_role"].eq("train").sum()),
                "approved_design_name": CONTEXT_SPLIT_NAME,
            }
        ]
    )

    materialize_dataframe_table(
        ctx,
        df=group_inventory,
        table_name="table_context_group_split_design",
        stage_id="2.3",
        label="C2.3 output: context-held-out group split design",
    )
    materialize_dataframe_table(
        ctx,
        df=table_context_group_split_summary,
        table_name="table_context_group_split_summary",
        stage_id="2.3",
        label="C2.3 output: context-held-out group split summary",
    )
    log_stage_end("2.3")


def stage_c2_4_context_survival_split(ctx: PipelineContext) -> None:
    log_stage_start("2.4", "Attach the context-held-out design to the survival backbone")

    survival_df = fetch_required_dataframe(
        ctx,
        table_name="enrollment_survival_ready",
        required_columns=REQUIRED_SURVIVAL_READY_COLUMNS,
    )
    context_design = ctx.con.execute("SELECT * FROM table_context_group_split_design").fetchdf()
    survival_df["group_id"] = (
        survival_df["code_module"].astype(str) + "||" + survival_df["code_presentation"].astype(str)
    )
    context_split = survival_df.merge(
        context_design[["group_id", "code_module", "code_presentation", "is_test_group", "split_role"]],
        on=["group_id", "code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    if context_split["split_role"].isna().any():
        raise RuntimeError("Some survival enrollments did not receive a context-held-out split assignment.")
    context_split["context_split_name"] = CONTEXT_SPLIT_NAME
    context_split["context_is_test"] = context_split["split_role"].eq("test").astype(int)

    materialize_dataframe_table(
        ctx,
        df=context_split,
        table_name="enrollment_survival_ready_context_split",
        stage_id="2.4",
        label="C2.4 output: survival backbone with context split assignment",
    )
    log_stage_end("2.4")


def stage_c2_5_context_assignment(ctx: PipelineContext) -> None:
    log_stage_start("2.5", "Link the main split and context-held-out split contracts")

    main_assignment = ctx.con.execute("SELECT * FROM table_enrollment_split_assignment").fetchdf()
    model_backbone = fetch_required_dataframe(
        ctx,
        table_name="enrollment_model_table",
        required_columns=REQUIRED_ENROLLMENT_MODEL_COLUMNS,
    )
    context_design = ctx.con.execute("SELECT * FROM table_context_group_split_design").fetchdf()

    model_backbone = normalize_enrollment_backbone(model_backbone)
    comparison = model_backbone.merge(
        main_assignment.rename(columns={"split": "main_split"}),
        on="enrollment_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_main"),
    )
    comparison["group_id"] = (
        comparison["code_module"].astype(str) + "||" + comparison["code_presentation"].astype(str)
    )
    comparison = comparison.merge(
        context_design[["group_id", "code_module", "code_presentation", "split_role", "is_test_group"]],
        on=["group_id", "code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    if comparison["split_role"].isna().any():
        raise RuntimeError("Some modeling-backbone enrollments did not receive a context-held-out assignment.")

    comparison["context_split_name"] = CONTEXT_SPLIT_NAME
    comparison["context_is_test"] = comparison["split_role"].eq("test").astype(int)
    comparison["context_split"] = comparison["split_role"].astype(str)
    n_missing_main_split = int(comparison["main_split"].isna().sum())
    if n_missing_main_split != 0:
        raise RuntimeError(
            f"Some modeling-backbone enrollments are missing the main split assignment: {n_missing_main_split}"
        )

    table_context_enrollment_split_assignment = comparison[
        [
            "enrollment_id",
            "id_student",
            "code_module",
            "code_presentation",
            "group_id",
            "main_split",
            "context_split",
            "context_split_name",
            "context_is_test",
        ]
    ].rename(columns={"context_split": "split"})
    materialize_dataframe_table(
        ctx,
        df=table_context_enrollment_split_assignment,
        table_name="table_context_enrollment_split_assignment",
        stage_id="2.5",
        label="C2.5 output: canonical context-held-out enrollment assignment",
    )

    table_c2_5_linkage_audit = pd.DataFrame(
        [
            {
                "n_model_backbone_enrollments": int(model_backbone["enrollment_id"].nunique()),
                "n_main_split_enrollments": int(main_assignment["enrollment_id"].nunique()),
                "n_context_assignment_enrollments": int(comparison["enrollment_id"].nunique()),
                "n_missing_main_split_alignment": n_missing_main_split,
                "main_split_alignment_complete": bool(n_missing_main_split == 0),
            }
        ]
    )
    materialize_dataframe_table(
        ctx,
        df=table_c2_5_linkage_audit,
        table_name="table_c2_5_linkage_audit",
        stage_id="2.5",
        label="C2.5 output: main-versus-context linkage audit",
    )

    summary_rows: list[dict[str, Any]] = []
    for design_name, split_column in [
        ("main_paper_split", "main_split"),
        (CONTEXT_SPLIT_NAME, "context_split"),
    ]:
        for split_value, block in comparison.groupby(split_column, dropna=False):
            if pd.isna(split_value):
                continue
            summary_rows.append(
                {
                    "design_name": design_name,
                    "split": str(split_value),
                    "n_enrollments": int(block["enrollment_id"].nunique()),
                    "n_context_groups": int(block["group_id"].nunique()),
                    "n_unique_students": int(block["id_student"].nunique()),
                    "n_events": int(pd.to_numeric(block["event"], errors="raise").astype(int).sum()),
                    "event_rate": float(pd.to_numeric(block["event"], errors="raise").astype(int).mean()),
                    "median_time_week": float(pd.to_numeric(block["duration"], errors="raise").median()),
                }
            )

    table_c2_5_split_comparison_audit = pd.DataFrame(summary_rows).sort_values(
        ["design_name", "split"]
    ).reset_index(drop=True)
    materialize_dataframe_table(
        ctx,
        df=table_c2_5_split_comparison_audit,
        table_name="table_c2_5_split_comparison_audit",
        stage_id="2.5",
        label="C2.5 output: split comparison audit",
    )

    group_rows: list[dict[str, Any]] = []
    for group_key, block in comparison.groupby(
        ["code_module", "code_presentation", "group_id", "context_split"],
        dropna=False,
        sort=True,
    ):
        code_module, code_presentation, group_id, context_split = group_key
        group_rows.append(
            {
                "code_module": code_module,
                "code_presentation": code_presentation,
                "group_id": group_id,
                "context_split": context_split,
                "n_enrollments": int(block["enrollment_id"].nunique()),
                "n_unique_students": int(block["id_student"].nunique()),
                "n_events": int(pd.to_numeric(block["event"], errors="raise").astype(int).sum()),
                "event_rate": float(pd.to_numeric(block["event"], errors="raise").astype(int).mean()),
                "main_split_roles_present": ", ".join(sorted(block["main_split"].dropna().astype(str).unique())),
            }
        )

    table_c2_5_context_group_audit = pd.DataFrame(group_rows).sort_values(
        ["context_split", "code_module", "code_presentation"]
    ).reset_index(drop=True)
    materialize_dataframe_table(
        ctx,
        df=table_c2_5_context_group_audit,
        table_name="table_c2_5_context_group_audit",
        stage_id="2.5",
        label="C2.5 output: context group composition audit",
    )
    log_stage_end("2.5")


def stage_c3_main_ready_tables(ctx: PipelineContext) -> None:
    log_stage_start("3", "Propagate the main split onto the modeling inputs")

    sensitivity_windows = get_sensitivity_windows(ctx)
    required_tables = [
        "table_enrollment_split_assignment",
        *REQUIRED_MODELING_INPUT_TABLES,
        *(f"enrollment_cox_input_w{window_weeks}" for window_weeks in sensitivity_windows),
        *(f"enrollment_deepsurv_input_w{window_weeks}" for window_weeks in sensitivity_windows),
    ]

    require_tables(ctx.con, required_tables, stage_id="3")
    require_columns(ctx.con, "table_enrollment_split_assignment", REQUIRED_SPLIT_JOIN_COLUMNS + ["split"])
    base_to_split = {
        "pp_linear_hazard_input": "pp_linear_hazard_input_split",
        "pp_neural_hazard_input": "pp_neural_hazard_input_split",
        "enrollment_cox_input_configured": "enrollment_cox_input_configured_split",
        "enrollment_deepsurv_input_configured": "enrollment_deepsurv_input_configured_split",
    }
    for window_weeks in sensitivity_windows:
        base_to_split[f"enrollment_cox_input_w{window_weeks}"] = f"enrollment_cox_input_w{window_weeks}_split"
        base_to_split[f"enrollment_deepsurv_input_w{window_weeks}"] = f"enrollment_deepsurv_input_w{window_weeks}_split"

    propagation_rows: list[dict[str, Any]] = []
    for base_table, split_table in base_to_split.items():
        require_columns(ctx.con, base_table, REQUIRED_SPLIT_JOIN_COLUMNS + ["enrollment_id"])
        materialize_select_table(
            ctx,
            select_sql=f"""
            SELECT
                base_table.*,
                split_table.split
            FROM {base_table} AS base_table
            LEFT JOIN table_enrollment_split_assignment AS split_table
              ON {enrollment_join_condition('base_table', 'split_table')}
            """,
            table_name=split_table,
            stage_id="3",
            label=f"C3 output: {split_table}",
        )

        rows_before = int(ctx.con.execute(f"SELECT COUNT(*) FROM {base_table}").fetchone()[0])
        rows_after = int(ctx.con.execute(f"SELECT COUNT(*) FROM {split_table}").fetchone()[0])
        distinct_before = int(
            ctx.con.execute(f"SELECT COUNT(DISTINCT enrollment_id) FROM {base_table}").fetchone()[0]
        )
        distinct_after = int(
            ctx.con.execute(f"SELECT COUNT(DISTINCT enrollment_id) FROM {split_table}").fetchone()[0]
        )
        missing_split_count = int(
            ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE split IS NULL").fetchone()[0]
        )
        train_rows = int(ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE split = 'train'").fetchone()[0])
        test_rows = int(ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE split = 'test'").fetchone()[0])
        propagation_ok = rows_before == rows_after and distinct_before == distinct_after and missing_split_count == 0
        propagation_rows.append(
            {
                "base_table": base_table,
                "split_table": split_table,
                "n_rows_before": rows_before,
                "n_rows_after": rows_after,
                "n_distinct_enrollments_before": distinct_before,
                "n_distinct_enrollments_after": distinct_after,
                "n_train_rows": train_rows,
                "n_test_rows": test_rows,
                "n_missing_split": missing_split_count,
                "row_count_preserved": bool(rows_before == rows_after),
                "enrollment_count_preserved": bool(distinct_before == distinct_after),
                "propagation_ok": bool(propagation_ok),
            }
        )

    table_c3_split_propagation_audit = pd.DataFrame(propagation_rows)
    if not bool(table_c3_split_propagation_audit["propagation_ok"].all()):
        raise RuntimeError("Main split propagation failed. Inspect table_c3_split_propagation_audit.")
    materialize_dataframe_table(
        ctx,
        df=table_c3_split_propagation_audit,
        table_name="table_c3_split_propagation_audit",
        stage_id="3",
        label="C3 output: main split propagation audit",
    )

    ready_specs = [
        ("pp_linear_hazard_input_split", "pp_linear_hazard_ready_train", "train"),
        ("pp_linear_hazard_input_split", "pp_linear_hazard_ready_test", "test"),
        ("pp_neural_hazard_input_split", "pp_neural_hazard_ready_train", "train"),
        ("pp_neural_hazard_input_split", "pp_neural_hazard_ready_test", "test"),
        ("enrollment_cox_input_configured_split", "enrollment_cox_ready_train", "train"),
        ("enrollment_cox_input_configured_split", "enrollment_cox_ready_test", "test"),
        ("enrollment_deepsurv_input_configured_split", "enrollment_deepsurv_ready_train", "train"),
        ("enrollment_deepsurv_input_configured_split", "enrollment_deepsurv_ready_test", "test"),
    ]
    for window_weeks in sensitivity_windows:
        ready_specs.extend(
            [
                (f"enrollment_cox_input_w{window_weeks}_split", f"enrollment_cox_ready_train_w{window_weeks}", "train"),
                (f"enrollment_cox_input_w{window_weeks}_split", f"enrollment_cox_ready_test_w{window_weeks}", "test"),
                (f"enrollment_deepsurv_input_w{window_weeks}_split", f"enrollment_deepsurv_ready_train_w{window_weeks}", "train"),
                (f"enrollment_deepsurv_input_w{window_weeks}_split", f"enrollment_deepsurv_ready_test_w{window_weeks}", "test"),
            ]
        )
    for source_table, target_table, split_value in ready_specs:
        materialize_select_table(
            ctx,
            select_sql=f"SELECT * FROM {source_table} WHERE split = '{split_value}'",
            table_name=target_table,
            stage_id="3",
            label=f"C3 output: {target_table}",
        )

    ready_tables = [target_table for _, target_table, _ in ready_specs]
    table_c3_ready_dataset_audit = build_ready_table_audit(ctx, ready_tables=ready_tables, split_column="split")
    materialize_dataframe_table(
        ctx,
        df=table_c3_ready_dataset_audit,
        table_name="table_c3_ready_dataset_audit",
        stage_id="3",
        label="C3 output: main ready dataset audit",
    )
    log_stage_end("3")


def stage_c3_1_context_ready_tables(ctx: PipelineContext) -> None:
    log_stage_start("3.1", "Propagate the context-held-out split onto the modeling inputs")

    require_tables(
        ctx.con,
        ["table_context_enrollment_split_assignment", *REQUIRED_MODELING_INPUT_TABLES],
        stage_id="3.1",
    )
    require_columns(
        ctx.con,
        "table_context_enrollment_split_assignment",
        REQUIRED_SPLIT_JOIN_COLUMNS + ["split", "context_split_name"],
    )
    base_to_context_split = {
        "pp_linear_hazard_input": "pp_linear_hazard_input_context_split",
        "pp_neural_hazard_input": "pp_neural_hazard_input_context_split",
        "enrollment_cox_input_configured": "enrollment_cox_input_configured_context_split",
        "enrollment_deepsurv_input_configured": "enrollment_deepsurv_input_configured_context_split",
    }

    propagation_rows: list[dict[str, Any]] = []
    for base_table, split_table in base_to_context_split.items():
        require_columns(ctx.con, base_table, REQUIRED_SPLIT_JOIN_COLUMNS + ["enrollment_id"])
        materialize_select_table(
            ctx,
            select_sql=f"""
            SELECT
                base_table.*,
                split_table.split AS context_split,
                split_table.context_split_name
            FROM {base_table} AS base_table
            LEFT JOIN table_context_enrollment_split_assignment AS split_table
              ON {enrollment_join_condition('base_table', 'split_table')}
            """,
            table_name=split_table,
            stage_id="3.1",
            label=f"C3.1 output: {split_table}",
        )

        rows_before = int(ctx.con.execute(f"SELECT COUNT(*) FROM {base_table}").fetchone()[0])
        rows_after = int(ctx.con.execute(f"SELECT COUNT(*) FROM {split_table}").fetchone()[0])
        distinct_before = int(
            ctx.con.execute(f"SELECT COUNT(DISTINCT enrollment_id) FROM {base_table}").fetchone()[0]
        )
        distinct_after = int(
            ctx.con.execute(f"SELECT COUNT(DISTINCT enrollment_id) FROM {split_table}").fetchone()[0]
        )
        missing_context_split_count = int(
            ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE context_split IS NULL").fetchone()[0]
        )
        train_rows = int(
            ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE context_split = 'train'").fetchone()[0]
        )
        test_rows = int(
            ctx.con.execute(f"SELECT COUNT(*) FROM {split_table} WHERE context_split = 'test'").fetchone()[0]
        )
        propagation_ok = (
            rows_before == rows_after and distinct_before == distinct_after and missing_context_split_count == 0
        )
        propagation_rows.append(
            {
                "base_table": base_table,
                "split_table": split_table,
                "n_rows_before": rows_before,
                "n_rows_after": rows_after,
                "n_distinct_enrollments_before": distinct_before,
                "n_distinct_enrollments_after": distinct_after,
                "n_train_rows": train_rows,
                "n_test_rows": test_rows,
                "n_missing_context_split": missing_context_split_count,
                "row_count_preserved": bool(rows_before == rows_after),
                "enrollment_count_preserved": bool(distinct_before == distinct_after),
                "propagation_ok": bool(propagation_ok),
            }
        )

    table_c3_1_context_split_propagation_audit = pd.DataFrame(propagation_rows)
    if not bool(table_c3_1_context_split_propagation_audit["propagation_ok"].all()):
        raise RuntimeError(
            "Context split propagation failed. Inspect table_c3_1_context_split_propagation_audit."
        )
    materialize_dataframe_table(
        ctx,
        df=table_c3_1_context_split_propagation_audit,
        table_name="table_c3_1_context_split_propagation_audit",
        stage_id="3.1",
        label="C3.1 output: context split propagation audit",
    )

    ready_specs = [
        ("pp_linear_hazard_input_context_split", "pp_linear_hazard_context_ready_train", "train"),
        ("pp_linear_hazard_input_context_split", "pp_linear_hazard_context_ready_test", "test"),
        ("pp_neural_hazard_input_context_split", "pp_neural_hazard_context_ready_train", "train"),
        ("pp_neural_hazard_input_context_split", "pp_neural_hazard_context_ready_test", "test"),
        (
            "enrollment_cox_input_configured_context_split",
            "enrollment_cox_context_ready_train",
            "train",
        ),
        (
            "enrollment_cox_input_configured_context_split",
            "enrollment_cox_context_ready_test",
            "test",
        ),
        (
            "enrollment_deepsurv_input_configured_context_split",
            "enrollment_deepsurv_context_ready_train",
            "train",
        ),
        (
            "enrollment_deepsurv_input_configured_context_split",
            "enrollment_deepsurv_context_ready_test",
            "test",
        ),
    ]
    for source_table, target_table, split_value in ready_specs:
        materialize_select_table(
            ctx,
            select_sql=f"SELECT * FROM {source_table} WHERE context_split = '{split_value}'",
            table_name=target_table,
            stage_id="3.1",
            label=f"C3.1 output: {target_table}",
        )

    ready_tables = [target_table for _, target_table, _ in ready_specs]
    table_c3_1_context_ready_dataset_audit = build_ready_table_audit(
        ctx,
        ready_tables=ready_tables,
        split_column="context_split",
    )
    materialize_dataframe_table(
        ctx,
        df=table_c3_1_context_ready_dataset_audit,
        table_name="table_c3_1_context_ready_dataset_audit",
        stage_id="3.1",
        label="C3.1 output: context ready dataset audit",
    )
    log_stage_end("3.1")


def main() -> None:
    from util import close_duckdb_connection

    ctx = initialize_context()
    try:
        stage_c1_event_time_audit(ctx)
        stage_c2_main_split(ctx)
        stage_c2_1_context_overlap(ctx)
        stage_c2_2_compact_audit(ctx)
        stage_c2_3_context_design(ctx)
        stage_c2_4_context_survival_split(ctx)
        stage_c2_5_context_assignment(ctx)
        stage_c3_main_ready_tables(ctx)
        stage_c3_1_context_ready_tables(ctx)
    finally:
        log_stage_start("4", "Close the DuckDB connection")
        ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
        log_stage_end("4")


if __name__ == "__main__":
    main()