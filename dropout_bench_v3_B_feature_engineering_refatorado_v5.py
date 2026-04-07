from __future__ import annotations

"""
Production feature engineering module for benchmark stage B.

Purpose:
- transform the stage A survival backbone into deterministic modeling inputs
- materialize weekly person-period features and enrollment-level comparable features
- export the canonical modeling contract consumed by downstream stages C and D

Input contract:
- benchmark_shared_config.toml
- DuckDB table enrollment_survival_ready
- DuckDB table studentVle

Output contract:
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/metadata_p10_canonical_modeling_configuration.json
- outputs_benchmark_survival/metadata/metadata_p10_1_benchmark_architecture.json
- outputs_benchmark_survival/metadata/metadata_p12_1_window_sensitivity_design.json
- outputs_benchmark_survival/metadata/metadata_p12_2_window_sensitivity_comparison.json
- DuckDB tables person_period_min, vle_weekly_features, person_period_enriched,
  enrollment_model_table, enrollment_window_features, pp_linear_hazard_input,
  pp_neural_hazard_input, enrollment_cox_input, enrollment_deepsurv_input,
  enrollment_cox_input_configured, enrollment_deepsurv_input_configured,
  enrollment_window_features_sensitivity, enrollment_cox_input_w2,
  enrollment_cox_input_w4, enrollment_cox_input_w6,
  enrollment_deepsurv_input_w2, enrollment_deepsurv_input_w4,
  enrollment_deepsurv_input_w6, and the analytical audit tables materialized below

Failure policy:
- missing DuckDB source tables or required columns raise immediately
- negative survival horizons from stage A raise immediately
- no CSV fallback, notebook fallback, or silent schema degradation is permitted
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_PREFIX = "B"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREVIEW_ROWS = 20
DEFAULT_SENSITIVITY_WINDOWS = (2, 4, 6, 8, 10)
REQUIRED_SURVIVAL_READY_COLUMNS = [
    "enrollment_id",
    "id_student",
    "code_module",
    "code_presentation",
    "event_observed",
    "t_event_week",
    "t_last_obs_week",
    "t_final_week",
    "used_zero_week_fallback_for_censoring",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
]
REQUIRED_STUDENT_VLE_COLUMNS = [
    "id_student",
    "code_module",
    "code_presentation",
    "date",
    "sum_click",
    "id_site",
]


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
    modeling_contract_toml_path: Path
    run_id: str
    run_metadata_path: Path
    benchmark_config: dict[str, Any]
    modeling_config: dict[str, Any]
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


def print_artifact(label: str, location: Any) -> None:
    print(f"ARTIFACT | {label} | {location}")


def print_dataframe_audit(label: str, name: str, df: pd.DataFrame, preview_rows: int = PREVIEW_ROWS) -> None:
    preview = df.head(preview_rows)
    print(f"TABLE_LABEL | {label}")
    print(f"TABLE_NAME | {name}")
    print(f"TABLE_ROWS | {df.shape[0]}")
    print(f"TABLE_COLS | {df.shape[1]}")
    print("TABLE_COLUMNS | " + ", ".join(str(column) for column in df.columns))
    if preview.empty:
        print("TABLE_PREVIEW | [empty table]")
    else:
        print(preview.to_string(index=False))


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def available_tables(con: Any) -> set[str]:
    return set(con.execute("SHOW TABLES").fetchdf()["name"].astype(str).tolist())


def get_table_columns(con: Any, table_name: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]


def require_columns(con: Any, table_name: str, required_columns: list[str]) -> None:
    actual_columns = set(get_table_columns(con, table_name))
    missing_columns = sorted(set(required_columns) - actual_columns)
    if missing_columns:
        raise KeyError(f"{table_name} is missing required columns: {', '.join(missing_columns)}")


def duplicated_names(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def build_window_select_fragment(extra_columns: list[str], alias: str) -> str:
    if not extra_columns:
        return ""
    return ",\n    " + ",\n    ".join(f"{alias}.{column}" for column in extra_columns)


def append_benchmark_architecture_row(
    rows: list[dict[str, Any]],
    benchmark_arm: str,
    arm_role: str,
    family: str,
    model_label: str,
    status_in_project: str,
    current_or_future: str,
    representation_type: str,
    update_regime: str,
    expected_input_table: str,
    expected_split_protocol: str,
    comparability_scope: str,
    included_in_current_main_results: bool,
    notes: str,
) -> None:
    rows.append(
        {
            "benchmark_arm": benchmark_arm,
            "arm_role": arm_role,
            "family": family,
            "model_label": model_label,
            "status_in_project": status_in_project,
            "current_or_future": current_or_future,
            "representation_type": representation_type,
            "update_regime": update_regime,
            "expected_input_table": expected_input_table,
            "expected_split_protocol": expected_split_protocol,
            "comparability_scope": comparability_scope,
            "included_in_current_main_results": included_in_current_main_results,
            "notes": notes,
        }
    )


def register_existing_table(ctx: PipelineContext, table_name: str, stage_id: str, label: str) -> None:
    from util import print_duckdb_table, refresh_pipeline_catalog_schema_view, register_duckdb_table

    register_duckdb_table(
        con=ctx.con,
        table_name=table_name,
        notebook_name=ctx.script_name,
        cell_name=stage_id,
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_duckdb_table(ctx.con, table_name, title=label, limit=PREVIEW_ROWS)
    print_artifact(label, f"duckdb://{table_name}")


def materialize_dataframe_table(
    ctx: PipelineContext,
    df: pd.DataFrame,
    table_name: str,
    stage_id: str,
    label: str,
) -> None:
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_b_materialize_df__"
    ctx.con.execute(f"DROP TABLE IF EXISTS {table_name}")
    ctx.con.register(temp_view_name, df)
    ctx.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_view_name}")
    ctx.con.unregister(temp_view_name)
    register_duckdb_table(
        con=ctx.con,
        table_name=table_name,
        notebook_name=ctx.script_name,
        cell_name=stage_id,
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_dataframe_audit(label=label, name=table_name, df=df)
    print_artifact(label, f"duckdb://{table_name}")


def build_window_variant(
    ctx: PipelineContext,
    base_table: str,
    out_table: str,
    window_weeks: int,
    feature_table: str = "enrollment_window_features_sensitivity",
) -> dict[str, Any]:
    base_columns = get_table_columns(ctx.con, base_table)
    target_columns = [
        f"clicks_first_{window_weeks}_weeks",
        f"active_weeks_first_{window_weeks}",
        f"mean_clicks_first_{window_weeks}_weeks",
    ]
    extra_columns = [column for column in target_columns if column not in base_columns]
    fragment = build_window_select_fragment(extra_columns, alias="ewf")

    ctx.con.execute(f"DROP TABLE IF EXISTS {out_table}")
    ctx.con.execute(
        f"""
        CREATE TABLE {out_table} AS
        SELECT
            b.*{fragment}
        FROM {base_table} AS b
        LEFT JOIN {feature_table} AS ewf
          ON b.enrollment_id = REPLACE(ewf.enrollment_id, '|', '||')
        """
    )

    final_columns = get_table_columns(ctx.con, out_table)
    duplicates = duplicated_names(final_columns)
    if duplicates:
        raise ValueError(f"Duplicated columns detected in {out_table}: {duplicates}")

    return {
        "target_table": out_table,
        "base_table": base_table,
        "window_weeks": int(window_weeks),
        "n_base_columns": int(len(base_columns)),
        "n_target_window_columns": int(len(target_columns)),
        "n_window_columns_attached": int(len(extra_columns)),
        "attached_window_columns": ", ".join(extra_columns),
        "n_window_columns_already_present": int(len(target_columns) - len(extra_columns)),
    }


def bootstrap_context() -> PipelineContext:
    from util import ensure_pipeline_catalog, ensure_run_metadata, load_shared_config, open_duckdb_connection, resolve_project_path

    log_stage_start("0", "Runtime Bootstrap")

    config, config_toml_path = load_shared_config(PROJECT_ROOT)
    paths_cfg = config.get("paths", {})
    benchmark_cfg = config.get("benchmark", {})

    output_dir = resolve_project_path(PROJECT_ROOT, paths_cfg.get("output_dir", "outputs_benchmark_survival"))
    tables_dir = output_dir / paths_cfg.get("tables_subdir", "tables")
    metadata_dir = output_dir / paths_cfg.get("metadata_subdir", "metadata")
    duckdb_path = output_dir / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")
    modeling_contract_toml_path = PROJECT_ROOT / "benchmark_modeling_contract.toml"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_metadata, run_metadata_path = ensure_run_metadata(metadata_dir=metadata_dir, notebook_name=SCRIPT_NAME)
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
        modeling_contract_toml_path=modeling_contract_toml_path,
        run_id=str(run_metadata["run_id"]),
        run_metadata_path=run_metadata_path,
        benchmark_config=benchmark_cfg,
        modeling_config={},
        con=con,
    )

    print_artifact("shared_config_toml", config_toml_path)
    print_artifact("duckdb_database", duckdb_path)
    print_artifact("run_metadata_json", run_metadata_path)
    log_stage_end("0")
    return ctx


def stage_b1_validate_source_contract(ctx: PipelineContext) -> None:
    log_stage_start("1", "Validate DuckDB Source Contract")

    present_tables = available_tables(ctx.con)
    required_tables = {"enrollment_survival_ready", "studentVle"}
    missing_tables = sorted(required_tables - present_tables)
    if missing_tables:
        raise RuntimeError(
            "Stage B requires DuckDB source tables materialized by stage A. Missing: "
            + ", ".join(missing_tables)
        )

    require_columns(ctx.con, "enrollment_survival_ready", REQUIRED_SURVIVAL_READY_COLUMNS)
    require_columns(ctx.con, "studentVle", REQUIRED_STUDENT_VLE_COLUMNS)

    source_audit = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT enrollment_id) AS n_distinct_enrollments,
            SUM(CASE WHEN t_final_week < 0 THEN 1 ELSE 0 END) AS n_negative_t_final_week,
            SUM(CASE WHEN t_last_obs_week < 0 THEN 1 ELSE 0 END) AS n_negative_t_last_obs_week,
            SUM(CASE WHEN used_zero_week_fallback_for_censoring <> 0 THEN 1 ELSE 0 END) AS n_nonzero_zero_week_fallback,
            SUM(CASE WHEN event_observed = 1 AND t_event_week IS NULL THEN 1 ELSE 0 END) AS n_event_rows_missing_t_event_week,
            SUM(CASE WHEN event_observed = 0 AND t_event_week IS NOT NULL THEN 1 ELSE 0 END) AS n_censored_rows_with_t_event_week
        FROM enrollment_survival_ready
        """
    ).fetchdf()
    materialize_dataframe_table(
        ctx,
        source_audit,
        "table_b_source_contract_audit",
        "B1",
        "B1 audit output",
    )

    audit_row = source_audit.iloc[0]
    if int(audit_row["n_rows"]) != int(audit_row["n_distinct_enrollments"]):
        raise ValueError("enrollment_survival_ready is not unique by enrollment_id.")
    if int(audit_row["n_negative_t_final_week"]) != 0:
        raise ValueError("Stage A contract violated: t_final_week contains negative values.")
    if int(audit_row["n_negative_t_last_obs_week"]) != 0:
        raise ValueError("Stage A contract violated: t_last_obs_week contains negative values.")
    if int(audit_row["n_nonzero_zero_week_fallback"]) != 0:
        raise ValueError("Stage A contract violated: zero-week fallback flag must remain zero.")
    if int(audit_row["n_event_rows_missing_t_event_week"]) != 0:
        raise ValueError("Stage A contract violated: observed events are missing t_event_week.")
    if int(audit_row["n_censored_rows_with_t_event_week"]) != 0:
        raise ValueError("Stage A contract violated: censored rows unexpectedly contain t_event_week.")

    print("Validated DuckDB source tables: enrollment_survival_ready, studentVle")
    log_stage_end("1")


def stage_b2_build_person_period_min(ctx: PipelineContext) -> None:
    log_stage_start("2", "Build Minimal Person-Period Skeleton")

    ctx.con.execute("DROP TABLE IF EXISTS person_period_min")
    ctx.con.execute(
        """
        CREATE TABLE person_period_min AS
        WITH expanded AS (
            SELECT
                esr.id_student,
                esr.code_module,
                esr.code_presentation,
                esr.gender,
                esr.region,
                esr.highest_education,
                esr.imd_band,
                esr.age_band,
                esr.num_of_prev_attempts,
                esr.studied_credits,
                esr.disability,
                esr.final_result,
                esr.date_registration,
                esr.date_unregistration,
                esr.is_withdrawn,
                esr.has_valid_unregistration_date,
                esr.event_observed,
                esr.is_withdrawn_without_valid_unregistration,
                esr.has_any_vle_activity,
                esr.max_vle_day,
                esr.n_vle_rows,
                esr.total_clicks_all_time,
                esr.t_event_week,
                esr.t_last_obs_week,
                esr.t_final_week,
                esr.used_zero_week_fallback_for_censoring,
                gs.week
            FROM enrollment_survival_ready AS esr
            CROSS JOIN LATERAL generate_series(0, esr.t_final_week) AS gs(week)
        )
        SELECT
            *,
            CAST(id_student AS VARCHAR) || '|' || code_module || '|' || code_presentation AS enrollment_id,
            CASE
                WHEN event_observed = 1 AND week = t_event_week THEN 1
                ELSE 0
            END AS event_t
        FROM expanded
        """
    )

    register_existing_table(ctx, "person_period_min", "B2", "B2 main output")

    audit_summary = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_person_period_rows,
            COUNT(DISTINCT enrollment_id) AS n_distinct_enrollments,
            MIN(week) AS min_week,
            MAX(week) AS max_week,
            MAX(t_final_week) AS max_t_final_week,
            MIN(t_final_week) AS min_t_final_week,
            SUM(event_t) AS n_total_event_rows,
            SUM(CASE WHEN used_zero_week_fallback_for_censoring = 1 THEN 1 ELSE 0 END) AS n_rows_from_zero_week_fallback,
            COUNT(DISTINCT CASE WHEN used_zero_week_fallback_for_censoring = 1 THEN enrollment_id END) AS n_enrollments_from_zero_week_fallback,
            COUNT(DISTINCT CASE WHEN t_final_week < 0 THEN enrollment_id END) AS n_enrollments_with_negative_t_final_week
        FROM person_period_min
        """
    ).fetchdf()
    event_check = ctx.con.execute(
        """
        WITH per_enrollment AS (
            SELECT
                enrollment_id,
                MAX(event_observed) AS event_observed,
                SUM(event_t) AS n_event_rows
            FROM person_period_min
            GROUP BY 1
        )
        SELECT
            event_observed,
            n_event_rows,
            COUNT(*) AS n_enrollments
        FROM per_enrollment
        GROUP BY event_observed, n_event_rows
        ORDER BY event_observed, n_event_rows
        """
    ).fetchdf()
    week_distribution = ctx.con.execute(
        """
        SELECT
            week,
            COUNT(*) AS n_rows,
            SUM(event_t) AS n_events
        FROM person_period_min
        GROUP BY week
        ORDER BY week
        """
    ).fetchdf()
    coverage_check = ctx.con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM enrollment_survival_ready) AS n_enrollments_survival_ready,
            (SELECT COUNT(DISTINCT enrollment_id) FROM person_period_min) AS n_enrollments_person_period
        """
    ).fetchdf()
    coverage_check["n_missing_enrollments_after_expansion"] = (
        coverage_check["n_enrollments_survival_ready"] - coverage_check["n_enrollments_person_period"]
    )

    materialize_dataframe_table(ctx, audit_summary, "table_person_period_min_audit", "B2", "B2 audit output")
    materialize_dataframe_table(ctx, event_check, "table_person_period_event_check", "B2", "B2 audit output")
    materialize_dataframe_table(ctx, week_distribution, "table_person_period_week_distribution", "B2", "B2 analytical output")
    materialize_dataframe_table(ctx, coverage_check, "table_person_period_coverage_check", "B2", "B2 audit output")

    audit_row = audit_summary.iloc[0]
    if int(audit_row["n_rows_from_zero_week_fallback"]) != 0:
        raise ValueError("B2 detected unexpected zero-week fallback rows.")
    if int(coverage_check.iloc[0]["n_missing_enrollments_after_expansion"]) != 0:
        raise ValueError("B2 failed to preserve all enrollments during person-period expansion.")

    log_stage_end("2")


def stage_b3_build_weekly_vle_features(ctx: PipelineContext) -> None:
    log_stage_start("3", "Build Weekly VLE Feature Layer")

    ctx.con.execute("DROP TABLE IF EXISTS vle_weekly_features")
    ctx.con.execute(
        """
        CREATE TABLE vle_weekly_features AS
        SELECT
            id_student,
            code_module,
            code_presentation,
            CAST(FLOOR(date / 7.0) AS INTEGER) AS week,
            SUM(sum_click) AS total_clicks_week,
            COUNT(*) AS n_vle_rows_week,
            1 AS active_this_week,
            COUNT(DISTINCT id_site) AS n_distinct_sites_week
        FROM studentVle
        GROUP BY
            id_student,
            code_module,
            code_presentation,
            CAST(FLOOR(date / 7.0) AS INTEGER)
        """
    )

    register_existing_table(ctx, "vle_weekly_features", "B3", "B3 main output")

    audit_summary = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_enrollment_week_rows,
            COUNT(DISTINCT CAST(id_student AS VARCHAR) || '|' || code_module || '|' || code_presentation) AS n_distinct_enrollments,
            MIN(week) AS min_week,
            MAX(week) AS max_week,
            SUM(CASE WHEN week < 0 THEN 1 ELSE 0 END) AS n_rows_negative_weeks,
            SUM(CASE WHEN total_clicks_week = 0 THEN 1 ELSE 0 END) AS n_rows_zero_clicks,
            MIN(total_clicks_week) AS min_total_clicks_week,
            MAX(total_clicks_week) AS max_total_clicks_week
        FROM vle_weekly_features
        """
    ).fetchdf()
    week_distribution = ctx.con.execute(
        """
        SELECT
            week,
            COUNT(*) AS n_rows,
            SUM(total_clicks_week) AS total_clicks_sum,
            AVG(total_clicks_week) AS avg_total_clicks_week
        FROM vle_weekly_features
        GROUP BY week
        ORDER BY week
        """
    ).fetchdf()
    coverage_check = ctx.con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM enrollment_survival_ready) AS n_enrollments_survival_ready,
            (SELECT COUNT(DISTINCT CAST(id_student AS VARCHAR) || '|' || code_module || '|' || code_presentation)
             FROM vle_weekly_features) AS n_enrollments_with_weekly_vle
        """
    ).fetchdf()
    coverage_check["n_enrollments_without_weekly_vle"] = (
        coverage_check["n_enrollments_survival_ready"] - coverage_check["n_enrollments_with_weekly_vle"]
    )

    materialize_dataframe_table(ctx, audit_summary, "table_vle_weekly_features_audit", "B3", "B3 audit output")
    materialize_dataframe_table(ctx, week_distribution, "table_vle_weekly_features_week_distribution", "B3", "B3 analytical output")
    materialize_dataframe_table(ctx, coverage_check, "table_vle_weekly_features_coverage_check", "B3", "B3 audit output")

    log_stage_end("3")


def stage_b4_build_person_period_enriched(ctx: PipelineContext) -> None:
    log_stage_start("4", "Enrich Person-Period with Temporal Features")

    ctx.con.execute("DROP TABLE IF EXISTS person_period_enriched")
    ctx.con.execute(
        """
        CREATE TABLE person_period_enriched AS
        WITH nonnegative_vle AS (
            SELECT *
            FROM vle_weekly_features
            WHERE week >= 0
        ),
        joined AS (
            SELECT
                ppm.*,
                COALESCE(vwf.total_clicks_week, 0) AS total_clicks_week,
                COALESCE(vwf.n_vle_rows_week, 0) AS n_vle_rows_week,
                COALESCE(vwf.active_this_week, 0) AS active_this_week,
                COALESCE(vwf.n_distinct_sites_week, 0) AS n_distinct_sites_week
            FROM person_period_min AS ppm
            LEFT JOIN nonnegative_vle AS vwf
              ON ppm.id_student = vwf.id_student
             AND ppm.code_module = vwf.code_module
             AND ppm.code_presentation = vwf.code_presentation
             AND ppm.week = vwf.week
        ),
        with_running_totals AS (
            SELECT
                *,
                SUM(total_clicks_week) OVER (
                    PARTITION BY enrollment_id
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_clicks_until_t,
                SUM(active_this_week) OVER (
                    PARTITION BY enrollment_id
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_active_weeks_until_t,
                MAX(CASE WHEN active_this_week = 1 THEN week ELSE NULL END) OVER (
                    PARTITION BY enrollment_id
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS last_active_week_so_far
            FROM joined
        ),
        with_recency AS (
            SELECT
                *,
                CASE
                    WHEN active_this_week = 1 THEN 0
                    WHEN last_active_week_so_far IS NOT NULL THEN week - last_active_week_so_far
                    ELSE week + 1
                END AS recency
            FROM with_running_totals
        ),
        with_groups AS (
            SELECT
                *,
                SUM(CASE WHEN active_this_week = 0 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY enrollment_id
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS inactivity_group
            FROM with_recency
        )
        SELECT
            *,
            CASE
                WHEN active_this_week = 1 THEN
                    ROW_NUMBER() OVER (
                        PARTITION BY enrollment_id, inactivity_group
                        ORDER BY week
                    )
                ELSE 0
            END AS streak
        FROM with_groups
        """
    )

    register_existing_table(ctx, "person_period_enriched", "B4", "B4 main output")

    audit_summary = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_person_period_rows,
            COUNT(DISTINCT enrollment_id) AS n_distinct_enrollments,
            MAX(week) AS max_week,
            AVG(total_clicks_week) AS avg_total_clicks_week,
            AVG(cum_clicks_until_t) AS avg_cum_clicks_until_t,
            AVG(active_this_week) AS avg_active_this_week,
            MAX(recency) AS max_recency,
            MAX(streak) AS max_streak
        FROM person_period_enriched
        """
    ).fetchdf()
    missing_check = ctx.con.execute(
        """
        SELECT
            SUM(CASE WHEN total_clicks_week IS NULL THEN 1 ELSE 0 END) AS n_missing_total_clicks_week,
            SUM(CASE WHEN n_vle_rows_week IS NULL THEN 1 ELSE 0 END) AS n_missing_n_vle_rows_week,
            SUM(CASE WHEN active_this_week IS NULL THEN 1 ELSE 0 END) AS n_missing_active_this_week,
            SUM(CASE WHEN n_distinct_sites_week IS NULL THEN 1 ELSE 0 END) AS n_missing_n_distinct_sites_week,
            SUM(CASE WHEN cum_clicks_until_t IS NULL THEN 1 ELSE 0 END) AS n_missing_cum_clicks_until_t,
            SUM(CASE WHEN recency IS NULL THEN 1 ELSE 0 END) AS n_missing_recency,
            SUM(CASE WHEN streak IS NULL THEN 1 ELSE 0 END) AS n_missing_streak
        FROM person_period_enriched
        """
    ).fetchdf()
    sample_rows = ctx.con.execute(
        """
        SELECT
            enrollment_id,
            week,
            event_t,
            total_clicks_week,
            active_this_week,
            cum_clicks_until_t,
            recency,
            streak
        FROM person_period_enriched
        ORDER BY enrollment_id, week
        LIMIT 30
        """
    ).fetchdf()

    materialize_dataframe_table(ctx, audit_summary, "table_person_period_enriched_audit", "B4", "B4 audit output")
    materialize_dataframe_table(ctx, missing_check, "table_person_period_enriched_missing_check", "B4", "B4 audit output")
    materialize_dataframe_table(ctx, sample_rows, "table_person_period_enriched_sample", "B4", "B4 analytical output")

    log_stage_end("4")


def stage_b5_build_enrollment_model_table(ctx: PipelineContext) -> None:
    log_stage_start("5", "Build Enrollment-Level Model Table")

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_model_table")
    ctx.con.execute(
        """
        CREATE TABLE enrollment_model_table AS
        SELECT
            enrollment_id,
            id_student,
            code_module,
            code_presentation,
            event_observed AS event,
            t_last_obs_week AS duration_raw,
            t_last_obs_week AS duration,
            used_zero_week_fallback_for_censoring,
            gender,
            region,
            highest_education,
            imd_band,
            age_band,
            num_of_prev_attempts,
            studied_credits,
            disability
        FROM enrollment_survival_ready
        """
    )

    register_existing_table(ctx, "enrollment_model_table", "B5", "B5 main output")

    summary_df = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT enrollment_id) AS n_distinct_enrollments,
            MIN(duration_raw) AS min_duration_raw,
            MIN(duration) AS min_duration,
            AVG(duration) AS avg_duration,
            MAX(duration) AS max_duration,
            SUM(event) AS n_events,
            100.0 * AVG(event) AS pct_events,
            SUM(CASE WHEN duration_raw < 0 THEN 1 ELSE 0 END) AS n_negative_duration_raw,
            CASE WHEN COUNT(*) = COUNT(DISTINCT enrollment_id) THEN TRUE ELSE FALSE END AS is_unique_by_enrollment_id
        FROM enrollment_model_table
        """
    ).fetchdf()
    event_distribution_df = ctx.con.execute(
        """
        SELECT
            event,
            COUNT(*) AS n
        FROM enrollment_model_table
        GROUP BY event
        ORDER BY event
        """
    ).fetchdf()
    missing_check_df = ctx.con.execute(
        """
        SELECT
            SUM(CASE WHEN duration_raw IS NULL THEN 1 ELSE 0 END) AS n_missing_duration_raw,
            SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS n_missing_duration,
            SUM(CASE WHEN event IS NULL THEN 1 ELSE 0 END) AS n_missing_event,
            SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) AS n_missing_gender,
            SUM(CASE WHEN region IS NULL THEN 1 ELSE 0 END) AS n_missing_region,
            SUM(CASE WHEN highest_education IS NULL THEN 1 ELSE 0 END) AS n_missing_highest_education,
            SUM(CASE WHEN imd_band IS NULL THEN 1 ELSE 0 END) AS n_missing_imd_band,
            SUM(CASE WHEN age_band IS NULL THEN 1 ELSE 0 END) AS n_missing_age_band,
            SUM(CASE WHEN num_of_prev_attempts IS NULL THEN 1 ELSE 0 END) AS n_missing_num_of_prev_attempts,
            SUM(CASE WHEN studied_credits IS NULL THEN 1 ELSE 0 END) AS n_missing_studied_credits,
            SUM(CASE WHEN disability IS NULL THEN 1 ELSE 0 END) AS n_missing_disability,
            SUM(CASE WHEN used_zero_week_fallback_for_censoring IS NULL THEN 1 ELSE 0 END) AS n_missing_zero_week_fallback_flag
        FROM enrollment_model_table
        """
    ).fetchdf()
    sample_df = ctx.con.execute(
        """
        SELECT *
        FROM enrollment_model_table
        ORDER BY enrollment_id
        LIMIT 30
        """
    ).fetchdf()

    materialize_dataframe_table(ctx, summary_df, "table_enrollment_model_table_audit", "B5", "B5 audit output")
    materialize_dataframe_table(ctx, missing_check_df, "table_enrollment_model_table_missing_check", "B5", "B5 audit output")
    materialize_dataframe_table(ctx, event_distribution_df, "table_enrollment_model_table_event_distribution", "B5", "B5 analytical output")
    materialize_dataframe_table(ctx, sample_df, "table_enrollment_model_table_sample", "B5", "B5 analytical output")

    summary_row = summary_df.iloc[0]
    if int(summary_row["n_negative_duration_raw"]) != 0:
        raise ValueError("B5 detected negative durations after stage A validation.")
    if not bool(summary_row["is_unique_by_enrollment_id"]):
        raise ValueError("B5 produced duplicate enrollment_id rows.")

    log_stage_end("5")


def stage_b6_define_modeling_configuration(ctx: PipelineContext) -> None:
    log_stage_start("6", "Define Canonical Modeling Configuration Layer")

    from dropout_bench_v3_D_00_common import resolve_early_window_sensitivity_weeks

    benchmark_cfg = ctx.benchmark_config
    random_seed = int(benchmark_cfg.get("seed", 42))
    test_size = float(benchmark_cfg.get("test_size", 0.30))
    early_window_weeks = int(benchmark_cfg.get("early_window_weeks", 4))
    main_enrollment_window_weeks = int(benchmark_cfg.get("main_enrollment_window_weeks", early_window_weeks))
    early_window_sensitivity_weeks = resolve_early_window_sensitivity_weeks(benchmark_cfg)

    main_clicks_feature = f"clicks_first_{main_enrollment_window_weeks}_weeks"
    main_active_feature = f"active_weeks_first_{main_enrollment_window_weeks}"
    main_mean_clicks_feature = f"mean_clicks_first_{main_enrollment_window_weeks}_weeks"

    static_features = ["studied_credits", "num_of_prev_attempts"]
    temporal_features_discrete = ["week", "total_clicks"]
    main_enrollment_window_features = ["studied_credits", "num_of_prev_attempts", main_clicks_feature]
    optional_comparable_window_features = [main_active_feature, main_mean_clicks_feature]
    dynamic_arm_features_linear = ["week", "total_clicks", "studied_credits", "num_of_prev_attempts"]
    dynamic_arm_features_neural = ["week", "total_clicks", "studied_credits", "num_of_prev_attempts"]

    benchmark_arms = {
        "comparable_arm": {
            "description": "Enrollment-level or early-window comparable benchmark arm.",
            "representation_type": "enrollment_level_or_early_window_summary",
            "update_regime": "static_after_early_window",
            "families": ["continuous_time", "continuous_time_neural", "discrete_time_comparable_minimal"],
        },
        "dynamic_arm": {
            "description": "Weekly-updating person-period benchmark arm.",
            "representation_type": "dynamic_weekly_person_period",
            "update_regime": "weekly_updating",
            "families": ["discrete_time", "discrete_time_neural"],
        },
    }
    evaluation_overlays = {
        "main_split": {
            "description": "Main benchmark split used by the paper-aligned benchmark.",
            "table_or_protocol": "existing train/test ready tables",
            "currently_active": True,
        },
        "context_heldout_v1": {
            "description": "Alternative grouped context-held-out split for transportability analysis.",
            "table_or_protocol": "enrollment_survival_ready_context_split",
            "currently_active": False,
        },
    }
    model_input_variants = {
        "continuous_time": {
            "benchmark_arm": "comparable_arm",
            "family": "continuous_time",
            "representation_type": "enrollment_level_or_early_window_summary",
            "update_regime": "static_after_early_window",
            "input_table_train": "enrollment_cox_ready_train",
            "input_table_test": "enrollment_cox_ready_test",
            "feature_set": main_enrollment_window_features,
            "optional_feature_set": optional_comparable_window_features,
            "currently_materialized": True,
            "included_in_main_results": True,
        },
        "continuous_time_neural": {
            "benchmark_arm": "comparable_arm",
            "family": "continuous_time_neural",
            "representation_type": "enrollment_level_or_early_window_summary",
            "update_regime": "static_after_early_window",
            "input_table_train": "enrollment_deepsurv_ready_train",
            "input_table_test": "enrollment_deepsurv_ready_test",
            "feature_set": main_enrollment_window_features,
            "optional_feature_set": optional_comparable_window_features,
            "currently_materialized": True,
            "included_in_main_results": True,
        },
        "discrete_time_comparable_minimal": {
            "benchmark_arm": "comparable_arm",
            "family": "discrete_time_comparable_minimal",
            "representation_type": "enrollment_level_or_early_window_summary",
            "update_regime": "static_after_early_window",
            "input_table_train": "enrollment_discrete_time_comparable_ready_train",
            "input_table_test": "enrollment_discrete_time_comparable_ready_test",
            "feature_set": main_enrollment_window_features,
            "optional_feature_set": optional_comparable_window_features,
            "currently_materialized": False,
            "included_in_main_results": False,
        },
        "discrete_time": {
            "benchmark_arm": "dynamic_arm",
            "family": "discrete_time",
            "representation_type": "dynamic_weekly_person_period",
            "update_regime": "weekly_updating",
            "input_table_train": "pp_linear_hazard_ready_train",
            "input_table_test": "pp_linear_hazard_ready_test",
            "feature_set": dynamic_arm_features_linear,
            "optional_feature_set": [],
            "currently_materialized": True,
            "included_in_main_results": True,
        },
        "discrete_time_neural": {
            "benchmark_arm": "dynamic_arm",
            "family": "discrete_time_neural",
            "representation_type": "dynamic_weekly_person_period",
            "update_regime": "weekly_updating",
            "input_table_train": "pp_neural_hazard_ready_train",
            "input_table_test": "pp_neural_hazard_ready_test",
            "feature_set": dynamic_arm_features_neural,
            "optional_feature_set": [],
            "currently_materialized": True,
            "included_in_main_results": True,
        },
    }

    ctx.modeling_config = {
        "random_seed": random_seed,
        "test_size": test_size,
        "early_window_weeks": early_window_weeks,
        "main_enrollment_window_weeks": main_enrollment_window_weeks,
        "main_clicks_feature": main_clicks_feature,
        "main_active_feature": main_active_feature,
        "main_mean_clicks_feature": main_mean_clicks_feature,
        "static_features": static_features,
        "temporal_features_discrete": temporal_features_discrete,
        "main_enrollment_window_features": main_enrollment_window_features,
        "optional_comparable_window_features": optional_comparable_window_features,
        "dynamic_arm_features_linear": dynamic_arm_features_linear,
        "dynamic_arm_features_neural": dynamic_arm_features_neural,
        "benchmark_arms": benchmark_arms,
        "evaluation_overlays": evaluation_overlays,
        "model_input_variants": model_input_variants,
    }

    configuration_rows = []
    for variant_name, variant_cfg in model_input_variants.items():
        configuration_rows.append(
            {
                "variant_name": variant_name,
                "benchmark_arm": variant_cfg["benchmark_arm"],
                "family": variant_cfg["family"],
                "representation_type": variant_cfg["representation_type"],
                "update_regime": variant_cfg["update_regime"],
                "input_table_train": variant_cfg["input_table_train"],
                "input_table_test": variant_cfg["input_table_test"],
                "feature_set": " | ".join(variant_cfg["feature_set"]),
                "optional_feature_set": " | ".join(variant_cfg["optional_feature_set"]),
                "currently_materialized": bool(variant_cfg["currently_materialized"]),
                "included_in_main_results": bool(variant_cfg["included_in_main_results"]),
            }
        )
    table_p10_canonical_modeling_configuration = (
        pd.DataFrame(configuration_rows)
        .sort_values(["benchmark_arm", "variant_name"])
        .reset_index(drop=True)
    )

    feature_contract_rows = []
    for feature_name in static_features:
        feature_contract_rows.append(
            {
                "feature_name": feature_name,
                "feature_group": "static_features",
                "applies_to_family": "continuous_time | continuous_time_neural | discrete_time_comparable_minimal | discrete_time | discrete_time_neural",
                "window_weeks": np.nan,
                "paper_locked": True,
                "notes": "Stable structural feature used across benchmark families.",
            }
        )
    for feature_name in temporal_features_discrete:
        feature_contract_rows.append(
            {
                "feature_name": feature_name,
                "feature_group": "temporal_features_discrete",
                "applies_to_family": "discrete_time | discrete_time_neural",
                "window_weeks": np.nan,
                "paper_locked": True,
                "notes": "Weekly-updating dynamic feature used in person-period arms.",
            }
        )
    for feature_name in main_enrollment_window_features:
        feature_contract_rows.append(
            {
                "feature_name": feature_name,
                "feature_group": "main_enrollment_window_features",
                "applies_to_family": "continuous_time | continuous_time_neural | discrete_time_comparable_minimal",
                "window_weeks": int(main_enrollment_window_weeks),
                "paper_locked": True,
                "notes": "Canonical early-window comparable feature.",
            }
        )
    for feature_name in optional_comparable_window_features:
        feature_contract_rows.append(
            {
                "feature_name": feature_name,
                "feature_group": "optional_comparable_window_features",
                "applies_to_family": "continuous_time | continuous_time_neural | discrete_time_comparable_minimal",
                "window_weeks": int(main_enrollment_window_weeks),
                "paper_locked": False,
                "notes": "Optional comparable feature; not required in the minimal paper-aligned set.",
            }
        )
    table_b_feature_contract = (
        pd.DataFrame(feature_contract_rows)
        .drop_duplicates()
        .sort_values(["feature_group", "feature_name"])
        .reset_index(drop=True)
    )

    table_b_modeling_contract_key_value = pd.DataFrame(
        [
            {"config_key": "random_seed", "config_value": str(random_seed), "paper_locked": True, "notes": "Random seed used across benchmark design."},
            {"config_key": "test_size", "config_value": str(test_size), "paper_locked": True, "notes": "Main train/test split proportion."},
            {"config_key": "early_window_weeks", "config_value": str(early_window_weeks), "paper_locked": True, "notes": "Canonical early window used in benchmark framing."},
            {"config_key": "main_enrollment_window_weeks", "config_value": str(main_enrollment_window_weeks), "paper_locked": True, "notes": "Canonical comparable enrollment window used by downstream stage D."},
            {"config_key": "early_window_sensitivity_weeks", "config_value": " | ".join(str(value) for value in early_window_sensitivity_weeks), "paper_locked": False, "notes": "Sensitivity grid for comparable early-window analyses."},
            {"config_key": "main_clicks_feature", "config_value": main_clicks_feature, "paper_locked": True, "notes": "Canonical clicks feature for comparable models."},
            {"config_key": "main_active_feature", "config_value": main_active_feature, "paper_locked": False, "notes": "Optional active-weeks comparable feature."},
            {"config_key": "main_mean_clicks_feature", "config_value": main_mean_clicks_feature, "paper_locked": False, "notes": "Optional mean-clicks comparable feature."},
            {"config_key": "static_features", "config_value": " | ".join(static_features), "paper_locked": True, "notes": "Static feature contract."},
            {"config_key": "temporal_features_discrete", "config_value": " | ".join(temporal_features_discrete), "paper_locked": True, "notes": "Dynamic weekly feature contract."},
            {"config_key": "main_enrollment_window_features", "config_value": " | ".join(main_enrollment_window_features), "paper_locked": True, "notes": "Comparable enrollment feature contract."},
        ]
    )

    materialize_dataframe_table(
        ctx,
        table_p10_canonical_modeling_configuration,
        "table_p10_canonical_modeling_configuration",
        "B6",
        "B6 analytical output",
    )
    materialize_dataframe_table(ctx, table_b_feature_contract, "table_b_feature_contract", "B6", "B6 analytical output")
    materialize_dataframe_table(
        ctx,
        table_b_modeling_contract_key_value,
        "table_b_modeling_contract_key_value",
        "B6",
        "B6 analytical output",
    )

    toml_lines = [
        "# Benchmark modeling contract exported by stage B.",
        "# This file contains stable modeling configuration consumed downstream.",
        "# Values marked as paper-aligned should not be changed casually because they",
        "# affect benchmark comparability with the written paper.",
        "",
        "[benchmark]",
        "# Random seed used by benchmark design. Paper-aligned and locked.",
        f"seed = {random_seed}",
        "# Main train/test split proportion. Paper-aligned and locked.",
        f"test_size = {test_size}",
        "# Canonical early-window framing in weeks. Paper-aligned and locked.",
        f"early_window_weeks = {early_window_weeks}",
        "# Canonical comparable enrollment window in weeks. Paper-aligned and locked.",
        f"main_enrollment_window_weeks = {main_enrollment_window_weeks}",
        "# Sensitivity grid for comparable early-window analyses.",
        f"early_window_sensitivity_weeks = [{', '.join(str(value) for value in early_window_sensitivity_weeks)}]",
        "",
        "[modeling]",
        "# Canonical clicks feature for comparable enrollment-level models.",
        f'main_clicks_feature = "{main_clicks_feature}"',
        "# Optional active-weeks feature for comparable models.",
        f'main_active_feature = "{main_active_feature}"',
        "# Optional mean-clicks feature for comparable models.",
        f'main_mean_clicks_feature = "{main_mean_clicks_feature}"',
        "",
        "[feature_contract]",
        "# Stable structural features used across benchmark families.",
        "static_features = [",
    ]
    for feature_name in static_features:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
        "# Weekly-updating dynamic features used in person-period models.",
        "temporal_features_discrete = [",
    ])
    for feature_name in temporal_features_discrete:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
        "# Canonical early-window comparable features. Changing these alters comparability.",
        "main_enrollment_window_features = [",
    ])
    for feature_name in main_enrollment_window_features:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
        "# Optional comparable features that are not part of the minimal paper-aligned set.",
        "optional_comparable_window_features = [",
    ])
    for feature_name in optional_comparable_window_features:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
        "# Dynamic person-period feature contract for linear discrete-time models.",
        "dynamic_arm_features_linear = [",
    ])
    for feature_name in dynamic_arm_features_linear:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
        "# Dynamic person-period feature contract for neural discrete-time models.",
        "dynamic_arm_features_neural = [",
    ])
    for feature_name in dynamic_arm_features_neural:
        toml_lines.append(f'  "{feature_name}",')
    toml_lines.extend([
        "]",
        "",
    ])
    ctx.modeling_contract_toml_path.write_text("\n".join(toml_lines), encoding="utf-8")
    print_artifact("benchmark_modeling_contract_toml", ctx.modeling_contract_toml_path)

    metadata_path = ctx.metadata_dir / "metadata_p10_canonical_modeling_configuration.json"
    metadata = {
        "step": "B6",
        "title": "Define Canonical Modeling Configuration Layer",
        "random_seed": random_seed,
        "test_size": test_size,
        "early_window_weeks": early_window_weeks,
        "main_enrollment_window_weeks": main_enrollment_window_weeks,
        "benchmark_arms": list(benchmark_arms.keys()),
        "evaluation_overlays": list(evaluation_overlays.keys()),
        "model_variants": list(model_input_variants.keys()),
        "canonical_comparable_click_feature": main_clicks_feature,
        "modeling_contract_toml": str(ctx.modeling_contract_toml_path),
        "output_tables": [
            "table_p10_canonical_modeling_configuration",
            "table_b_feature_contract",
            "table_b_modeling_contract_key_value",
        ],
    }
    save_json(metadata, metadata_path)
    print_artifact("metadata_p10_canonical_modeling_configuration_json", metadata_path)

    log_stage_end("6")


def stage_b7_define_benchmark_architecture(ctx: PipelineContext) -> None:
    log_stage_start("7", "Define Corrected Benchmark Architecture")

    present_tables = available_tables(ctx.con)
    rows: list[dict[str, Any]] = []

    append_benchmark_architecture_row(
        rows,
        benchmark_arm="comparable_arm",
        arm_role="model_family",
        family="continuous_time",
        model_label="Comparable Cox arm",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="enrollment_level_or_early_window_summary",
        update_regime="static_after_early_window",
        expected_input_table="enrollment_cox_*",
        expected_split_protocol="main_split and future context-held-out reruns",
        comparability_scope="within comparable_arm",
        included_in_current_main_results=True,
        notes="Continuous-time comparable arm on enrollment-level static and early-window inputs.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="comparable_arm",
        arm_role="model_family",
        family="continuous_time_neural",
        model_label="Comparable DeepSurv arm",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="enrollment_level_or_early_window_summary",
        update_regime="static_after_early_window",
        expected_input_table="enrollment_deepsurv_*",
        expected_split_protocol="main_split and future context-held-out reruns",
        comparability_scope="within comparable_arm",
        included_in_current_main_results=True,
        notes="Neural continuous-time comparable arm on enrollment-level static and early-window inputs.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="comparable_arm",
        arm_role="model_family",
        family="discrete_time_comparable_minimal",
        model_label="Comparable discrete-time minimal arm",
        status_in_project="to_be_materialized",
        current_or_future="future",
        representation_type="enrollment_level_or_early_window_summary",
        update_regime="static_after_early_window",
        expected_input_table="to_be_defined_from_enrollment_level_comparable_features",
        expected_split_protocol="main_split first, context-held-out optionally later",
        comparability_scope="within comparable_arm",
        included_in_current_main_results=False,
        notes="Future minimal discrete-time comparable arm aligned to the comparable representation regime.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="dynamic_arm",
        arm_role="model_family",
        family="discrete_time",
        model_label="Dynamic linear hazard arm",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="dynamic_weekly_person_period",
        update_regime="weekly_updating",
        expected_input_table="pp_linear_hazard_*",
        expected_split_protocol="main_split and future context-held-out reruns if materialized",
        comparability_scope="within dynamic_arm",
        included_in_current_main_results=True,
        notes="Dynamic discrete-time linear hazard arm on weekly person-period inputs.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="dynamic_arm",
        arm_role="model_family",
        family="discrete_time_neural",
        model_label="Dynamic neural hazard arm",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="dynamic_weekly_person_period",
        update_regime="weekly_updating",
        expected_input_table="pp_neural_hazard_*",
        expected_split_protocol="main_split and future context-held-out reruns if materialized",
        comparability_scope="within dynamic_arm",
        included_in_current_main_results=True,
        notes="Dynamic neural hazard arm on weekly person-period inputs.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="shared_backbone",
        arm_role="upstream_data",
        family="enrollment_survival_backbone",
        model_label="Enrollment survival backbone",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="enrollment_level_survival",
        update_regime="not_a_model",
        expected_input_table="enrollment_survival_ready",
        expected_split_protocol="feeds all benchmark arms",
        comparability_scope="shared_input_backbone",
        included_in_current_main_results=False,
        notes="Canonical event-time enrollment-level backbone from stage A.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="shared_backbone",
        arm_role="upstream_data",
        family="person_period_backbone",
        model_label="Person-period dynamic backbone",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="dynamic_weekly_person_period",
        update_regime="not_a_model",
        expected_input_table="person_period_* / pp_*",
        expected_split_protocol="feeds dynamic benchmark arm",
        comparability_scope="shared_input_backbone",
        included_in_current_main_results=False,
        notes="Canonical weekly person-period backbone for discrete-time models.",
    )
    append_benchmark_architecture_row(
        rows,
        benchmark_arm="evaluation_overlay",
        arm_role="evaluation_protocol",
        family="main_random_enrollment_split",
        model_label="Main benchmark split",
        status_in_project="already_present",
        current_or_future="current",
        representation_type="split_protocol",
        update_regime="not_a_model",
        expected_input_table="existing train/test ready tables",
        expected_split_protocol="main_split",
        comparability_scope="applies_across_arms",
        included_in_current_main_results=True,
        notes="Main benchmark protocol used in the paper-aligned workflow.",
    )
    if "enrollment_survival_ready_context_split" in present_tables:
        append_benchmark_architecture_row(
            rows,
            benchmark_arm="evaluation_overlay",
            arm_role="evaluation_protocol",
            family="context_heldout_split",
            model_label="Context-held-out split",
            status_in_project="already_present",
            current_or_future="current",
            representation_type="split_protocol",
            update_regime="not_a_model",
            expected_input_table="enrollment_survival_ready_context_split",
            expected_split_protocol="context_heldout_v1",
            comparability_scope="applies_across_arms",
            included_in_current_main_results=False,
            notes="Alternative transportability-oriented split protocol.",
        )

    table_architecture = pd.DataFrame(rows)
    table_arm_summary = (
        table_architecture
        .groupby(["benchmark_arm", "representation_type", "update_regime", "comparability_scope"], dropna=False)
        .size()
        .reset_index(name="n_entries")
        .sort_values(["benchmark_arm", "representation_type", "update_regime"])
        .reset_index(drop=True)
    )
    table_overlay_summary = (
        table_architecture.loc[table_architecture["arm_role"] == "evaluation_protocol"]
        .copy()
        .reset_index(drop=True)
    )

    materialize_dataframe_table(ctx, table_architecture, "table_p10_1_benchmark_architecture", "B7", "B7 analytical output")
    materialize_dataframe_table(ctx, table_arm_summary, "table_p10_1_benchmark_arm_summary", "B7", "B7 analytical output")
    materialize_dataframe_table(ctx, table_overlay_summary, "table_p10_1_benchmark_overlay_summary", "B7", "B7 analytical output")

    metadata_path = ctx.metadata_dir / "metadata_p10_1_benchmark_architecture.json"
    save_json(
        {
            "step": "B7",
            "title": "Define Corrected Benchmark Architecture",
            "output_tables": [
                "table_p10_1_benchmark_architecture",
                "table_p10_1_benchmark_arm_summary",
                "table_p10_1_benchmark_overlay_summary",
            ],
            "n_rows_architecture": int(table_architecture.shape[0]),
            "n_rows_overlay_summary": int(table_overlay_summary.shape[0]),
        },
        metadata_path,
    )
    print_artifact("metadata_p10_1_benchmark_architecture_json", metadata_path)

    log_stage_end("7")


def stage_b8_materialize_enrollment_window_features(ctx: PipelineContext) -> None:
    log_stage_start("8", "Materialize Canonical Enrollment Window Features")

    window_weeks = int(ctx.modeling_config["main_enrollment_window_weeks"])
    clicks_column = f"clicks_first_{window_weeks}_weeks"
    active_column = f"active_weeks_first_{window_weeks}"
    mean_column = f"mean_clicks_first_{window_weeks}_weeks"

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_window_features")
    ctx.con.execute(
        f"""
        CREATE TABLE enrollment_window_features AS
        SELECT
            enrollment_id,
            SUM(CASE WHEN week < {window_weeks} THEN total_clicks_week ELSE 0 END) AS {clicks_column},
            SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END) AS {active_column},
            CASE
                WHEN SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END) > 0
                THEN
                    SUM(CASE WHEN week < {window_weeks} THEN total_clicks_week ELSE 0 END) * 1.0
                    / SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END)
                ELSE 0
            END AS {mean_column}
        FROM person_period_enriched
        GROUP BY enrollment_id
        """
    )

    register_existing_table(ctx, "enrollment_window_features", "B8", "B8 main output")

    audit_df = ctx.con.execute(
        f"""
        SELECT
            {window_weeks} AS window_weeks,
            COUNT(*) AS n_rows,
            COUNT(DISTINCT enrollment_id) AS n_distinct_enrollments,
            CASE WHEN COUNT(*) = COUNT(DISTINCT enrollment_id) THEN TRUE ELSE FALSE END AS is_unique_by_enrollment_id,
            AVG({clicks_column}) AS avg_clicks,
            AVG({active_column}) AS avg_active_weeks,
            AVG({mean_column}) AS avg_mean_clicks
        FROM enrollment_window_features
        """
    ).fetchdf()
    missing_df = ctx.con.execute(
        f"""
        SELECT
            SUM(CASE WHEN {clicks_column} IS NULL THEN 1 ELSE 0 END) AS n_missing_clicks,
            SUM(CASE WHEN {active_column} IS NULL THEN 1 ELSE 0 END) AS n_missing_active_weeks,
            SUM(CASE WHEN {mean_column} IS NULL THEN 1 ELSE 0 END) AS n_missing_mean_clicks
        FROM enrollment_window_features
        """
    ).fetchdf()

    materialize_dataframe_table(ctx, audit_df, "table_enrollment_window_features_audit", "B8", "B8 audit output")
    materialize_dataframe_table(ctx, missing_df, "table_enrollment_window_features_missing_check", "B8", "B8 audit output")

    log_stage_end("8")


def stage_b9_build_model_input_views(ctx: PipelineContext) -> None:
    log_stage_start("9", "Build Model-Specific Input Tables")

    window_weeks = int(ctx.modeling_config["main_enrollment_window_weeks"])
    main_window_columns = [
        f"clicks_first_{window_weeks}_weeks",
        f"active_weeks_first_{window_weeks}",
        f"mean_clicks_first_{window_weeks}_weeks",
    ]

    ctx.con.execute("DROP TABLE IF EXISTS pp_linear_hazard_input")
    ctx.con.execute(
        """
        CREATE TABLE pp_linear_hazard_input AS
        SELECT
            enrollment_id,
            id_student,
            code_module,
            code_presentation,
            week,
            event_t,
            event_observed,
            t_event_week,
            t_final_week,
            used_zero_week_fallback_for_censoring,
            gender,
            region,
            highest_education,
            imd_band,
            age_band,
            num_of_prev_attempts,
            studied_credits,
            disability,
            total_clicks_week,
            active_this_week,
            n_vle_rows_week,
            n_distinct_sites_week,
            cum_clicks_until_t,
            recency,
            streak
        FROM person_period_enriched
        """
    )

    ctx.con.execute("DROP TABLE IF EXISTS pp_neural_hazard_input")
    ctx.con.execute("CREATE TABLE pp_neural_hazard_input AS SELECT * FROM pp_linear_hazard_input")

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_cox_input")
    ctx.con.execute(
        f"""
        CREATE TABLE enrollment_cox_input AS
        SELECT
            b.enrollment_id,
            b.id_student,
            b.code_module,
            b.code_presentation,
            b.event,
            b.duration,
            b.duration_raw,
            b.used_zero_week_fallback_for_censoring,
            b.gender,
            b.region,
            b.highest_education,
            b.imd_band,
            b.age_band,
            b.num_of_prev_attempts,
            b.studied_credits,
            b.disability,
            w.{main_window_columns[0]},
            w.{main_window_columns[1]},
            w.{main_window_columns[2]}
        FROM enrollment_model_table AS b
        LEFT JOIN enrollment_window_features AS w
          ON b.enrollment_id = REPLACE(w.enrollment_id, '|', '||')
        """
    )

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_deepsurv_input")
    ctx.con.execute("CREATE TABLE enrollment_deepsurv_input AS SELECT * FROM enrollment_cox_input")

    for table_name in [
        "pp_linear_hazard_input",
        "pp_neural_hazard_input",
        "enrollment_cox_input",
        "enrollment_deepsurv_input",
    ]:
        register_existing_table(ctx, table_name, "B9", f"B9 main output {table_name}")

    audit_rows = []
    for table_name, family, target_column, duration_column in [
        ("pp_linear_hazard_input", "discrete_time", "event_t", None),
        ("pp_neural_hazard_input", "discrete_time", "event_t", None),
        ("enrollment_cox_input", "continuous_time", "event", "duration"),
        ("enrollment_deepsurv_input", "continuous_time", "event", "duration"),
    ]:
        df = ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()
        audit_rows.append(
            {
                "table_name": table_name,
                "family": family,
                "n_rows": int(df.shape[0]),
                "n_distinct_enrollments": int(df["enrollment_id"].nunique()) if "enrollment_id" in df.columns else np.nan,
                "n_columns": int(df.shape[1]),
                "target_column": target_column,
                "target_sum": float(df[target_column].sum()) if target_column in df.columns else np.nan,
                "duration_column": duration_column,
                "min_duration": float(df[duration_column].min()) if duration_column and duration_column in df.columns else np.nan,
                "max_duration": float(df[duration_column].max()) if duration_column and duration_column in df.columns else np.nan,
            }
        )
    audit_df = pd.DataFrame(audit_rows)
    sample_pp_linear = ctx.con.execute(
        """
        SELECT *
        FROM pp_linear_hazard_input
        ORDER BY enrollment_id, week
        LIMIT 10
        """
    ).fetchdf()
    sample_cox = ctx.con.execute(
        """
        SELECT *
        FROM enrollment_cox_input
        ORDER BY enrollment_id
        LIMIT 10
        """
    ).fetchdf()

    materialize_dataframe_table(ctx, audit_df, "table_model_input_views_audit", "B9", "B9 audit output")
    materialize_dataframe_table(ctx, sample_pp_linear, "table_pp_linear_hazard_input_sample", "B9", "B9 analytical output")
    materialize_dataframe_table(ctx, sample_cox, "table_enrollment_cox_input_sample", "B9", "B9 analytical output")

    log_stage_end("9")


def stage_b10_build_configured_inputs(ctx: PipelineContext) -> None:
    log_stage_start("10", "Attach Canonical Window Features and Build Configured Inputs")

    window_columns = [column for column in get_table_columns(ctx.con, "enrollment_window_features") if column != "enrollment_id"]
    cox_base_columns = get_table_columns(ctx.con, "enrollment_cox_input")
    deepsurv_base_columns = get_table_columns(ctx.con, "enrollment_deepsurv_input")

    cox_extra_window_columns = [column for column in window_columns if column not in cox_base_columns]
    deepsurv_extra_window_columns = [column for column in window_columns if column not in deepsurv_base_columns]

    cox_fragment = build_window_select_fragment(cox_extra_window_columns, alias="ewf")
    deepsurv_fragment = build_window_select_fragment(deepsurv_extra_window_columns, alias="ewf")

    attach_plan_df = pd.DataFrame(
        [
            {
                "target_table": "enrollment_cox_input_configured",
                "base_table": "enrollment_cox_input",
                "n_base_columns": len(cox_base_columns),
                "n_window_columns_available": len(window_columns),
                "n_window_columns_attached": len(cox_extra_window_columns),
                "attached_window_columns": ", ".join(cox_extra_window_columns),
                "n_window_columns_skipped_as_already_present": len([column for column in window_columns if column in cox_base_columns]),
                "skipped_window_columns": ", ".join([column for column in window_columns if column in cox_base_columns]),
            },
            {
                "target_table": "enrollment_deepsurv_input_configured",
                "base_table": "enrollment_deepsurv_input",
                "n_base_columns": len(deepsurv_base_columns),
                "n_window_columns_available": len(window_columns),
                "n_window_columns_attached": len(deepsurv_extra_window_columns),
                "attached_window_columns": ", ".join(deepsurv_extra_window_columns),
                "n_window_columns_skipped_as_already_present": len([column for column in window_columns if column in deepsurv_base_columns]),
                "skipped_window_columns": ", ".join([column for column in window_columns if column in deepsurv_base_columns]),
            },
        ]
    )

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_cox_input_configured")
    ctx.con.execute(
        f"""
        CREATE TABLE enrollment_cox_input_configured AS
        SELECT
            eci.*{cox_fragment}
        FROM enrollment_cox_input AS eci
        LEFT JOIN enrollment_window_features AS ewf
          ON eci.enrollment_id = ewf.enrollment_id
        """
    )

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_deepsurv_input_configured")
    ctx.con.execute(
        f"""
        CREATE TABLE enrollment_deepsurv_input_configured AS
        SELECT
            edi.*{deepsurv_fragment}
        FROM enrollment_deepsurv_input AS edi
        LEFT JOIN enrollment_window_features AS ewf
          ON edi.enrollment_id = ewf.enrollment_id
        """
    )

    configured_cox_columns = get_table_columns(ctx.con, "enrollment_cox_input_configured")
    configured_deepsurv_columns = get_table_columns(ctx.con, "enrollment_deepsurv_input_configured")
    cox_duplicates = duplicated_names(configured_cox_columns)
    deepsurv_duplicates = duplicated_names(configured_deepsurv_columns)
    if cox_duplicates:
        raise ValueError(f"Duplicated columns detected in enrollment_cox_input_configured: {cox_duplicates}")
    if deepsurv_duplicates:
        raise ValueError(f"Duplicated columns detected in enrollment_deepsurv_input_configured: {deepsurv_duplicates}")

    for table_name in ["enrollment_cox_input_configured", "enrollment_deepsurv_input_configured"]:
        register_existing_table(ctx, table_name, "B10", f"B10 main output {table_name}")

    created_columns_df = pd.DataFrame({"created_feature_name": window_columns})
    sample_rows = ctx.con.execute(
        """
        SELECT *
        FROM enrollment_cox_input_configured
        ORDER BY enrollment_id
        LIMIT 20
        """
    ).fetchdf()

    materialize_dataframe_table(
        ctx,
        created_columns_df,
        "table_enrollment_window_features_created_columns",
        "B10",
        "B10 analytical output",
    )
    materialize_dataframe_table(
        ctx,
        attach_plan_df,
        "table_enrollment_window_feature_attach_plan",
        "B10",
        "B10 analytical output",
    )
    materialize_dataframe_table(
        ctx,
        sample_rows,
        "table_enrollment_cox_input_configured_sample",
        "B10",
        "B10 analytical output",
    )

    log_stage_end("10")


def stage_b11_build_window_sensitivity_design(ctx: PipelineContext) -> None:
    log_stage_start("11", "Early-Window Length Sensitivity Design")

    from dropout_bench_v3_D_00_common import resolve_early_window_sensitivity_weeks

    sensitivity_windows = resolve_early_window_sensitivity_weeks(
        ctx.benchmark_config,
        default=DEFAULT_SENSITIVITY_WINDOWS,
    )

    window_feature_sql_parts = []
    created_feature_names: list[str] = []
    for window_weeks in sensitivity_windows:
        clicks_column = f"clicks_first_{window_weeks}_weeks"
        active_column = f"active_weeks_first_{window_weeks}"
        mean_column = f"mean_clicks_first_{window_weeks}_weeks"
        created_feature_names.extend([clicks_column, active_column, mean_column])
        window_feature_sql_parts.append(
            f"""
            SUM(CASE WHEN week < {window_weeks} THEN total_clicks_week ELSE 0 END) AS {clicks_column},
            SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END) AS {active_column},
            CASE
                WHEN SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END) > 0
                THEN
                    SUM(CASE WHEN week < {window_weeks} THEN total_clicks_week ELSE 0 END) * 1.0
                    / SUM(CASE WHEN week < {window_weeks} THEN active_this_week ELSE 0 END)
                ELSE 0
            END AS {mean_column}
            """.strip()
        )

    sensitivity_projection = ",\n            ".join(window_feature_sql_parts)

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_window_features_sensitivity")
    ctx.con.execute(
        f"""
        CREATE TABLE enrollment_window_features_sensitivity AS
        SELECT
            enrollment_id,
            {sensitivity_projection}
        FROM person_period_enriched
        GROUP BY enrollment_id
        """
    )
    register_existing_table(ctx, "enrollment_window_features_sensitivity", "B11", "B11 main output")

    attach_rows = []
    for window_weeks in sensitivity_windows:
        attach_rows.append(build_window_variant(ctx, "enrollment_cox_input", f"enrollment_cox_input_w{window_weeks}", window_weeks))
        attach_rows.append(build_window_variant(ctx, "enrollment_deepsurv_input", f"enrollment_deepsurv_input_w{window_weeks}", window_weeks))

    for table_name in [
        *(f"enrollment_cox_input_w{window_weeks}" for window_weeks in sensitivity_windows),
        *(f"enrollment_deepsurv_input_w{window_weeks}" for window_weeks in sensitivity_windows),
    ]:
        register_existing_table(ctx, table_name, "B11", f"B11 main output {table_name}")

    design_rows = []
    for window_weeks in sensitivity_windows:
        for family, table_name in [
            ("continuous_time_cox", f"enrollment_cox_input_w{window_weeks}"),
            ("continuous_time_deepsurv", f"enrollment_deepsurv_input_w{window_weeks}"),
        ]:
            df = ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()
            design_rows.append(
                {
                    "window_weeks": int(window_weeks),
                    "family": family,
                    "table_name": table_name,
                    "n_rows": int(df.shape[0]),
                    "n_distinct_enrollments": int(df["enrollment_id"].nunique()) if "enrollment_id" in df.columns else None,
                    "n_columns": int(df.shape[1]),
                    "has_clicks_feature": f"clicks_first_{window_weeks}_weeks" in df.columns,
                    "has_active_feature": f"active_weeks_first_{window_weeks}" in df.columns,
                    "has_mean_feature": f"mean_clicks_first_{window_weeks}_weeks" in df.columns,
                }
            )
    design_df = pd.DataFrame(design_rows)
    attach_plan_df = pd.DataFrame(attach_rows)

    materialize_dataframe_table(ctx, design_df, "table_p12_1_window_sensitivity_design", "B11", "B11 analytical output")
    materialize_dataframe_table(ctx, attach_plan_df, "table_p12_1_window_sensitivity_attach_plan", "B11", "B11 analytical output")
    materialize_dataframe_table(
        ctx,
        pd.DataFrame({"created_feature_name": created_feature_names}),
        "table_p12_1_window_sensitivity_created_columns",
        "B11",
        "B11 analytical output",
    )

    metadata_path = ctx.metadata_dir / "metadata_p12_1_window_sensitivity_design.json"
    save_json(
        {
            "step": "B11",
            "title": "Early-Window Length Sensitivity Design",
            "windows_sensitivity": sensitivity_windows,
            "materialized_duckdb_tables": [
                "enrollment_window_features_sensitivity",
                *(f"enrollment_cox_input_w{window_weeks}" for window_weeks in sensitivity_windows),
                *(f"enrollment_deepsurv_input_w{window_weeks}" for window_weeks in sensitivity_windows),
            ],
        },
        metadata_path,
    )
    print_artifact("metadata_p12_1_window_sensitivity_design_json", metadata_path)

    log_stage_end("11")


def stage_b12_build_window_sensitivity_comparison(ctx: PipelineContext) -> None:
    log_stage_start("12", "Early-Window Length Sensitivity Comparison")

    from dropout_bench_v3_D_00_common import resolve_early_window_sensitivity_weeks

    sensitivity_windows = resolve_early_window_sensitivity_weeks(
        ctx.benchmark_config,
        default=DEFAULT_SENSITIVITY_WINDOWS,
    )

    present_tables = available_tables(ctx.con)
    families = [
        ("continuous_time_cox", "enrollment_cox_input_w{w}"),
        ("continuous_time_deepsurv", "enrollment_deepsurv_input_w{w}"),
    ]

    registry_rows = []
    for family_name, template in families:
        for window_weeks in sensitivity_windows:
            table_name = template.format(w=window_weeks)
            exists = table_name in present_tables
            row = {
                "window_weeks": int(window_weeks),
                "family": family_name,
                "table_name": table_name,
                "table_exists": bool(exists),
                "n_rows": np.nan,
                "n_distinct_enrollments": np.nan,
                "n_columns": np.nan,
                "has_clicks_feature": False,
                "has_active_feature": False,
                "has_mean_feature": False,
                "event_sum": np.nan,
                "duration_min": np.nan,
                "duration_max": np.nan,
                "ready_for_model_training": False,
                "notes": "",
            }
            if exists:
                df = ctx.con.execute(f"SELECT * FROM {table_name}").fetchdf()
                columns = set(df.columns)
                clicks_column = f"clicks_first_{window_weeks}_weeks"
                active_column = f"active_weeks_first_{window_weeks}"
                mean_column = f"mean_clicks_first_{window_weeks}_weeks"
                row["n_rows"] = int(df.shape[0])
                row["n_distinct_enrollments"] = int(df["enrollment_id"].nunique()) if "enrollment_id" in df.columns else np.nan
                row["n_columns"] = int(df.shape[1])
                row["has_clicks_feature"] = clicks_column in columns
                row["has_active_feature"] = active_column in columns
                row["has_mean_feature"] = mean_column in columns
                if "event" in columns:
                    row["event_sum"] = float(pd.to_numeric(df["event"], errors="coerce").fillna(0).sum())
                if "duration" in columns:
                    row["duration_min"] = float(pd.to_numeric(df["duration"], errors="coerce").min())
                    row["duration_max"] = float(pd.to_numeric(df["duration"], errors="coerce").max())
                row["ready_for_model_training"] = bool(
                    row["has_clicks_feature"]
                    and row["has_active_feature"]
                    and row["has_mean_feature"]
                    and ("event" in columns)
                    and ("duration" in columns)
                )
                row["notes"] = (
                    "Main comparable early-window benchmark design."
                    if window_weeks == 4
                    else "Sensitivity variant materialized and structurally ready."
                )
            else:
                row["notes"] = "Variant table not materialized."
            registry_rows.append(row)

    registry_df = pd.DataFrame(registry_rows).sort_values(["family", "window_weeks"]).reset_index(drop=True)

    comparison_rows = []
    for family_name, group_df in registry_df.groupby("family", dropna=False):
        reference_df = group_df.loc[group_df["window_weeks"] == 4].copy()
        if reference_df.empty:
            for _, row in group_df.iterrows():
                comparison_rows.append(
                    {
                        "family": family_name,
                        "window_weeks": int(row["window_weeks"]),
                        "table_name": row["table_name"],
                        "table_exists": bool(row["table_exists"]),
                        "ready_for_model_training": bool(row["ready_for_model_training"]),
                        "delta_n_columns_vs_w4": np.nan,
                        "delta_event_sum_vs_w4": np.nan,
                        "delta_duration_max_vs_w4": np.nan,
                        "comparison_status": "missing_w4_reference",
                        "notes": "No 4-week reference row available for comparison.",
                    }
                )
            continue

        reference_row = reference_df.iloc[0]
        for _, row in group_df.iterrows():
            comparison_rows.append(
                {
                    "family": family_name,
                    "window_weeks": int(row["window_weeks"]),
                    "table_name": row["table_name"],
                    "table_exists": bool(row["table_exists"]),
                    "ready_for_model_training": bool(row["ready_for_model_training"]),
                    "delta_n_columns_vs_w4": (
                        pd.to_numeric(row["n_columns"], errors="coerce")
                        - pd.to_numeric(reference_row["n_columns"], errors="coerce")
                    ) if pd.notna(row["n_columns"]) and pd.notna(reference_row["n_columns"]) else np.nan,
                    "delta_event_sum_vs_w4": (
                        pd.to_numeric(row["event_sum"], errors="coerce")
                        - pd.to_numeric(reference_row["event_sum"], errors="coerce")
                    ) if pd.notna(row["event_sum"]) and pd.notna(reference_row["event_sum"]) else np.nan,
                    "delta_duration_max_vs_w4": (
                        pd.to_numeric(row["duration_max"], errors="coerce")
                        - pd.to_numeric(reference_row["duration_max"], errors="coerce")
                    ) if pd.notna(row["duration_max"]) and pd.notna(reference_row["duration_max"]) else np.nan,
                    "comparison_status": (
                        "reference_main_window"
                        if int(row["window_weeks"]) == 4
                        else "sensitivity_variant_ready"
                        if bool(row["ready_for_model_training"])
                        else "not_ready"
                    ),
                    "notes": (
                        "4-week reference window."
                        if int(row["window_weeks"]) == 4
                        else "Sensitivity variant can be used for downstream window-length comparison."
                    ),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["family", "window_weeks"]).reset_index(drop=True)
    summary_rows = []
    for family_name, group_df in registry_df.groupby("family", dropna=False):
        summary_rows.append(
            {
                "family": family_name,
                "n_windows_expected": len(sensitivity_windows),
                "n_windows_materialized": int(pd.to_numeric(group_df["table_exists"], errors="coerce").fillna(False).sum()),
                "n_windows_ready_for_model_training": int(pd.to_numeric(group_df["ready_for_model_training"], errors="coerce").fillna(False).sum()),
                "all_windows_materialized": bool(group_df["table_exists"].all()),
                "all_windows_ready_for_model_training": bool(group_df["ready_for_model_training"].all()),
                "window_list_materialized": ", ".join(str(int(value)) for value in group_df.loc[group_df["table_exists"], "window_weeks"].tolist()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("family").reset_index(drop=True)

    materialize_dataframe_table(ctx, registry_df, "table_p12_2_window_sensitivity_registry", "B12", "B12 analytical output")
    materialize_dataframe_table(ctx, comparison_df, "table_p12_2_window_sensitivity_comparison", "B12", "B12 analytical output")
    materialize_dataframe_table(ctx, summary_df, "table_p12_2_window_sensitivity_summary", "B12", "B12 analytical output")

    metadata_path = ctx.metadata_dir / "metadata_p12_2_window_sensitivity_comparison.json"
    save_json(
        {
            "step": "B12",
            "title": "Early-Window Length Sensitivity Comparison",
            "windows_evaluated_structurally": sensitivity_windows,
            "families": [family_name for family_name, _ in families],
            "purpose": "Consolidate the currently materialized early-window sensitivity variants into a canonical comparison table.",
        },
        metadata_path,
    )
    print_artifact("metadata_p12_2_window_sensitivity_comparison_json", metadata_path)

    log_stage_end("12")


def stage_b13_shutdown(ctx: PipelineContext) -> None:
    from util import close_duckdb_connection

    log_stage_start("13", "DuckDB Shutdown")
    close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
    ctx.con = None
    log_stage_end("13")


def main() -> None:
    ctx = bootstrap_context()
    try:
        stage_b1_validate_source_contract(ctx)
        stage_b2_build_person_period_min(ctx)
        stage_b3_build_weekly_vle_features(ctx)
        stage_b4_build_person_period_enriched(ctx)
        stage_b5_build_enrollment_model_table(ctx)
        stage_b6_define_modeling_configuration(ctx)
        stage_b7_define_benchmark_architecture(ctx)
        stage_b8_materialize_enrollment_window_features(ctx)
        stage_b9_build_model_input_views(ctx)
        stage_b10_build_configured_inputs(ctx)
        stage_b11_build_window_sensitivity_design(ctx)
        stage_b12_build_window_sensitivity_comparison(ctx)
    finally:
        if ctx.con is not None:
            stage_b13_shutdown(ctx)


if __name__ == "__main__":
    main()