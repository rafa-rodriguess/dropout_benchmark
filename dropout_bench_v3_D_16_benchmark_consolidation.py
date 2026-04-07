from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd

import dropout_bench_v3_D_00_common as base


NOTEBOOK_NAME = "dropout_bench_v3_D_16_benchmark_consolidation.py"
STAGE_ID = "5.16"

PRIMARY_METRIC_SUFFIXES = [
    "tuning_results",
    "tuned_test_predictions",
    "tuned_primary_metrics",
    "tuned_brier_by_horizon",
    "tuned_secondary_metrics",
    "tuned_td_auc_support_audit",
    "tuned_row_diagnostics",
    "tuned_support_by_horizon",
    "tuned_calibration_summary",
    "tuned_calibration_bins_by_horizon",
    "tuned_predicted_vs_observed_survival",
]

MODEL_SPECS = base.get_d16_model_specs(include_contract=True)

WEIGHT_SENSITIVITY_SPECS = base.get_d16_weight_sensitivity_specs()

COMPARABLE_WINDOW_SPECS = base.get_d16_comparable_window_specs()

CROSS_ARM_MODEL_SPECS = base.get_d16_cross_arm_model_specs()


def table_exists(con, table_name: str | None) -> bool:
    return bool(table_name) and base.duckdb_table_exists(con, str(table_name))


def source_table_for_window(prefix: str, window_weeks: int, canonical_window_weeks: int) -> str:
    if int(window_weeks) == int(canonical_window_weeks):
        return f"{prefix}_tuned_primary_metrics"
    return f"{prefix}_tuned_primary_metrics_w{int(window_weeks)}"


def read_metric_values(con, table_name: str | None) -> dict[str, float]:
    if not table_exists(con, table_name):
        return {}
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    if df.empty or "metric_name" not in df.columns or "metric_value" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        metric_name = str(row.get("metric_name", "")).strip().lower()
        metric_value = pd.to_numeric(pd.Series([row.get("metric_value")]), errors="coerce").iloc[0]
        if metric_name and pd.notna(metric_value):
            out[metric_name] = float(metric_value)
    return out


def metric_or_nan(metric_map: dict[str, float], key: str) -> float:
    return float(metric_map[key]) if key in metric_map else math.nan


def materialize_prefix_alias_tables(runtime: dict, alias_specs: list[dict], canonical_window_weeks: int) -> list[dict]:
    con = runtime["con"]
    alias_events = []
    for spec in alias_specs:
        source_prefix = spec.get("source_prefix")
        alias_prefix = spec.get("alias_prefix")
        if not source_prefix or not alias_prefix:
            continue
        for suffix in PRIMARY_METRIC_SUFFIXES:
            source_table = f"{source_prefix}_{suffix}_w{int(canonical_window_weeks)}"
            alias_table = f"{alias_prefix}_{suffix}"
            if not table_exists(con, source_table):
                continue
            df = con.execute(f"SELECT * FROM {source_table}").fetchdf()
            base.materialize_dataframe(con, df, alias_table, STAGE_ID, runtime["NOTEBOOK_NAME"], runtime["RUN_ID"])
            alias_events.append({"alias_table": alias_table, "source_table": source_table})
    return alias_events


def materialize_alias_tables(runtime: dict, canonical_window_weeks: int) -> list[dict]:
    canonical_alias_specs = [
        {
            "source_prefix": spec.get("source_prefix"),
            "alias_prefix": spec.get("alias_prefix"),
        }
        for spec in MODEL_SPECS
    ]
    weighted_alias_specs = [
        {
            "source_prefix": spec.get("weighted_prefix"),
            "alias_prefix": spec.get("weighted_prefix"),
        }
        for spec in WEIGHT_SENSITIVITY_SPECS
    ]
    alias_events = materialize_prefix_alias_tables(runtime, canonical_alias_specs, canonical_window_weeks)
    alias_events.extend(materialize_prefix_alias_tables(runtime, weighted_alias_specs, canonical_window_weeks))
    return alias_events


def build_inventory(runtime: dict, canonical_window_weeks: int) -> pd.DataFrame:
    con = runtime["con"]
    rows = []
    for spec in MODEL_SPECS:
        primary_table = spec.get("primary_metrics_table")
        primary_exists = table_exists(con, primary_table)
        artifact_status = "available" if (spec["family_group"] == "contract" or primary_exists) else "missing"
        rows.append({
            "stage_id": spec["stage_id"],
            "stage_order": int(spec["stage_order"]),
            "model_name": spec["model_name"],
            "family_group": spec["family_group"],
            "artifact_status": artifact_status,
            "canonical_window_weeks": int(canonical_window_weeks),
            "primary_metrics_table": primary_table,
            "brier_table": primary_table.replace("_primary_metrics", "_brier_by_horizon") if primary_table else pd.NA,
            "secondary_metrics_table": primary_table.replace("_primary_metrics", "_secondary_metrics") if primary_table else pd.NA,
            "support_table": primary_table.replace("_primary_metrics", "_support_by_horizon") if primary_table else pd.NA,
            "calibration_table": primary_table.replace("_primary_metrics", "_calibration_summary") if primary_table else pd.NA,
        })
    inventory_df = pd.DataFrame(rows).sort_values("stage_order", kind="mergesort").reset_index(drop=True)
    predictive_count = int((inventory_df["family_group"] != "contract").sum())
    if predictive_count != 14:
        raise ValueError(f"{STAGE_ID}: expected 14 predictive models in canonical inventory, found {predictive_count}.")
    return inventory_df


def build_primary_summary(runtime: dict, inventory_df: pd.DataFrame) -> pd.DataFrame:
    con = runtime["con"]
    rows = []
    predictive_inventory = inventory_df[inventory_df["family_group"] != "contract"].copy()
    for _, row in predictive_inventory.iterrows():
        primary_table = str(row["primary_metrics_table"])
        metric_map = read_metric_values(con, primary_table)
        rows.append({
            "stage_id": row["stage_id"],
            "stage_order": int(row["stage_order"]),
            "model_name": row["model_name"],
            "family_group": row["family_group"],
            "primary_metrics_table": primary_table,
            "ibs": metric_or_nan(metric_map, "ibs"),
            "c_index": metric_or_nan(metric_map, "c_index"),
        })
    return pd.DataFrame(rows).sort_values("stage_order", kind="mergesort").reset_index(drop=True)


def build_item6_loss_sensitivity(runtime: dict, canonical_window_weeks: int, sensitivity_weeks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = runtime["con"]
    rows = []
    for spec in WEIGHT_SENSITIVITY_SPECS:
        for window_weeks in sensitivity_weeks:
            not_weighted_table = f"{spec['not_weighted_prefix']}_tuned_primary_metrics_w{int(window_weeks)}"
            weighted_table = f"{spec['weighted_prefix']}_tuned_primary_metrics_w{int(window_weeks)}"
            not_weighted_metrics = read_metric_values(con, not_weighted_table)
            weighted_metrics = read_metric_values(con, weighted_table)
            availability_status = "available" if not_weighted_metrics and weighted_metrics else "partial_or_missing"
            rows.append({
                "model_name": spec["model_name"],
                "family_group": spec["family_group"],
                "window_weeks": int(window_weeks),
                "selected_variant": "not_weighted",
                "comparison_variant": "weighted",
                "selected_primary_table": not_weighted_table,
                "comparison_primary_table": weighted_table,
                "ibs_not_weighted": metric_or_nan(not_weighted_metrics, "ibs"),
                "ibs_weighted": metric_or_nan(weighted_metrics, "ibs"),
                "delta_ibs_weighted_minus_not_weighted": metric_or_nan(weighted_metrics, "ibs") - metric_or_nan(not_weighted_metrics, "ibs"),
                "c_index_not_weighted": metric_or_nan(not_weighted_metrics, "c_index"),
                "c_index_weighted": metric_or_nan(weighted_metrics, "c_index"),
                "delta_c_index_weighted_minus_not_weighted": metric_or_nan(weighted_metrics, "c_index") - metric_or_nan(not_weighted_metrics, "c_index"),
                "canonical_window": bool(int(window_weeks) == int(canonical_window_weeks)),
                "availability_status": availability_status,
            })
    sensitivity_df = pd.DataFrame(rows)
    decision_rows = []
    for model_name, group_df in sensitivity_df.groupby("model_name", sort=False):
        canonical_df = group_df[group_df["canonical_window"]].copy()
        canonical_row = canonical_df.iloc[0] if not canonical_df.empty else group_df.iloc[0]
        decision_rows.append({
            "model_name": model_name,
            "family_group": str(canonical_row["family_group"]),
            "selected_variant": "not_weighted",
            "selection_scope": "main_benchmark_roster",
            "canonical_window_weeks": int(canonical_row["window_weeks"]),
            "canonical_ibs_not_weighted": float(canonical_row["ibs_not_weighted"]),
            "canonical_ibs_weighted": float(canonical_row["ibs_weighted"]),
            "canonical_c_index_not_weighted": float(canonical_row["c_index_not_weighted"]),
            "canonical_c_index_weighted": float(canonical_row["c_index_weighted"]),
            "decision_rationale": "The paper-aligned dynamic roster remains unweighted at the canonical window; weighted runs are retained as sensitivity checks only.",
        })
    return sensitivity_df.sort_values(["model_name", "window_weeks"]), pd.DataFrame(decision_rows)


def comparable_primary_table(table_prefix: str, canonical_window_weeks: int, window_weeks: int) -> str:
    if int(window_weeks) == int(canonical_window_weeks):
        return f"{table_prefix}_tuned_primary_metrics"
    return f"{table_prefix}_tuned_primary_metrics_w{int(window_weeks)}"


def cross_arm_primary_table(table_prefix: str, canonical_window_weeks: int, window_weeks: int, canonical_uses_unsuffixed: bool) -> str:
    if bool(canonical_uses_unsuffixed) and int(window_weeks) == int(canonical_window_weeks):
        return f"{table_prefix}_tuned_primary_metrics"
    return f"{table_prefix}_tuned_primary_metrics_w{int(window_weeks)}"


def build_comparable_window_tables(runtime: dict, canonical_window_weeks: int, sensitivity_weeks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = runtime["con"]
    rows = []
    for spec in COMPARABLE_WINDOW_SPECS:
        for window_weeks in sensitivity_weeks:
            primary_table = comparable_primary_table(spec["table_prefix"], canonical_window_weeks, window_weeks)
            metric_map = read_metric_values(con, primary_table)
            rows.append({
                "model_name": spec["model_name"],
                "family_group": spec["family_group"],
                "window_weeks": int(window_weeks),
                "primary_metrics_table": primary_table,
                "ibs": metric_or_nan(metric_map, "ibs"),
                "c_index": metric_or_nan(metric_map, "c_index"),
                "availability_status": "available" if metric_map else "missing",
                "canonical_window": bool(int(window_weeks) == int(canonical_window_weeks)),
            })
    sensitivity_df = pd.DataFrame(rows)
    available_df = sensitivity_df[sensitivity_df["availability_status"] == "available"].copy()
    champions_rows = []
    for window_weeks, group_df in available_df.groupby("window_weeks", sort=True):
        champion_row = group_df.sort_values(["ibs", "c_index"], ascending=[True, False], kind="mergesort").iloc[0]
        champions_rows.append({
            "window_weeks": int(window_weeks),
            "champion_model_name": champion_row["model_name"],
            "champion_family_group": champion_row["family_group"],
            "champion_primary_metrics_table": champion_row["primary_metrics_table"],
            "champion_ibs": float(champion_row["ibs"]),
            "champion_c_index": float(champion_row["c_index"]),
        })
    champions_df = pd.DataFrame(champions_rows)
    stability_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "champion_window_count": int(group_df.shape[0]),
                "champion_window_share": float(group_df.shape[0]) / float(len(sensitivity_weeks)),
            }
            for model_name, group_df in champions_df.groupby("champion_model_name", sort=False)
        ]
    ).rename(columns={"champion_model_name": "model_name"})
    return sensitivity_df.sort_values(["window_weeks", "model_name"]), champions_df.sort_values("window_weeks"), stability_df.sort_values(["champion_window_count", "model_name"], ascending=[False, True])


def build_cross_arm_tables(runtime: dict, inventory_df: pd.DataFrame, canonical_window_weeks: int, sensitivity_weeks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = runtime["con"]
    predictive_df = inventory_df[inventory_df["family_group"] != "contract"].copy()
    protocol_df = pd.DataFrame(
        [
            {
                "protocol_order": 1,
                "protocol_key": "arm_definition",
                "protocol_value": "dynamic_weekly_person_period_vs_comparable_continuous_time_early_window",
                "notes": "Cross-arm interpretation must remain contract-aware because the two arms operate on different data representations.",
            },
            {
                "protocol_order": 2,
                "protocol_key": "canonical_window_weeks",
                "protocol_value": str(int(canonical_window_weeks)),
                "notes": "The canonical enrollment window remains fixed at four weeks for paper-aligned comparability.",
            },
            {
                "protocol_order": 3,
                "protocol_key": "sensitivity_windows",
                "protocol_value": ",".join(str(int(value)) for value in sensitivity_weeks),
                "notes": "Window sensitivity is tracked separately from the canonical roster and does not redefine the main benchmark inventory.",
            },
        ]
    )
    sensitivity_rows = []
    for spec in CROSS_ARM_MODEL_SPECS:
        for window_weeks in sensitivity_weeks:
            primary_table = cross_arm_primary_table(
                spec["table_prefix"],
                canonical_window_weeks,
                window_weeks,
                spec["canonical_uses_unsuffixed"],
            )
            metric_map = read_metric_values(con, primary_table)
            sensitivity_rows.append(
                {
                    "window_weeks": int(window_weeks),
                    "arm_name": spec["arm_name"],
                    "stage_id": spec["stage_id"],
                    "stage_order": int(spec["stage_order"]),
                    "model_name": spec["model_name"],
                    "family_group": spec["family_group"],
                    "primary_metrics_table": primary_table,
                    "ibs": metric_or_nan(metric_map, "ibs"),
                    "c_index": metric_or_nan(metric_map, "c_index"),
                    "availability_status": "available" if metric_map else "missing",
                }
            )
    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(["window_weeks", "arm_name", "stage_order"], kind="mergesort").reset_index(drop=True)
    available_df = sensitivity_df[sensitivity_df["availability_status"] == "available"].copy()
    arm_champion_rows = []
    for (window_weeks, arm_name), group_df in available_df.groupby(["window_weeks", "arm_name"], sort=True):
        champion_row = group_df.sort_values(["ibs", "c_index", "stage_order"], ascending=[True, False, True], kind="mergesort").iloc[0]
        arm_champion_rows.append(
            {
                "window_weeks": int(window_weeks),
                "arm_name": arm_name,
                "champion_stage_id": champion_row["stage_id"],
                "champion_stage_order": int(champion_row["stage_order"]),
                "champion_model_name": champion_row["model_name"],
                "champion_family_group": champion_row["family_group"],
                "champion_primary_metrics_table": champion_row["primary_metrics_table"],
                "champion_ibs": float(champion_row["ibs"]),
                "champion_c_index": float(champion_row["c_index"]),
            }
        )
    arm_champions_df = pd.DataFrame(arm_champion_rows).sort_values(["window_weeks", "arm_name"], kind="mergesort").reset_index(drop=True)
    cross_arm_champion_rows = []
    for window_weeks, group_df in arm_champions_df.groupby("window_weeks", sort=True):
        champion_row = group_df.sort_values(["champion_ibs", "champion_c_index", "champion_stage_order"], ascending=[True, False, True], kind="mergesort").iloc[0]
        runner_up_df = group_df[group_df["arm_name"] != champion_row["arm_name"]].copy()
        runner_up_row = runner_up_df.iloc[0] if not runner_up_df.empty else None
        cross_arm_champion_rows.append(
            {
                "window_weeks": int(window_weeks),
                "winning_arm_name": champion_row["arm_name"],
                "winning_stage_id": champion_row["champion_stage_id"],
                "winning_model_name": champion_row["champion_model_name"],
                "winning_family_group": champion_row["champion_family_group"],
                "winning_primary_metrics_table": champion_row["champion_primary_metrics_table"],
                "winning_ibs": float(champion_row["champion_ibs"]),
                "winning_c_index": float(champion_row["champion_c_index"]),
                "runner_up_arm_name": None if runner_up_row is None else runner_up_row["arm_name"],
                "runner_up_stage_id": None if runner_up_row is None else runner_up_row["champion_stage_id"],
                "runner_up_model_name": None if runner_up_row is None else runner_up_row["champion_model_name"],
                "runner_up_ibs": math.nan if runner_up_row is None else float(runner_up_row["champion_ibs"]),
                "runner_up_c_index": math.nan if runner_up_row is None else float(runner_up_row["champion_c_index"]),
                "delta_ibs_winner_minus_runner_up": math.nan if runner_up_row is None else float(champion_row["champion_ibs"] - runner_up_row["champion_ibs"]),
                "delta_c_index_winner_minus_runner_up": math.nan if runner_up_row is None else float(champion_row["champion_c_index"] - runner_up_row["champion_c_index"]),
            }
        )
    cross_arm_champions_df = pd.DataFrame(cross_arm_champion_rows).sort_values("window_weeks", kind="mergesort").reset_index(drop=True)
    leadership_stability_df = pd.DataFrame(
        [
            {
                "winning_arm_name": arm_name,
                "winning_window_count": int(group_df.shape[0]),
                "winning_window_share": float(group_df.shape[0]) / float(len(sensitivity_weeks)),
            }
            for arm_name, group_df in cross_arm_champions_df.groupby("winning_arm_name", sort=False)
        ]
    ).sort_values(["winning_window_count", "winning_arm_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    arm_sequence = ",".join(cross_arm_champions_df["winning_arm_name"].astype(str).tolist())
    model_sequence = ",".join(cross_arm_champions_df["winning_model_name"].astype(str).tolist())
    decision_summary_df = pd.DataFrame(
        [
            {
                "decision_key": "cross_arm_window_winners",
                "decision_value": model_sequence,
                "notes": "Ordered winning models across sensitivity windows 2,4,6,8,10 under the primary ranking rule lowest IBS then highest c-index then lowest stage order.",
            },
            {
                "decision_key": "cross_arm_window_winning_arms",
                "decision_value": arm_sequence,
                "notes": "Ordered winning arms across sensitivity windows 2,4,6,8,10.",
            },
            {
                "decision_key": "cross_arm_changes_main_benchmark_narrative",
                "decision_value": "yes" if any(arm == "dynamic" for arm in cross_arm_champions_df["winning_arm_name"].tolist()) else "no",
                "notes": "Set to yes if any sensitivity window selects the dynamic arm over the comparable arm.",
            },
        ]
    )
    execution_scope_df = pd.DataFrame(
        [
            {
                "scope_key": "predictive_model_count",
                "scope_value": int(predictive_df.shape[0]),
                "notes": "Expanded benchmark roster excluding the contract stage.",
            },
            {
                "scope_key": "dynamic_arm_model_count",
                "scope_value": int(predictive_df["family_group"].isin(["dynamic_weekly", "dynamic_neural"]).sum()),
                "notes": "Dynamic weekly person-period models retained in the expanded roster.",
            },
            {
                "scope_key": "comparable_arm_model_count",
                "scope_value": int(predictive_df[~predictive_df["family_group"].isin(["dynamic_weekly", "dynamic_neural"])].shape[0]),
                "notes": "Comparable early-window continuous-time models retained in the expanded roster.",
            },
            {
                "scope_key": "contract_stage_count",
                "scope_value": int((inventory_df["family_group"] == "contract").sum()),
                "notes": "Non-predictive runtime contract materialization stage kept in the inventory for provenance.",
            },
        ]
    )
    return protocol_df, execution_scope_df, sensitivity_df, arm_champions_df, cross_arm_champions_df, leadership_stability_df, decision_summary_df


def write_metadata(runtime: dict, inventory_df: pd.DataFrame, primary_summary_df: pd.DataFrame, alias_events: list[dict]) -> None:
    metadata = {
        "stage_id": STAGE_ID,
        "notebook_name": runtime["NOTEBOOK_NAME"],
        "run_id": runtime["RUN_ID"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictive_model_count": int((inventory_df["family_group"] != "contract").sum()),
        "inventory_rows": int(inventory_df.shape[0]),
        "primary_summary_rows": int(primary_summary_df.shape[0]),
        "alias_tables_materialized": alias_events,
        "outputs": {
            "table_5_16_model_inventory": "duckdb://table_5_16_model_inventory",
            "table_5_16_model_primary_summary": "duckdb://table_5_16_model_primary_summary",
            "table_5_16_item6_loss_sensitivity": "duckdb://table_5_16_item6_loss_sensitivity",
            "table_5_16_item6_decision_summary": "duckdb://table_5_16_item6_decision_summary",
            "table_5_16_comparable_window_sensitivity": "duckdb://table_5_16_comparable_window_sensitivity",
            "table_5_16_comparable_window_champions": "duckdb://table_5_16_comparable_window_champions",
            "table_5_16_comparable_window_leadership_stability": "duckdb://table_5_16_comparable_window_leadership_stability",
            "table_5_16_cross_arm_parity_protocol": "duckdb://table_5_16_cross_arm_parity_protocol",
            "table_5_16_cross_arm_execution_scope": "duckdb://table_5_16_cross_arm_execution_scope",
            "table_5_16_cross_arm_window_sensitivity": "duckdb://table_5_16_cross_arm_window_sensitivity",
            "table_5_16_cross_arm_arm_champions": "duckdb://table_5_16_cross_arm_arm_champions",
            "table_5_16_cross_arm_window_champions": "duckdb://table_5_16_cross_arm_window_champions",
            "table_5_16_cross_arm_window_leadership_stability": "duckdb://table_5_16_cross_arm_window_leadership_stability",
            "table_5_16_cross_arm_decision_summary": "duckdb://table_5_16_cross_arm_decision_summary",
        },
    }
    metadata_path = Path(runtime["METADATA_DIR"]) / "metadata_5_16_model_inventory.json"
    runtime["save_json"](metadata, metadata_path)


def main() -> None:
    runtime = base.open_notebook_runtime(NOTEBOOK_NAME)
    try:
        shared_benchmark = runtime["SHARED_MODELING_CONTRACT"].get("benchmark", {})
        canonical_window_weeks = int(shared_benchmark["early_window_weeks"])
        sensitivity_weeks = base.resolve_early_window_sensitivity_weeks(shared_benchmark)

        alias_events = materialize_alias_tables(runtime, canonical_window_weeks)
        inventory_df = build_inventory(runtime, canonical_window_weeks)
        primary_summary_df = build_primary_summary(runtime, inventory_df)
        item6_sensitivity_df, item6_decision_df = build_item6_loss_sensitivity(runtime, canonical_window_weeks, sensitivity_weeks)
        comparable_sensitivity_df, comparable_champions_df, comparable_stability_df = build_comparable_window_tables(runtime, canonical_window_weeks, sensitivity_weeks)
        (
            cross_arm_protocol_df,
            cross_arm_scope_df,
            cross_arm_sensitivity_df,
            cross_arm_arm_champions_df,
            cross_arm_window_champions_df,
            cross_arm_leadership_stability_df,
            cross_arm_decision_summary_df,
        ) = build_cross_arm_tables(runtime, inventory_df, canonical_window_weeks, sensitivity_weeks)

        base.materialize_dataframe(runtime["con"], inventory_df, "table_5_16_model_inventory", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], primary_summary_df, "table_5_16_model_primary_summary", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], item6_sensitivity_df, "table_5_16_item6_loss_sensitivity", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], item6_decision_df, "table_5_16_item6_decision_summary", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], comparable_sensitivity_df, "table_5_16_comparable_window_sensitivity", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], comparable_champions_df, "table_5_16_comparable_window_champions", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], comparable_stability_df, "table_5_16_comparable_window_leadership_stability", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_protocol_df, "table_5_16_cross_arm_parity_protocol", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_scope_df, "table_5_16_cross_arm_execution_scope", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_sensitivity_df, "table_5_16_cross_arm_window_sensitivity", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_arm_champions_df, "table_5_16_cross_arm_arm_champions", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_window_champions_df, "table_5_16_cross_arm_window_champions", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_leadership_stability_df, "table_5_16_cross_arm_window_leadership_stability", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])
        base.materialize_dataframe(runtime["con"], cross_arm_decision_summary_df, "table_5_16_cross_arm_decision_summary", STAGE_ID, NOTEBOOK_NAME, runtime["RUN_ID"])

        write_metadata(runtime, inventory_df, primary_summary_df, alias_events)
        print(f"[OK] {STAGE_ID} consolidated {int(primary_summary_df.shape[0])} predictive models.")
    finally:
        base.close_notebook_runtime(runtime)


if __name__ == "__main__":
    main()