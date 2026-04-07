from __future__ import annotations

"""
Production runtime-contract materialization module for benchmark stage D5.1.

What this file does:
- initializes the D-stage runtime deterministically from shared configuration artifacts
- validates the exported modeling contract created upstream by stage B
- materializes auditable DuckDB contract tables consumed by the D-stage workflow
- persists the stage D5.1 modeling configuration metadata artifact

Main processing purpose:
- provide a single, explicit source of truth for D-stage runtime and modeling configuration

Expected inputs:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json

Expected outputs:
- DuckDB table table_5_1_runtime_contract
- DuckDB table table_5_1_modeling_contract
- outputs_benchmark_survival/metadata/metadata_5_1_modeling_config.json

Main DuckDB tables used as inputs:
- pipeline_table_catalog

Main DuckDB tables created or updated as outputs:
- table_5_1_runtime_contract
- table_5_1_modeling_contract
- pipeline_table_catalog
- vw_pipeline_table_catalog_schema

Main file artifacts read or generated:
- benchmark_shared_config.toml
- benchmark_modeling_contract.toml
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/metadata_5_1_modeling_config.json

Failure policy:
- missing required artifacts or contract keys raise immediately
- no fallback values, silent defaults, or degraded execution paths are permitted
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


if sys.version_info >= (3, 11):
    import tomllib as toml_reader
else:
    import tomli as toml_reader


STAGE_PREFIX = "5.1"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_NAME = "dropout_bench_v3_D_5_1.ipynb"
PREVIEW_ROWS = 20
REQUIRED_SHARED_PATH_KEYS = [
    "data_dir",
    "output_dir",
    "tables_subdir",
    "metadata_subdir",
    "data_output_subdir",
    "duckdb_filename",
]
REQUIRED_MODELING_BENCHMARK_KEYS = [
    "seed",
    "test_size",
    "benchmark_horizons",
    "calibration_bins",
    "early_window_weeks",
    "main_enrollment_window_weeks",
    "early_window_sensitivity_weeks",
]
REQUIRED_MODELING_MODEL_KEYS = [
    "main_clicks_feature",
    "main_active_feature",
    "main_mean_clicks_feature",
]
REQUIRED_MODELING_FEATURE_KEYS = [
    "static_features",
    "temporal_features_discrete",
    "main_enrollment_window_features",
    "optional_comparable_window_features",
    "dynamic_arm_features_linear",
    "dynamic_arm_features_neural",
]
ITEM_6_DYNAMIC_STAGE_IDS = ["D5.2", "D5.3", "D5.6", "D5.7", "D5.8"]
ITEM_6_WEIGHTED_BASELINE_LABELS = {
    "D5.2": "official_baseline_vs_weighted_variant",
    "D5.3": "official_baseline_vs_weighted_variant",
    "D5.6": "official_baseline_vs_weighted_variant",
    "D5.7": "gb_weekly_hazard_unweighted_vs_weighted_variant",
    "D5.8": "official_baseline_vs_weighted_variant",
}


@dataclass
class PipelineContext:
    project_root: Path
    script_name: str
    notebook_name: str
    config_toml_path: Path
    modeling_contract_toml_path: Path
    run_metadata_path: Path
    output_dir: Path
    tables_dir: Path
    metadata_dir: Path
    data_output_dir: Path
    duckdb_path: Path
    run_id: str
    cpu_cores: int
    shared_config: dict[str, Any]
    shared_modeling_contract: dict[str, Any]
    run_metadata: dict[str, Any]
    con: Any


def log_stage_start(block_number: str, title: str) -> None:
    print(f"[START] {STAGE_PREFIX} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# ==============================================================")
    print(f"# {block_number} - {title}")
    print("# ==============================================================")


def log_stage_end(block_number: str) -> None:
    print(f"[END] {block_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_artifact(label: str, location: str) -> None:
    print(f"ARTIFACT | {label} | {location}")


def require_mapping_keys(mapping: dict[str, Any], required_keys: list[str], mapping_name: str) -> None:
    missing_keys = [key for key in required_keys if key not in mapping]
    if missing_keys:
        raise KeyError(f"{mapping_name} is missing required keys: {', '.join(missing_keys)}")


def require_list_of_strings(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")
    return list(value)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_table_audit(con: Any, table_name: str, label: str, preview_rows: int = PREVIEW_ROWS) -> None:
    row_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    preview_df = con.execute(f"SELECT * FROM {table_name} LIMIT {preview_rows}").fetchdf()
    column_count = int(len(preview_df.columns))
    print(f"[{label}]")
    print(f"table_name={table_name}")
    print(f"rows={row_count}, cols={column_count}")
    if preview_df.empty:
        print("[empty table]")
    else:
        print(preview_df.to_string(index=False))


def materialize_dataframe_table(
    ctx: PipelineContext,
    df: pd.DataFrame,
    table_name: str,
    block_number: str,
    label: str,
) -> None:
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_d_5_1_materialize_df__"
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
        notebook_name=ctx.notebook_name,
        cell_name=block_number,
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_table_audit(ctx.con, table_name, label=label)
    print_artifact(label, f"duckdb://{table_name}")


def initialize_context() -> PipelineContext:
    from dropout_bench_v3_A_2_runtime_config import configure_runtime_cpu_cores
    from util import close_duckdb_connection, ensure_pipeline_catalog, open_duckdb_connection

    # Inputs:
    # - benchmark_shared_config.toml
    # - benchmark_modeling_contract.toml
    # - outputs_benchmark_survival/metadata/run_metadata.json
    # Outputs:
    # - active DuckDB connection
    # - validated runtime context in memory
    # - pipeline catalog availability in DuckDB
    log_stage_start("5.1.1", "Lightweight runtime bootstrap")

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
        shared_config = toml_reader.load(file_obj)
    with open(modeling_contract_toml_path, "rb") as file_obj:
        shared_modeling_contract = toml_reader.load(file_obj)
    with open(run_metadata_path, "r", encoding="utf-8") as file_obj:
        run_metadata = json.load(file_obj)

    require_mapping_keys(shared_config, ["paths", "benchmark"], "benchmark_shared_config.toml")
    require_mapping_keys(shared_modeling_contract, ["benchmark", "modeling", "feature_contract"], "benchmark_modeling_contract.toml")
    require_mapping_keys(run_metadata, ["run_id"], "run_metadata.json")

    paths_config = shared_config["paths"]
    require_mapping_keys(paths_config, REQUIRED_SHARED_PATH_KEYS, "benchmark_shared_config.toml [paths]")

    benchmark_config = shared_modeling_contract["benchmark"]
    require_mapping_keys(benchmark_config, REQUIRED_MODELING_BENCHMARK_KEYS, "benchmark_modeling_contract.toml [benchmark]")
    cpu_cores = configure_runtime_cpu_cores(shared_config)

    output_dir = PROJECT_ROOT / str(paths_config["output_dir"])
    tables_dir = output_dir / str(paths_config["tables_subdir"])
    metadata_dir = output_dir / str(paths_config["metadata_subdir"])
    data_output_dir = output_dir / str(paths_config["data_output_subdir"])
    duckdb_path = output_dir / str(paths_config["duckdb_filename"])

    for directory_path in [output_dir, tables_dir, metadata_dir, data_output_dir]:
        directory_path.mkdir(parents=True, exist_ok=True)

    run_id = str(run_metadata["run_id"]).strip()
    if not run_id:
        raise ValueError("run_metadata.json contains an empty run_id.")

    con = open_duckdb_connection(duckdb_path)
    ensure_pipeline_catalog(con)

    ctx = PipelineContext(
        project_root=PROJECT_ROOT,
        script_name=SCRIPT_NAME,
        notebook_name=NOTEBOOK_NAME,
        config_toml_path=config_toml_path,
        modeling_contract_toml_path=modeling_contract_toml_path,
        run_metadata_path=run_metadata_path,
        output_dir=output_dir,
        tables_dir=tables_dir,
        metadata_dir=metadata_dir,
        data_output_dir=data_output_dir,
        duckdb_path=duckdb_path,
        run_id=run_id,
        cpu_cores=cpu_cores,
        shared_config=shared_config,
        shared_modeling_contract=shared_modeling_contract,
        run_metadata=run_metadata,
        con=con,
    )

    print(f"- SCRIPT_NAME: {ctx.script_name}")
    print(f"- NOTEBOOK_NAME: {ctx.notebook_name}")
    print(f"- RUN_ID: {ctx.run_id}")
    print(f"- CPU_CORES: {ctx.cpu_cores}")
    print(f"- DUCKDB_PATH: {ctx.duckdb_path}")
    print_artifact("shared_config", str(ctx.config_toml_path))
    print_artifact("modeling_contract", str(ctx.modeling_contract_toml_path))
    print_artifact("run_metadata", str(ctx.run_metadata_path))
    log_stage_end("5.1.1")
    return ctx


def build_contract_outputs(ctx: PipelineContext) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    # Inputs:
    # - benchmark_shared_config.toml
    # - benchmark_modeling_contract.toml
    # - outputs_benchmark_survival/metadata/run_metadata.json
    # Outputs:
    # - runtime contract dataframe in memory
    # - modeling contract dataframe in memory
    # - validated modeling configuration dictionary in memory
    # - metadata_5_1_modeling_config.json output path
    log_stage_start("5.1.2", "Validate the shared runtime and modeling contract")

    from dropout_bench_v3_D_00_common import build_calibration_contract, resolve_early_window_sensitivity_weeks

    benchmark_contract = ctx.shared_modeling_contract["benchmark"]
    modeling_contract = ctx.shared_modeling_contract["modeling"]
    feature_contract = ctx.shared_modeling_contract["feature_contract"]

    require_mapping_keys(modeling_contract, REQUIRED_MODELING_MODEL_KEYS, "benchmark_modeling_contract.toml [modeling]")
    require_mapping_keys(feature_contract, REQUIRED_MODELING_FEATURE_KEYS, "benchmark_modeling_contract.toml [feature_contract]")

    static_features = require_list_of_strings(feature_contract["static_features"], "feature_contract.static_features")
    temporal_features_discrete = require_list_of_strings(
        feature_contract["temporal_features_discrete"],
        "feature_contract.temporal_features_discrete",
    )
    main_enrollment_window_features = require_list_of_strings(
        feature_contract["main_enrollment_window_features"],
        "feature_contract.main_enrollment_window_features",
    )
    optional_comparable_window_features = require_list_of_strings(
        feature_contract["optional_comparable_window_features"],
        "feature_contract.optional_comparable_window_features",
    )
    dynamic_arm_features_linear = require_list_of_strings(
        feature_contract["dynamic_arm_features_linear"],
        "feature_contract.dynamic_arm_features_linear",
    )
    dynamic_arm_features_neural = require_list_of_strings(
        feature_contract["dynamic_arm_features_neural"],
        "feature_contract.dynamic_arm_features_neural",
    )
    early_window_sensitivity_weeks = resolve_early_window_sensitivity_weeks(benchmark_contract)
    calibration_contract = build_calibration_contract(benchmark_contract)
    selected_loss_variant_by_stage = {
        stage_id: "weighted" for stage_id in ITEM_6_DYNAMIC_STAGE_IDS
    }
    item_6_protocol = {
        "comparison_rule": "official_baseline_vs_weighted_variant_same_split_same_features_same_evaluation",
        "promotion_rule": "promote_only_if_weighted_variant_consistently_outperforms_official_baseline",
        "tuning_policy": "full_variant_specific_retuning_required",
        "stage_weighting_protocol": ITEM_6_WEIGHTED_BASELINE_LABELS,
    }

    runtime_contract_df = pd.DataFrame(
        [
            {"config_key": "run_id", "config_value": ctx.run_id, "source": "run_metadata.json"},
            {"config_key": "duckdb_path", "config_value": str(ctx.duckdb_path), "source": "benchmark_shared_config.toml"},
            {"config_key": "output_dir", "config_value": str(ctx.output_dir), "source": "benchmark_shared_config.toml"},
            {"config_key": "tables_dir", "config_value": str(ctx.tables_dir), "source": "benchmark_shared_config.toml"},
            {"config_key": "metadata_dir", "config_value": str(ctx.metadata_dir), "source": "benchmark_shared_config.toml"},
        ]
    )

    modeling_contract_df = pd.DataFrame(
        [
            {"config_key": "random_seed", "config_value": str(int(benchmark_contract["seed"])), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "test_size", "config_value": str(float(benchmark_contract["test_size"])), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "benchmark_horizons", "config_value": " | ".join(str(value) for value in calibration_contract["benchmark_horizons"]), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "calibration_bins", "config_value": str(int(calibration_contract["calibration_bins"])), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "early_window_weeks", "config_value": str(int(benchmark_contract["early_window_weeks"])), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "main_enrollment_window_weeks", "config_value": str(int(benchmark_contract["main_enrollment_window_weeks"])), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "early_window_sensitivity_weeks", "config_value": " | ".join(str(value) for value in early_window_sensitivity_weeks), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "main_clicks_feature", "config_value": str(modeling_contract["main_clicks_feature"]), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "main_active_feature", "config_value": str(modeling_contract["main_active_feature"]), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "main_mean_clicks_feature", "config_value": str(modeling_contract["main_mean_clicks_feature"]), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "static_features", "config_value": " | ".join(static_features), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "temporal_features_discrete", "config_value": " | ".join(temporal_features_discrete), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "main_enrollment_window_features", "config_value": " | ".join(main_enrollment_window_features), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "optional_comparable_window_features", "config_value": " | ".join(optional_comparable_window_features), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "dynamic_arm_features_linear", "config_value": " | ".join(dynamic_arm_features_linear), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "dynamic_arm_features_neural", "config_value": " | ".join(dynamic_arm_features_neural), "source": "benchmark_modeling_contract.toml"},
            {"config_key": "calibration_contract_version", "config_value": str(calibration_contract["calibration_contract_version"]), "source": "dropout_bench_v3_D_00_common.py"},
            {"config_key": "calibration_observed_risk_method", "config_value": str(calibration_contract["calibration_observed_risk_method"]), "source": "dropout_bench_v3_D_00_common.py"},
            {"config_key": "item_6_weighting_protocol", "config_value": json.dumps(item_6_protocol, ensure_ascii=False, sort_keys=True), "source": "dropout_bench_v3_D_01_contract_runtime_materialization.py"},
            {"config_key": "selected_official_window_weeks", "config_value": "pending", "source": "dropout_bench_v3_D_01_contract_runtime_materialization.py"},
            {"config_key": "selected_loss_variant_by_stage", "config_value": json.dumps(selected_loss_variant_by_stage, ensure_ascii=False, sort_keys=True), "source": "dropout_bench_v3_D_01_contract_runtime_materialization.py"},
        ]
    )

    modeling_config = {
        "random_seed": int(benchmark_contract["seed"]),
        "test_size": float(benchmark_contract["test_size"]),
        "benchmark_horizons": calibration_contract["benchmark_horizons"],
        "calibration_bins": int(calibration_contract["calibration_bins"]),
        "early_window_weeks": int(benchmark_contract["early_window_weeks"]),
        "main_enrollment_window_weeks": int(benchmark_contract["main_enrollment_window_weeks"]),
        "early_window_sensitivity_weeks": early_window_sensitivity_weeks,
        "main_clicks_feature": str(modeling_contract["main_clicks_feature"]),
        "main_active_feature": str(modeling_contract["main_active_feature"]),
        "main_mean_clicks_feature": str(modeling_contract["main_mean_clicks_feature"]),
        "static_features": static_features,
        "temporal_features_discrete": temporal_features_discrete,
        "main_enrollment_window_features": main_enrollment_window_features,
        "optional_comparable_window_features": optional_comparable_window_features,
        "dynamic_arm_features_linear": dynamic_arm_features_linear,
        "dynamic_arm_features_neural": dynamic_arm_features_neural,
        "calibration_contract_version": calibration_contract["calibration_contract_version"],
        "calibration_observed_risk_method": calibration_contract["calibration_observed_risk_method"],
        "item_6_weighting_protocol": item_6_protocol,
        "selected_official_window_weeks": "pending",
        "selected_loss_variant_by_stage": selected_loss_variant_by_stage,
    }
    metadata_output_path = ctx.metadata_dir / "metadata_5_1_modeling_config.json"

    print(runtime_contract_df.to_string(index=False))
    print(modeling_contract_df.to_string(index=False))
    print_artifact("modeling_config_metadata", str(metadata_output_path))
    log_stage_end("5.1.2")
    return runtime_contract_df, modeling_contract_df, modeling_config, metadata_output_path


def materialize_contract_tables(
    ctx: PipelineContext,
    runtime_contract_df: pd.DataFrame,
    modeling_contract_df: pd.DataFrame,
) -> None:
    # Inputs:
    # - runtime contract dataframe from block 5.1.2
    # - modeling contract dataframe from block 5.1.2
    # - active DuckDB connection and pipeline catalog
    # Outputs:
    # - DuckDB table table_5_1_runtime_contract
    # - DuckDB table table_5_1_modeling_contract
    # - updated pipeline_table_catalog and vw_pipeline_table_catalog_schema
    log_stage_start("5.1.3", "Materialize auditable D5.1 contract tables")

    materialize_dataframe_table(
        ctx,
        df=runtime_contract_df,
        table_name="table_5_1_runtime_contract",
        block_number="5.1.3",
        label="Stage 5.1.3 table_5_1_runtime_contract — Runtime configuration contract",
    )
    materialize_dataframe_table(
        ctx,
        df=modeling_contract_df,
        table_name="table_5_1_modeling_contract",
        block_number="5.1.3",
        label="Stage 5.1.3 table_5_1_modeling_contract — Modeling configuration contract",
    )
    log_stage_end("5.1.3")


def persist_metadata_artifact(ctx: PipelineContext, modeling_config: dict[str, Any], metadata_output_path: Path) -> None:
    # Inputs:
    # - validated modeling configuration dictionary from block 5.1.2
    # - metadata output path under outputs_benchmark_survival/metadata
    # Outputs:
    # - metadata_5_1_modeling_config.json
    # - auditable artifact location printed to stdout
    log_stage_start("5.1.4", "Persist the stage D5.1 metadata artifact")

    save_json(modeling_config, metadata_output_path)
    print_artifact("metadata_5_1_modeling_config", str(metadata_output_path))
    print(metadata_output_path.read_text(encoding="utf-8"))
    log_stage_end("5.1.4")


def close_context(ctx: PipelineContext) -> None:
    from util import close_duckdb_connection

    # Inputs:
    # - active DuckDB connection
    # Outputs:
    # - closed DuckDB connection
    log_stage_start("5.1.5", "Close the DuckDB runtime cleanly")
    ctx.con = close_duckdb_connection(ctx.con, checkpoint=True, quiet=False)
    log_stage_end("5.1.5")


def main() -> None:
    ctx = initialize_context()
    try:
        runtime_contract_df, modeling_contract_df, modeling_config, metadata_output_path = build_contract_outputs(ctx)
        materialize_contract_tables(ctx, runtime_contract_df, modeling_contract_df)
        persist_metadata_artifact(ctx, modeling_config, metadata_output_path)
    finally:
        close_context(ctx)


if __name__ == "__main__":
    main()