from __future__ import annotations

"""
Production bootstrap module for benchmark stage A.

Purpose:
- prepare the project runtime deterministically
- validate and materialize the raw OULAD sources into DuckDB
- build the enrollment backbone and the canonical survival-ready table

Input contract:
- content/studentInfo.csv
- content/studentRegistration.csv
- content/studentVle.csv
- content/courses.csv
- content/vle.csv
- content/studentAssessment.csv
- content/assessments.csv

Output contract:
- benchmark_shared_config.toml
- outputs_benchmark_survival/metadata/config.json
- outputs_benchmark_survival/metadata/environment_summary.json
- outputs_benchmark_survival/metadata/benchmark_shared_config.json
- outputs_benchmark_survival/metadata/run_metadata.json
- outputs_benchmark_survival/metadata/dataset_setup.json
- outputs_benchmark_survival/metadata/duckdb_setup.json
- DuckDB catalog objects and the stage A analytical tables

Failure policy:
- missing required dependencies raise immediately after install attempts
- broken dataset contracts raise immediately
- missing administrative censoring information raises immediately
- no silent degradation paths are permitted in this module
"""

import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


STAGE_PREFIX = "A"
SCRIPT_NAME = Path(__file__).name
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
SEED = 42
DATASET_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"
ZIP_FILENAME = "anonymisedData.zip"
EXPECTED_SOURCE_FILES = {
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentVle": "studentVle.csv",
    "courses": "courses.csv",
    "vle": "vle.csv",
    "studentAssessment": "studentAssessment.csv",
    "assessments": "assessments.csv",
}
ENROLLMENT_KEY = ["id_student", "code_module", "code_presentation"]
PREVIEW_ROWS = 20


@dataclass(frozen=True)
class PackageRequirement:
    install_spec: str
    distribution_name: str


def stage_label(number: str) -> str:
    return f"{STAGE_PREFIX}.{number}"


def log_stage_start(number: str, title: str) -> None:
    print(f"[START] {stage_label(number)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# ==============================================================")
    print(f"# {stage_label(number)} - {title}")
    print("# ==============================================================")


def log_stage_end(number: str) -> None:
    print(f"[END] {stage_label(number)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_kv(label: str, value: Any) -> None:
    print(f"- {label}: {value}")


def print_artifact(label: str, location: Any) -> None:
    print(f"ARTIFACT | {label} | {location}")


def print_dataframe_audit(label: str, name: str, df, preview_rows: int = PREVIEW_ROWS) -> None:
    preview = df.head(preview_rows)
    print(f"TABLE_LABEL | {label}")
    print(f"TABLE_NAME | {name}")
    print(f"TABLE_ROWS | {df.shape[0]}")
    print(f"TABLE_COLS | {df.shape[1]}")
    print("TABLE_COLUMNS | " + ", ".join(str(col) for col in df.columns))
    if preview.empty:
        print("TABLE_PREVIEW | [empty table]")
    else:
        print(preview.to_string(index=False))


def parse_distribution_name(requirement_spec: str) -> str:
    normalized = requirement_spec.split(";", 1)[0].strip()
    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", normalized)
    if match is None:
        raise ValueError(f"Invalid requirement entry: {requirement_spec!r}")
    return match.group(0)


def load_required_packages(requirements_path: Path) -> list[PackageRequirement]:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    requirements: list[PackageRequirement] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement_spec = raw_line.split("#", 1)[0].strip()
        if not requirement_spec:
            continue
        requirements.append(
            PackageRequirement(
                install_spec=requirement_spec,
                distribution_name=parse_distribution_name(requirement_spec),
            )
        )

    if not requirements:
        raise RuntimeError(f"Requirements file is empty: {requirements_path}")
    return requirements


def install_package(package_name: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package_name],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return result.returncode == 0, output


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def distribution_is_installed(distribution_name: str) -> bool:
    return package_version(distribution_name) != "not-installed"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def save_toml_sections(payload: dict[str, dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for section_name, section_values in payload.items():
        lines.append(f"[{section_name}]")
        for key, value in section_values.items():
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_zip_file(zip_path: Path) -> bool:
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            corrupt_member = zip_file.testzip()
        return corrupt_member is None
    except zipfile.BadZipFile:
        return False


def download_dataset_zip(url: str, zip_path: Path) -> None:
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(zip_path)
    if not validate_zip_file(zip_path):
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Downloaded zip file is invalid: {zip_path}")


def extract_dataset_zip(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip file not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)


def expected_source_paths(data_dir: Path) -> dict[str, Path]:
    return {table_name: data_dir / filename for table_name, filename in EXPECTED_SOURCE_FILES.items()}


def missing_source_files(data_dir: Path) -> list[str]:
    return [
        filename
        for filename in EXPECTED_SOURCE_FILES.values()
        if not (data_dir / filename).exists()
    ]


def package_environment_summary() -> dict[str, Any]:
    import torch

    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "duckdb_version": package_version("duckdb"),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "matplotlib_version": package_version("matplotlib"),
        "catboost_version": package_version("catboost"),
        "xgboost_version": package_version("xgboost"),
        "scikit_learn_version": package_version("scikit-learn"),
        "scikit_survival_version": package_version("scikit-survival"),
        "lifelines_version": package_version("lifelines"),
        "torch_version": package_version("torch"),
        "torchtuples_version": package_version("torchtuples"),
        "pycox_version": package_version("pycox"),
        "requests_version": package_version("requests"),
        "cuda_available": torch.cuda.is_available(),
        "seed": SEED,
    }


@dataclass
class PipelineContext:
    project_root: Path
    script_name: str
    seed: int
    config: dict[str, Any]
    shared_config: dict[str, Any]
    config_toml_path: Path
    data_dir: Path
    output_dir: Path
    data_output_dir: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    metadata_dir: Path
    duckdb_path: Path
    run_metadata_path: Path
    run_metadata: dict[str, Any]
    run_id: str
    env_summary: dict[str, Any]
    dataset_setup_metadata: dict[str, Any] | None = None
    con: Any = None


def require_columns(df, required_columns: list[str], table_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {', '.join(missing)}")


def materialize_dataframe_table(ctx: PipelineContext, df, table_name: str, stage_id: str, label: str) -> None:
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    temp_view_name = "__stage_a_materialize_df__"
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


def stage_a1_runtime_bootstrap() -> None:
    log_stage_start("1", "Runtime Bootstrap")
    print_kv("SCRIPT_NAME", SCRIPT_NAME)
    print_kv("SCRIPT_PATH", SCRIPT_PATH)
    print_kv("PROJECT_ROOT", PROJECT_ROOT)
    print_kv("PYTHON_EXECUTABLE", sys.executable)
    log_stage_end("1")


def stage_a2_dependency_bootstrap() -> None:
    log_stage_start("2", "Dependency Validation and Install")
    required_packages = load_required_packages(REQUIREMENTS_PATH)
    missing_requirements = [
        requirement
        for requirement in required_packages
        if not distribution_is_installed(requirement.distribution_name)
    ]
    if not missing_requirements:
        print("All required packages are already available.")
        log_stage_end("2")
        return

    install_failures: list[tuple[PackageRequirement, str]] = []
    for requirement in missing_requirements:
        print(f"Installing required package: {requirement.install_spec}")
        success, output = install_package(requirement.install_spec)
        if not success:
            install_failures.append((requirement, output))

    still_missing = [
        requirement
        for requirement in required_packages
        if not distribution_is_installed(requirement.distribution_name)
    ]
    if still_missing:
        print("Required dependency installation failed.")
        for requirement, output in install_failures:
            print(f"FAILED_PACKAGE | {requirement.distribution_name} | {requirement.install_spec}")
            print(output.splitlines()[-1] if output else "[no output]")
        raise RuntimeError("Stage A.2 could not provision all required dependencies.")

    print("All required packages are available after installation.")
    log_stage_end("2")


def stage_a3_import_runtime_stack() -> tuple[Any, Any, Any]:
    log_stage_start("3", "Runtime Imports and Determinism")
    import numpy as np
    import pandas as pd
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    warnings.simplefilter("default")
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    print_kv("SEED", SEED)
    print_kv("CUDA_AVAILABLE", torch.cuda.is_available())
    print_kv("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"))
    log_stage_end("3")
    return np, pd, torch


def stage_a4_prepare_context(np, pd, torch) -> PipelineContext:
    log_stage_start("4", "Configuration and Execution Context")
    from util import load_shared_config, resolve_project_path

    config_toml_path = PROJECT_ROOT / "benchmark_shared_config.toml"
    existing_shared_config = {}
    if config_toml_path.exists():
        existing_shared_config, _ = load_shared_config(PROJECT_ROOT)

    shared_config = {
        "paths": {
            "data_dir": "content",
            "output_dir": "outputs_benchmark_survival",
            "tables_subdir": "tables",
            "figures_subdir": "figures",
            "models_subdir": "models",
            "metadata_subdir": "metadata",
            "data_output_subdir": "data",
            "duckdb_filename": "benchmark_survival.duckdb",
        },
        "benchmark": {
            "seed": SEED,
            "test_size": 0.30,
            "temporal_buckets_q": 4,
            "early_window_weeks": 4,
            "main_enrollment_window_weeks": 4,
            "unit_of_analysis": "enrollment",
            "time_granularity": "weekly",
            "event_definition": "Withdrawn with valid date_unregistration",
        },
        "keys": {
            "enrollment_key": ENROLLMENT_KEY,
        },
    }
    if isinstance(existing_shared_config.get("runtime"), dict):
        shared_config["runtime"] = dict(existing_shared_config["runtime"])
    save_toml_sections(shared_config, config_toml_path)
    print_artifact("shared_config_toml", config_toml_path)

    loaded_shared_config, loaded_config_path = load_shared_config(PROJECT_ROOT)
    paths_cfg = loaded_shared_config.get("paths", {})
    data_dir = resolve_project_path(PROJECT_ROOT, paths_cfg.get("data_dir", "content"))
    output_dir = resolve_project_path(PROJECT_ROOT, paths_cfg.get("output_dir", "outputs_benchmark_survival"))
    data_output_dir = output_dir / paths_cfg.get("data_output_subdir", "data")
    tables_dir = output_dir / paths_cfg.get("tables_subdir", "tables")
    figures_dir = output_dir / paths_cfg.get("figures_subdir", "figures")
    models_dir = output_dir / paths_cfg.get("models_subdir", "models")
    metadata_dir = output_dir / paths_cfg.get("metadata_subdir", "metadata")
    duckdb_path = output_dir / paths_cfg.get("duckdb_filename", "benchmark_survival.duckdb")
    run_metadata_path = metadata_dir / "run_metadata.json"

    now = datetime.now(timezone.utc).astimezone().isoformat()
    run_metadata = {
        "run_id": uuid4().hex,
        "created_at": now,
        "last_seen_at": now,
        "created_by_notebook": SCRIPT_NAME,
        "last_notebook": SCRIPT_NAME,
    }
    config = {
        "seed": SEED,
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "data_output_dir": str(data_output_dir),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "models_dir": str(models_dir),
        "metadata_dir": str(metadata_dir),
        "unit_of_analysis": "enrollment",
        "enrollment_key": ENROLLMENT_KEY,
        "time_granularity": "weekly",
        "event_definition": "Withdrawn with valid date_unregistration",
        "test_size": 0.30,
        "temporal_buckets_q": 4,
        "script_name": SCRIPT_NAME,
    }

    ctx = PipelineContext(
        project_root=PROJECT_ROOT,
        script_name=SCRIPT_NAME,
        seed=SEED,
        config=config,
        shared_config=loaded_shared_config,
        config_toml_path=loaded_config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        data_output_dir=data_output_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        metadata_dir=metadata_dir,
        duckdb_path=duckdb_path,
        run_metadata_path=run_metadata_path,
        run_metadata=run_metadata,
        run_id=run_metadata["run_id"],
        env_summary=package_environment_summary(),
    )

    print_kv("CONFIG_TOML_PATH", ctx.config_toml_path)
    print_kv("DATA_DIR", ctx.data_dir)
    print_kv("OUTPUT_DIR", ctx.output_dir)
    print_kv("DUCKDB_PATH", ctx.duckdb_path)
    print_kv("RUN_ID", ctx.run_id)
    print_dataframe_audit(
        label="environment_summary",
        name="environment_summary",
        df=pd.DataFrame(sorted(ctx.env_summary.items()), columns=["key", "value"]),
    )
    log_stage_end("4")
    return ctx


def stage_a5_ensure_dataset_contract(ctx: PipelineContext) -> None:
    log_stage_start("5", "Dataset Acquisition and Validation")
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ctx.project_root / ZIP_FILENAME
    missing_before = missing_source_files(ctx.data_dir)

    if missing_before:
        if zip_path.exists() and not validate_zip_file(zip_path):
            print(f"Removing invalid dataset zip: {zip_path}")
            zip_path.unlink()
        if not zip_path.exists():
            download_dataset_zip(DATASET_URL, zip_path)
            print_artifact("dataset_zip", zip_path)
        extract_dataset_zip(zip_path, ctx.data_dir)

    missing_after = missing_source_files(ctx.data_dir)
    if missing_after:
        raise FileNotFoundError(
            "Missing OULAD source files after dataset setup: " + ", ".join(missing_after)
        )

    source_paths = expected_source_paths(ctx.data_dir)
    ctx.dataset_setup_metadata = {
        "dataset_url": DATASET_URL,
        "zip_filename": ZIP_FILENAME,
        "zip_path": str(zip_path),
        "data_dir": str(ctx.data_dir),
        "expected_csvs": list(EXPECTED_SOURCE_FILES.values()),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "resolved_source_files": {key: str(path) for key, path in source_paths.items()},
    }

    import pandas as pd

    dataset_contract_df = pd.DataFrame(
        [
            {"table_name": table_name, "csv_path": str(path), "exists": path.exists()}
            for table_name, path in source_paths.items()
        ]
    ).sort_values("table_name").reset_index(drop=True)
    print_dataframe_audit("dataset_contract", "dataset_contract", dataset_contract_df)
    log_stage_end("5")


def stage_a6_reset_outputs(ctx: PipelineContext) -> None:
    log_stage_start("6", "Output Reset")
    from util import close_duckdb_connection

    if ctx.con is not None:
        ctx.con = close_duckdb_connection(ctx.con, checkpoint=False, quiet=False)

    if ctx.output_dir.exists():
        shutil.rmtree(ctx.output_dir)
        print(f"Removed output directory: {ctx.output_dir}")
    else:
        print(f"Output directory already absent: {ctx.output_dir}")
    log_stage_end("6")


def stage_a7_initialize_outputs_and_duckdb(ctx: PipelineContext) -> None:
    log_stage_start("7", "Output Tree, Metadata, and DuckDB Bootstrap")
    from util import ensure_pipeline_catalog, open_duckdb_connection, refresh_pipeline_catalog_schema_view

    required_dirs = [
        ctx.output_dir,
        ctx.data_output_dir,
        ctx.tables_dir,
        ctx.tables_dir / "paper_main",
        ctx.tables_dir / "paper_appendix",
        ctx.figures_dir,
        ctx.models_dir,
        ctx.metadata_dir,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print_artifact("directory", directory)

    ctx.run_metadata["last_notebook"] = ctx.script_name
    ctx.run_metadata["last_seen_at"] = datetime.now(timezone.utc).astimezone().isoformat()

    save_json(ctx.config, ctx.metadata_dir / "config.json")
    save_json(ctx.env_summary, ctx.metadata_dir / "environment_summary.json")
    save_json(ctx.shared_config, ctx.metadata_dir / "benchmark_shared_config.json")
    save_json(ctx.run_metadata, ctx.run_metadata_path)
    if ctx.dataset_setup_metadata is None:
        raise RuntimeError("Dataset metadata must be prepared before DuckDB bootstrap.")
    save_json(ctx.dataset_setup_metadata, ctx.metadata_dir / "dataset_setup.json")

    print_artifact("config_json", ctx.metadata_dir / "config.json")
    print_artifact("environment_summary_json", ctx.metadata_dir / "environment_summary.json")
    print_artifact("shared_config_json", ctx.metadata_dir / "benchmark_shared_config.json")
    print_artifact("run_metadata_json", ctx.run_metadata_path)
    print_artifact("dataset_setup_json", ctx.metadata_dir / "dataset_setup.json")

    ctx.con = open_duckdb_connection(ctx.duckdb_path)
    ensure_pipeline_catalog(ctx.con)
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_artifact("duckdb_database", ctx.duckdb_path)
    log_stage_end("7")


def stage_a8_register_source_views(ctx: PipelineContext, pd) -> None:
    log_stage_start("8", "DuckDB Source Registration")
    source_paths = expected_source_paths(ctx.data_dir)
    for table_name, csv_path in source_paths.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Required source file does not exist: {csv_path}")
        ctx.con.execute(
            f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT *
            FROM read_csv_auto('{csv_path.as_posix()}', HEADER=TRUE)
            """
        )

    registry_rows = []
    for table_name, csv_path in source_paths.items():
        describe_df = ctx.con.execute(f"DESCRIBE {table_name}").fetchdf()
        registry_rows.append(
            {
                "view_name": table_name,
                "n_rows": int(ctx.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]),
                "n_cols": int(describe_df.shape[0]),
                "columns": ", ".join(describe_df["column_name"].astype(str).tolist()),
                "source_csv": str(csv_path),
            }
        )
    registry_df = pd.DataFrame(registry_rows).sort_values("view_name").reset_index(drop=True)
    materialize_dataframe_table(
        ctx=ctx,
        df=registry_df,
        table_name="table_duckdb_registered_views",
        stage_id=stage_label("8"),
        label="registered_views_summary",
    )

    duckdb_setup_metadata = {
        "run_id": ctx.run_id,
        "duckdb_path": str(ctx.duckdb_path),
        "registered_views": sorted(source_paths.keys()),
        "source_files": {name: str(path) for name, path in source_paths.items()},
        "catalog_table": "pipeline_table_catalog",
        "catalog_view": "vw_pipeline_table_catalog_schema",
    }
    save_json(duckdb_setup_metadata, ctx.metadata_dir / "duckdb_setup.json")
    print_artifact("duckdb_setup_json", ctx.metadata_dir / "duckdb_setup.json")
    log_stage_end("8")


def stage_a9_backbone_raw_audit(ctx: PipelineContext, np, pd) -> None:
    log_stage_start("9", "Backbone-Oriented Raw Audit")
    audit_tables = {
        "studentInfo": {
            "role": "backbone_core",
            "critical_columns": ["id_student", "code_module", "code_presentation", "final_result"],
        },
        "studentRegistration": {
            "role": "backbone_core",
            "critical_columns": [
                "id_student",
                "code_module",
                "code_presentation",
                "date_registration",
                "date_unregistration",
            ],
        },
        "studentVle": {
            "role": "transactional_minimal",
            "critical_columns": ["id_student", "code_module", "code_presentation", "id_site", "date", "sum_click"],
        },
        "studentAssessment": {
            "role": "transactional_optional",
            "critical_columns": ["id_student", "code_module", "code_presentation", "id_assessment", "date_submitted"],
        },
        "assessments": {
            "role": "transactional_optional",
            "critical_columns": ["id_assessment", "code_module", "code_presentation"],
        },
        "courses": {
            "role": "auxiliary",
            "critical_columns": ["code_module", "code_presentation", "module_presentation_length"],
        },
        "vle": {
            "role": "auxiliary",
            "critical_columns": ["id_site", "code_module", "code_presentation"],
        },
    }

    structural_rows = []
    missing_rows = []
    uniqueness_rows = []

    for table_name, table_meta in audit_tables.items():
        describe_df = ctx.con.execute(f"DESCRIBE {table_name}").fetchdf()
        columns = describe_df["column_name"].astype(str).tolist()
        critical_columns = table_meta["critical_columns"]
        present_critical = [column for column in critical_columns if column in columns]
        missing_critical = [column for column in critical_columns if column not in columns]
        structural_rows.append(
            {
                "table_name": table_name,
                "role": table_meta["role"],
                "n_rows": int(ctx.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]),
                "n_cols": int(len(columns)),
                "critical_columns_expected": ", ".join(critical_columns),
                "critical_columns_present": ", ".join(present_critical),
                "critical_columns_missing": ", ".join(missing_critical),
                "all_columns": ", ".join(columns),
            }
        )
        for column_name in critical_columns:
            if column_name not in columns:
                missing_rows.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "exists_in_table": False,
                        "n_total": None,
                        "n_missing": None,
                        "pct_missing": None,
                    }
                )
                continue
            stats = ctx.con.execute(
                f"""
                SELECT
                    COUNT(*) AS n_total,
                    SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) AS n_missing
                FROM {table_name}
                """
            ).fetchdf().iloc[0]
            n_total = int(stats["n_total"])
            n_missing = int(stats["n_missing"]) if pd.notna(stats["n_missing"]) else 0
            missing_rows.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "exists_in_table": True,
                    "n_total": n_total,
                    "n_missing": n_missing,
                    "pct_missing": (n_missing / n_total * 100.0) if n_total else np.nan,
                }
            )

    for table_name in ["studentInfo", "studentRegistration"]:
        total_rows = int(ctx.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        distinct_rows = int(
            ctx.con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT id_student, code_module, code_presentation
                    FROM {table_name}
                ) AS dedup
                """
            ).fetchone()[0]
        )
        uniqueness_rows.append(
            {
                "table_name": table_name,
                "keys_checked": ", ".join(ENROLLMENT_KEY),
                "n_rows_total": total_rows,
                "n_distinct_by_keys": distinct_rows,
                "n_excess_duplicate_rows": total_rows - distinct_rows,
                "is_unique_by_keys": total_rows == distinct_rows,
            }
        )

    materialize_dataframe_table(
        ctx,
        pd.DataFrame(structural_rows).sort_values(["role", "table_name"]).reset_index(drop=True),
        "table_oulad_backbone_oriented_audit",
        stage_label("9"),
        "backbone_structural_audit",
    )
    materialize_dataframe_table(
        ctx,
        pd.DataFrame(missing_rows).sort_values(["table_name", "column_name"]).reset_index(drop=True),
        "table_oulad_critical_missing_audit",
        stage_label("9"),
        "critical_missing_audit",
    )
    materialize_dataframe_table(
        ctx,
        pd.DataFrame(uniqueness_rows).reset_index(drop=True),
        "table_oulad_backbone_key_uniqueness",
        stage_label("9"),
        "backbone_key_uniqueness",
    )
    log_stage_end("9")


def stage_a10_build_enrollment_backbone(ctx: PipelineContext, pd) -> None:
    log_stage_start("10", "Enrollment Backbone Materialization")
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_backbone")
    ctx.con.execute(
        """
        CREATE TABLE enrollment_backbone AS
        SELECT
            si.id_student,
            si.code_module,
            si.code_presentation,
            si.gender,
            si.region,
            si.highest_education,
            si.imd_band,
            si.age_band,
            si.num_of_prev_attempts,
            si.studied_credits,
            si.disability,
            si.final_result,
            sr.date_registration,
            sr.date_unregistration
        FROM studentInfo AS si
        LEFT JOIN studentRegistration AS sr
          ON si.id_student = sr.id_student
         AND si.code_module = sr.code_module
         AND si.code_presentation = sr.code_presentation
        """
    )
    register_duckdb_table(
        con=ctx.con,
        table_name="enrollment_backbone",
        notebook_name=ctx.script_name,
        cell_name=stage_label("10"),
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)

    backbone_df = ctx.con.execute("SELECT * FROM enrollment_backbone").fetchdf()
    print_dataframe_audit("enrollment_backbone_preview", "enrollment_backbone", backbone_df.head(PREVIEW_ROWS))
    print(f"TABLE_TOTAL_ROWS | {backbone_df.shape[0]}")
    print_artifact("enrollment_backbone", "duckdb://enrollment_backbone")

    n_rows_total = int(backbone_df.shape[0])
    n_distinct_keys = int(
        (
            backbone_df["id_student"].astype(str)
            + "||"
            + backbone_df["code_module"].astype(str)
            + "||"
            + backbone_df["code_presentation"].astype(str)
        ).nunique()
    )
    date_missing_audit = ctx.con.execute(
        """
        SELECT
            COUNT(*) AS n_total,
            SUM(CASE WHEN date_registration IS NULL THEN 1 ELSE 0 END) AS n_missing_date_registration,
            SUM(CASE WHEN date_unregistration IS NULL THEN 1 ELSE 0 END) AS n_missing_date_unregistration
        FROM enrollment_backbone
        """
    ).fetchdf().iloc[0]
    audit_df = pd.DataFrame(
        [
            {
                "table_name": "enrollment_backbone",
                "n_rows_total": n_rows_total,
                "n_distinct_enrollment_keys": n_distinct_keys,
                "is_unique_by_enrollment_key": n_rows_total == n_distinct_keys,
                "n_missing_date_registration": int(date_missing_audit["n_missing_date_registration"]),
                "pct_missing_date_registration": float(date_missing_audit["n_missing_date_registration"] / date_missing_audit["n_total"] * 100.0),
                "n_missing_date_unregistration": int(date_missing_audit["n_missing_date_unregistration"]),
                "pct_missing_date_unregistration": float(date_missing_audit["n_missing_date_unregistration"] / date_missing_audit["n_total"] * 100.0),
            }
        ]
    )
    final_result_dist = (
        backbone_df.groupby("final_result", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    final_result_dist["pct"] = final_result_dist["n"] / final_result_dist["n"].sum() * 100.0

    materialize_dataframe_table(
        ctx,
        audit_df,
        "table_enrollment_backbone_audit",
        stage_label("10"),
        "enrollment_backbone_audit",
    )
    print_dataframe_audit("final_result_distribution", "final_result_distribution", final_result_dist)
    log_stage_end("10")


def stage_a11_build_survival_ready(ctx: PipelineContext, np, pd) -> None:
    log_stage_start("11", "Enrollment Survival-Ready Construction")
    from util import refresh_pipeline_catalog_schema_view, register_duckdb_table

    ctx.con.execute(
        """
        CREATE OR REPLACE TEMP VIEW tmp_vle_enrollment_agg AS
        SELECT
            id_student,
            code_module,
            code_presentation,
            MAX(date) AS max_vle_day,
            COUNT(*) AS n_vle_rows,
            COALESCE(SUM(sum_click), 0) AS total_clicks_all_time,
            CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END AS has_any_vle_activity
        FROM studentVle
        GROUP BY id_student, code_module, code_presentation
        """
    )
    ctx.con.execute(
        """
        CREATE OR REPLACE TEMP VIEW enrollment_survival_ready_raw AS
        SELECT
            eb.*,
            c.module_presentation_length,
            COALESCE(v.has_any_vle_activity, 0) AS has_any_vle_activity,
            v.max_vle_day,
            COALESCE(v.n_vle_rows, 0) AS n_vle_rows,
            COALESCE(v.total_clicks_all_time, 0) AS total_clicks_all_time
        FROM enrollment_backbone AS eb
        LEFT JOIN courses AS c
          ON eb.code_module = c.code_module
         AND eb.code_presentation = c.code_presentation
        LEFT JOIN tmp_vle_enrollment_agg AS v
          ON eb.id_student = v.id_student
         AND eb.code_module = v.code_module
         AND eb.code_presentation = v.code_presentation
        """
    )

    base_df = ctx.con.execute("SELECT * FROM enrollment_survival_ready_raw").fetchdf()
    require_columns(
        base_df,
        [
            "id_student",
            "code_module",
            "code_presentation",
            "final_result",
            "date_registration",
            "date_unregistration",
            "module_presentation_length",
            "has_any_vle_activity",
            "max_vle_day",
            "n_vle_rows",
            "total_clicks_all_time",
        ],
        "enrollment_survival_ready_raw",
    )

    df = base_df.copy()
    df["enrollment_id"] = (
        df["id_student"].astype(str)
        + "||"
        + df["code_module"].astype(str)
        + "||"
        + df["code_presentation"].astype(str)
    )

    date_unregistration_num = pd.to_numeric(df["date_unregistration"], errors="coerce")
    max_vle_day_num = pd.to_numeric(df["max_vle_day"], errors="coerce")
    module_length_days = pd.to_numeric(df["module_presentation_length"], errors="coerce")
    if module_length_days.isna().any():
        missing_rows = int(module_length_days.isna().sum())
        raise RuntimeError(f"Missing module_presentation_length for {missing_rows} enrollment rows.")
    if (module_length_days < 0).any():
        raise RuntimeError("Negative module_presentation_length values are not allowed.")

    final_result_clean = df["final_result"].astype(str).str.strip().str.upper()
    df["is_withdrawn"] = (final_result_clean == "WITHDRAWN").astype(int)
    df["has_valid_unregistration_date"] = (
        date_unregistration_num.notna() & (date_unregistration_num >= 0)
    ).astype(int)
    df["event_observed"] = (
        (df["is_withdrawn"] == 1) & (df["has_valid_unregistration_date"] == 1)
    ).astype(int)
    df["is_withdrawn_without_valid_unregistration"] = (
        (df["is_withdrawn"] == 1) & (df["event_observed"] == 0)
    ).astype(int)

    df["module_presentation_length_days"] = module_length_days.astype(int)
    df["t_administrative_censor_week"] = np.floor(module_length_days / 7.0).astype(int)
    df["t_event_week"] = np.where(
        df["event_observed"] == 1,
        np.floor(date_unregistration_num / 7.0),
        np.nan,
    )
    df["t_last_vle_week_raw"] = np.where(max_vle_day_num.notna(), np.floor(max_vle_day_num / 7.0), np.nan)
    df["has_pre_start_only_activity"] = (
        (pd.to_numeric(df["has_any_vle_activity"], errors="coerce").fillna(0).astype(int) == 1)
        & max_vle_day_num.notna()
        & (max_vle_day_num < 0)
    ).astype(int)
    df["t_last_obs_week_raw"] = df["t_last_vle_week_raw"]
    df["t_last_obs_week"] = np.where(
        df["event_observed"] == 1,
        np.floor(date_unregistration_num / 7.0),
        df["t_administrative_censor_week"],
    )
    df["t_last_obs_week"] = pd.to_numeric(df["t_last_obs_week"], errors="coerce")
    if df["t_last_obs_week"].isna().any():
        raise RuntimeError("Stage A.11 produced missing t_last_obs_week values.")
    if (df["t_last_obs_week"] < 0).any():
        raise RuntimeError("Stage A.11 produced negative t_last_obs_week values.")
    df["t_last_obs_week"] = df["t_last_obs_week"].astype(int)
    df["t_final_week"] = df["t_last_obs_week"].astype(int)
    df["used_zero_week_fallback_for_censoring"] = 0
    df["t_last_obs_week_was_clamped"] = 0
    df["t_final_week_was_clamped"] = 0
    df["censoring_strategy"] = np.where(
        df["event_observed"] == 1,
        "observed_withdrawal",
        "administrative_course_end",
    )

    priority_columns = [
        "enrollment_id",
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "date_registration",
        "date_unregistration",
        "module_presentation_length",
        "module_presentation_length_days",
        "is_withdrawn",
        "has_valid_unregistration_date",
        "event_observed",
        "is_withdrawn_without_valid_unregistration",
        "has_any_vle_activity",
        "max_vle_day",
        "n_vle_rows",
        "total_clicks_all_time",
        "t_event_week",
        "t_last_vle_week_raw",
        "t_last_obs_week_raw",
        "t_administrative_censor_week",
        "t_last_obs_week",
        "t_final_week",
        "has_pre_start_only_activity",
        "used_zero_week_fallback_for_censoring",
        "t_last_obs_week_was_clamped",
        "t_final_week_was_clamped",
        "censoring_strategy",
    ]
    remaining_columns = [column for column in df.columns if column not in priority_columns]
    df = df[priority_columns + remaining_columns]

    ctx.con.execute("DROP TABLE IF EXISTS enrollment_survival_ready")
    ctx.con.register("__stage_a_survival_ready_df__", df)
    ctx.con.execute(
        """
        CREATE TABLE enrollment_survival_ready AS
        SELECT *
        FROM __stage_a_survival_ready_df__
        """
    )
    ctx.con.unregister("__stage_a_survival_ready_df__")
    register_duckdb_table(
        con=ctx.con,
        table_name="enrollment_survival_ready",
        notebook_name=ctx.script_name,
        cell_name=stage_label("11"),
        run_id=ctx.run_id,
    )
    refresh_pipeline_catalog_schema_view(ctx.con)
    print_dataframe_audit("enrollment_survival_ready_preview", "enrollment_survival_ready", df.head(PREVIEW_ROWS))
    print(f"TABLE_TOTAL_ROWS | {df.shape[0]}")
    print_artifact("enrollment_survival_ready", "duckdb://enrollment_survival_ready")

    audit_df = pd.DataFrame(
        [
            {
                "table_name": "enrollment_survival_ready",
                "n_rows_total": int(df.shape[0]),
                "n_unique_enrollments": int(df["enrollment_id"].nunique()),
                "is_unique_by_enrollment": bool(df["enrollment_id"].nunique() == df.shape[0]),
                "n_withdrawn": int(df["is_withdrawn"].sum()),
                "n_event_observed": int(df["event_observed"].sum()),
                "event_rate_over_all_enrollments": float(df["event_observed"].mean()),
                "n_withdrawn_without_valid_unregistration": int(df["is_withdrawn_without_valid_unregistration"].sum()),
                "n_no_vle_activity": int((pd.to_numeric(df["has_any_vle_activity"], errors="coerce").fillna(0).astype(int) == 0).sum()),
                "n_pre_start_only_activity": int(df["has_pre_start_only_activity"].sum()),
                "n_zero_week_fallback_for_censoring": 0,
                "n_administrative_censoring_rows": int((df["censoring_strategy"] == "administrative_course_end").sum()),
                "max_t_event_week": float(pd.to_numeric(df["t_event_week"], errors="coerce").dropna().max()) if pd.to_numeric(df["t_event_week"], errors="coerce").notna().any() else np.nan,
                "max_t_last_obs_week": float(pd.to_numeric(df["t_last_obs_week"], errors="coerce").max()),
                "max_t_final_week": float(pd.to_numeric(df["t_final_week"], errors="coerce").max()),
            }
        ]
    )
    final_result_by_event_df = (
        df.groupby(["final_result", "event_observed"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["final_result", "event_observed"], ascending=[True, True])
        .reset_index(drop=True)
    )
    final_result_by_event_df["pct_within_final_result"] = (
        final_result_by_event_df["n"]
        / final_result_by_event_df.groupby("final_result")["n"].transform("sum")
        * 100.0
    )

    materialize_dataframe_table(
        ctx,
        audit_df,
        "table_enrollment_survival_ready_audit",
        stage_label("11"),
        "enrollment_survival_ready_audit",
    )
    materialize_dataframe_table(
        ctx,
        final_result_by_event_df,
        "table_enrollment_survival_ready_final_result_by_event",
        stage_label("11"),
        "enrollment_survival_ready_final_result_by_event",
    )
    log_stage_end("11")


def stage_a12_shutdown(ctx: PipelineContext) -> None:
    log_stage_start("12", "DuckDB Shutdown")
    from util import shutdown_duckdb_connection_from_globals

    globals_dict = {"con": ctx.con}
    shutdown_duckdb_connection_from_globals(globals_dict)
    ctx.con = globals_dict["con"]
    log_stage_end("12")


def main() -> None:
    np = pd = torch = None
    ctx: PipelineContext | None = None
    try:
        stage_a1_runtime_bootstrap()
        stage_a2_dependency_bootstrap()
        np, pd, torch = stage_a3_import_runtime_stack()
        ctx = stage_a4_prepare_context(np=np, pd=pd, torch=torch)
        stage_a5_ensure_dataset_contract(ctx)
        stage_a6_reset_outputs(ctx)
        stage_a7_initialize_outputs_and_duckdb(ctx)
        stage_a8_register_source_views(ctx, pd=pd)
        stage_a9_backbone_raw_audit(ctx, np=np, pd=pd)
        stage_a10_build_enrollment_backbone(ctx, pd=pd)
        stage_a11_build_survival_ready(ctx, np=np, pd=pd)
    finally:
        if ctx is not None and ctx.con is not None:
            stage_a12_shutdown(ctx)


if __name__ == "__main__":
    main()