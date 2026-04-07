from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb

try:
    from notebook_lib.duckdb_notebook_utils import parse_notebook_table_ops, duckdb_table_columns
except Exception:  # pragma: no cover
    parse_notebook_table_ops = None

    def duckdb_table_columns(con, table_name):
        try:
            rows = con.execute(f"DESCRIBE {table_name}").fetchall()
            return [row[0] for row in rows]
        except Exception:
            return []

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("dropout_bench_v3*.ipynb"))
META_DIR = ROOT / "outputs_benchmark_survival" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)


def _load_duckdb_path() -> Path:
    cfg_path = ROOT / "benchmark_shared_config.toml"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        paths = cfg.get("paths", {})
        output_dir = Path(paths.get("output_dir", "outputs_benchmark_survival"))
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        db_name = paths.get("duckdb_filename", "benchmark_survival.duckdb")
        return output_dir / db_name
    return ROOT / "outputs_benchmark_survival" / "benchmark_survival.duckdb"


def collect_function_defs(nb_path: Path) -> Set[str]:
    data = json.loads(nb_path.read_text())
    names: Set[str] = set()
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
    return names


def infer_created_table_fields(nb_path: Path) -> Dict[str, List[str]]:
    """Infer output field names from CREATE TABLE ... AS SELECT ... SQL snippets."""
    data = json.loads(nb_path.read_text())
    out: Dict[str, List[str]] = {}

    create_sql_pat = re.compile(
        r"CREATE\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+SELECT([\s\S]*?)\bFROM\b",
        re.I,
    )

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for m in create_sql_pat.finditer(src):
            table = m.group(1)
            select_block = m.group(2)

            raw_items = [x.strip() for x in select_block.split(",") if x.strip()]
            fields: List[str] = []
            for item in raw_items:
                one_line = " ".join(item.split())
                alias_match = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)$", one_line, re.I)
                if alias_match:
                    fields.append(alias_match.group(1))
                    continue

                if one_line == "*":
                    fields.append("*")
                    continue

                last_token = one_line.split(".")[-1].split()[-1]
                last_token = re.sub(r"[^A-Za-z0-9_]", "", last_token)
                if last_token:
                    fields.append(last_token)

            if fields:
                out[table] = fields

    return out


def main() -> None:
    func_to_notebooks: Dict[str, Set[str]] = {}
    table_creator: Dict[str, List[Tuple[str, int]]] = {}
    static_table_fields: Dict[str, List[str]] = {}
    cell_review_lines: List[str] = []

    for nb in NOTEBOOKS:
        funcs = collect_function_defs(nb)
        inferred_fields = infer_created_table_fields(nb)
        for tbl, cols in inferred_fields.items():
            static_table_fields[tbl] = cols

        for fn in funcs:
            func_to_notebooks.setdefault(fn, set()).add(nb.name)

        ops = parse_notebook_table_ops(nb)
        cell_review_lines.append(f"Notebook: {nb.name}")
        for op in ops:
            creates = ", ".join(op["creates"]) if op["creates"] else "-"
            reads = ", ".join(op["reads"]) if op["reads"] else "-"
            status = "OK"
            if op["uses_pandas_io"]:
                status = "NEEDS_MIGRATION_TO_DUCKDB"
            cell_review_lines.append(
                f"  Cell {op['cell_number']:>3} | creates: {creates} | reads: {reads} | pandas_io: {op['uses_pandas_io']} | status: {status}"
            )
            for t in op["creates"]:
                table_creator.setdefault(t, []).append((nb.name, op["cell_number"]))
        cell_review_lines.append("")

    reusable = {
        fn: sorted(nbs)
        for fn, nbs in func_to_notebooks.items()
        if len(nbs) >= 2
    }

    # Function reuse map
    function_lines = [
        "Function Reuse Map",
        "===================",
        "",
        "Functions defined in >=2 notebooks and good candidates for migration to .py libs:",
    ]
    for fn in sorted(reusable):
        function_lines.append(f"- {fn}: {', '.join(reusable[fn])}")

    function_lines.extend(
        [
            "",
            "Priority candidates already addressed or strongly recommended:",
            "- initialize_notebook_runtime / close_duckdb_connection (runtime and DB lifecycle)",
            "- table/schema helpers (table existence, schema checks, table I/O wrappers)",
            "- split helper functions reused in C and combined notebook",
        ]
    )
    (META_DIR / "function_reuse_map.txt").write_text("\n".join(function_lines))

    # Cell-level review
    review_header = [
        "DuckDB Table Usage Review (Cell-Level)",
        "====================================",
        "",
        "Goal: all table I/O should flow through DuckDB.",
        "Status field marks code cells with pandas file I/O as NEEDS_MIGRATION_TO_DUCKDB.",
        "",
    ]
    (META_DIR / "table_cell_review.txt").write_text("\n".join(review_header + cell_review_lines))

    # Data catalog from created tables + current DuckDB schema
    db_path = _load_duckdb_path()
    con = None
    if db_path.exists():
        try:
            con = duckdb.connect(str(db_path), read_only=True)
        except Exception as conn_err:
            print(f"Warning: could not open DuckDB for schema inspection: {conn_err}")
            con = None

    catalog_lines = [
        "Data Catalog (DuckDB-Centric)",
        "============================",
        "",
        f"duckdb_path: {db_path}",
        "",
        "Columns: table_name | creator_notebook | creator_cells | fields",
        "",
    ]

    for table_name in sorted(table_creator):
        creators = table_creator[table_name]
        creator_notebooks = sorted({c[0] for c in creators})
        creator_cells = ", ".join([f"{nb}:cell{cell}" for nb, cell in creators])

        fields = []
        if con is not None:
            fields = duckdb_table_columns(con, table_name)
        if not fields:
            fields = static_table_fields.get(table_name, [])

        fields_str = ", ".join(fields) if fields else "(table not found in current duckdb or schema unavailable)"
        catalog_lines.append(
            f"{table_name} | {', '.join(creator_notebooks)} | {creator_cells} | {fields_str}"
        )

    if con is not None:
        con.close()

    (META_DIR / "data_catalog.txt").write_text("\n".join(catalog_lines))

    print("Wrote:")
    print("-", META_DIR / "function_reuse_map.txt")
    print("-", META_DIR / "table_cell_review.txt")
    print("-", META_DIR / "data_catalog.txt")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Notebook refactoring helpers (DuckDB-first pattern)
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
from uuid import uuid4


def load_shared_config(
    project_root: Path | str,
    config_filename: str = "benchmark_shared_config.toml",
) -> tuple[dict, Path]:
    project_root = Path(project_root)
    config_path = project_root / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config TOML not found: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config, config_path


def resolve_project_path(project_root: Path | str, raw_path: str) -> Path:
    project_root = Path(project_root)
    p = Path(raw_path)
    return p if p.is_absolute() else project_root / p


def ensure_run_metadata(
    metadata_dir: Path | str,
    notebook_name: str,
    metadata_filename: str = "run_metadata.json",
) -> tuple[dict, Path]:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / metadata_filename

    now = datetime.now(timezone.utc).astimezone().isoformat()
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["last_notebook"] = notebook_name
        payload["last_seen_at"] = now
    else:
        payload = {
            "run_id": uuid4().hex,
            "created_at": now,
            "last_seen_at": now,
            "created_by_notebook": notebook_name,
            "last_notebook": notebook_name,
        }

    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload, metadata_path


def open_duckdb_connection(duckdb_path: Path | str):
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(duckdb_path))


def close_duckdb_connection(con, checkpoint: bool = True, quiet: bool = False):
    if con is None:
        if not quiet:
            print("No active DuckDB connection found.")
        return None

    if checkpoint:
        try:
            con.execute("CHECKPOINT")
        except Exception as exc:
            if not quiet:
                print(f"Warning during DuckDB CHECKPOINT: {exc}")

    try:
        con.close()
        if not quiet:
            print("DuckDB connection closed.")
    except Exception as exc:
        if not quiet:
            print(f"Warning while closing DuckDB connection: {exc}")
    return None


def shutdown_duckdb_connection_from_globals(
    global_vars: dict,
    connection_name: str = "con",
):
    if connection_name in global_vars:
        global_vars[connection_name] = close_duckdb_connection(
            global_vars[connection_name],
            checkpoint=True,
            quiet=False,
        )
    else:
        print(f"No active DuckDB connection found in variable '{connection_name}'.")


def ensure_pipeline_catalog(
    con,
    catalog_table: str = "pipeline_table_catalog",
    catalog_view: str = "vw_pipeline_table_catalog_schema",
) -> None:
    con.execute(f'''
    CREATE TABLE IF NOT EXISTS {catalog_table} (
        table_name VARCHAR,
        created_by_notebook VARCHAR,
        created_in_cell VARCHAR,
        created_at TIMESTAMP,
        run_id VARCHAR
    )
    ''')
    refresh_pipeline_catalog_schema_view(con, catalog_table=catalog_table, catalog_view=catalog_view)


def register_duckdb_table(
    con,
    table_name: str,
    notebook_name: str,
    cell_name: str,
    run_id: str,
    catalog_table: str = "pipeline_table_catalog",
) -> None:
    con.execute(
        f'''
        INSERT INTO {catalog_table} (
            table_name, created_by_notebook, created_in_cell, created_at, run_id
        )
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
        ''',
        [table_name, notebook_name, cell_name, run_id],
    )


def refresh_pipeline_catalog_schema_view(
    con,
    catalog_table: str = "pipeline_table_catalog",
    catalog_view: str = "vw_pipeline_table_catalog_schema",
) -> None:
    con.execute(f'''
    CREATE OR REPLACE VIEW {catalog_view} AS
    WITH catalog_dedup AS (
        SELECT
            table_name,
            created_by_notebook,
            created_in_cell,
            created_at,
            run_id,
            ROW_NUMBER() OVER (
                PARTITION BY table_name
                ORDER BY created_at DESC, run_id DESC
            ) AS rn
        FROM {catalog_table}
    )
    SELECT
        c.table_schema,
        c.table_name,
        c.column_name,
        c.data_type,
        c.ordinal_position,
        d.created_by_notebook,
        d.created_in_cell,
        d.created_at,
        d.run_id
    FROM information_schema.columns AS c
    LEFT JOIN catalog_dedup AS d
      ON c.table_name = d.table_name
     AND d.rn = 1
    WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog')
    ORDER BY c.table_schema, c.table_name, c.ordinal_position
    ''')


def print_duckdb_table(
    con,
    table_name: str,
    title: str | None = None,
    limit: int = 20,
) -> None:
    title = title or table_name
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    schema_rows = con.execute(f"DESCRIBE {table_name}").fetchall()
    preview = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()

    print(f"\n{title}")
    print(f"Table: {table_name}")
    print(f"Row count: {n_rows}")
    print("Schema:")
    for row in schema_rows:
        print(f" - {row[0]}: {row[1]}")
    try:
        from IPython.display import display as ipy_display

        ipy_display(preview)
    except Exception:
        print(preview.to_string(index=False) if not preview.empty else "[empty table]")
