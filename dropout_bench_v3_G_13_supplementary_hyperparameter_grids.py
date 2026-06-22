"""
G13 — Generate supplementary/S1_hyperparameter_grids.md

Reads hyperparameter search grids from two sources (no hardcoding):
  - Family B: JSON config files in outputs_benchmark_survival/metadata/
  - Family A: Python AST extraction from pipeline source scripts

Output: supplementary/S1_hyperparameter_grids.md
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import textwrap
from datetime import datetime as _dt

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).parent.resolve()
META_DIR = ROOT / "outputs_benchmark_survival" / "metadata"
OUT_FILE = ROOT / "S1_hyperparameter_grids.md"

print(f"[START] G13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_val(v: object) -> str:
    """Format a value for markdown: lists as [a, b], floats in scientific if small."""
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_val(x) for x in v) + "]"
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.0e}"
        return str(v)
    return str(v)


def _md_table(rows: list[dict], param_keys: list[str]) -> str:
    """Render a list of candidate dicts as a markdown table."""
    headers = ["Candidate"] + param_keys
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cid = str(row.get("candidate_id", "—"))
        vals = [_fmt_val(row.get(k, "—")) for k in param_keys]
        lines.append("| " + " | ".join([cid] + vals) + " |")
    return "\n".join(lines)


# ── Family B: read from JSON metadata ────────────────────────────────────────

FAMILY_B_JSON_MAP = {
    "Cox Comparable":      "cox_tuned_model_config.json",
    "DeepSurv":            "deepsurv_tuned_model_config.json",
    "Random Survival Forest": "rsf_tuned_model_config.json",
    "Neural-MTLR":         "mtlr_tuned_model_config.json",
    "DeepHit":             "deephit_tuned_model_config.json",
    "GB Cox":              "gb_cox_tuned_model_config.json",
    "Royston-Parmar":      "royston_parmar_tuned_model_config.json",
    "Weibull AFT":         "weibull_aft_tuned_model_config.json",
    "XGBoost AFT":         "xgb_aft_tuned_model_config.json",
}

FAMILY_B_SELECTION: dict[str, str] = {
    "Cox Comparable":         "highest validation C-index",
    "DeepSurv":               "lowest validation loss (early stopping, patience=10)",
    "Random Survival Forest": "lowest validation IBS",
    "Neural-MTLR":            "lowest validation IBS (early stopping, patience=8)",
    "DeepHit":                "lowest validation IBS (early stopping, patience=8)",
    "GB Cox":                 "lowest validation IBS",
    "Royston-Parmar":         "lowest validation IBS",
    "Weibull AFT":            "lowest validation IBS",
    "XGBoost AFT":            "lowest validation IBS",
}

FAMILY_B_NOTES: dict[str, str] = {
    "DeepSurv":   "Architecture grid: [32,16], [64,32], [128,64]. Early stopping on val loss.",
    "Neural-MTLR":"num_durations controls survival-time discretization. Early stopping on val IBS.",
    "DeepHit":    "alpha/sigma are DeepHit-specific ranking loss weights.",
}


def load_family_b_grid(model_name: str, json_name: str) -> tuple[list[dict], list[str]]:
    path = META_DIR / json_name
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    candidates = data.get("search_space", [])
    if not candidates or not isinstance(candidates, list):
        return [], []
    param_keys = [k for k in candidates[0].keys() if k != "candidate_id"]
    return candidates, param_keys


# ── Family A: extract constants via AST from source scripts ──────────────────

FAMILY_A_SCRIPT_MAP = {
    "Linear Discrete-Time Hazard": (
        "dropout_bench_v3_D_02_A_dynamic_weekly_linear_discrete_time_hazard.py",
        "LINEAR_TUNING_GRID",
        "highest row-level validation C-index",
        "Penalty type (L1/L2) × C regularization strength.",
    ),
    "Poisson Piecewise-Exponential": (
        "dropout_bench_v3_D_06_A_dynamic_weekly_poisson_piecewise_exponential.py",
        "ALPHA_GRID",
        "lowest validation log-loss on discrete hazard",
        "1-D grid over L2 regularization strength (alpha). Single-param sweep via statsmodels GLM.",
    ),
    "GB Weekly Hazard": (
        "dropout_bench_v3_D_07_A_dynamic_weekly_gb_weekly_hazard.py",
        "HGB_CANDIDATE_GRID",
        "lowest validation log-loss (early stopping)",
        "HistGradientBoostingClassifier with early stopping. Row budget capped by feature count.",
    ),
    "Neural Discrete-Time": (
        "dropout_bench_v3_D_03_A_dynamic_neural_neural_discrete_time_survival.py",
        "NEURAL_GRID",
        "lowest validation loss (early stopping, patience=5)",
        "Full grid product: 2 arch × 2 dropout × 2 lr × 2 wd = 16 candidates. "
        "Override: candidate_id=9 uses dropout=0.05.",
    ),
}


def _ast_eval_node(node: ast.expr) -> object:
    """Safely evaluate a constant AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_eval_node(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_ast_eval_node(k): _ast_eval_node(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_eval_node(node.operand)
    raise ValueError(f"Cannot evaluate node type: {type(node).__name__}")


def _extract_constant_from_source(script_path: pathlib.Path, const_name: str) -> object | None:
    """Parse Python source and return the value of a top-level constant by name."""
    source = script_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  [WARN] AST parse error in {script_path.name}: {e}")
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == const_name:
                    try:
                        return _ast_eval_node(node.value)
                    except (ValueError, AttributeError):
                        return None
    return None


def load_family_a_grid(
    script_name: str, const_name: str
) -> tuple[list[dict], list[str]]:
    script_path = ROOT / script_name
    if not script_path.exists():
        return [], []
    value = _extract_constant_from_source(script_path, const_name)
    if value is None:
        return [], []

    # ALPHA_GRID is a flat tuple of floats → convert to list of dicts
    if const_name == "ALPHA_GRID" and isinstance(value, (tuple, list)):
        candidates = [
            {"candidate_id": i + 1, "alpha": float(v)} for i, v in enumerate(value)
        ]
        return candidates, ["alpha"]

    # NEURAL_GRID is a generator expression — AST extracts the itertools.product args
    # but it's defined as a tuple(... for ...) which AST cannot fully evaluate.
    # Fallback: re-parse to find the product arguments directly.
    if const_name == "NEURAL_GRID":
        return _extract_neural_grid(script_path)

    # LINEAR_TUNING_GRID and HGB_CANDIDATE_GRID are tuples of dicts
    if isinstance(value, (tuple, list)) and len(value) > 0:
        rows = []
        for i, item in enumerate(value):
            if isinstance(item, dict):
                if "candidate_id" not in item:
                    item = {"candidate_id": i + 1, **item}
                rows.append(item)
        if rows:
            param_keys = [k for k in rows[0].keys() if k != "candidate_id"]
            return rows, param_keys
    return [], []


def _extract_neural_grid(script_path: pathlib.Path) -> tuple[list[dict], list[str]]:
    """Special parser for NEURAL_GRID which is built from itertools.product."""
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find NEURAL_GRID_OVERRIDES
    overrides: dict[int, dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "NEURAL_GRID_OVERRIDES":
                    try:
                        overrides = _ast_eval_node(node.value)
                    except (ValueError, AttributeError):
                        pass

    # Find the itertools.product(...) call inside NEURAL_GRID
    # The structure is: tuple({...} for cid, (...) in enumerate(itertools.product(...), start=1))
    # We look for the Call node that is itertools.product
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "product"
        ):
            try:
                axes = [_ast_eval_node(arg) for arg in node.args]
            except (ValueError, AttributeError):
                continue
            if not axes:
                continue
            import itertools
            candidates = []
            param_names = ["hidden_dims", "dropout", "learning_rate", "weight_decay"]
            for cid, combo in enumerate(itertools.product(*axes), start=1):
                row: dict[str, object] = {"candidate_id": cid}
                for pname, pval in zip(param_names, combo):
                    row[pname] = pval
                # Apply overrides
                if cid in overrides:
                    for ok, ov in overrides[cid].items():
                        row[ok] = ov
                candidates.append(row)
            return candidates, param_names

    return [], []


# ── Markdown generation ───────────────────────────────────────────────────────

lines: list[str] = []
lines += [
    "# S1 — Hyperparameter Search Grids",
    "",
    "This supplementary file is **auto-generated** by `dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py`.",
    "It documents the complete candidate search grids for all tuned models in the benchmark.",
    "For Family B, grids are read from JSON metadata files produced at pipeline run-time.",
    "For Family A, grids are extracted from the pipeline source constants via AST parsing.",
    "",
    f"*Generated: {_dt.now().strftime('%Y-%m-%d %H:%M UTC')}*",
    "",
    "---",
    "",
    "## Family B — Static Early-Window (Comparable Arm)",
    "",
    "All Family B models use enrollment-level early-window features.",
    "Validation uses a 20% enrollment-level hold-out (GroupShuffleSplit) stratified by event,",
    "except Neural-MTLR and DeepHit which use a 10% internal fraction.",
    "",
]

for model_name, json_name in FAMILY_B_JSON_MAP.items():
    candidates, param_keys = load_family_b_grid(model_name, json_name)
    selection = FAMILY_B_SELECTION.get(model_name, "—")
    note = FAMILY_B_NOTES.get(model_name, "")
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append(f"**Candidates:** {len(candidates)}  |  **Selection criterion:** {selection}")
    if note:
        lines.append(f"*{note}*")
    lines.append("")
    if candidates and param_keys:
        lines.append(_md_table(candidates, param_keys))
    else:
        lines.append("*(grid data not found in metadata)*")
    lines.append("")

lines += [
    "---",
    "",
    "## Family A — Dynamic Weekly (Person-Period Arm)",
    "",
    "All Family A models use person-period rows (one row per enrollment per week).",
    "Validation uses a 10–20% enrollment-level GroupShuffleSplit.",
    "",
]

for model_name, (script_name, const_name, selection, note) in FAMILY_A_SCRIPT_MAP.items():
    candidates, param_keys = load_family_a_grid(script_name, const_name)
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append(f"**Candidates:** {len(candidates)}  |  **Selection criterion:** {selection}")
    lines.append(f"*{note}*")
    lines.append("")
    if candidates and param_keys:
        lines.append(_md_table(candidates, param_keys))
    else:
        lines.append("*(grid data not available — check source script)*")
    lines.append("")

lines += [
    "---",
    "",
    "## Notes on tuning scope",
    "",
    "Tuning was deliberately controlled rather than exhaustive: the goal was",
    "comparable benchmark representatives, not unbounded per-model optimization.",
    "All candidates were evaluated under the same preprocessing pipeline",
    "(median imputation, constant-missing categoricals, one-hot encoding,",
    "standard scaling fitted on training rows only).",
    "",
    "The asymmetry in candidate counts (24 for DeepSurv vs. 4 for MTLR/DeepHit)",
    "reflects the larger regularization surface of the DeepSurv formulation.",
    "MTLR and DeepHit results should be read as lower bounds on achievable",
    "neural performance within this benchmark contract.",
    "",
    "Full machine-readable candidate records are available in",
    "`outputs_benchmark_survival/metadata/` as JSON files.",
]

OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Written: {OUT_FILE}")
print(f"  Family B models: {len(FAMILY_B_JSON_MAP)}")
print(f"  Family A models: {len(FAMILY_A_SCRIPT_MAP)}")
print(f"[END] G13 - {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
