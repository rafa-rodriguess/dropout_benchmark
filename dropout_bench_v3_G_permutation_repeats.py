#!/usr/bin/env python3
"""G14 — Repeated grouped permutation importance for all 14 tuned models.

Standalone stage (safe to run in a separate terminal). Also listed after G13
in exe/run_exp_aa.sh so a full pipeline freeze includes this audit.

Family A: importance = drop in row-level ROC-AUC of predicted hazard.
Family B: importance = drop in sksurv concordance (higher risk = earlier event).
CIs: percentile 2.5–97.5 over --n-repeats independent shuffles (default 30).

DuckDB is opened read-only and closed after tables are loaded into RAM.

Outputs:
  outputs_benchmark_survival/tables/table_permutation_repeats_feature.csv
  outputs_benchmark_survival/tables/table_permutation_repeats_block.csv
  outputs_benchmark_survival/logs/permutation_repeats_status.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N_PERM_REPEATS_DEFAULT = 30
RANDOM_SEED = 42
CI_LO, CI_HI = 2.5, 97.5

PP_FEATURES = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
    "num_of_prev_attempts", "studied_credits", "week",
    "total_clicks_week", "active_this_week", "n_vle_rows_week", "n_distinct_sites_week",
    "cum_clicks_until_t", "recency", "streak",
]
PP_CATS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
ENR_FEATURES = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
    "num_of_prev_attempts", "studied_credits",
    "clicks_first_4_weeks", "active_weeks_first_4", "mean_clicks_first_4_weeks",
]
PP_AUX = {
    "enrollment_id", "id_student", "code_module", "code_presentation",
    "event_observed", "event_t", "t_event_week", "t_final_week",
    "used_zero_week_fallback_for_censoring", "split",
    "time_for_split", "time_bucket", "event_time_bucket_label",
}
ENR_AUX = {
    "enrollment_id", "id_student", "code_module", "code_presentation", "split",
    "time_for_split", "time_bucket", "event_time_bucket_label",
    "event", "duration", "duration_raw", "used_zero_week_fallback_for_censoring",
}


def infer_block(name: str) -> str:
    if name == "week":
        return "discrete_time_index"
    if name in {
        "total_clicks_week", "active_this_week", "n_vle_rows_week",
        "n_distinct_sites_week", "cum_clicks_until_t", "recency", "streak",
    }:
        return "dynamic_temporal_behavioral"
    if name in {"clicks_first_4_weeks", "active_weeks_first_4", "mean_clicks_first_4_weeks"}:
        return "early_window_behavior"
    return "static_structural"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def percentile_ci(values: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(np.mean(arr)), float(np.percentile(arr, CI_LO)), float(np.percentile(arr, CI_HI))


def to_dense(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def sklearn_scores(model, x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if hasattr(model, "params"):
        params = np.asarray(model.params, dtype=float).reshape(-1)
        if arr.ndim == 2 and arr.shape[1] + 1 == params.size:
            arr = np.column_stack([np.ones(arr.shape[0], dtype=arr.dtype), arr])
        return np.asarray(model.predict(arr), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).reshape(-1)
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all() or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def c_index(event: np.ndarray, duration: np.ndarray, risk: np.ndarray) -> float:
    from sksurv.metrics import concordance_index_censored
    risk = np.asarray(risk, dtype=float)
    if not np.isfinite(risk).all():
        return float("nan")
    c, *_ = concordance_index_censored(event.astype(bool), duration.astype(float), risk)
    return float(c)


def permute_column(df: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    vals = out[col].to_numpy(copy=True)
    rng.shuffle(vals)
    out[col] = vals
    return out


def permute_block(df: pd.DataFrame, cols: List[str], rng: np.random.Generator) -> pd.DataFrame:
    out = df
    for col in cols:
        if col in out.columns:
            out = permute_column(out, col, rng)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def best_params(cfg: dict) -> dict:
    bc = cfg.get("best_candidate") or {}
    if isinstance(bc.get("params"), dict):
        return bc["params"]
    return bc


def fcols_pp(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in PP_AUX]


def fcols_enr(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ENR_AUX]


class Wrapper:
    def __init__(self, model_id, display, family, features, metric_name, metric_fn):
        self.model_id = model_id
        self.display = display
        self.family = family
        self.features = features
        self.metric_name = metric_name
        self.metric = metric_fn


def make_sklearn_a(model_id, display, model, prep, df, y_col) -> Wrapper:
    cols = fcols_pp(df)
    y = df[y_col].to_numpy()

    def metric(frame: pd.DataFrame) -> float:
        x = to_dense(prep.transform(frame[cols]))
        return roc_auc(y, sklearn_scores(model, x))

    return Wrapper(model_id, display, "A", list(PP_FEATURES), "row_auc", metric)


def make_catboost(df: pd.DataFrame, model_path: Path) -> Wrapper:
    from catboost import CatBoostClassifier
    cats = [c for c in PP_CATS if c in df.columns]
    nums = [c for c in PP_FEATURES if c not in cats]
    ycol = "event_t" if "event_t" in df.columns else "event_observed"
    y = df[ycol].to_numpy()
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    feat_order = list(model.feature_names_) if model.feature_names_ else list(PP_FEATURES)

    def sanitize(frame: pd.DataFrame) -> pd.DataFrame:
        safe = frame.copy()
        for c in cats:
            col = safe[c].astype("string").fillna("missing")
            safe[c] = col.replace({"<NA>": "missing", "nan": "missing", "None": "missing"}).astype(str)
        for c in nums:
            if c in safe.columns:
                safe[c] = pd.to_numeric(safe[c], errors="coerce").astype(float)
        return safe[feat_order]

    def metric(frame: pd.DataFrame) -> float:
        s = np.asarray(model.predict_proba(sanitize(frame))[:, 1], dtype=float)
        return roc_auc(y, s)

    return Wrapper("catboost_weekly_tuned", "CatBoost Weekly Hazard", "A", list(PP_FEATURES), "row_auc", metric)


def make_neural(df, prep, cfg, pt_path) -> Wrapper:
    import torch
    import torch.nn as nn

    class TunedHazardMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, tensor):
            return self.network(tensor)

    cols = fcols_pp(df)
    y = df["event_t"].to_numpy()
    x0 = to_dense(prep.transform(df[cols]))
    bp = best_params(cfg)
    net = TunedHazardMLP(x0.shape[1], list(bp["hidden_dims"]), float(bp["dropout"]))
    state = torch.load(str(pt_path), map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    def metric(frame: pd.DataFrame) -> float:
        x = torch.from_numpy(to_dense(prep.transform(frame[cols])))
        with torch.no_grad():
            logits = net(x).cpu().numpy().reshape(-1)
        s = 1.0 / (1.0 + np.exp(-logits))
        return roc_auc(y, s)

    return Wrapper("neural_tuned", "Neural Discrete-Time Survival", "A", list(PP_FEATURES), "row_auc", metric)


def _enr_xy(prep, df, cols):
    x = prep.transform(df[cols])
    if hasattr(x, "toarray"):
        x = x.toarray()
    names = list(prep.get_feature_names_out()) if hasattr(prep, "get_feature_names_out") else None
    return pd.DataFrame(np.asarray(x), columns=names)


def make_cox(df, model, prep) -> Wrapper:
    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)

    def metric(frame: pd.DataFrame) -> float:
        xdf = _enr_xy(prep, frame, cols)
        if hasattr(model, "predict_partial_hazard"):
            risk = np.asarray(model.predict_partial_hazard(xdf), dtype=float).reshape(-1)
        else:
            risk = np.asarray(model.predict(xdf), dtype=float).reshape(-1)
        return c_index(event, dur, risk)

    return Wrapper("cox_tuned", "Cox Comparable", "B", list(ENR_FEATURES), "c_index", metric)


def make_sksurv(model_id, display, df, model, prep) -> Wrapper:
    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)

    def metric(frame: pd.DataFrame) -> float:
        x = to_dense(prep.transform(frame[cols]))
        risk = np.asarray(model.predict(x), dtype=float).reshape(-1)
        return c_index(event, dur, risk)

    return Wrapper(model_id, display, "B", list(ENR_FEATURES), "c_index", metric)


def make_lifelines_aft(model_id, display, df, model, prep, formula_map=None) -> Wrapper:
    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)

    def metric(frame: pd.DataFrame) -> float:
        xdf = _enr_xy(prep, frame, cols)
        if formula_map:
            xdf = xdf.rename(columns=formula_map)
        if hasattr(model, "predict_partial_hazard") and model_id != "royston_parmar_tuned":
            risk = np.asarray(model.predict_partial_hazard(xdf), dtype=float).reshape(-1)
        elif hasattr(model, "predict_expectation"):
            risk = -np.asarray(model.predict_expectation(xdf), dtype=float).reshape(-1)
        elif hasattr(model, "predict_median"):
            risk = -np.asarray(model.predict_median(xdf), dtype=float).reshape(-1)
        else:
            raise RuntimeError(f"{model_id}: no supported predict method")
        return c_index(event, dur, risk)

    return Wrapper(model_id, display, "B", list(ENR_FEATURES), "c_index", metric)


def make_xgb_aft(df, booster, prep) -> Wrapper:
    import xgboost as xgb
    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)

    def metric(frame: pd.DataFrame) -> float:
        x = to_dense(prep.transform(frame[cols]))
        if hasattr(booster, "predict") and not str(type(booster)).endswith("Booster'>"):
            loc = np.asarray(booster.predict(x), dtype=float).reshape(-1)
        else:
            loc = np.asarray(booster.predict(xgb.DMatrix(x)), dtype=float).reshape(-1)
        return c_index(event, dur, -loc)

    return Wrapper("xgboost_aft_tuned", "XGBoost AFT", "B", list(ENR_FEATURES), "c_index", metric)


def make_deepsurv(df, prep, cfg, pt_path) -> Wrapper:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH

    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)
    x0 = to_dense(prep.transform(df[cols]))
    bp = best_params(cfg)
    net = tt.practical.MLPVanilla(
        in_features=x0.shape[1], num_nodes=list(bp["hidden_dims"]), out_features=1,
        batch_norm=True, dropout=float(bp["dropout"]), output_bias=False,
    )
    model = CoxPH(net, tt.optim.AdamW(lr=float(bp.get("learning_rate", 1e-3))))
    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and not hasattr(state, "to"):
        net.load_state_dict(state)
        model.net = net
    else:
        model.net = state
    model.net.eval()

    def metric(frame: pd.DataFrame) -> float:
        x = to_dense(prep.transform(frame[cols]))
        return c_index(event, dur, np.asarray(model.predict(x), dtype=float).reshape(-1))

    return Wrapper("deepsurv_tuned", "DeepSurv", "B", list(ENR_FEATURES), "c_index", metric)


def make_pycox_discrete(model_id, display, df, prep, cfg, pt_path, kind: str) -> Wrapper:
    import torch
    import torchtuples as tt
    from pycox.models import DeepHitSingle, MTLR

    cols = fcols_enr(df)
    event, dur = df["event"].to_numpy().astype(int), df["duration"].to_numpy().astype(float)
    x0 = to_dense(prep.transform(df[cols])).astype(np.float32)
    bp = best_params(cfg)
    cuts = np.asarray(cfg["duration_index"], dtype=float)
    net = tt.practical.MLPVanilla(
        in_features=x0.shape[1], num_nodes=list(bp["hidden_dims"]),
        out_features=int(bp["num_durations"]), batch_norm=True,
        dropout=float(bp["dropout"]), output_bias=False,
    )
    lr, wd = float(bp.get("learning_rate", 1e-3)), float(bp.get("weight_decay", 0.0))
    if kind == "mtlr":
        model = MTLR(net, tt.optim.Adam(lr=lr, weight_decay=wd), duration_index=cuts)
    else:
        model = DeepHitSingle(
            net, tt.optim.Adam(lr=lr, weight_decay=wd), duration_index=cuts,
            alpha=float(bp.get("alpha", 0.2)), sigma=float(bp.get("sigma", 0.1)),
        )
    try:
        model.load_net(str(pt_path))
    except Exception:
        state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            model.net.load_state_dict(state)
        else:
            model.net = state

    def metric(frame: pd.DataFrame) -> float:
        x = to_dense(prep.transform(frame[cols])).astype(np.float32)
        surv = model.predict_surv_df(x)
        idx = min(surv.index, key=lambda t: abs(float(t) - 20.0))
        risk = 1.0 - np.asarray(surv.loc[idx], dtype=float).reshape(-1)
        return c_index(event, dur, risk)

    return Wrapper(model_id, display, "B", list(ENR_FEATURES), "c_index", metric)


def run_one(wrapper: Wrapper, df: pd.DataFrame, n_repeats: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    baseline = wrapper.metric(df)
    log(f"  baseline {wrapper.metric_name}={baseline:.6f}")
    feat_rows = []
    block_map: Dict[str, List[str]] = {}
    for feat in wrapper.features:
        if feat not in df.columns:
            continue
        block_map.setdefault(infer_block(feat), []).append(feat)
        deltas = []
        for r in range(n_repeats):
            m = wrapper.metric(permute_column(df, feat, rng))
            deltas.append(baseline - m)
            if (r + 1) % 10 == 0:
                log(f"    {wrapper.model_id} feat={feat} {r+1}/{n_repeats}")
        mean, lo, hi = percentile_ci(np.array(deltas))
        feat_rows.append({
            "model_id": wrapper.model_id, "display_name": wrapper.display, "family": wrapper.family,
            "feature": feat, "feature_block": infer_block(feat), "metric": wrapper.metric_name,
            "baseline": baseline, "n_repeats": n_repeats,
            "importance_mean": mean, "importance_ci_low": lo, "importance_ci_high": hi,
        })
    block_rows = []
    for block, cols in block_map.items():
        deltas = []
        for r in range(n_repeats):
            m = wrapper.metric(permute_block(df, cols, rng))
            deltas.append(baseline - m)
            if (r + 1) % 10 == 0:
                log(f"    {wrapper.model_id} BLOCK={block} {r+1}/{n_repeats}")
        mean, lo, hi = percentile_ci(np.array(deltas))
        block_rows.append({
            "model_id": wrapper.model_id, "display_name": wrapper.display, "family": wrapper.family,
            "feature_block": block, "n_features_in_block": len(cols), "metric": wrapper.metric_name,
            "baseline": baseline, "n_repeats": n_repeats,
            "importance_mean": mean, "importance_ci_low": lo, "importance_ci_high": hi,
        })
    return pd.DataFrame(feat_rows), pd.DataFrame(block_rows)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="G14 repeated permutation importance (14 models).")
    parser.add_argument("--n-repeats", type=int, default=N_PERM_REPEATS_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", type=str, default="")
    args = parser.parse_args()

    import duckdb
    import joblib
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    shared = tomllib.load((ROOT / "benchmark_shared_config.toml").open("rb"))
    out_dir = ROOT / shared.get("paths", {}).get("output_dir", "outputs_benchmark_survival")
    models_dir, meta_dir = out_dir / "models", out_dir / "metadata"
    tables_dir, log_dir = out_dir / "tables", out_dir / "logs"
    feat_csv = tables_dir / "table_permutation_repeats_feature.csv"
    block_csv = tables_dir / "table_permutation_repeats_block.csv"
    status_path = log_dir / "permutation_repeats_status.json"
    duckdb_path = out_dir / shared.get("paths", {}).get("duckdb_filename", "benchmark_survival.duckdb")

    status = json.loads(status_path.read_text()) if args.resume and status_path.exists() else {"done": [], "failed": {}}
    if not args.resume:
        for p in (feat_csv, block_csv):
            if p.exists():
                p.unlink()

    log(f"DuckDB read-only: {duckdb_path}")
    con = duckdb.connect(str(duckdb_path), read_only=True)

    def q(name: str) -> pd.DataFrame:
        return con.execute(f"SELECT * FROM {name}").fetchdf()

    pp_lin = q("pp_linear_hazard_ready_test")
    try:
        pp_neu = q("pp_neural_hazard_ready_test")
    except Exception:
        pp_neu = pp_lin
    enr_cox = q("enrollment_cox_ready_test")
    try:
        enr_ds = q("enrollment_deepsurv_ready_test")
    except Exception:
        enr_ds = enr_cox
    con.close()
    log("DuckDB closed. Tables in RAM.")

    y_pp = "event_t" if "event_t" in pp_lin.columns else "event_observed"
    builders = [
        ("linear_tuned", lambda: make_sklearn_a(
            "linear_tuned", "Linear Discrete-Time Hazard",
            joblib.load(models_dir / "linear_discrete_time_hazard_not_weighted_tuned_w4.joblib"),
            joblib.load(models_dir / "linear_discrete_time_not_weighted_preprocessor_w4.joblib"),
            pp_lin, y_pp)),
        ("poisson_pexp_tuned", lambda: make_sklearn_a(
            "poisson_pexp_tuned", "Poisson Piecewise-Exponential",
            joblib.load(models_dir / "poisson_piecewise_exponential_not_weighted_tuned_w4.joblib"),
            joblib.load(models_dir / "poisson_piecewise_exponential_not_weighted_preprocessor_w4.joblib"),
            pp_lin, y_pp)),
        ("gb_weekly_tuned", lambda: make_sklearn_a(
            "gb_weekly_tuned", "GB Weekly Hazard",
            joblib.load(models_dir / "gb_weekly_hazard_not_weighted_tuned_w4.joblib"),
            joblib.load(models_dir / "gb_weekly_hazard_not_weighted_preprocessor_w4.joblib"),
            pp_lin, y_pp)),
        ("catboost_weekly_tuned", lambda: make_catboost(
            pp_lin, models_dir / "catboost_weekly_hazard_not_weighted_tuned_w4.cbm")),
        ("neural_tuned", lambda: make_neural(
            pp_neu,
            joblib.load(models_dir / "neural_discrete_time_not_weighted_preprocessor_w4.joblib"),
            load_json(meta_dir / "neural_not_weighted_tuned_model_config_w4.json"),
            models_dir / "neural_discrete_time_survival_not_weighted_tuned_w4.pt")),
        ("cox_tuned", lambda: make_cox(
            enr_cox, joblib.load(models_dir / "cox_early_window_tuned.joblib"),
            joblib.load(models_dir / "cox_preprocessor.joblib"))),
        ("rsf_tuned", lambda: make_sksurv(
            "rsf_tuned", "Random Survival Forest", enr_cox,
            joblib.load(models_dir / "rsf_tuned.joblib"),
            joblib.load(models_dir / "rsf_preprocessor.joblib"))),
        ("gb_cox_tuned", lambda: make_sksurv(
            "gb_cox_tuned", "Gradient-Boosted Cox", enr_cox,
            joblib.load(models_dir / "gb_cox_tuned.joblib"),
            joblib.load(models_dir / "gb_cox_preprocessor.joblib"))),
        ("weibull_aft_tuned", lambda: make_lifelines_aft(
            "weibull_aft_tuned", "Weibull AFT", enr_cox,
            joblib.load(models_dir / "weibull_aft_tuned.joblib"),
            joblib.load(models_dir / "weibull_aft_preprocessor.joblib"))),
        ("royston_parmar_tuned", lambda: make_lifelines_aft(
            "royston_parmar_tuned", "Royston-Parmar", enr_cox,
            joblib.load(models_dir / "royston_parmar_tuned.joblib"),
            joblib.load(models_dir / "royston_parmar_preprocessor.joblib"),
            formula_map=load_json(meta_dir / "royston_parmar_preprocessing_config.json").get("formula_feature_map"),
        )),
        ("xgboost_aft_tuned", lambda: make_xgb_aft(
            enr_cox, joblib.load(models_dir / "xgb_aft_tuned.joblib"),
            joblib.load(models_dir / "xgb_aft_preprocessor.joblib"))),
        ("deepsurv_tuned", lambda: make_deepsurv(
            enr_ds, joblib.load(models_dir / "deepsurv_preprocessor.joblib"),
            load_json(meta_dir / "deepsurv_tuned_model_config.json"),
            models_dir / "deepsurv_tuned.pt")),
        ("mtlr_tuned", lambda: make_pycox_discrete(
            "mtlr_tuned", "Neural-MTLR", enr_cox,
            joblib.load(models_dir / "mtlr_preprocessor.joblib"),
            load_json(meta_dir / "mtlr_tuned_model_config.json"),
            models_dir / "mtlr_tuned.pt", "mtlr")),
        ("deephit_tuned", lambda: make_pycox_discrete(
            "deephit_tuned", "DeepHit", enr_cox,
            joblib.load(models_dir / "deephit_preprocessor.joblib"),
            load_json(meta_dir / "deephit_tuned_model_config.json"),
            models_dir / "deephit_tuned.pt", "deephit")),
    ]

    wanted = {m.strip() for m in args.models.split(",") if m.strip()} or {m for m, _ in builders}
    n_ok = n_fail = 0
    log_dir.mkdir(parents=True, exist_ok=True)

    for model_id, builder in builders:
        if model_id not in wanted:
            continue
        if args.resume and model_id in status.get("done", []):
            log(f"SKIP (resume) {model_id}")
            continue
        if model_id in {"linear_tuned", "poisson_pexp_tuned", "gb_weekly_tuned", "catboost_weekly_tuned"}:
            df_use = pp_lin
        elif model_id == "neural_tuned":
            df_use = pp_neu
        elif model_id == "deepsurv_tuned":
            df_use = enr_ds
        else:
            df_use = enr_cox
        log(f"=== START {model_id} ===")
        try:
            wrapper = builder()
            feat_df, block_df = run_one(wrapper, df_use, args.n_repeats, RANDOM_SEED)
            append_csv(feat_csv, feat_df)
            append_csv(block_csv, block_df)
            status.setdefault("done", []).append(model_id)
            status.setdefault("failed", {}).pop(model_id, None)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            log(f"=== DONE {model_id} ===")
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            log(f"=== FAIL {model_id}: {exc} ===")
            print(traceback.format_exc(), flush=True)
            status.setdefault("failed", {})[model_id] = str(exc)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    log(f"Finished ok={n_ok} fail={n_fail}")
    log(f"Feature: {feat_csv}")
    log(f"Block:   {block_csv}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
