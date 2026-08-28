#!/usr/bin/env bash
# Manual / separate-terminal runner for G14 permutation repeats.
# The same script is also the last step of exe/run_exp_aa.sh (reproducibility).
#
# Usage (from repo root or from anywhere):
#   bash exe/run_permutation_repeats.sh
#   bash exe/run_permutation_repeats.sh --reset     # wipe CSVs + status, start over
#   bash exe/run_permutation_repeats.sh --resume    # default; skip finished models
#
# Optional extra args are passed to the Python script, e.g.:
#   bash exe/run_permutation_repeats.sh --models linear_tuned,rsf_tuned
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/Users/rafars/.pyenv/versions/3.9.13/bin/python}"
LOGDIR="$ROOT/outputs_benchmark_survival/logs"
TABLES="$ROOT/outputs_benchmark_survival/tables"
LOGFILE="$LOGDIR/permutation_repeats.log"

mkdir -p "$LOGDIR" "$TABLES"
cd "$ROOT"

RESET=0
PY_ARGS=(--n-repeats 30 --resume)
PASS=()
for arg in "$@"; do
    case "$arg" in
        --reset)
            RESET=1
            ;;
        --resume)
            ;;
        *)
            PASS+=("$arg")
            ;;
    esac
done

if [[ "$RESET" -eq 1 ]]; then
    rm -f "$LOGDIR/permutation_repeats_status.json"
    rm -f "$TABLES/table_permutation_repeats_feature.csv"
    rm -f "$TABLES/table_permutation_repeats_block.csv"
    PY_ARGS=(--n-repeats 30)
    echo "[permutation] reset: status + CSVs removed"
fi

echo "[permutation] ROOT=$ROOT"
echo "[permutation] log=$LOGFILE"
echo "[permutation] python=$PY ${PY_ARGS[*]} ${PASS[*]:-}"
echo "[permutation] DuckDB is opened read-only and closed after load."

"$PY" -u dropout_bench_v3_G_permutation_repeats.py "${PY_ARGS[@]}" ${PASS[@]+"${PASS[@]}"} 2>&1 | tee -a "$LOGFILE"
