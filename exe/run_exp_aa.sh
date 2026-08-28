#!/usr/bin/env bash
# run_exp_aa.sh — Execução sequencial exp-AA (grids padronizados para 24 candidatos)
# Suporte a resume: re-executa apenas etapas não concluídas.
# Em caso de erro, para imediatamente e registra a etapa falhada.
#
# Uso:
#   bash exe/run_exp_aa.sh            # run normal
#   bash exe/run_exp_aa.sh --reset    # limpa estado e recomeça do zero

set -euo pipefail

PY=/Users/rafars/.pyenv/versions/3.9.13/bin/python
ROOT=/Users/rafars/dropout_benchmark_desenv
LOGDIR="$ROOT/outputs_benchmark_survival/logs"
STATE_FILE="/tmp/exp_aa_done_steps.txt"
MASTER_LOG="/tmp/exp_aa_master.log"

mkdir -p "$LOGDIR"

if [[ "${1:-}" == "--reset" ]]; then
    rm -f "$STATE_FILE"
    echo "[exp-AA] Estado resetado."
fi

touch "$STATE_FILE"

STEPS=(
    "dropout_bench_v3_D_02_A_dynamic_weekly_linear_discrete_time_hazard.py"
    "dropout_bench_v3_D_02_B_dynamic_weekly_linear_discrete_time_hazard_sensibility_weighted_test.py"
    "dropout_bench_v3_D_03_A_dynamic_neural_neural_discrete_time_survival.py"
    "dropout_bench_v3_D_03_B_dynamic_neural_neural_discrete_time_survival_sensibility_weighted_test.py"
    "dropout_bench_v3_D_06_A_dynamic_weekly_poisson_piecewise_exponential.py"
    "dropout_bench_v3_D_06_B_dynamic_weekly_poisson_piecewise_exponential_sensibility_weighted_test.py"
    "dropout_bench_v3_D_07_A_dynamic_weekly_gb_weekly_hazard.py"
    "dropout_bench_v3_D_07_B_dynamic_weekly_gb_weekly_hazard_sensibility_weighted_test.py"
    "dropout_bench_v3_D_04_comparable_continuous_time_cox_early_window.py"
    "dropout_bench_v3_D_09_comparable_tree_survival_random_survival_forest.py"
    "dropout_bench_v3_D_10_comparable_tree_survival_gradient_boosted_cox.py"
    "dropout_bench_v3_D_11_comparable_parametric_weibull_aft.py"
    "dropout_bench_v3_D_12_comparable_parametric_royston_parmar.py"
    "dropout_bench_v3_D_13_comparable_tree_survival_xgboost_aft.py"
    "dropout_bench_v3_D_14_comparable_neural_neural_mtlr.py"
    "dropout_bench_v3_D_15_comparable_neural_deephit.py"
    "dropout_bench_v3_D_16_benchmark_consolidation.py"
    "dropout_bench_v3_E_posthoc_audits_refatorado_v21.py"
    "dropout_bench_v3_F_ablation_stability_refatorado_v30.py"
    "dropout_bench_v3_G_explainability_paper_refatorado_v7.py"
    "dropout_bench_v3_G_13_supplementary_hyperparameter_grids.py"
    "dropout_bench_v3_G_permutation_repeats.py"
)

TOTAL=${#STEPS[@]}

is_done() {
    grep -qxF "$1" "$STATE_FILE" 2>/dev/null
}

mark_done() {
    echo "$1" >> "$STATE_FILE"
}

log() {
    local msg="$1"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $msg" | tee -a "$MASTER_LOG"
}

cd "$ROOT"

log "=== exp-AA iniciado (total: $TOTAL etapas) ==="
log "Master log: $MASTER_LOG"
log "Estado de resume: $STATE_FILE"
echo ""

STEP_NUM=0
for SCRIPT in "${STEPS[@]}"; do
    STEP_NUM=$((STEP_NUM + 1))
    STEPLOG="$LOGDIR/${SCRIPT%.py}.log"

    if is_done "$SCRIPT"; then
        log "[$STEP_NUM/$TOTAL] SKIP (ja concluido)  $SCRIPT"
        continue
    fi

    log "[$STEP_NUM/$TOTAL] INICIO  $SCRIPT"
    T_START=$(date +%s)

    EXTRA=()
    if [[ "$SCRIPT" == "dropout_bench_v3_G_permutation_repeats.py" ]]; then
        EXTRA=(--n-repeats 30 --resume)
    fi

    if $PY "$SCRIPT" "${EXTRA[@]}" > "$STEPLOG" 2>&1; then
        T_END=$(date +%s)
        ELAPSED=$((T_END - T_START))
        mark_done "$SCRIPT"
        log "[$STEP_NUM/$TOTAL] DONE em ${ELAPSED}s  $SCRIPT"
    else
        T_END=$(date +%s)
        ELAPSED=$((T_END - T_START))
        log "[$STEP_NUM/$TOTAL] FAILED em ${ELAPSED}s  $SCRIPT"
        log ">>> Ultimo log ($STEPLOG):"
        tail -30 "$STEPLOG" | tee -a "$MASTER_LOG"
        log "=== PIPELINE INTERROMPIDA na etapa $STEP_NUM/$TOTAL ==="
        exit 1
    fi

    echo ""
done

log "=== exp-AA CONCLUIDO — todas as $TOTAL etapas OK ==="
