#!/usr/bin/env bash
# run_plan_priority.sh — EXPERIMENT_PLAN.md §4 Phase B/C 우선순위대로 일괄 실행.
#
# Phase B (B-III metadata 공유 축): S-V → E-III sweep → FL-III detect → E-II sweep
# Phase C (재조합/확장): int_01~03 → E-I sweep → F3 Tiered
# 실패해도 다음 실험으로 진행. 각 실험 종료 후 HISTORY/CATALOG/ID_MIGRATION
# 업데이트는 별도로 수행 (CLAUDE.md rule).
#
# Precondition: vLLM 서버가 http://localhost:8000 에서 Qwen3-Coder-30B 를 서빙 중이어야 한다.
# 필요 GPU: GAT 추론용 + vLLM 용. s06_a01_07 종료 후 실행.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="logs/plan_priority_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/run.log"
echo "=== Plan priority run started at $(date -Iseconds) ===" | tee "$RUN_LOG"

# ── Phase B ────────────────────────────────────────────────────────────
PHASE_B=(
  "experiments/abl/sel/ns_l1/abl_sel_ns_l1_01"                          # 1 S-V NSL1
  "experiments/abl/a06_ext_fkprior/a06_01_fkprior_discount03"           # 2a E-III
  "experiments/abl/a06_ext_fkprior/a06_02_fkprior_bridge05"             # 2b
  "experiments/abl/a06_ext_fkprior/a06_03_fkprior_aggressive"           # 2c
  "experiments/abl/a06_ext_fkprior/a06_04_fkprior_discount03_xiyan"     # 2d E-III + XiYan
  "experiments/abl/a05_filter_agentic/a05_21_symverify_xiyan_detect"    # 3 FL-III detect
  "experiments/abl/a07_ext_path_ensemble/a07_01_path_union_k5"          # 4a E-II
  "experiments/abl/a07_ext_path_ensemble/a07_02_path_2pass_k5"          # 4b
  "experiments/abl/a07_ext_path_ensemble/a07_03_path_union_k3"          # 4c
  "experiments/abl/a07_ext_path_ensemble/a07_04_path_2pass_k10"         # 4d
  "experiments/abl/a07_ext_path_ensemble/a07_05_path_union_k5_xiyan"    # 4e E-II + XiYan
)

# ── Phase C ────────────────────────────────────────────────────────────
PHASE_C=(
  "experiments/abl/integration/int_01_e1_refl"                          # 5a int E1×Refl
  "experiments/abl/integration/int_02_e2_refl"                          # 5b int E2×Refl
  "experiments/abl/integration/int_03_e2_path_refl"                     # 5c int E2×Path×Refl
  "experiments/abl/a08_ext_louvain/a08_01_louvain_top2"                 # 6a E-I
  "experiments/abl/a08_ext_louvain/a08_02_louvain_res05"                # 6b
  "experiments/abl/a08_ext_louvain/a08_03_louvain_adaptive_top3"        # 6c
  "experiments/abl/a05_filter_agentic/a05_06_tiered_full_tools"         # 7 F3 Tiered
)

run_one () {
  local cfg="$1"
  local tag="${cfg##*/}"
  local log="${LOG_DIR}/${tag}.log"
  local t0 t1
  t0=$(date +%s)
  echo "--- [$(date -Iseconds)] START $cfg" | tee -a "$RUN_LOG"
  conda run -n base --no-capture-output \
    python src/main.py --config "$cfg" >"$log" 2>&1 || true
  t1=$(date +%s)
  echo "--- [$(date -Iseconds)] END   $cfg  (elapsed $((t1-t0))s)" | tee -a "$RUN_LOG"
}

echo "=== Phase B (${#PHASE_B[@]} experiments) ===" | tee -a "$RUN_LOG"
for cfg in "${PHASE_B[@]}"; do run_one "$cfg"; done

echo "=== Phase C (${#PHASE_C[@]} experiments) ===" | tee -a "$RUN_LOG"
for cfg in "${PHASE_C[@]}"; do run_one "$cfg"; done

echo "=== Plan priority run finished at $(date -Iseconds) ===" | tee -a "$RUN_LOG"
echo "Next: write configs/experiments/abl/integration/int_04_ns_full.yaml"
echo "       after inspecting the best hyperparameters from the a06/a07/a08 sweeps."
echo "Remember to update EXPERIMENT_HISTORY.md / CATALOG.md / ID_MIGRATION.md."
