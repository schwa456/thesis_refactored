#!/usr/bin/env bash
# V7-W1 (FKH, RFP #3) — 25 cells × 5 seeds 순차 launcher
# anchor stack = c01_01_wave7_relog (XiYanFilter glm-4.7 EX 0.5176 baseline)
# Refs: planning/extractor/extractor_redesign_v7_plan_2026-06-04.md §1.3
#       planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md §3
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1   # root 세션 위 결정 (memory feedback_gpu_allocation)
#   bash scripts/run_v7_w1_fkh_sweep.sh
#
# 실패 cell 은 || true 위 skip (다음 cell 진행). GLM API rate limit 시 sleep 추가
# 권장 — 본 script 는 default sequential.

set -u

cd "$(dirname "$0")/.."

CELLS=("fkh_00" "fkh_01" "fkh_02" "fkh_03" "fkh_04")
SEEDS=("42" "123" "7" "456" "789")

TOTAL=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))
COUNTER=0
SUCCESS=0
FAILED=0

LOG_DIR="logs/v7_w1_fkh"
mkdir -p "$LOG_DIR"

echo "[V7-W1 FKH sweep] launching $TOTAL configs (5 cells × 5 seeds)"
echo "[V7-W1 FKH sweep] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[V7-W1 FKH sweep] log_dir=$LOG_DIR"
echo

for cell in "${CELLS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNTER=$((COUNTER + 1))
        CFG="experiments/abl/v7_extractor_redesign/${cell}_seed${seed}"
        LOGFILE="${LOG_DIR}/${cell}_seed${seed}.log"
        echo "[$COUNTER/$TOTAL] $CFG"
        if conda run -n base python src/main.py --config "$CFG" > "$LOGFILE" 2>&1; then
            SUCCESS=$((SUCCESS + 1))
            echo "    OK ($LOGFILE)"
        else
            FAILED=$((FAILED + 1))
            echo "    FAILED — see $LOGFILE"
            # 다음 cell 로 진행 (|| true equivalent)
        fi
    done
done

echo
echo "[V7-W1 FKH sweep] DONE — success=$SUCCESS failed=$FAILED total=$TOTAL"
echo "[V7-W1 FKH sweep] outputs at outputs/experiments/abl/v7_extractor_redesign/fkh_*/"
