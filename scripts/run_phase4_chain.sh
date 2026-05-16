#!/bin/bash
# Phase 4.1 + 4.2 통합 Chain Sweep — 9 cells parallel (학술 agent plan §Phase 4, DECISIONS 2026-05-16 §4)
#
# Phase 4.1 (c04_phase4_alpha_sweep, 6 cells, extractor commit 1e2c46a):
#   MSTPCSTUnionExtractor seed_selection_mode="integrated_score"
#   α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, 고정 (θ, K) = anchor c01_01 (θ=0.1, K=20)
#   p4_01 (α=0.0) = anchor c01_01 deterministic 일치 검증 cell (F1=0.8664 정합)
#   p4_06 (α=1.0) = effective seed = Selector top-K only
#
# Phase 4.2 (c05_phase4_conditional_filter, 3 cells, filter commit e0685eb):
#   ConditionalFilterWrapper(inner=XiYanFilter GLM 4.7) TCR-gated voluntary skip
#   tcr_threshold ∈ {0.3, 0.5, 0.7}, 고정 (θ, K) = anchor c01_01 (θ=0.1, K=20)
#
# 사용자 5/16 spec: 9 cells parallel launch + kill 금지
# GPU 분배: GPU 0 × 4 (Phase 4.1 α=0.0~0.6) + GPU 1 × 5 (Phase 4.1 α=0.8/1.0 + Phase 4.2 3)
# Phase 4.2 의 TCR-gated skip 효과로 effective concurrency ~7 (8-conc 검증 위)
# ETA: ~2-3h wall (Phase 2 8-conc 정합 위)
#
# Launch:
#   nohup bash scripts/run_phase4_chain.sh > logs/phase4_chain_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/phase4_chain

echo "=== Phase 4.1 + 4.2 Chain START $(date -Iseconds) ==="
echo "  9 cells parallel — GPU 0 × 4 (Phase 4.1) + GPU 1 × 5 (Phase 4.1 + 4.2)"
echo "  ETA ~2-3h wall (8-conc 검증 정합)"

run_one() {
  local cell=$1
  local cfg=$2
  local gpu=$3
  local log="logs/phase4_chain/${cell}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${cell}"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py \
    --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end ${cell}"
}

# GPU 0 × 4 cells (Phase 4.1 α=0.0 ~ α=0.6)
run_one p4_01_alpha_0.0 experiments/abl/c04_phase4_alpha_sweep/p4_01_alpha_0.0 0 &
PID_01=$!
run_one p4_02_alpha_0.2 experiments/abl/c04_phase4_alpha_sweep/p4_02_alpha_0.2 0 &
PID_02=$!
run_one p4_03_alpha_0.4 experiments/abl/c04_phase4_alpha_sweep/p4_03_alpha_0.4 0 &
PID_03=$!
run_one p4_04_alpha_0.6 experiments/abl/c04_phase4_alpha_sweep/p4_04_alpha_0.6 0 &
PID_04=$!

# GPU 1 × 5 cells (Phase 4.1 α=0.8/1.0 + Phase 4.2 thr=0.3/0.5/0.7)
run_one p4_05_alpha_0.8 experiments/abl/c04_phase4_alpha_sweep/p4_05_alpha_0.8 1 &
PID_05=$!
run_one p4_06_alpha_1.0 experiments/abl/c04_phase4_alpha_sweep/p4_06_alpha_1.0 1 &
PID_06=$!
run_one p4_2_thr_0.3 experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.3 1 &
PID_07=$!
run_one p4_2_thr_0.5 experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.5 1 &
PID_08=$!
run_one p4_2_thr_0.7 experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.7 1 &
PID_09=$!

wait $PID_01 $PID_02 $PID_03 $PID_04 $PID_05 $PID_06 $PID_07 $PID_08 $PID_09

echo ""
echo "=========================================="
echo "  Phase 4.1 + 4.2 Chain 9 cells DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics 요약:"

echo ""
echo "[Phase 4.1 — 6 cells α sweep, anchor c01_01 F1=0.8664]"
for d in outputs/experiments/abl/c04_phase4_alpha_sweep/p4_*/; do
  cell=$(basename "$d")
  m="${d}metrics.txt"
  if [ -f "$m" ]; then
    R=$(grep "^recall:" "$m" | awk '{print $2}')
    P=$(grep "^precision:" "$m" | awk '{print $2}')
    EX=$(grep "^ex:" "$m" | awk '{print $2}')
    F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
    printf "  %-25s R=%s P=%s F1=%s EX=%s\n" "${cell}" "${R}" "${P}" "${F1}" "${EX}"
  else
    printf "  %-25s (metrics 미생성)\n" "${cell}"
  fi
done

echo ""
echo "[Phase 4.2 — 3 cells TCR-conditional filter]"
for d in outputs/experiments/abl/c05_phase4_conditional_filter/p4_*/; do
  cell=$(basename "$d")
  m="${d}metrics.txt"
  if [ -f "$m" ]; then
    R=$(grep "^recall:" "$m" | awk '{print $2}')
    P=$(grep "^precision:" "$m" | awk '{print $2}')
    EX=$(grep "^ex:" "$m" | awk '{print $2}')
    F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
    printf "  %-25s R=%s P=%s F1=%s EX=%s\n" "${cell}" "${R}" "${P}" "${F1}" "${EX}"
  else
    printf "  %-25s (metrics 미생성)\n" "${cell}"
  fi
done

echo ""
echo "=== Phase 4.1 + 4.2 Chain DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (Phase 4.1 + 4.2 통합)"
echo "  2. Analyzer 핸드오프 (Phase 4.1: phase4_1_integrated_alpha_sweep_2026-05-XX.md,"
echo "                       Phase 4.2: phase4_2_conditional_filter_2026-05-XX.md)"
