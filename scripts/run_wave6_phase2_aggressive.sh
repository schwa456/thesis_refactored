#!/bin/bash
# Wave 6 Phase 2 (a+aggressive) — M3 + M4 + M5 동시 launch (학술 agent §5+§6+§7)
# DECISIONS 2026-05-16 (Wave 6 Phase 2 a+aggressive M2~M5 동시 launch) §2 + §3
# Module:Filter commit 88ad47e (M3 MultiPromptVotingFilter + M4 BidirectionalFilter + M5 TwoStageFilter)
#
# M2 (이미 launch, wrapper PID 1249922) 와 wall 정합 — 전체 Phase 2 wall ~3h 정합
# 3 cells parallel — GPU 0+1 split, GLM API rate limit 정합 (~3 streams conservative)
# Total LLM calls: M3 4602 + M4 3068 + M5 3068 = 10,738 calls, ~$25-50 GLM 4.7, ~2-3h wall
#
# Spec:
#   M3 MultiPromptVotingFilter: 3 prompts × 3 voting (OR/MAJORITY/AND), single config measures all
#   M4 BidirectionalFilter: Forward (M1-A) + Backward (SQL Schema Analyst) union
#   M5 TwoStageFilter: Sequential Stage1 (Coarse Recall) → Stage2 (Fine Precision)
#
# Launch:
#   nohup bash scripts/run_wave6_phase2_aggressive.sh > logs/wave6_phase2_aggressive_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/wave6_phase2_aggressive

echo "=== Wave 6 Phase 2 (a+aggressive) START $(date -Iseconds) ==="
echo "  3 cells parallel — GPU 0 × 2 (M3 + M4) + GPU 1 × 1 (M5)"
echo "  Total: 10,738 LLM calls, ~\$25-50 GLM 4.7, ~2-3h wall"
echo "  M2 (w6_p2a_m2cot_strong, PID 1249922) 공존 — 종료 ~23:13"

run_one() {
  local cell=$1
  local cfg=$2
  local gpu=$3
  local log="logs/wave6_phase2_aggressive/${cell}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${cell}"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py \
    --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end ${cell}"
}

# GPU 0: M3 + M4 (2-conc, lighter — Voting + Bidirectional non-sequential)
run_one w6_p2_m3_voting \
        experiments/abl/wave6_recall_biased/w6_p2_m3_voting 0 &
PID_M3=$!

run_one w6_p2_m4_bidirectional \
        experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional 0 &
PID_M4=$!

# GPU 1: M5 (1-conc, sequential 2-stage)
run_one w6_p2_m5_two_stage \
        experiments/abl/wave6_recall_biased/w6_p2_m5_two_stage 1 &
PID_M5=$!

wait $PID_M3 $PID_M4 $PID_M5

echo ""
echo "=========================================="
echo "  Wave 6 Phase 2 (a+aggressive) M3+M4+M5 DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics 요약 (anchor c01_01 F1=0.8664 EX=0.5176 비교):"

for cell in w6_p2_m3_voting w6_p2_m4_bidirectional w6_p2_m5_two_stage; do
  m="outputs/experiments/abl/wave6_recall_biased/${cell}/metrics.txt"
  if [ -f "$m" ]; then
    R=$(grep "^recall:" "$m" | awk '{print $2}')
    P=$(grep "^precision:" "$m" | awk '{print $2}')
    EX=$(grep "^ex:" "$m" | awk '{print $2}')
    F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
    printf "  %-30s R=%s P=%s F1=%s EX=%s\n" "${cell}" "${R}" "${P}" "${F1}" "${EX}"
  else
    printf "  %-30s (metrics 미생성)\n" "${cell}"
  fi
done

echo ""
echo "=== Wave 6 Phase 2 (a+aggressive) M3+M4+M5 DONE $(date -Iseconds) ==="
echo "Next (M2 종료 후 통합):"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (M2+M3+M4+M5 통합 entry)"
echo "  2. Analyzer 핸드오프 (Phase 3 분석):"
echo "     - results_all_methods.csv 통합 (M2+M3+M4+M5)"
echo "     - Pareto frontier R ≥ 0.90 ∧ P ≥ 0.75"
echo "     - axis #15 정식 채택 결정 (F1 > 0.8672 통계 robust)"
echo "     - axis #11 narrative Option A retain / Option B reinterpret"
echo "     - Top 2 methodology 조합 candidate"
