#!/bin/bash
# Wave 6 Phase 1 M1 Recall-Biased Prompt sweep — 3 variants parallel
# DECISIONS 2026-05-16 Wave 6 신규 활성 entry §2 Phase 1 Spec
# 학술 agent filter improve plan §3 방법론 1
# Module:Filter commit 07d2fda — XiYanFilter prompt_mode + sanitize_filter_output
#
# Variants:
#   M1-A mild: RELEVANT or POTENTIALLY RELEVANT + WHEN IN DOUBT INCLUDE
#   M1-B strong: Default decision is INCLUDE + 명시적 exclusion criteria
#   M1-C exclusion_rule: 4-rule conjunctive exclusion + UNSURE → KEEP
#
# Anchor stack 그대로 (c01_01: QCondGAT + MSTPCSTUnion + XiYanFilter GLM 4.7 + LLMSQL)
# Filter prompt 만 교체 (XiYanFilter.prompt_mode parameter)
# 1 LLM call/q × 3 variants × 1534q = 4602 calls, cost ~$3-6
# GPU 0 × 2 + GPU 1 × 1 = 3-conc parallel, ETA ~1.5h
# sanitize_filter_output() default-on (Hallucination 방지, 학술 agent §2.3)
#
# Launch:
#   nohup bash scripts/run_wave6_phase1_recall_biased.sh > logs/wave6_phase1_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/wave6_phase1

echo "=== Wave 6 Phase 1 M1 Recall-Biased START $(date -Iseconds) ==="
echo "  3 variants parallel — GPU 0 × 2 (mild + strong) + GPU 1 × 1 (exclusion_rule)"
echo "  4602 LLM calls (1×/q × 3 variants × 1534q), cost ~\$3-6, ETA ~1.5h wall"

run_one() {
  local cell=$1
  local cfg=$2
  local gpu=$3
  local log="logs/wave6_phase1/${cell}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${cell}"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py \
    --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end ${cell}"
}

# GPU 0: mild + strong (2-conc)
run_one wave6_p1_recall_biased_mild \
        experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_mild 0 &
PID_MILD=$!

run_one wave6_p1_recall_biased_strong \
        experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_strong 0 &
PID_STRONG=$!

# GPU 1: exclusion_rule (1-conc)
run_one wave6_p1_recall_biased_exclusion_rule \
        experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_exclusion_rule 1 &
PID_EXCL=$!

wait $PID_MILD $PID_STRONG $PID_EXCL

echo ""
echo "=========================================="
echo "  Wave 6 Phase 1 3 cells DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics 요약:"
echo ""
echo "[Wave 6 Phase 1 — M1 Recall-Biased 3 variants, anchor c01_01 F1=0.8664]"
for d in outputs/experiments/abl/wave6_recall_biased/wave6_p1_*/; do
  cell=$(basename "$d")
  m="${d}metrics.txt"
  if [ -f "$m" ]; then
    R=$(grep "^recall:" "$m" | awk '{print $2}')
    P=$(grep "^precision:" "$m" | awk '{print $2}')
    EX=$(grep "^ex:" "$m" | awk '{print $2}')
    F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
    printf "  %-50s R=%s P=%s F1=%s EX=%s\n" "${cell}" "${R}" "${P}" "${F1}" "${EX}"
  else
    printf "  %-50s (metrics 미생성)\n" "${cell}"
  fi
done

echo ""
echo "=== Wave 6 Phase 1 DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (Wave 6 Phase 1 신규 entry)"
echo "  2. Analyzer 핸드오프: notebooks/analysis_results/wave6_phase1_recall_biased_2026-05-16.md"
echo "     - R_fil / P_fil / F1_fil / FNR / FPR / Prune% / LLM_calls per variant"
echo "     - Hallucination rate (filter_hallucination_removed_count / filter_input_node_count)"
echo "     - Phase 2 분기 권고: R_fil 기준 (≥0.92 → a, 0.88-0.92 → b, <0.88 → c)"
echo "  3. 학술 agent §10 성공 기준: F1_fil ≥ 0.8672 (anchor 하한선, 필수)"
