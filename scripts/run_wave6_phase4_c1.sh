#!/bin/bash
# Wave 6 Phase 4 Top 2 조합 C1 (M4 + M1-B strong) launch
# DECISIONS 2026-05-17 §4 Top 2 C1 Spec
# Module:Filter commit 60b6988 (BidirectionalFilter bidirectional_forward_prompt_mode)
#
# Single cell: w6_p4_c1_m4_strong (Forward=strong + Backward=bidirectional_backward)
# Cost: 3068 LLM calls (1534 × 2), ~$2-4 GLM 4.7, ~1.5h wall
# Baselines:
#   anchor c01_01: F1=0.8664 EX=0.5176
#   M4 baseline (mild Forward): F1=0.8370 EX=0.5300 ★ EX-max
#   M1-B strong baseline: F1=0.8655 ★ F1-best M1 EX=0.5130
# Expected: F1 sweet spot 0.85~0.87 + EX gain +0.01~0.02 + Pareto frontier 진입
#
# Launch:
#   nohup bash scripts/run_wave6_phase4_c1.sh > logs/wave6_phase4_c1_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/wave6_phase4_c1

echo "=== Wave 6 Phase 4 Top 2 C1 START $(date -Iseconds) ==="
echo "  Single cell: w6_p4_c1_m4_strong"
echo "  Stack: M4 BidirectionalFilter (Forward=strong + Backward=bidirectional_backward)"
echo "  Cost: 3068 LLM calls (1534 × 2), ~\$2-4, ETA ~1.5h wall"
echo "  GPU 0 (single launch, no parallelism)"

cell=w6_p4_c1_m4_strong
cfg="experiments/abl/wave6_recall_biased/${cell}"
log="logs/wave6_phase4_c1/${cell}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] start ${cell}"
CUDA_VISIBLE_DEVICES=0 conda run -n base python -u src/main.py \
  --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
echo "[$(date +%H:%M:%S)] end ${cell}"

echo ""
echo "=========================================="
echo "  Wave 6 Phase 4 Top 2 C1 DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics:"

d="outputs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong/"
m="${d}metrics.txt"
if [ -f "$m" ]; then
  R=$(grep "^recall:" "$m" | awk '{print $2}')
  P=$(grep "^precision:" "$m" | awk '{print $2}')
  EX=$(grep "^ex:" "$m" | awk '{print $2}')
  F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
  echo ""
  echo "  C1 result: R=${R} P=${P} F1=${F1} EX=${EX}"
  echo ""
  echo "  Baselines:"
  echo "    anchor c01_01:    R=0.8748 P=0.8582 F1=0.8664 EX=0.5176"
  echo "    M4 baseline:      R=0.9325 P=0.7593 F1=0.8370 EX=0.5300 ★ EX-max"
  echo "    M1-B strong:      R=0.9022 P=0.8316 F1=0.8655 ★ F1-best M1 EX=0.5130"
  echo "  → 학술 agent §10: F1 ≥ 0.8672?"
  echo "  → Pareto frontier: R ≥ 0.90 ∧ P ≥ 0.75?"
else
  echo "  (metrics.txt 미생성)"
fi

echo ""
echo "=== Wave 6 Phase 4 Top 2 C1 DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (Phase 4 C1 entry)"
echo "  2. Analyzer 핸드오프: notebooks/analysis_results/wave6_phase4_c1_2026-05-17.md"
echo "     - C1 vs M4 baseline + M1-B strong baseline 의 ΔF1/ΔEX 분석 (synergy or additive)"
echo "     - C1 의 backward_added mechanism 변동 (strong < mild inclusive)"
echo "     - C2 (M4 + M3 MAJORITY) launch 결정"
