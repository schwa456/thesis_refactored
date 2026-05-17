#!/bin/bash
# Wave 6 Phase 5 Top 2 C2 — M4 + M3 MAJORITY voting Forward launch
# DECISIONS 2026-05-17 §6 C2 launch 결정
# Module:Filter commit 7a07a6b — BidirectionalFilter + voting_multi_prompt Forward composition
# (M3 MultiPromptVotingFilter logic 의 voting Forward 를 BidirectionalFilter Forward 로 swap-in)
#
# Single cell: w6_p5_c2_m4_majority
# Stack:
#   - base: anchor c01_01 (Enriched + QCondGAT + MSTPCSTUnion + GLM 4.7 + SQL gen)
#   - filter: BidirectionalFilter
#     * Forward: voting_multi_prompt (3 prompts × MAJORITY ≥2 votes)
#     * Backward: bidirectional_backward (SQL Schema Analyst, retain from M4)
#
# Cost: 4 LLM call/q × 1534 = 6136 calls (3 voting Forward + 1 Backward)
# Wall: ~3h (BidirectionalFilter 내부 Forward 3 prompts parallel + Backward sequential)
# Cost: ~$10-15 GLM 4.7
#
# 3 hypothesis 검증 (DECISIONS §6):
#   H1 — Forward inclusiveness dominant: C2 EX ≈ M4 EX (0.5300) 시 confirm
#   H2 — Forward mechanism dominant   : C2 EX ≈ C1 (0.5150) 시 confirm
#   H3 — Partial entanglement         : C2 EX intermediate (0.52~0.53) 시 confirm
#
# Baselines:
#   anchor c01_01:        R=0.8748 P=0.8582 F1=0.8664 EX=0.5176
#   M4 (mild Forward):    R=0.9325 P=0.7593 F1=0.8370 EX=0.5300 ★ EX-max
#   C1 (strong Forward):  R=0.9177 P=0.8109 F1=0.8610 EX=0.5150 (Partial Degrade)
#   M3 MAJORITY (post-hoc): R=0.9290 P=0.7934 F1=0.8433
#
# Launch:
#   nohup bash scripts/run_wave6_phase5_c2.sh > logs/wave6_phase5_c2_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/wave6_phase5_c2

echo "=== Wave 6 Phase 5 Top 2 C2 START $(date -Iseconds) ==="
echo "  Single cell: w6_p5_c2_m4_majority"
echo "  Stack: BidirectionalFilter + voting_multi_prompt MAJORITY Forward + bidirectional_backward"
echo "  Cost: 6136 LLM calls (4 LLM/q × 1534), ~\$10-15, ETA ~3h wall"
echo "  GPU 0 (single launch, BidirectionalFilter 내부 Forward 3 parallel)"

cell=w6_p5_c2_m4_majority
cfg="experiments/abl/wave6_recall_biased/${cell}"
log="logs/wave6_phase5_c2/${cell}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] start ${cell}"
CUDA_VISIBLE_DEVICES=0 conda run -n base python -u src/main.py \
  --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
echo "[$(date +%H:%M:%S)] end ${cell}"

echo ""
echo "=========================================="
echo "  Wave 6 Phase 5 Top 2 C2 DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics:"

d="outputs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority/"
m="${d}metrics.txt"
if [ -f "$m" ]; then
  R=$(grep "^recall:" "$m" | awk '{print $2}')
  P=$(grep "^precision:" "$m" | awk '{print $2}')
  EX=$(grep "^ex:" "$m" | awk '{print $2}')
  F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
  echo ""
  echo "  C2 result: R=${R} P=${P} F1=${F1} EX=${EX}"
  echo ""
  echo "  Baselines:"
  echo "    anchor c01_01:        R=0.8748 P=0.8582 F1=0.8664 EX=0.5176"
  echo "    M4 (mild Forward):    R=0.9325 P=0.7593 F1=0.8370 EX=0.5300 ★ EX-max"
  echo "    C1 (strong Forward):  R=0.9177 P=0.8109 F1=0.8610 EX=0.5150"
  echo ""
  echo "  3 hypothesis 판정 (C2 EX ${EX}):"
  echo "    H1 inclusiveness dominant : C2 EX ≈ M4 (0.5300)? → Forward inclusiveness 단독 효과"
  echo "    H2 mechanism dominant     : C2 EX ≈ C1 (0.5150)? → Voting noise pruning 영향"
  echo "    H3 partial entanglement   : C2 EX intermediate (0.52~0.53)?"
else
  echo "  (metrics.txt 미생성)"
fi

echo ""
echo "=== Wave 6 Phase 5 Top 2 C2 DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (Phase 5 C2 entry)"
echo "  2. Analyzer 핸드오프: notebooks/analysis_results/wave6_phase5_c2_2026-05-17.md"
echo "     - 3 hypothesis (H1/H2/H3) 판정"
echo "     - Forward Dominance 3-cell complete coverage (M4 mild + C1 strong + C2 voting)"
echo "     - axis #15 triple → quadruple evidence (M1+M4+C1+C2)"
