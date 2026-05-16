#!/bin/bash
# Wave 6 Phase 2 (a) — M2 CoT + Confidence-Gated + M1 best (strong) 조합 launch
# DECISIONS 2026-05-16 (Phase 2 (a) 분기 활성 결정) §2 Phase 2 (a) Spec
# 학술 agent §3.3 Phase 2 + §4 M2 CoT spec
# Module:Filter commit 7dac875 (cot_reasoning + confidence_gated + confidence_threshold)
#
# Single cell launch (1 config) — anchor 비교: c01_01 (F1=0.8664) + M1 strong (F1=0.8655)
# Confidence threshold 0.5 → "medium" gate level (학술 agent §4.1)
# Sanitize default-on (Hallucination 방지)
#
# Cost:
#   LLM call/q: 2 (M1 prompt + M2 CoT)
#   Total: 3068 calls (1534 × 2)
#   ETA: ~1.7h wall (anchor 2873 정합 — CoT prompt length 약간 증가)
#   GLM 4.7: ~$2-4
#
# Expected outcomes (학술 agent §10 success criterion):
#   F1_fil > 0.8672 (anchor 하한선)
#   R_fil ≥ M1 strong baseline 0.9022 (R lift retain)
#   P_fil ≥ M1 strong baseline 0.8316 (Confidence-Gated P recovery)
#
# Launch:
#   nohup bash scripts/run_wave6_phase2a_cot.sh > logs/wave6_phase2a_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/wave6_phase2a

echo "=== Wave 6 Phase 2 (a) M2 CoT + Confidence-Gated START $(date -Iseconds) ==="
echo "  Single cell: w6_p2a_m2cot_strong"
echo "  Stack: M1 best (recall_biased_strong) + M2 CoT + Confidence-Gated (thr=0.5)"
echo "  Cost: 3068 LLM calls (1534 × 2), ~\$2-4, ETA ~1.7h wall"
echo "  GPU 0 (single launch, no parallelism)"

cell=w6_p2a_m2cot_strong
cfg="experiments/abl/wave6_recall_biased/${cell}"
log="logs/wave6_phase2a/${cell}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date +%H:%M:%S)] start ${cell}"
CUDA_VISIBLE_DEVICES=0 conda run -n base python -u src/main.py \
  --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
echo "[$(date +%H:%M:%S)] end ${cell}"

echo ""
echo "=========================================="
echo "  Wave 6 Phase 2 (a) DONE"
echo "=========================================="
echo "[$(date -Iseconds)] Metrics:"

d="outputs/experiments/abl/wave6_recall_biased/w6_p2a_m2cot_strong/"
m="${d}metrics.txt"
if [ -f "$m" ]; then
  R=$(grep "^recall:" "$m" | awk '{print $2}')
  P=$(grep "^precision:" "$m" | awk '{print $2}')
  EX=$(grep "^ex:" "$m" | awk '{print $2}')
  F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
  echo ""
  echo "  Phase 2 (a) result: R=${R} P=${P} F1=${F1} EX=${EX}"
  echo ""
  echo "  Baselines:"
  echo "    anchor c01_01:    R=0.8748 P=0.8582 F1=0.8664 EX=0.5176"
  echo "    M1 strong:        R=0.9022 P=0.8316 F1=0.8655 EX=0.5130"
  echo "    학술 agent §10:    F1 ≥ 0.8672 (success criterion)"
  echo "  → F1=${F1} 비교 ≥ 0.8672?"
else
  echo "  (metrics.txt 미생성)"
fi

echo ""
echo "=== Wave 6 Phase 2 (a) DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 (Phase 2 (a) entry)"
echo "  2. Analyzer 핸드오프: notebooks/analysis_results/wave6_phase2a_cot_2026-05-16.md"
echo "     - axis #15 정식 채택 결정 (F1>0.8672 시 Option B reinterpret)"
echo "     - axis #11 narrative Option A retain (F1 미달 시) 또는 Option B reinterpret (달성 시)"
