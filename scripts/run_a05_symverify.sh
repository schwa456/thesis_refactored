#!/bin/bash
# FL-III (Symbolic Verifier) batch — sequential execution of 4 configs:
#   a05_21: XiYan + detect-only         (measures disconnected rate on anchor)
#   a05_19: XiYan + auto_repair         (vs anchor a03_17 F1=0.6940)
#   a05_20: Reflection + auto_repair    (vs current best a05_02 F1=0.7068)
#   a05_22: Reflection+Verifier+Symverify stacked (3-layer ceiling)
#
# Prerequisite: Qwen vLLM server on localhost:8000 (scripts/run_vllm_server.sh).
#
# After completion, update EXPERIMENT_HISTORY.md / EXPERIMENT_CATALOG.md /
# EXPERIMENT_ID_MIGRATION.md per CLAUDE.md §2-1 and memory rule.
#
# Usage (from project root):
#   bash scripts/run_a05_symverify.sh                  # all four
#   bash scripts/run_a05_symverify.sh a05_21           # single by substring
#   bash scripts/run_a05_symverify.sh a05_19 a05_20    # subset by substring

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

CONFIGS=(
  "experiments/abl/a05_filter_agentic/a05_21_symverify_xiyan_detect"
  "experiments/abl/a05_filter_agentic/a05_19_symverify_xiyan_repair"
  "experiments/abl/a05_filter_agentic/a05_20_symverify_reflection_repair"
  "experiments/abl/a05_filter_agentic/a05_22_symverify_reflection_verifier_stacked"
)

if [ $# -gt 0 ]; then
  SELECTED=()
  for pat in "$@"; do
    for c in "${CONFIGS[@]}"; do
      [[ "$c" == *"$pat"* ]] && SELECTED+=("$c")
    done
  done
  CONFIGS=("${SELECTED[@]}")
  if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No config matched arg(s): $*" >&2
    exit 2
  fi
fi

LOGS_ROOT="logs/experiments/abl/a05_filter_agentic"
mkdir -p "$LOGS_ROOT"

GLOBAL_START=$(date +%s)
echo "================================================================"
echo "FL-III SymVerify batch — ${#CONFIGS[@]} experiment(s)"
echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg")
  log="${LOGS_ROOT}/${name}_wrapper.log"
  echo
  echo "▶ [$(date '+%Y-%m-%d %H:%M:%S')] START $name"
  echo "  config: $cfg"
  echo "  log:    $log"

  exp_start=$(date +%s)
  python src/main.py --config "$cfg" > "$log" 2>&1 || {
    echo "✗ $name FAILED (exit $?). Continuing to next experiment."
    continue
  }
  exp_end=$(date +%s)
  exp_min=$(( (exp_end - exp_start) / 60 ))

  metrics="outputs/${cfg}/metrics.txt"
  preds="outputs/${cfg}/predictions.jsonl"
  echo "✓ $name DONE in ${exp_min}m"
  if [ -f "$metrics" ]; then
    echo "--- metrics ---"
    cat "$metrics"
  fi
  if [ -f "$preds" ]; then
    total=$(wc -l < "$preds")
    invalid=$(grep -c '"connectivity_valid": *false' "$preds" 2>/dev/null || echo 0)
    repaired=$(grep -c '"status": *"repaired"' "$preds" 2>/dev/null || echo 0)
    echo "--- connectivity ---"
    echo "  total predictions: $total"
    echo "  connectivity_valid=false: $invalid"
    echo "  status=repaired:          $repaired"
  fi
done

GLOBAL_END=$(date +%s)
TOTAL_MIN=$(( (GLOBAL_END - GLOBAL_START) / 60 ))
echo
echo "================================================================"
echo "All FL-III SymVerify experiments done in ${TOTAL_MIN}m total."
echo "Next step (per CLAUDE.md §2-1):"
echo "  1. EXPERIMENT_HISTORY.md  — add rows for a05_19/20/21/22"
echo "  2. EXPERIMENT_CATALOG.md  — update a05 table + construction spec"
echo "  3. EXPERIMENT_ID_MIGRATION.md — add 4 new entries"
echo "Metric format: R, P, F1 (4 decimal places)."
echo "================================================================"
