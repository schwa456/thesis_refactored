#!/bin/bash
# Sequentially runs the agentic filter experiments with gpt-4o-mini backbone.
# a05_16 (Reflection 3iter + gpt) is intentionally skipped for cost.
#
# After completion, update EXPERIMENT_HISTORY.md / EXPERIMENT_CATALOG.md /
# EXPERIMENT_ID_MIGRATION.md with the new rows (per CLAUDE.md §2-1 rule).
#
# Usage (from project root):
#   bash scripts/run_a05_gpt4omini_filters.sh            # all three
#   bash scripts/run_a05_gpt4omini_filters.sh a05_15     # single
#
# Requires OPENAI_API_KEY in .env (already loaded by src/main.py).

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

CONFIGS=(
  "experiments/abl/a05_filter_agentic/a05_14_adaptive_multi_agent_gpt4omini"
  "experiments/abl/a05_filter_agentic/a05_15_reflection_1iter_gpt4omini"
  "experiments/abl/a05_filter_agentic/a05_17_verifier_gpt4omini"
)

if [ $# -gt 0 ]; then
  FILTER="$1"
  SELECTED=()
  for c in "${CONFIGS[@]}"; do
    [[ "$c" == *"$FILTER"* ]] && SELECTED+=("$c")
  done
  CONFIGS=("${SELECTED[@]}")
  if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No config matched '$FILTER'. Aborting." >&2
    exit 2
  fi
fi

mkdir -p logs

GLOBAL_START=$(date +%s)
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg")
  log="logs/${name}.log"
  echo "================================================================"
  echo "▶ [$(date '+%Y-%m-%d %H:%M:%S')] START $name"
  echo "  config: $cfg"
  echo "  log:    $log"
  echo "================================================================"

  exp_start=$(date +%s)
  python src/main.py --config "$cfg" > "$log" 2>&1 || {
    echo "✗ $name FAILED (exit $?). Continuing to next experiment."
    continue
  }
  exp_end=$(date +%s)
  exp_min=$(( (exp_end - exp_start) / 60 ))

  metrics="outputs/${name}/metrics.txt"
  tokens="outputs/${name}/token_usage.json"
  echo "✓ $name DONE in ${exp_min}m"
  if [ -f "$metrics" ]; then
    echo "--- metrics ---"
    cat "$metrics"
  fi
  if [ -f "$tokens" ]; then
    echo "--- token_usage ---"
    cat "$tokens"
  fi
  echo
done

GLOBAL_END=$(date +%s)
TOTAL_MIN=$(( (GLOBAL_END - GLOBAL_START) / 60 ))
echo "================================================================"
echo "All a05 gpt-4o-mini experiments done in ${TOTAL_MIN}m total."
echo "Next step: update EXPERIMENT_HISTORY.md / EXPERIMENT_CATALOG.md /"
echo "           EXPERIMENT_ID_MIGRATION.md with the new a05_14/15/17 rows."
echo "================================================================"
