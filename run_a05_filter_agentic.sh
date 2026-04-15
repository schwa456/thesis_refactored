#!/bin/bash
# a05 Filter Agentic sweep: F1-F5 filter module enhancements.
# Anchor: a03_17 components (SuperNode Direct + Fixed PCST).
#
# Prereqs:
#   1) vLLM server running: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 on localhost:8000
#   2) For a05_11/12 (GPT-4o-mini): export OPENAI_API_KEY in .env
#
# Usage:
#   bash run_a05_filter_agentic.sh              # all 12
#   bash run_a05_filter_agentic.sh a05_06       # one
#   bash run_a05_filter_agentic.sh a05_02 a05_06 a05_09  # subset

set -e

ALL_CONFIGS=(
    "a05_01_adaptive_multi_agent"
    "a05_02_reflection_1iter"
    "a05_03_reflection_3iter"
    "a05_04_verifier"
    "a05_05_tiered_no_tools"
    "a05_06_tiered_full_tools"
    "a05_07_adaptive_depth"
    "a05_08_tiered_verifier_stack"
    "a05_09_tiered_retry"
    "a05_10_adaptive_retry"
    "a05_11_tiered_gpt4omini"
    "a05_12_adaptive_retry_gpt4omini"
)

if [ "$#" -eq 0 ]; then
    CONFIGS=("${ALL_CONFIGS[@]}")
else
    CONFIGS=()
    for arg in "$@"; do
        for full in "${ALL_CONFIGS[@]}"; do
            if [[ "$full" == "$arg"* ]]; then
                CONFIGS+=("$full")
            fi
        done
    done
fi

BASE="abl/a05_filter_agentic"

for name in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "  Running ${name}"
    echo "========================================"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=2,3 python src/main.py --config "experiments/${BASE}/${name}"
    echo ""
done

echo "ALL ${#CONFIGS[@]} EXPERIMENTS COMPLETE"
for name in "${CONFIGS[@]}"; do
    echo "--- ${name} ---"
    cat "outputs/experiments/${BASE}/${name}/metrics.txt" 2>/dev/null || echo "NO METRICS"
done
