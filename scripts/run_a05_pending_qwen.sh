#!/bin/bash
# Run the remaining Qwen-backboned a05 filter ablations in sequence.
# GPT-4o-mini variants (a05_11, a05_12) are intentionally excluded.
# vLLM server must be running on localhost:8000 (GPU 0,1).
# Main pipeline runs on GPU 2,3 (GAT inference only).

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CONFIGS=(
    "a05_05_tiered_no_tools"
    "a05_06_tiered_full_tools"
    "a05_07_adaptive_depth"
    "a05_08_tiered_verifier_stack"
    "a05_09_tiered_retry"
    "a05_10_adaptive_retry"
)

BASE="abl/a05_filter_agentic"

for name in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "  Running ${name}  ($(date -Iseconds))"
    echo "========================================"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 python src/main.py --config "experiments/${BASE}/${name}" \
        || echo "[!] ${name} FAILED — continuing with next"
    echo ""
done

echo "=== ALL QWEN a05 PENDING COMPLETE ==="
for name in "${CONFIGS[@]}"; do
    echo "--- ${name} ---"
    cat "outputs/experiments/${BASE}/${name}/metrics.txt" 2>/dev/null || echo "NO METRICS"
done
