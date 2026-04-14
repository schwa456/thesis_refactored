#!/bin/bash
# SuperNode Direct Binary threshold sweep: Steiner + XiYan
# Thresholds: 0.05, 0.10, 0.15, 0.20 (신규 ID: abl_a04_01 ~ _04)

set -e

CONFIGS=(
    "abl/a04_direct_binary_steiner_sweep/abl_a04_01_supernode_t005_steiner_xiyan"
    "abl/a04_direct_binary_steiner_sweep/abl_a04_02_supernode_t010_steiner_xiyan"
    "abl/a04_direct_binary_steiner_sweep/abl_a04_03_supernode_t015_steiner_xiyan"
    "abl/a04_direct_binary_steiner_sweep/abl_a04_04_supernode_t020_steiner_xiyan"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg")
    echo "========================================"
    echo "  Running ${name}"
    echo "========================================"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/main.py --config "experiments/${cfg}"
    echo ""
done

echo "ALL 4 EXPERIMENTS COMPLETE"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg")
    echo "--- ${name} ---"
    cat "outputs/experiments/${cfg}/metrics.txt" 2>/dev/null || echo "NO METRICS"
done
