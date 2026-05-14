#!/usr/bin/env bash
# Quick single experiment runner (nohup).
# Usage: bash scripts/run_main.sh <config_name>
#   e.g.: bash scripts/run_main.sh experiments/s03_gat_ensemble/a02_adaptive_pcst/s03_a02_03_xiyan_filter
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CONFIG="${1:?Usage: $0 <config_name>}"
EXP_NAME=$(basename "$CONFIG")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
nohup python src/main.py --config "$CONFIG" > "logs/${EXP_NAME}.log" 2>&1 &
echo "PID=$! — logs/${EXP_NAME}.log"
