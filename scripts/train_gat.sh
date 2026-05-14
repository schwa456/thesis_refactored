#!/usr/bin/env bash
# Train GAT model in background.
# Usage: bash scripts/train_gat.sh [config_yaml]
#   default: configs/training/train_gat_config.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CONFIG="${1:-configs/training/train_gat_config.yaml}"
LOG_NAME=$(basename "$CONFIG" .yaml)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
nohup python src/train_gat.py --config "$CONFIG" > "logs/train/${LOG_NAME}.log" 2>&1 &
echo "PID=$! — logs/train/${LOG_NAME}.log"
