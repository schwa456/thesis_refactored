#!/usr/bin/env bash
# Run benchmark suite in background.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup python src/benchmark.py > logs/benchmark_log.log 2>&1 &
echo "PID=$! — logs/benchmark_log.log"
