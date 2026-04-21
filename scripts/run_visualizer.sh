#!/usr/bin/env bash
# Schema Linking Graph Visualizer — Streamlit 실행 스크립트
# Usage: bash run_visualizer.sh [PORT]

set -e

cd "$(dirname "$0")/.."
PORT="${1:-8501}"

# conda base 환경 활성화
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate base

echo "Starting Streamlit on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."

streamlit run src/analysis/visualize_graph_app.py \
    --server.port "$PORT" \
    --server.headless true
