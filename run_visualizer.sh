#!/usr/bin/env bash
# Schema Linking Graph Visualizer — Streamlit 실행 스크립트
# Usage: bash run_visualizer.sh [PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8501}"

# conda base 환경 활성화
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate base

cd "$SCRIPT_DIR"

echo "Starting Streamlit on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."

streamlit run notebooks/visualize_graph_app.py \
    --server.port "$PORT" \
    --server.headless true
