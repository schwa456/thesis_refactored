#!/usr/bin/env bash
# Schema Linking Graph Visualizer — Streamlit 실행 스크립트
# Usage: bash run_visualizer.sh [PORT]
#
# 주의: `streamlit` CLI 의 shebang 이 /usr/bin/python3 (torch 미설치) 인 경우가 있어,
#       conda base 의 python 으로 `python -m streamlit` 을 강제 실행한다.

set -e

cd "$(dirname "$0")/.."
export TMPDIR=/tmp
PORT="${1:-8501}"

# conda base 의 python 경로 확정
CONDA_PY="${CONDA_PYTHON:-/home/hyeonjin/miniconda3/bin/python}"
if [[ ! -x "$CONDA_PY" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate base
    CONDA_PY="$(command -v python)"
fi

echo "Using python: $CONDA_PY"
"$CONDA_PY" -c "import torch, streamlit, pyvis, networkx" \
    || { echo "Required deps missing in $CONDA_PY"; exit 1; }

echo "Starting Streamlit on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."

exec "$CONDA_PY" -m streamlit run src/analysis/visualize_graph_app.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
