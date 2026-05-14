#!/bin/bash
# Ablation 1 (Builder × Stage) cumulative backfill — 5 cells
# 근거: planning/DECISIONS.md 2026-04-26 (보강 — Option C 채택) §결정 (a)(b)
# LLM 호출 0 (Filter=None, Extractor=None or Basic PCST)
# GPU 0/1 병렬: GPU 0 = selector_only 3개 sequential, GPU 1 = no_filter 2개 sequential
# 예상: ~30~45min wall clock, 비용 ₩0

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CFGS_GPU0=(
    "experiments/s04_ablation/stagewise/qcond_gat_basic_selector_only"
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_gat_selector_only"
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_02_edge_prize_selector_only"
)

CFGS_GPU1=(
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_gat_no_filter"
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_02_edge_prize_no_filter"
)

mkdir -p /tmp/builder_cumulative_logs

echo "========================================"
echo "  Ablation 1 (Builder × Stage) cumulative — 5 cells"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# GPU 0 batch (sequential, 3 selector_only cells)
(
    for cfg in "${CFGS_GPU0[@]}"; do
        echo "[GPU 0] Inference ${cfg} ($(date -Iseconds))"
        CUDA_VISIBLE_DEVICES=2 conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/builder_cumulative_logs/$(basename ${cfg}).log" 2>&1 \
            || echo "[!] ${cfg} FAILED — continuing"
    done
    echo "[GPU 0] batch complete: $(date -Iseconds)"
) &
GPU0_PID=$!

# GPU 1 batch (sequential, 2 no_filter cells)
(
    for cfg in "${CFGS_GPU1[@]}"; do
        echo "[GPU 1] Inference ${cfg} ($(date -Iseconds))"
        CUDA_VISIBLE_DEVICES=3 conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/builder_cumulative_logs/$(basename ${cfg}).log" 2>&1 \
            || echo "[!] ${cfg} FAILED — continuing"
    done
    echo "[GPU 1] batch complete: $(date -Iseconds)"
) &
GPU1_PID=$!

wait $GPU0_PID $GPU1_PID

echo ""
echo "========================================"
echo "  Metrics summary"
echo "========================================"
for cfg in "${CFGS_GPU0[@]}" "${CFGS_GPU1[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    head -3 "${mpath}" 2>/dev/null || echo "NO METRICS at ${mpath}"
    echo ""
done

echo "========================================"
echo "  Builder cumulative finished: $(date -Iseconds)"
echo "========================================"
