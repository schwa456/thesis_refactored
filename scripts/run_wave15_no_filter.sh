#!/bin/bash
# Wave 1.5 +Extractor (no filter) cell backfill — analyzer §4 pending cells
# 근거: notebooks/analysis_results/stagewise_qcond_ablation.md §4 (2026-04-22 analyzer 요청)
# 실행: 3 pipeline × 1534 queries × NoneFilter (LLM 호출 0, pure Extractor cell)
# GPU: 0,1 (vLLM 없음 — NoneFilter 는 pass-through, 그래도 embedding load 용 CUDA 필요)

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CONFIGS=(
    "experiments/s04_ablation/stagewise/no_filter/ensemble_raw_a0_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/qcond_raw_basic_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/qcond_gat_basic_no_filter"
)

echo "========================================"
echo "  Wave 1.5 no-filter backfill started ($(date -Iseconds))"
echo "  Total: ${#CONFIGS[@]} configs × 1534 queries × NoneFilter"
echo "========================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
    echo "----------------------------------------"
    echo "  Running ${cfg}  ($(date -Iseconds))"
    echo "----------------------------------------"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n base python src/main.py --config "${cfg}" \
        || echo "[!] ${cfg} FAILED — continuing with next"
    echo ""
done

echo "========================================"
echo "  Wave 1.5 no-filter backfill finished ($(date -Iseconds))"
echo "========================================"
for cfg in "${CONFIGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    cat "${mpath}" 2>/dev/null || echo "NO METRICS at ${mpath}"
    echo ""
done

echo "========================================"
echo "  다음 단계 (analyzer handoff):"
echo "  1) notebooks/analysis_results/stagewise_qcond_ablation.md §1.1"
echo "     +Extractor (Basic PCST, no filter) 3 cell (W1/W2/W3) 을 이번 metrics.txt 수치로 채움"
echo "  2) §4 Pending Cells 섹션 업데이트 (pending → resolved, 출처 metrics 파일 경로 추가)"
echo "  3) §5 발표 슬라이드 3 '전환점' 보강 — Extractor 단독 stage 의 R/P 추이 정량화"
echo "  4) EXPERIMENT_HISTORY.md §8 에 no-filter 결과 부기록"
echo "========================================"
