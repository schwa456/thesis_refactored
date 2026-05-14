#!/bin/bash
# Wave 1.5 backfill — Proposal A stagewise matrix Extractor 축 통일 (Basic PCST)
# 근거: planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md §9 Root 행
#       notebooks/analysis_results/stagewise_qcond_ablation.md §1.1 Extractor 불일치 caveat
# 실행: 3 pipeline × 1534 queries × XiYan (vLLM Qwen3-Coder-30B)
# vLLM 서버는 localhost:8000 (GPU 0,1) 에서 ready 상태여야 함.
# 메인 파이프라인도 GPU 0,1 공유 (GAT 추론; GPU 2,3 은 다른 연구자 예약)

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

CONFIGS=(
    "experiments/s04_ablation/stagewise/ensemble_raw_a0"
    "experiments/s04_ablation/stagewise/qcond_raw_basic"
    "experiments/s04_ablation/stagewise/qcond_gat_basic"
)

echo "========================================"
echo "  Wave 1.5 backfill started ($(date -Iseconds))"
echo "  Total: ${#CONFIGS[@]} configs × 1534 queries × XiYan"
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
echo "  Wave 1.5 backfill finished ($(date -Iseconds))"
echo "========================================"
for cfg in "${CONFIGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    cat "${mpath}" 2>/dev/null || echo "NO METRICS at ${mpath}"
    echo ""
done

echo "========================================"
echo "  다음 단계 (memory rule — 필수):"
echo "  1) EXPERIMENT_HISTORY.md 에 3 실험 결과 엔트리 추가"
echo "     (R/P/F1 4자리, Selector/Extractor/Filter cumulative 표기)"
echo "  2) EXPERIMENT_CATALOG.md 에 하이퍼파라미터 블록 추가"
echo "  3) EXPERIMENT_ID_MIGRATION.md 에 ID 매핑 추가"
echo "  4) Planner 세션에 완료 신호 → analyzer 2차 리포트 요청 트리거"
echo "     (stagewise_qcond_ablation.md §1.1 Extractor 통일 후 5×3 매트릭스 재작성)"
echo "========================================"
