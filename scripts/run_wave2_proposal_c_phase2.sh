#!/bin/bash
# Wave 2 Proposal C — Phase 2 inference (XiYan filter, vLLM 필요)
# 선행: Phase 1 (학습) 완료 + vLLM 재기동 완료
# 출력: outputs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}/metrics.txt

set -u
cd "$(dirname "$0")/.."

# vLLM health check
if ! pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "[!] vLLM 서버가 실행 중이 아닙니다. 먼저 vLLM 기동 후 다시 실행하세요."
    exit 1
fi

INFER_CFGS=(
    "experiments/s04_ablation/diameter_layers/layers_L1"
    "experiments/s04_ablation/diameter_layers/layers_L2"
    "experiments/s04_ablation/diameter_layers/layers_L3"
    "experiments/s04_ablation/diameter_layers/layers_L6"
    "experiments/s04_ablation/diameter_layers/layers_L7"
)

echo "========================================"
echo "  Wave 2 Proposal C Phase 2 — 5 inference"
echo "  Started: $(date -Iseconds)"
echo "========================================"

for cfg in "${INFER_CFGS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "  Inference ${cfg}  ($(date -Iseconds))"
    echo "----------------------------------------"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n base python src/main.py --config "${cfg}" \
        || echo "[!] ${cfg} FAILED — continuing"
done

echo ""
echo "========================================"
echo "  Phase 2 metrics summary"
echo "========================================"
for cfg in "${INFER_CFGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    cat "${mpath}" 2>/dev/null || echo "NO METRICS at ${mpath}"
    echo ""
done

echo "========================================"
echo "  다음 단계 (수작업):"
echo "  1) EXPERIMENT_HISTORY.md §9 에 Wave 2 Proposal C entry 추가 (5 row)"
echo "  2) EXPERIMENT_CATALOG.md 에 s04_diameter_layers cluster 추가"
echo "  3) EXPERIMENT_ID_MIGRATION.md 에 abl_sel_diameter_layers_nl{1,2,3,6,7} 명명"
echo "  4) planner 에 핸드오프 — F1 peak 위치로 H1 검증 결론 도출"
echo "========================================"
echo "  Phase 2 finished: $(date -Iseconds)"
