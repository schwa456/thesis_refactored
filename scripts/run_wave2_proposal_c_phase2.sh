#!/bin/bash
# Wave 2 Proposal C GLM era — Phase 2 inference (XiYan filter on GLM-4.7 Live API)
# 선행: Phase 1 (학습) 완료 + .env 에 GLM_BASE_URL / GLM_API_KEY 설정
# 출력: outputs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}_glm/metrics.txt
# 근거: planning/DECISIONS.md 2026-04-24 LLM era transition

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

# GLM endpoint health check (.env 의 GLM_BASE_URL / GLM_API_KEY 사용)
set -a; source .env 2>/dev/null; set +a
if [ -z "${GLM_BASE_URL:-}" ] || [ -z "${GLM_API_KEY:-}" ]; then
    echo "[!] .env 에 GLM_BASE_URL / GLM_API_KEY 미설정."
    exit 1
fi
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -H "Authorization: Bearer ${GLM_API_KEY}" "${GLM_BASE_URL}/models")
if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[!] GLM endpoint ${GLM_BASE_URL}/models 응답 ${HTTP_STATUS}."
    echo "    endpoint 형식(표준 /v1) 또는 api key 를 .env 에서 재확인하세요."
    exit 1
fi
echo "[OK] GLM endpoint ready (${GLM_BASE_URL})"

INFER_CFGS=(
    "experiments/s04_ablation/diameter_layers/layers_L1_glm"
    "experiments/s04_ablation/diameter_layers/layers_L2_glm"
    "experiments/s04_ablation/diameter_layers/layers_L3_glm"
    "experiments/s04_ablation/diameter_layers/layers_L6_glm"
    "experiments/s04_ablation/diameter_layers/layers_L7_glm"
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
