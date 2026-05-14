#!/bin/bash
# GLM era kickoff (post-sanity): 5-cell sweep + 1 new anchor sequential
# 근거: planning/DECISIONS.md 2026-04-24 "Sanity check 합격 기준 재정의" 엔트리
# 선행: .env GLM_BASE_URL=https://mlapi.run/<api_id>/v1 + GLM_API_KEY 설정, sanity 합격
# 예상: 총 ~5h (6 × 50min), 총 ~₩4,585
# 출력: outputs/experiments/s04_ablation/{diameter_layers/layers_L{1,2,3,6,7}_glm, stagewise/qcond_gat_basic_glm}/

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

# GLM endpoint health check — minimal POST /chat/completions (기존 /models GET 은 400 반환 확인됨)
set -a; source .env 2>/dev/null; set +a
if [ -z "${GLM_BASE_URL:-}" ] || [ -z "${GLM_API_KEY:-}" ]; then
    echo "[!] .env 에 GLM_BASE_URL / GLM_API_KEY 미설정."
    exit 1
fi
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST \
    -H "Authorization: Bearer ${GLM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/glm-4.7","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}' \
    "${GLM_BASE_URL}/chat/completions")
if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[!] GLM chat endpoint 응답 ${HTTP_STATUS} (${GLM_BASE_URL}/chat/completions)."
    echo "    .env 의 GLM_BASE_URL/API_KEY 를 확인하세요."
    exit 1
fi
echo "[OK] GLM chat endpoint ready (${GLM_BASE_URL}/chat/completions)"
echo

CFGS=(
    "experiments/s04_ablation/diameter_layers/layers_L1_glm"
    "experiments/s04_ablation/diameter_layers/layers_L2_glm"
    "experiments/s04_ablation/diameter_layers/layers_L3_glm"
    "experiments/s04_ablation/diameter_layers/layers_L6_glm"
    "experiments/s04_ablation/diameter_layers/layers_L7_glm"
    "experiments/s04_ablation/stagewise/qcond_gat_basic_glm"
)

echo "========================================"
echo "  GLM era kickoff — 6 inference (5 sweep + 1 new anchor)"
echo "  Started: $(date -Iseconds)"
echo "========================================"

for cfg in "${CFGS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "  Inference ${cfg}  ($(date -Iseconds))"
    echo "----------------------------------------"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n base python -u src/main.py --config "${cfg}" \
        || echo "[!] ${cfg} FAILED — continuing to next"
done

echo ""
echo "========================================"
echo "  Metrics summary"
echo "========================================"
for cfg in "${CFGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    head -3 "${mpath}" 2>/dev/null || echo "NO METRICS at ${mpath}"
    echo ""
done

echo "========================================"
echo "  GLM era kickoff finished: $(date -Iseconds)"
echo "========================================"
