#!/bin/bash
# H2 truncate inference 2 cell parallel (GLM API, GPU 미점유)
# 근거: planning/DECISIONS.md 2026-04-26 "Selector H2 inference 2 cell 실측 승인" 엔트리
# Selector impl mechanism: nl=6/7 ckpt 의 layer 수 동적 truncate forward (D_max<6 DB 도 per-DB 동적 depth)
# Analyzer reconstruction (0.5805, D_max=4/5 → nl=6 fallback) 과 다른 mechanism 의 첫 실측
#
# Configs (selector 세션 산출물, 2026-04-25 01:05/01:06):
#   - layers_Ldbmax_glm.yaml       (num_layers_mode=D_max,      nl=6 ckpt truncate)
#   - layers_Ldbmax_plus1_glm.yaml (num_layers_mode=D_max_plus1, nl=7 ckpt truncate)
#
# 예상: 2 cell 병렬, ~50min total, ~₩1,528 (2 × ₩764)
# 비교 anchor: layers_L6_glm F1=0.5824 (global fixed nl=6)
# 4-way 결과 해석 분기: DECISIONS.md 2026-04-26 엔트리 §영향 범위 표 참조

set -u
cd "$(dirname "$0")/.."

# GLM endpoint health check
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
    exit 1
fi
echo "[OK] GLM chat endpoint ready (${GLM_BASE_URL})"
echo

mkdir -p /tmp/h2_truncate_logs

CFGS=(
    "experiments/s04_ablation/diameter_layers/layers_Ldbmax_glm"
    "experiments/s04_ablation/diameter_layers/layers_Ldbmax_plus1_glm"
)

echo "========================================"
echo "  H2 truncate inference — 2 cell parallel"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# Launch both cells in parallel, GPU 0/1 공유 (GLM API 호출 — GPU 실제 부하는 Encoder/GAT forward 만)
for cfg in "${CFGS[@]}"; do
    logfile="/tmp/h2_truncate_logs/$(basename ${cfg}).log"
    echo "  Launching ${cfg} → ${logfile}"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n base python -u src/main.py --config "${cfg}" \
        > "${logfile}" 2>&1 &
done

# wait for both to finish
wait

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
echo "  H2 truncate finished: $(date -Iseconds)"
echo "========================================"
