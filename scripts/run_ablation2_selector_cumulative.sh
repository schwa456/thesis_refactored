#!/bin/bash
# Ablation 2 (Encoder × Score × Stage) cumulative — 19 cells
# 근거: planning/DECISIONS.md 2026-04-26 (Ablation 2 Option B + 보강 #2 9-cell 매트릭스)
# Final 4 cells: GLM API ~₩3,056, No-filter 6 + Selector_only 9: LLM-free
# GPU 분배: GPU 2 = LLM-free 15 cells (sequential), GPU 3 = Final 4 cells (sequential, GLM API)
# 예상: ~3-3.5h parallel wall clock
# 사전: SuperNode + EnsembleSelector smoke test (supernode_gat_a0_selector_only) 단독 검증 권장
# 주의: GPU 2/3 사용 (memory rule swap, settings.json `ask` rule 적용)

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

# GLM endpoint health check (Final 4 cells 호출용)
set -a; source .env 2>/dev/null; set +a
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST \
    -H "Authorization: Bearer ${GLM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/glm-4.7","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}' \
    "${GLM_BASE_URL}/chat/completions")
if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — Final 4 cells 보류, LLM-free 15 cells 만 진행."
    GLM_OK=0
else
    echo "[OK] GLM endpoint ready (${GLM_BASE_URL})"
    GLM_OK=1
fi
echo

mkdir -p /tmp/ablation2_logs

# 2026-04-26 02:49 SuperNode smoke FAILED (size mismatch ckpt[256,384] vs model[256,768])
# → SuperNode 9 cells 전부 보류 (selector_only 3 + no_filter 3 + final 3). DECISIONS.md 후속 엔트리 기록.
# 진행: 10 cells (Plain/QCond LLM-free 9 + qcond_cos_a1_glm 1)

# GPU 2 batch (LLM-free, 9 cells: 6 selector_only + 3 no_filter, Plain/QCond 만)
CFGS_GPU2=(
    "experiments/s04_ablation/stagewise/selector_only/plain_gat_a0_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/plain_cos_a1_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/plain_ens_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/qcond_gat_a0_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/qcond_cos_a1_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/qcond_ens_selector_only"
    "experiments/s04_ablation/stagewise/no_filter/plain_cos_a1_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/plain_ens_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/qcond_cos_a1_no_filter"
)

# GPU 3 batch (Final 1 cell, GLM API — SuperNode 3 cells 제거)
CFGS_GPU3=(
    "experiments/s04_ablation/stagewise/qcond_cos_a1_glm"
)

echo "========================================"
echo "  Ablation 2 (Encoder × Score × Stage) — 18 cells (smoke 1 cell 별도 launch)"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# GPU 2 batch (sequential)
(
    for cfg in "${CFGS_GPU2[@]}"; do
        echo "[GPU 2] $(date +%H:%M:%S) ${cfg}"
        CUDA_VISIBLE_DEVICES=2 conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/ablation2_logs/$(basename ${cfg}).log" 2>&1 \
            || echo "[!] ${cfg} FAILED — continuing"
    done
    echo "[GPU 2] batch complete: $(date -Iseconds)"
) &
GPU2_PID=$!

# GPU 3 batch (sequential, only if GLM OK)
if [ "${GLM_OK}" -eq 1 ]; then
    (
        for cfg in "${CFGS_GPU3[@]}"; do
            echo "[GPU 3] $(date +%H:%M:%S) ${cfg}"
            CUDA_VISIBLE_DEVICES=3 conda run -n base python -u src/main.py --config "${cfg}" \
                > "/tmp/ablation2_logs/$(basename ${cfg}).log" 2>&1 \
                || echo "[!] ${cfg} FAILED — continuing"
        done
        echo "[GPU 3] batch complete: $(date -Iseconds)"
    ) &
    GPU3_PID=$!
    wait $GPU2_PID $GPU3_PID
else
    wait $GPU2_PID
fi

echo ""
echo "========================================"
echo "  Metrics summary"
echo "========================================"
for cfg in "${CFGS_GPU2[@]}" "${CFGS_GPU3[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    head -3 "${mpath}" 2>/dev/null || echo "NO METRICS"
done

echo "Ablation 2 finished: $(date -Iseconds)"
