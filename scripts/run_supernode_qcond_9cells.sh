#!/bin/bash
# SuperNode QCond 9-cell matrix — α∈{0, 0.5, 1} × {selector_only, no_filter, glm}
# 근거: planning/DECISIONS.md 2026-04-28 (H6 SuperNode 학습 완료 + 코드 fix + smoke PASS)
# Stack: SuperNode encoder (query_conditioned=true + query_supernode=true) + best_gat_query_supernode_qcond.pt
# 코드 수정 적용: src/modules/selectors/ensemble_selector.py:241-243 (query_emb=q_emb 전달)
# GPU 분배:
#   GPU 0: 3 GLM cells in 2 batches of 2+1 (~50min × 2 = ~100min)
#   GPU 1: 6 LLM-free cells in 2 batches of 3 (~10min × 2 = ~20min)
# Wall clock: ~100min (GPU 0 이 bottleneck)
# 비용: 3 GLM × ~₩764 = ~₩2,292
# α=0 selector_only 는 smoke 결과 재사용 가능 — 본 script 에서는 8 cells 실행 (smoke 결과 → 정식 ID 매핑 역할)
# 단 명명 일관성 위해 9 cells 모두 launch (smoke 결과는 호환 config 라 동일 결과 기대)

set -u
cd "$(dirname "$0")/.."

# GLM endpoint health check
set -a; source .env 2>/dev/null; set +a
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST \
    -H "Authorization: Bearer ${GLM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/glm-4.7","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}' \
    "${GLM_BASE_URL}/chat/completions")
if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — Final 3 GLM cells 보류, 6 LLM-free 만 진행"
    GLM_OK=0
else
    echo "[OK] GLM endpoint ready HTTP ${HTTP_STATUS}"
    GLM_OK=1
fi
echo

mkdir -p /tmp/supernode_9cells_logs

# GPU 0 — 3 GLM cells (2 concurrent + 1 single)
GPU0_BATCH1=(
    "experiments/s04_ablation/stagewise/supernode_qcond_a0_glm"
    "experiments/s04_ablation/stagewise/supernode_qcond_a05_glm"
)
GPU0_BATCH2=(
    "experiments/s04_ablation/stagewise/supernode_qcond_a1_glm"
)

# GPU 1 — 6 LLM-free cells (3 concurrent in 2 batches)
# Batch 1: 3 selector_only + a0 no_filter
GPU1_BATCH1=(
    "experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a0_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a05_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a1_selector_only"
)
GPU1_BATCH2=(
    "experiments/s04_ablation/stagewise/no_filter/supernode_qcond_a0_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/supernode_qcond_a05_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/supernode_qcond_a1_no_filter"
)

run_batch_concurrent() {
    local gpu=$1
    shift
    local pids=()
    for cfg in "$@"; do
        echo "[GPU ${gpu}] $(date +%H:%M:%S) start ${cfg}"
        CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/supernode_9cells_logs/$(basename ${cfg}).log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid || echo "[!] PID ${pid} on GPU ${gpu} FAILED — continuing"
    done
}

echo "========================================"
echo "  SuperNode QCond 9-cell matrix"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# GPU 0 (GLM cells, only if GLM OK)
if [ "${GLM_OK}" -eq 1 ]; then
    (
        run_batch_concurrent 0 "${GPU0_BATCH1[@]}"
        echo "[GPU 0] $(date +%H:%M:%S) GLM batch1 done"
        run_batch_concurrent 0 "${GPU0_BATCH2[@]}"
        echo "[GPU 0] $(date +%H:%M:%S) GLM batch2 done — 3 Final cells complete"
    ) &
    GPU0_PID=$!
fi

# GPU 1 (LLM-free, 3 concurrent batches)
(
    run_batch_concurrent 1 "${GPU1_BATCH1[@]}"
    echo "[GPU 1] $(date +%H:%M:%S) LLM-free batch1 done (3 selector_only)"
    run_batch_concurrent 1 "${GPU1_BATCH2[@]}"
    echo "[GPU 1] $(date +%H:%M:%S) LLM-free batch2 done (3 no_filter) — 6 LLM-free cells complete"
) &
GPU1_PID=$!

if [ "${GLM_OK}" -eq 1 ]; then
    wait $GPU0_PID $GPU1_PID
else
    wait $GPU1_PID
fi

echo ""
echo "========================================"
echo "  9 cells Metrics summary (R / P / F1)"
echo "========================================"
ALL_CFGS=("${GPU0_BATCH1[@]}" "${GPU0_BATCH2[@]}" "${GPU1_BATCH1[@]}" "${GPU1_BATCH2[@]}")
for cfg in "${ALL_CFGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    name=$(basename "${cfg}")
    if [ -f "${mpath}" ]; then
        P=$(grep "^precision:" "$mpath" | head -1 | awk '{print $2}')
        R=$(grep "^recall:" "$mpath" | head -1 | awk '{print $2}')
        F1=$(awk "BEGIN {if (${P}+${R} > 0) printf \"%.4f\", 2*${P}*${R}/(${P}+${R}); else printf \"NaN\"}")
        printf "%-50s R=%s P=%s F1=%s\n" "$name" "$R" "$P" "$F1"
    else
        printf "%-50s NO METRICS — likely failed\n" "$name"
    fi
done

echo ""
echo "SuperNode 9-cell matrix finished: $(date -Iseconds)"
