#!/bin/bash
# Ablation 1/2/3 α=0.5 재측정 (Option B) — 15 cells
# 근거: planning/DECISIONS.md 2026-04-27 (Ablation 1/2/3 α=0.5 재측정 결정)
# Ensemble baseline α convention 변경: 0.85 (Cosine 우세) → 0.5 (neutral, GAT/Cosine 동등)
# GPU 분배 (사용자 명시 GPU 2/3):
#   GPU 2: 6 Final GLM cells (~50min × 6 / 2 concurrent = ~2.5h)
#   GPU 3: 9 LLM-free cells (~15min × 9 / 2 concurrent = ~70min)
# Parallel total ~2.5-3h
# 사전: GLM API HTTP 200 ping OK (2026-04-27 launch 직전)

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

# GLM endpoint health check
set -a; source .env 2>/dev/null; set +a
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST \
    -H "Authorization: Bearer ${GLM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/glm-4.7","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}' \
    "${GLM_BASE_URL}/chat/completions")
if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — Final 6 GLM cells 보류, 9 LLM-free 만 진행 권장."
    GLM_OK=0
else
    echo "[OK] GLM endpoint ready (${GLM_BASE_URL}) HTTP ${HTTP_STATUS}"
    GLM_OK=1
fi
echo

mkdir -p /tmp/alpha05_logs

# GPU 2 — 6 Final GLM cells, 2 concurrent per batch (3 batches)
GPU2_BATCH1=(
    "experiments/s04_ablation/stagewise/plain_ens_a05_glm"
    "experiments/s04_ablation/stagewise/qcond_ens_a05_glm"
)
GPU2_BATCH2=(
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_a05_glm"
    "experiments/s04_ablation/extractor/plain_ens_a05_adaptive_glm"
)
GPU2_BATCH3=(
    "experiments/s04_ablation/extractor/plain_ens_a05_steiner_glm"
    "experiments/s04_ablation/extractor/plain_ens_a05_mst_glm"
)

# GPU 3 — 9 LLM-free cells, 2 concurrent per batch (5 batches: 4× pairs + 1 single)
GPU3_BATCH1=(
    "experiments/s04_ablation/stagewise/selector_only/plain_ens_a05_selector_only"
    "experiments/s04_ablation/stagewise/selector_only/qcond_ens_a05_selector_only"
)
GPU3_BATCH2=(
    "experiments/s04_ablation/stagewise/no_filter/plain_ens_a05_no_filter"
    "experiments/s04_ablation/stagewise/no_filter/qcond_ens_a05_no_filter"
)
GPU3_BATCH3=(
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_a05_selector_only"
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_a05_no_filter"
)
GPU3_BATCH4=(
    "experiments/s04_ablation/extractor/no_filter/plain_ens_a05_adaptive_no_filter"
    "experiments/s04_ablation/extractor/no_filter/plain_ens_a05_steiner_no_filter"
)
GPU3_BATCH5=(
    "experiments/s04_ablation/extractor/no_filter/plain_ens_a05_mst_no_filter"
)

run_batch_concurrent() {
    local gpu=$1
    shift
    local pids=()
    for cfg in "$@"; do
        echo "[GPU ${gpu}] $(date +%H:%M:%S) start ${cfg}"
        CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/alpha05_logs/$(basename ${cfg}).log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid || echo "[!] PID ${pid} on GPU ${gpu} FAILED — continuing"
    done
}

echo "========================================"
echo "  Ablation 1/2/3 α=0.5 재측정 — 15 cells"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# GPU 2 sequential batches (Final GLM cells, only if GLM OK)
if [ "${GLM_OK}" -eq 1 ]; then
    (
        run_batch_concurrent 2 "${GPU2_BATCH1[@]}"
        echo "[GPU 2] $(date +%H:%M:%S) GLM batch1 done"
        run_batch_concurrent 2 "${GPU2_BATCH2[@]}"
        echo "[GPU 2] $(date +%H:%M:%S) GLM batch2 done"
        run_batch_concurrent 2 "${GPU2_BATCH3[@]}"
        echo "[GPU 2] $(date +%H:%M:%S) GLM batch3 done — 6 Final cells complete"
    ) &
    GPU2_PID=$!
fi

# GPU 3 sequential batches (LLM-free cells)
(
    run_batch_concurrent 3 "${GPU3_BATCH1[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) LLM-free batch1 done"
    run_batch_concurrent 3 "${GPU3_BATCH2[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) LLM-free batch2 done"
    run_batch_concurrent 3 "${GPU3_BATCH3[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) LLM-free batch3 done"
    run_batch_concurrent 3 "${GPU3_BATCH4[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) LLM-free batch4 done"
    run_batch_concurrent 3 "${GPU3_BATCH5[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) LLM-free batch5 done — 9 LLM-free cells complete"
) &
GPU3_PID=$!

if [ "${GLM_OK}" -eq 1 ]; then
    wait $GPU2_PID $GPU3_PID
else
    wait $GPU3_PID
fi

echo ""
echo "========================================"
echo "  Metrics summary (R / P / F1)"
echo "========================================"
ALL_CFGS=(
    "${GPU2_BATCH1[@]}" "${GPU2_BATCH2[@]}" "${GPU2_BATCH3[@]}"
    "${GPU3_BATCH1[@]}" "${GPU3_BATCH2[@]}" "${GPU3_BATCH3[@]}" "${GPU3_BATCH4[@]}" "${GPU3_BATCH5[@]}"
)
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
echo "α=0.5 재측정 finished: $(date -Iseconds)"
