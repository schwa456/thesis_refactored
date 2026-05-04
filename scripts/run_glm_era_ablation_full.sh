#!/bin/bash
# Ablation 1/2/3 GLM era 일관 재측정 — 11 cells (8 final GLM + 3 LLM-free no-filter)
# 근거: planning/DECISIONS.md 2026-04-27 (GLM era 일관 재측정)
# GPU 분배 (2 concurrent per GPU = 4 cells 동시):
#   GPU 2: 6 cells in 3 batches of 2 (sequential within batch pair)
#   GPU 3: 5 cells in 3 batches (2+2+1)
# 타이밍: GLM cell ~55 min, LLM-free ~10 min. 2 concurrent → batch ~70 min. 3 batch ≈ 210 min = 3.5h
# 사전: MST smoke OK (2026-04-27 00:59 plain_ens_mst_no_filter 15 preds in 28s)
# 주의: GPU 2/3 사용 (사용자 명시 허가, settings.json ask rule)

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
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — 8 GLM cells 보류, 3 LLM-free 만 진행 권장."
    GLM_OK=0
else
    echo "[OK] GLM endpoint ready (${GLM_BASE_URL}) HTTP ${HTTP_STATUS}"
    GLM_OK=1
fi
echo

mkdir -p /tmp/glm_full_logs

# GPU 2 — 6 cells in 3 batches of 2 (concurrent within batch)
# Batch order: heaviest GLM first, LLM-free last
GPU2_BATCH1=(
    "experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_gat_glm"
    "experiments/s04_ablation/stagewise/plain_gat_a0_glm"
)
GPU2_BATCH2=(
    "experiments/s04_ablation/stagewise/plain_cos_a1_glm"
    "experiments/s04_ablation/stagewise/plain_ens_glm"
)
GPU2_BATCH3=(
    "experiments/s04_ablation/extractor/no_filter/plain_ens_adaptive_no_filter"
    "experiments/s04_ablation/extractor/no_filter/plain_ens_steiner_no_filter"
)

# GPU 3 — 5 cells in 3 batches (2+2+1)
GPU3_BATCH1=(
    "experiments/s04_ablation/stagewise/qcond_gat_a0_glm"
    "experiments/s04_ablation/extractor/plain_ens_adaptive_glm"
)
GPU3_BATCH2=(
    "experiments/s04_ablation/extractor/plain_ens_steiner_glm"
    "experiments/s04_ablation/extractor/plain_ens_mst_glm"
)
GPU3_BATCH3=(
    "experiments/s04_ablation/extractor/no_filter/plain_ens_mst_no_filter"
)

run_batch_concurrent() {
    local gpu=$1
    shift
    local pids=()
    for cfg in "$@"; do
        echo "[GPU ${gpu}] $(date +%H:%M:%S) start ${cfg}"
        CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/glm_full_logs/$(basename ${cfg}).log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid || echo "[!] PID ${pid} on GPU ${gpu} FAILED — continuing"
    done
}

echo "========================================"
echo "  GLM era full backfill — 11 cells"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# GPU 2 sequential batches
(
    run_batch_concurrent 2 "${GPU2_BATCH1[@]}"
    echo "[GPU 2] $(date +%H:%M:%S) batch1 done"
    run_batch_concurrent 2 "${GPU2_BATCH2[@]}"
    echo "[GPU 2] $(date +%H:%M:%S) batch2 done"
    run_batch_concurrent 2 "${GPU2_BATCH3[@]}"
    echo "[GPU 2] $(date +%H:%M:%S) batch3 done — all 6 cells complete"
) &
GPU2_PID=$!

# GPU 3 sequential batches
(
    run_batch_concurrent 3 "${GPU3_BATCH1[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) batch1 done"
    run_batch_concurrent 3 "${GPU3_BATCH2[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) batch2 done"
    run_batch_concurrent 3 "${GPU3_BATCH3[@]}"
    echo "[GPU 3] $(date +%H:%M:%S) batch3 done — all 5 cells complete"
) &
GPU3_PID=$!

wait $GPU2_PID $GPU3_PID

echo ""
echo "========================================"
echo "  Metrics summary"
echo "========================================"
ALL_CFGS=("${GPU2_BATCH1[@]}" "${GPU2_BATCH2[@]}" "${GPU2_BATCH3[@]}" "${GPU3_BATCH1[@]}" "${GPU3_BATCH2[@]}" "${GPU3_BATCH3[@]}")
for cfg in "${ALL_CFGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    echo "--- ${cfg} ---"
    if [ -f "${mpath}" ]; then
        head -3 "${mpath}"
    else
        echo "NO METRICS — likely failed"
    fi
done

echo ""
echo "GLM era full backfill finished: $(date -Iseconds)"
