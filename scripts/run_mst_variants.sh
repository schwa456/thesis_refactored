#!/bin/bash
# MST/Steiner 변형 4 cells 측정 (Steiner Tree threshold + MST Kruskal × {no_filter, glm})
# 근거: planning/DECISIONS.md 2026-04-27 (옵션 C — MST 명명 정정 + score-threshold 변형 + 진짜 MST Kruskal)
# GPU: GPU 1 only (CUDA_VISIBLE_DEVICES=1) — SuperNode 학습 GPU 0 보호 (~5h 학습 진행 중)
# 사용자 명시: GPU 2/3 사용 금지, CUDA_VISIBLE_DEVICES=0,1 만
# 분배: GPU 1 에 4 cells, 2 batches × 2 concurrent = ~60 min total
#   Batch 1 (LLM-free): steiner_threshold_no_filter + mst_kruskal_no_filter (~10 min)
#   Batch 2 (GLM): steiner_threshold_glm + mst_kruskal_glm (~50 min)
# 비용: 2 GLM × ~₩764 = ~₩1,528

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
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — Final 2 GLM cells 보류, 2 LLM-free 만 진행 권장."
    GLM_OK=0
else
    echo "[OK] GLM endpoint ready (${GLM_BASE_URL}) HTTP ${HTTP_STATUS}"
    GLM_OK=1
fi
echo

mkdir -p /tmp/mst_variants_logs

# Batch 1 (LLM-free, ~10 min)
BATCH1=(
    "experiments/s04_ablation/extractor/no_filter/plain_ens_a05_steiner_threshold_no_filter"
    "experiments/s04_ablation/extractor/no_filter/plain_ens_a05_mst_kruskal_no_filter"
)

# Batch 2 (GLM, ~50 min)
BATCH2=(
    "experiments/s04_ablation/extractor/plain_ens_a05_steiner_threshold_glm"
    "experiments/s04_ablation/extractor/plain_ens_a05_mst_kruskal_glm"
)

run_batch_concurrent() {
    local pids=()
    for cfg in "$@"; do
        echo "[GPU 1] $(date +%H:%M:%S) start ${cfg}"
        CUDA_VISIBLE_DEVICES=1 conda run -n base python -u src/main.py --config "${cfg}" \
            > "/tmp/mst_variants_logs/$(basename ${cfg}).log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait $pid || echo "[!] PID ${pid} FAILED — continuing"
    done
}

echo "========================================"
echo "  MST/Steiner 변형 4 cells 측정 (GPU 1)"
echo "  Started: $(date -Iseconds)"
echo "========================================"

# Sequential batches on GPU 1
run_batch_concurrent "${BATCH1[@]}"
echo "[GPU 1] $(date +%H:%M:%S) batch1 (LLM-free) done"

if [ "${GLM_OK}" -eq 1 ]; then
    run_batch_concurrent "${BATCH2[@]}"
    echo "[GPU 1] $(date +%H:%M:%S) batch2 (GLM) done — 4 cells complete"
else
    echo "[GPU 1] batch2 GLM 보류 — 2 LLM-free 만 완료"
fi

echo ""
echo "========================================"
echo "  Metrics summary (R / P / F1)"
echo "========================================"
ALL_CFGS=("${BATCH1[@]}" "${BATCH2[@]}")
for cfg in "${ALL_CFGS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    name=$(basename "${cfg}")
    if [ -f "${mpath}" ]; then
        P=$(grep "^precision:" "$mpath" | head -1 | awk '{print $2}')
        R=$(grep "^recall:" "$mpath" | head -1 | awk '{print $2}')
        F1=$(awk "BEGIN {if (${P}+${R} > 0) printf \"%.4f\", 2*${P}*${R}/(${P}+${R}); else printf \"NaN\"}")
        printf "%-55s R=%s P=%s F1=%s\n" "$name" "$R" "$P" "$F1"
    else
        printf "%-55s NO METRICS — likely failed or skipped\n" "$name"
    fi
done

echo ""
echo "MST 변형 측정 finished: $(date -Iseconds)"
