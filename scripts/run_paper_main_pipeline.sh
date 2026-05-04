#!/bin/bash
# Paper Main Pipeline 측정 — 옵션 A2, 2 cells (End-to-End Co-Design with Modular LLM Filter)
# 근거: planning/DECISIONS.md 2026-04-28 (방향 F' 최종 채택 + 옵션 A2)
# Stack: Enriched + QCond Ens α=0.5 + {MST Kruskal, MST ∪ PCST Union} + XiYan GLM
# GPU: GPU 1 only (옵션 3 사용자 결정 — SuperNode 학습 GPU 0 보호)
#   - 사용자 명시 default GPU 0/1 안 (GPU 2/3 사용 금지)
#   - 직전 MST 변형 6 cells 동일 패턴 (GPU 1 단독 + 학습 동시 진행) 성공 실증
# 분배: GPU 1, 2 cells parallel = ~50min wall clock

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
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — 측정 보류, 사용자 에스컬레이션 권장"
    exit 1
fi
echo "[OK] GLM endpoint ready (${GLM_BASE_URL}) HTTP ${HTTP_STATUS}"
echo

mkdir -p /tmp/paper_main_logs

CELLS=(
    "experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm"
    "experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm"
)

echo "========================================"
echo "  Paper Main Pipeline (옵션 A2) — 2 cells"
echo "  Started: $(date -Iseconds)"
echo "  GPU: 1 only (학습 GPU 0 보호)"
echo "========================================"

# 2 cells parallel on GPU 1
pids=()
for cfg in "${CELLS[@]}"; do
    echo "[GPU 1] $(date +%H:%M:%S) start ${cfg}"
    CUDA_VISIBLE_DEVICES=1 conda run -n base python -u src/main.py --config "${cfg}" \
        > "/tmp/paper_main_logs/$(basename ${cfg}).log" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid || echo "[!] PID ${pid} FAILED — continuing"
done

echo "[GPU 1] $(date +%H:%M:%S) 2 cells complete"
echo ""
echo "========================================"
echo "  Metrics summary (R / P / F1)"
echo "========================================"
for cfg in "${CELLS[@]}"; do
    mpath="outputs/${cfg}/metrics.txt"
    name=$(basename "${cfg}")
    if [ -f "${mpath}" ]; then
        P=$(grep "^precision:" "$mpath" | head -1 | awk '{print $2}')
        R=$(grep "^recall:" "$mpath" | head -1 | awk '{print $2}')
        F1=$(awk "BEGIN {if (${P}+${R} > 0) printf \"%.4f\", 2*${P}*${R}/(${P}+${R}); else printf \"NaN\"}")
        printf "%-60s R=%s P=%s F1=%s\n" "$name" "$R" "$P" "$F1"
    else
        printf "%-60s NO METRICS — likely failed\n" "$name"
    fi
done

echo ""
echo "Paper main pipeline 측정 finished: $(date -Iseconds)"
