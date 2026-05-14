#!/bin/bash
# S/E/F Ablation 6 cells — 실패 tolerant 실행 스크립트
# 근거: 사용자 요청 (2026-05-04) — Selector/Extractor/Filter 3-axis ablation
# 비교 baseline: t_00 (S-4=E-3=F-3, F-1=no_filter_sql) — 모두 기존 측정 재사용
# 신규 6 cells:
#   S-1 alpha0, S-2 alpha1 (typo 추정), S-3 no_qcond
#   E-1 mst_only, E-2 basic_pcst
#   F-2 no_examples (XiYan num_examples=0)
# GPU 분배: GPU 0 (3 cells: S-1, S-2, S-3), GPU 1 (3 cells: E-1, E-2, F-2)
# 모든 cells parallel — 각 cell 실패해도 다른 cell 영향 없음 ('|| true' + 개별 wait)
# 비용 추정: 6 × ~₩1500 = ~₩9000 (filter + sql gen × 1534)
# Wall clock 추정: ~50min parallel (GLM API contention 고려 시 ~70min)

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
    echo "[!] GLM endpoint 응답 ${HTTP_STATUS} — 실행 중단"
    exit 1
fi
echo "[OK] GLM endpoint ready HTTP ${HTTP_STATUS}"
echo

mkdir -p /tmp/sef_ablation_logs

# Cell ID → (config, gpu) 매핑
declare -A CELL_CFG
CELL_CFG["S1_alpha0"]="experiments/s04_ablation/pipeline/t00_S1_alpha0"
CELL_CFG["S2_alpha1"]="experiments/s04_ablation/pipeline/t00_S2_alpha1"
CELL_CFG["S3_no_qcond"]="experiments/s04_ablation/pipeline/t00_S3_no_qcond"
CELL_CFG["E1_mst_only"]="experiments/s04_ablation/pipeline/t00_E1_mst_only"
CELL_CFG["E2_basic_pcst"]="experiments/s04_ablation/pipeline/t00_E2_basic_pcst"
CELL_CFG["F2_no_examples"]="experiments/s04_ablation/pipeline/t00_F2_no_examples"

declare -A CELL_GPU
CELL_GPU["S1_alpha0"]=0
CELL_GPU["S2_alpha1"]=0
CELL_GPU["S3_no_qcond"]=0
CELL_GPU["E1_mst_only"]=1
CELL_GPU["E2_basic_pcst"]=1
CELL_GPU["F2_no_examples"]=1

declare -A CELL_PID

echo "========================================"
echo "  S/E/F Ablation — 6 cells parallel"
echo "  Started: $(date -Iseconds)"
echo "  GPU 0: S-1, S-2, S-3"
echo "  GPU 1: E-1, E-2, F-2"
echo "========================================"
echo

# 6 cells parallel launch (각 cell 독립 실행, 하나 실패해도 다른 cell 진행)
for cell_id in "${!CELL_CFG[@]}"; do
    cfg="${CELL_CFG[$cell_id]}"
    gpu="${CELL_GPU[$cell_id]}"
    log="/tmp/sef_ablation_logs/${cell_id}.log"
    echo "[GPU ${gpu}] $(date +%H:%M:%S) start ${cell_id} (${cfg})"
    CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py --config "${cfg}" \
        > "${log}" 2>&1 &
    CELL_PID[$cell_id]=$!
done

# 각 cell 개별 wait (failure tolerant)
for cell_id in "${!CELL_CFG[@]}"; do
    pid=${CELL_PID[$cell_id]}
    if wait $pid; then
        echo "[OK] $(date +%H:%M:%S) ${cell_id} done"
    else
        echo "[FAIL] $(date +%H:%M:%S) ${cell_id} — log /tmp/sef_ablation_logs/${cell_id}.log"
    fi
done

echo ""
echo "========================================"
echo "  6 cells Metrics summary (R / P / F1 / EX)"
echo "========================================"
for cell_id in S1_alpha0 S2_alpha1 S3_no_qcond E1_mst_only E2_basic_pcst F2_no_examples; do
    cfg="${CELL_CFG[$cell_id]}"
    mpath="outputs/${cfg}/metrics.txt"
    if [ -f "${mpath}" ]; then
        P=$(grep "^precision:" "$mpath" | head -1 | awk '{print $2}')
        R=$(grep "^recall:" "$mpath" | head -1 | awk '{print $2}')
        EX=$(grep "^ex:" "$mpath" | head -1 | awk '{print $2}')
        F1=$(awk "BEGIN {if (${P}+${R} > 0) printf \"%.4f\", 2*${P}*${R}/(${P}+${R}); else printf \"NaN\"}")
        printf "%-20s R=%s P=%s F1=%s EX=%s\n" "${cell_id}" "$R" "$P" "$F1" "$EX"
    else
        printf "%-20s NO METRICS — failed\n" "${cell_id}"
    fi
done

echo ""
echo "S/E/F Ablation finished: $(date -Iseconds)"
