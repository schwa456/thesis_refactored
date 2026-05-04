#!/bin/bash
# Alpha sweep — 8 cells (α∈{0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9})
# 근거: 사용자 요청 (2026-05-04) — Selection module α=0.0~1.0 0.1 단위 sweep
# 재사용: α=0.0 (S-1), α=0.5 (t_00), α=1.0 (S-2) 측정 완료
# Stack: t_00 base — Enriched + QCond + qcond_nl3 + α(변화) + MSTPCSTUnion + XiYan(GLM, 3 ex) + SQL gen
# GPU: 0/1 split, 4 cells per GPU (총 8 parallel)
# Failure-tolerant: 각 cell 독립 PID + wait, 실패 시 다른 cells 영향 없음
# 비용 추정: 8 × ~₩1,500 = ~₩12,000 (filter + sql gen)
# Wall clock 추정: ~3h (8 cells GLM API contention)

set -u
cd "$(dirname "$0")/.."

# GLM endpoint health check
set -a; source .env 2>/dev/null; set +a
HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -X POST \
    -H "Authorization: Bearer ${GLM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"zai-org/glm-4.7","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}' \
    "${GLM_BASE_URL}/chat/completions")
if [ "${HTTP}" != "200" ]; then
    echo "[!] GLM endpoint ${HTTP} — 실행 중단"
    exit 1
fi
echo "[OK] GLM endpoint ready HTTP ${HTTP}"
echo

mkdir -p /tmp/alpha_sweep_logs

# Cell ID → (config, gpu)
declare -A CELL_CFG
CELL_CFG["alpha_01"]="experiments/s04_ablation/pipeline/t00_alpha_01"
CELL_CFG["alpha_02"]="experiments/s04_ablation/pipeline/t00_alpha_02"
CELL_CFG["alpha_03"]="experiments/s04_ablation/pipeline/t00_alpha_03"
CELL_CFG["alpha_04"]="experiments/s04_ablation/pipeline/t00_alpha_04"
CELL_CFG["alpha_06"]="experiments/s04_ablation/pipeline/t00_alpha_06"
CELL_CFG["alpha_07"]="experiments/s04_ablation/pipeline/t00_alpha_07"
CELL_CFG["alpha_08"]="experiments/s04_ablation/pipeline/t00_alpha_08"
CELL_CFG["alpha_09"]="experiments/s04_ablation/pipeline/t00_alpha_09"

declare -A CELL_GPU
CELL_GPU["alpha_01"]=0
CELL_GPU["alpha_02"]=0
CELL_GPU["alpha_03"]=0
CELL_GPU["alpha_04"]=0
CELL_GPU["alpha_06"]=1
CELL_GPU["alpha_07"]=1
CELL_GPU["alpha_08"]=1
CELL_GPU["alpha_09"]=1

declare -A CELL_PID

echo "========================================"
echo "  Alpha Sweep — 8 cells parallel"
echo "  Started: $(date -Iseconds)"
echo "  GPU 0: α∈{0.1, 0.2, 0.3, 0.4}"
echo "  GPU 1: α∈{0.6, 0.7, 0.8, 0.9}"
echo "========================================"
echo

# 8 cells parallel launch
for cell_id in "${!CELL_CFG[@]}"; do
    cfg="${CELL_CFG[$cell_id]}"
    gpu="${CELL_GPU[$cell_id]}"
    log="/tmp/alpha_sweep_logs/${cell_id}.log"
    echo "[GPU ${gpu}] $(date +%H:%M:%S) start ${cell_id}"
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
        echo "[FAIL] $(date +%H:%M:%S) ${cell_id} — log /tmp/alpha_sweep_logs/${cell_id}.log"
    fi
done

echo ""
echo "========================================"
echo "  8 cells Metrics summary"
echo "========================================"
for cell_id in alpha_01 alpha_02 alpha_03 alpha_04 alpha_06 alpha_07 alpha_08 alpha_09; do
    cfg="${CELL_CFG[$cell_id]}"
    mpath="outputs/${cfg}/metrics.txt"
    if [ -f "${mpath}" ]; then
        P=$(grep "^precision:" "$mpath" | head -1 | awk '{print $2}')
        R=$(grep "^recall:" "$mpath" | head -1 | awk '{print $2}')
        EX=$(grep "^ex:" "$mpath" | head -1 | awk '{print $2}')
        F1=$(awk "BEGIN {if (${P}+${R} > 0) printf \"%.4f\", 2*${P}*${R}/(${P}+${R}); else printf \"NaN\"}")
        printf "%-12s R=%s P=%s F1=%s EX=%s\n" "${cell_id}" "$R" "$P" "$F1" "$EX"
    else
        printf "%-12s NO METRICS — failed\n" "${cell_id}"
    fi
done

echo ""
echo "Alpha sweep finished: $(date -Iseconds)"
