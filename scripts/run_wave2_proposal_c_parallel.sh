#!/bin/bash
# Wave 2 Proposal C — Parallel orchestrator v2 (PID-only waits)
# 기존 v1 버그: `until [ -f $ckpt ]` 가 이전 실험 잔존 파일 존재시 즉시 통과 → 4 중복 학습 kick
# v2 수정: 파일 기반 wait 제거, PID 기반만 사용. 기존 실행 중 nl1/nl2 PID 를 인자로 받음.
#
# 사용법:
#   bash scripts/run_wave2_proposal_c_parallel.sh <NL1_PID> <NL2_PID>
#
# 스케줄:
#   [GPU 0] nl1 (실행 중, arg1) → nl3 → nl7
#   [GPU 1] nl2 (실행 중, arg2) → nl6
#   ETA: ceil(5/2) × 5h = ~15h

set -u
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    echo "usage: $0 <NL1_PID> <NL2_PID>"
    exit 1
fi

NL1_PID="$1"
NL2_PID="$2"

NAS_CKPT_DIR="/SSL_NAS/peoples/khj/thesis/checkpoints"
LOG_DIR="/tmp/wave2_parallel_logs"
mkdir -p "${LOG_DIR}"

launch_train() {
    local gpu="$1"
    local nl="$2"
    local cfg="configs/training/diameter_layers/train_qcond_nl${nl}.yaml"
    local logfile="${LOG_DIR}/train_nl${nl}_gpu${gpu}.log"
    echo "[$(date -Iseconds)] GPU ${gpu} → nl${nl} launch" >&2
    CUDA_VISIBLE_DEVICES="${gpu}" nohup conda run -n base python src/train_gat.py --config "${cfg}" \
        > "${logfile}" 2>&1 &
    local pid=$!
    disown
    echo "  PID ${pid}, log ${logfile}" >&2
    echo "${pid}"
}

wait_for_pid() {
    local pid="$1"
    local tag="$2"
    echo "[$(date -Iseconds)] waiting for ${tag} PID ${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date -Iseconds)] ${tag} PID ${pid} exited"
}

echo "========================================"
echo "  Wave 2 Parallel v2 — $(date -Iseconds)"
echo "  nl1 PID ${NL1_PID} (GPU 0, running)"
echo "  nl2 PID ${NL2_PID} (GPU 1, running)"
echo "========================================"

# nl1 (GPU 0) 대기 → nl3 (GPU 0) launch
wait_for_pid "${NL1_PID}" "nl1"
PID_NL3=$(launch_train 0 3)

# nl2 (GPU 1) 대기 → nl6 (GPU 1) launch
wait_for_pid "${NL2_PID}" "nl2"
PID_NL6=$(launch_train 1 6)

# nl3 (GPU 0) 대기 → nl7 (GPU 0) launch
wait_for_pid "${PID_NL3}" "nl3"
PID_NL7=$(launch_train 0 7)

# nl6 + nl7 완료 대기
wait_for_pid "${PID_NL6}" "nl6"
wait_for_pid "${PID_NL7}" "nl7"

echo ""
echo "All 5 trainings finished — NAS symlink"

CKPT_BASENAMES=(
    "best_gat_qcond_nl1.pt"
    "best_gat_qcond_nl2.pt"
    "best_gat_qcond_nl3.pt"
    "best_gat_qcond_nl6.pt"
    "best_gat_qcond_nl7.pt"
)

for ckpt in "${CKPT_BASENAMES[@]}"; do
    LOCAL="outputs/checkpoints/${ckpt}"
    NAS="${NAS_CKPT_DIR}/${ckpt}"
    if [ -f "${LOCAL}" ] && [ ! -L "${LOCAL}" ]; then
        mv "${LOCAL}" "${NAS}"
        ln -s "${NAS}" "${LOCAL}"
        echo "  moved ${ckpt} → NAS, symlinked"
    else
        echo "  skip ${ckpt} (absent or already symlink)"
    fi
done

echo "Phase 1 완료: $(date -Iseconds)"
