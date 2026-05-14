#!/bin/bash
# Wave 2 Phase 1 — NAS move watcher
# 각 학습이 종료되는 대로 해당 nl 의 체크포인트를 NAS 로 이동 + symlink.
# orchestrator v2 의 최종 NAS 이동 단계와 race-free (symlink 상태면 orchestrator 가 skip).
#
# 처리 순서: nl2 → nl3 → nl6 → nl7 (nl1 은 이미 symlink 상태)

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

NAS_CKPT_DIR="/SSL_NAS/peoples/khj/thesis/checkpoints"

move_if_ready() {
    local nl="$1"
    local LOCAL="outputs/checkpoints/best_gat_qcond_nl${nl}.pt"
    local NAS="${NAS_CKPT_DIR}/best_gat_qcond_nl${nl}.pt"
    if [ -f "${LOCAL}" ] && [ ! -L "${LOCAL}" ]; then
        mv "${LOCAL}" "${NAS}"
        ln -s "${NAS}" "${LOCAL}"
        echo "[$(date -Iseconds)] nl${nl} → NAS + symlinked ($(ls -lh ${NAS} | awk '{print $5}'))"
    else
        echo "[$(date -Iseconds)] nl${nl} already symlink or absent"
    fi
}

wait_for_training() {
    local nl="$1"
    local pattern="train_gat.py --config.*nl${nl}.yaml"

    # Phase A: wait until training process exists
    echo "[$(date -Iseconds)] watcher: polling for nl${nl} training to start"
    until pgrep -f "${pattern}" > /dev/null; do
        sleep 60
    done
    echo "[$(date -Iseconds)] watcher: nl${nl} training detected"

    # Phase B: wait until training process ends
    while pgrep -f "${pattern}" > /dev/null; do
        sleep 60
    done
    echo "[$(date -Iseconds)] watcher: nl${nl} training ended"
}

echo "========================================"
echo "  NAS move watcher — $(date -Iseconds)"
echo "  Order: nl2 → nl3 → nl6 → nl7"
echo "========================================"

for nl in 2 3 6 7; do
    wait_for_training "${nl}"
    move_if_ready "${nl}"
done

echo ""
echo "[$(date -Iseconds)] All NAS moves completed"
