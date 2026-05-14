#!/bin/bash
# ============================================================
# Migrate loose outputs/ and logs/ directories to match configs/ structure.
# Dry-run first: bash scripts/migrate_directory_structure.sh --dry-run
# Execute:       bash scripts/migrate_directory_structure.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export TMPDIR=/tmp
ROOT="$(pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — no files will be moved ==="
fi

move_dir() {
    local src="$1"
    local dst="$2"
    if [[ ! -d "$src" ]]; then
        return
    fi
    # Skip if source is empty
    if [[ -z "$(ls -A "$src" 2>/dev/null)" ]]; then
        echo "  SKIP (empty): $src"
        if [[ "$DRY_RUN" == false ]]; then
            rmdir "$src" 2>/dev/null || true
        fi
        return
    fi
    # Skip if already at target
    if [[ "$src" == "$dst" ]]; then
        return
    fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "  MOVE: $src → $dst"
    else
        mkdir -p "$(dirname "$dst")"
        if [[ -d "$dst" ]]; then
            # Merge into existing dir
            echo "  MERGE: $src → $dst"
            cp -rn "$src"/* "$dst"/ 2>/dev/null || true
            rm -rf "$src"
        else
            echo "  MOVE: $src → $dst"
            mv "$src" "$dst"
        fi
    fi
}

remove_empty() {
    local dir="$1"
    if [[ -d "$dir" ]] && [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "  REMOVE (empty): $dir"
        else
            echo "  REMOVE (empty): $dir"
            rmdir "$dir"
        fi
    fi
}

echo ""
echo "=== 1. outputs/ — s03_a09 topology_cost → experiments/s03_gat_ensemble/a09_topology_cost/ ==="
for d in "$ROOT"/outputs/s03_a09_*; do
    name=$(basename "$d")
    move_dir "$d" "$ROOT/outputs/experiments/s03_gat_ensemble/a09_topology_cost/$name"
done
# s03_a09_02_topology_xiyan has no config but has output dir
move_dir "$ROOT/outputs/s03_a09_02_topology_xiyan" \
         "$ROOT/outputs/experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_02_topology_xiyan"

echo ""
echo "=== 2. outputs/ — s03_a10 fk_steiner → experiments/s03_gat_ensemble/a10_fk_steiner/ ==="
for d in "$ROOT"/outputs/s03_a10_*; do
    name=$(basename "$d")
    move_dir "$d" "$ROOT/outputs/experiments/s03_gat_ensemble/a10_fk_steiner/$name"
done

echo ""
echo "=== 3. outputs/ — a05 gpt4omini filters → experiments/abl/a05_filter_agentic/ ==="
move_dir "$ROOT/outputs/a05_14_adaptive_multi_agent_gpt4omini" \
         "$ROOT/outputs/experiments/abl/a05_filter_agentic/a05_14_adaptive_multi_agent_gpt4omini"
move_dir "$ROOT/outputs/a05_15_reflection_1iter_gpt4omini" \
         "$ROOT/outputs/experiments/abl/a05_filter_agentic/a05_15_reflection_1iter_gpt4omini"
move_dir "$ROOT/outputs/a05_17_verifier_gpt4omini" \
         "$ROOT/outputs/experiments/abl/a05_filter_agentic/a05_17_verifier_gpt4omini"

echo ""
echo "=== 4. outputs/ — legacy qcond/supernode root duplicates ==="
# These are duplicates of experiments/s04. Move to archive.
move_dir "$ROOT/outputs/qcond_idea24_a0_xiyan" \
         "$ROOT/outputs/archive/legacy_base_runs/qcond_idea24_a0_xiyan"
move_dir "$ROOT/outputs/supernode_idea24_a0_xiyan" \
         "$ROOT/outputs/archive/legacy_base_runs/supernode_idea24_a0_xiyan"

echo ""
echo "=== 5. outputs/ — empty legacy dirs ==="
remove_empty "$ROOT/outputs/experiment_b0_raw_pcst_baseline"
remove_empty "$ROOT/outputs/experiment_b1_adaptive_pcst"

echo ""
echo "=== 6. outputs/configs — stale experiment outputs (wrong path) → archive ==="
if [[ -d "$ROOT/outputs/configs" ]]; then
    move_dir "$ROOT/outputs/configs" "$ROOT/outputs/archive/stale_configs_outputs"
fi

echo ""
echo "=== 7. outputs/logs — misplaced wrapper logs → archive ==="
if [[ -d "$ROOT/outputs/logs" ]]; then
    move_dir "$ROOT/outputs/logs" "$ROOT/outputs/archive/wrapper_logs"
fi

echo ""
echo "========================================"
echo "=== logs/ — same structure migration ==="
echo "========================================"

echo ""
echo "=== 8. logs/ — s03_a09 → experiments/s03_gat_ensemble/a09_topology_cost/ ==="
for d in "$ROOT"/logs/s03_a09_*; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    move_dir "$d" "$ROOT/logs/experiments/s03_gat_ensemble/a09_topology_cost/$name"
done
move_dir "$ROOT/logs/s03_a09_02_topology_xiyan" \
         "$ROOT/logs/experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_02_topology_xiyan"

echo ""
echo "=== 9. logs/ — s03_a10 → experiments/s03_gat_ensemble/a10_fk_steiner/ ==="
for d in "$ROOT"/logs/s03_a10_*; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    move_dir "$d" "$ROOT/logs/experiments/s03_gat_ensemble/a10_fk_steiner/$name"
done

echo ""
echo "=== 10. logs/ — a05 gpt4omini → experiments/abl/a05_filter_agentic/ ==="
move_dir "$ROOT/logs/a05_14_adaptive_multi_agent_gpt4omini" \
         "$ROOT/logs/experiments/abl/a05_filter_agentic/a05_14_adaptive_multi_agent_gpt4omini"
move_dir "$ROOT/logs/a05_15_reflection_1iter_gpt4omini" \
         "$ROOT/logs/experiments/abl/a05_filter_agentic/a05_15_reflection_1iter_gpt4omini"
move_dir "$ROOT/logs/a05_17_verifier_gpt4omini" \
         "$ROOT/logs/experiments/abl/a05_filter_agentic/a05_17_verifier_gpt4omini"

echo ""
echo "=== 11. logs/ — loose .log files → archive ==="
if [[ "$DRY_RUN" == true ]]; then
    for f in "$ROOT"/logs/a05_*.log "$ROOT"/logs/a05_gpt4omini_wrapper.log; do
        [[ -f "$f" ]] && echo "  MOVE: $f → $ROOT/logs/archive/"
    done
else
    mkdir -p "$ROOT/logs/archive"
    for f in "$ROOT"/logs/a05_*.log "$ROOT"/logs/a05_gpt4omini_wrapper.log; do
        [[ -f "$f" ]] && mv "$f" "$ROOT/logs/archive/" && echo "  MOVE: $(basename "$f") → logs/archive/"
    done
fi

echo ""
echo "=== Done ==="
