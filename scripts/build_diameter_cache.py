"""B-III.b — Build per-DB schema diameter cache.

Writes {db_id: D_max} dict to ``data/processed/<split>_diameter.pt`` so
downstream Selector sweeps (advisor proposal C, num_layers ∈ {1,2,3,
D_max,D_max+1}) can read D_max without rebuilding the full HeteroData.

Mirrors the enriched/triplet cache pattern: file lives on NAS at
``/SSL_NAS/peoples/khj/thesis_refactored_offload/processed/`` and is
symlinked into ``data/processed/`` for code that uses the local path.

Usage:
    conda run -n base python scripts/build_diameter_cache.py --split dev
    conda run -n base python scripts/build_diameter_cache.py --split train
    conda run -n base python scripts/build_diameter_cache.py --split both
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.builders.graph_builder import EnrichedHeteroGraphBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
NAS_PROCESSED = Path("/SSL_NAS/peoples/khj/thesis_refactored_offload/processed")
LOCAL_PROCESSED = REPO_ROOT / "data" / "processed"

SPLIT_PATHS = {
    "dev": {
        "db_dir": REPO_ROOT / "data" / "raw" / "BIRD_dev" / "dev_databases",
        "tables_json": REPO_ROOT / "data" / "raw" / "BIRD_dev" / "dev_tables.json",
    },
    "train": {
        "db_dir": Path("/SSL_NAS/peoples/khj/thesis/train/train_databases"),
        "tables_json": Path("/SSL_NAS/peoples/khj/thesis/train/train_tables.json"),
    },
}


def list_dbs(db_dir: Path) -> list[str]:
    return sorted(p.name for p in db_dir.iterdir() if p.is_dir())


def build_split(split: str) -> Dict[str, int]:
    cfg = SPLIT_PATHS[split]
    builder = EnrichedHeteroGraphBuilder(tables_json_path=str(cfg["tables_json"]))
    out: Dict[str, int] = {}
    db_ids = list_dbs(cfg["db_dir"])
    print(f"[{split}] {len(db_ids)} DBs: {db_ids}")
    for db_id in db_ids:
        _, meta = builder.build(db_id, str(cfg["db_dir"]))
        d_max = int(meta.get("schema_diameter", 0) or 0)
        out[db_id] = d_max
        print(f"  {db_id:<30s} D_max={d_max}")
    return out


def write_cache(split: str, payload: Dict[str, int]) -> None:
    NAS_PROCESSED.mkdir(parents=True, exist_ok=True)
    LOCAL_PROCESSED.mkdir(parents=True, exist_ok=True)
    nas_path = NAS_PROCESSED / f"{split}_diameter.pt"
    local_path = LOCAL_PROCESSED / f"{split}_diameter.pt"
    torch.save(payload, nas_path)
    if local_path.exists() or local_path.is_symlink():
        local_path.unlink()
    local_path.symlink_to(nas_path)
    print(f"[{split}] wrote {nas_path}  ({nas_path.stat().st_size} B)")
    print(f"[{split}] symlinked {local_path} -> {nas_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "train", "both"], default="dev")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache file already exists.",
    )
    args = ap.parse_args()
    splits = ["dev", "train"] if args.split == "both" else [args.split]
    for sp in splits:
        nas_path = NAS_PROCESSED / f"{sp}_diameter.pt"
        if nas_path.exists() and not args.force:
            print(f"[{sp}] cache exists at {nas_path} — skip (use --force to rebuild)")
            continue
        payload = build_split(sp)
        write_cache(sp, payload)
    print("Done.")


if __name__ == "__main__":
    main()
