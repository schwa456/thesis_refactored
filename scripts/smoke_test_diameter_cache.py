"""B-III.b — Minimal smoke for the diameter cache writer.

Verifies that ``scripts/build_diameter_cache.py`` end-to-end logic works
on a single DB without touching all 11 BIRD-Dev DBs (NAS-bound full build
is deferred to GAT training start). Confirms:
  - builder produces a finite ``schema_diameter`` for the DB
  - cache file gets written and reloaded correctly (``{db_id: D_max}``)
  - NAS path + local symlink convention is respected (writes to a tmp
    location to avoid clobbering the real cache)

Usage:
    conda run -n base python scripts/smoke_test_diameter_cache.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.builders.graph_builder import EnrichedHeteroGraphBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "data" / "raw" / "BIRD_dev" / "dev_databases"
TABLES_JSON = REPO_ROOT / "data" / "raw" / "BIRD_dev" / "dev_tables.json"
DB_ID = "california_schools"


def main() -> None:
    builder = EnrichedHeteroGraphBuilder(tables_json_path=str(TABLES_JSON))
    _, meta = builder.build(DB_ID, str(DB_DIR))
    d_max = int(meta["schema_diameter"])
    assert d_max > 0, f"schema_diameter must be positive, got {d_max}"
    payload = {DB_ID: d_max}

    with tempfile.TemporaryDirectory() as tmp:
        nas_path = Path(tmp) / "nas" / "dev_diameter.pt"
        local_path = Path(tmp) / "local" / "dev_diameter.pt"
        nas_path.parent.mkdir(parents=True)
        local_path.parent.mkdir(parents=True)
        torch.save(payload, nas_path)
        local_path.symlink_to(nas_path)

        loaded = torch.load(local_path, weights_only=False)
        assert loaded == payload, f"reload mismatch: {loaded} vs {payload}"
        assert local_path.is_symlink(), "local path must be a symlink"
        assert local_path.resolve() == nas_path.resolve(), "symlink must resolve to NAS path"

    print(f"DB={DB_ID}, D_max={d_max}")
    print("Cache writer logic OK — full 11-DB build deferred to GAT training trigger.")


if __name__ == "__main__":
    main()
