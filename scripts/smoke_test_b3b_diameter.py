"""B-III.b smoke test — full hetero schema diameter precompute.

Per-DB profile of `metadata['schema_diameter']` and `schema_eccentricity`
across BIRD-Dev 11 DBs. Sanity check: california_schools (3 tables, 2 FK)
should have a small finite diameter.

Usage:
    conda run -n base python scripts/smoke_test_b3b_diameter.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import statistics
from modules.builders.graph_builder import EnrichedHeteroGraphBuilder

DB_DIR = "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_databases"
TABLES_JSON = "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_tables.json"


def load_db_ids():
    with open(TABLES_JSON, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return [t["db_id"] for t in tables]


def main():
    builder = EnrichedHeteroGraphBuilder(tables_json_path=TABLES_JSON)
    db_ids = load_db_ids()

    print(f"{'db_id':<32} {'T':>4} {'C':>5} {'FK':>4} {'D':>4} {'ecc_med':>8} {'ecc_max':>8}")
    diameters = []
    rows = []
    for db_id in db_ids:
        try:
            data, meta = builder.build(db_id, DB_DIR)
        except Exception as e:
            print(f"{db_id:<32}  build failed: {e}")
            continue
        T = len(meta["table_to_id"])
        C = len(meta["col_to_id"])
        FK = len(meta["fk_to_id"])
        D = int(meta["schema_diameter"])
        ecc = list(meta["schema_eccentricity"].values())
        emed = int(statistics.median(ecc)) if ecc else 0
        emax = int(max(ecc)) if ecc else 0
        diameters.append(D)
        rows.append((db_id, T, C, FK, D, emed, emax))
        print(f"{db_id:<32} {T:>4} {C:>5} {FK:>4} {D:>4} {emed:>8} {emax:>8}")

    if diameters:
        print()
        print(f"D_max profile across {len(diameters)} DBs:")
        print(f"  min    = {min(diameters)}")
        print(f"  median = {int(statistics.median(diameters))}")
        print(f"  mean   = {statistics.mean(diameters):.2f}")
        print(f"  max    = {max(diameters)}")

    # Sanity: california_schools should have a small finite diameter.
    cs_row = next((r for r in rows if r[0] == "california_schools"), None)
    if cs_row:
        D = cs_row[4]
        assert 1 <= D <= 10, f"california_schools D={D} out of expected range"
        print(f"\ncalifornia_schools D={D} (expected 1<=D<=10) OK.")

    print("\nB-III.b smoke OK.")


if __name__ == "__main__":
    main()
