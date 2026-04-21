"""B-I smoke test: RFMCompatibleBuilder serialize + per-DB token-length profile."""
import os
import sys
import json
import statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.builders.graph_builder import RFMCompatibleBuilder

DB_DIR = "data/raw/BIRD_dev/dev_databases"
TABLES_JSON = "data/raw/BIRD_dev/dev_tables.json"
DEV_JSON = "data/raw/BIRD_dev/dev.json"


def main():
    print("=" * 72)
    print("B-I smoke test (RFMCompatibleBuilder)")
    print("=" * 72)

    builder = RFMCompatibleBuilder(tables_json_path=TABLES_JSON)

    # 1. Manual inspection on california_schools
    print("\n[1] california_schools serialized snippet")
    txt = builder.serialize("california_schools", DB_DIR)
    print("  " + txt[:600] + ("..." if len(txt) > 600 else ""))
    print(f"  total chars: {len(txt)}, tokens: {len(txt.split())}")

    # 2. build() returns metadata with rfm_text + rfm_tokens
    print("\n[2] build() exposes rfm_* metadata")
    data, meta = builder.build("california_schools", DB_DIR)
    for k in ("rfm_text", "rfm_tokens", "rfm_special_tokens",
              "fk_reachability", "table_to_id"):
        assert k in meta, f"missing key {k}"
    print(f"  rfm_tokens count = {len(meta['rfm_tokens'])}")
    print(f"  fk_reachability shape = {meta['fk_reachability'].shape}")

    # 3. Token-length profile across dev DBs
    print("\n[3] Per-DB token-length profile (dev set)")
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        dev = json.load(f)
    db_ids = sorted({q["db_id"] for q in dev})
    lengths = []
    char_lengths = []
    longest = ("", 0)
    for db_id in db_ids:
        try:
            t = builder.serialize(db_id, DB_DIR)
        except Exception as e:
            print(f"  ! serialize failed for {db_id}: {e}")
            continue
        n_tok = len(t.split())
        lengths.append(n_tok)
        char_lengths.append(len(t))
        if n_tok > longest[1]:
            longest = (db_id, n_tok)

    print(f"  dev DB count           : {len(db_ids)}")
    print(f"  token count  min/median/mean/max : "
          f"{min(lengths)} / {statistics.median(lengths):.0f} / "
          f"{statistics.mean(lengths):.0f} / {max(lengths)}")
    print(f"  char count             : "
          f"{min(char_lengths)} / {statistics.median(char_lengths):.0f} / "
          f"{statistics.mean(char_lengths):.0f} / {max(char_lengths)}")
    print(f"  longest DB             : {longest[0]} ({longest[1]} tokens)")

    print("\n[OK] smoke test finished")


if __name__ == "__main__":
    main()
