"""Wave 8 D2 선결 — DB FK/PK metadata extractor (CLI thin wrapper).

DECISIONS 2026-05-18 Wave 8 §2 D2 Step 1. 실제 구현은
``src/modules/builders/db_fk_extractor.py``.

BIRD-Dev 11 DBs 의 SQLite PRAGMA (foreign_key_list / table_info) 를 읽어
``data/processed/db_fk_metadata.json`` 로 저장. LLM 0×, idempotent.

Usage:
    conda run -n base python scripts/extract_fk_metadata.py
    conda run -n base python scripts/extract_fk_metadata.py --force
    conda run -n base python scripts/extract_fk_metadata.py --db-root <path> --out <path>

40KB 정도라 NAS 저장 불필요 (로컬 ``data/processed/`` 유지).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.builders.db_fk_extractor import main


if __name__ == "__main__":
    main()
