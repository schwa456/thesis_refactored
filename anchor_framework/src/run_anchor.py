#!/usr/bin/env python
"""Anchor framework — 단일 질의 데모 entrypoint.

한 개의 (db_id, question) 에 대해 anchor 파이프라인을 돌려
선택된 부분 스키마(노드) + 생성된 SQL 을 출력한다.

사용:
  PYTHONPATH=src python src/run_anchor.py \
      --db_id california_schools \
      --question "What is the highest eligible free rate for K-12 students in Alameda County?" \
      --evidence "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"

전제: configs/anchor.yaml 의 weight_path(GAT 체크포인트), data/raw/BIRD_dev/ (스키마+sqlite),
      .env 의 ANTHROPIC_API_KEY. 자세한 준비는 README.md 참조.

BIRD-Dev 전체 평가는 src/main.py --config anchor 를 사용.
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/ 를 path 에
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utils.config_parser import load_and_merge_config
from utils.logger import setup_logger, get_logger
from pipeline import SchemaLinkingPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor framework single-query demo")
    ap.add_argument("--config", default="anchor", help="configs/<name>.yaml (기본 anchor)")
    ap.add_argument("--db_id", required=True, help="대상 DB id (data/raw/BIRD_dev/dev_databases/<db_id>)")
    ap.add_argument("--question", required=True, help="자연어 질의")
    ap.add_argument("--evidence", default="", help="외부 지식(BIRD evidence). 선택")
    ns = ap.parse_args()

    config = load_and_merge_config(ns.config)
    setup_logger(log_dir=config["paths"]["log_dir"], exp_name=config["experiment_name"])
    log = get_logger("run_anchor")

    log.info(f"Building anchor pipeline from configs/{ns.config}.yaml ...")
    pipeline = SchemaLinkingPipeline(config)

    result = pipeline.run(db_id=ns.db_id, query=ns.question, evidence=ns.evidence)

    print("\n" + "=" * 70)
    print(f"DB        : {ns.db_id}")
    print(f"Question  : {ns.question}")
    print("-" * 70)
    print(f"Selected schema nodes ({len(result.get('node_names', []))}):")
    for n in result.get("node_names", []):
        print(f"  - {n}")
    print("-" * 70)
    print("Generated SQL:")
    print(f"  {result.get('generated_sql', '(SQLGen disabled)')}")
    print("=" * 70)

    # 전체 결과 dict 도 JSON 으로
    out = {k: v for k, v in result.items() if k not in ("raw_scores",)}
    print("\n[full result JSON]")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
