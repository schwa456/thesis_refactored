#!/usr/bin/env python
"""M4 reproducibility package — smoke test.

두 단계로 구성된다:

  [Tier 1] 패키지 무결성 (항상 실행, GPU/API/데이터 불필요)
      - src/ import 가 깨지지 않는지
      - 레지스트리에 M4 anchor 6개 모듈이 모두 등록되는지
      - m4.yaml + base_config.yaml 병합이 M4 구성을 산출하는지

  [Tier 2] 단일 질의 end-to-end (자산이 모두 갖춰졌을 때만 실행)
      - checkpoints/best_gat_qcond_nl3.pt + data/raw/BIRD_dev/ + GLM_API_KEY 가
        모두 존재하면 BIRD dev 의 첫 질의를 파이프라인에 통과시켜
        Recall/Precision 를 측정하고 합리적 범위인지 확인한다.

사용:
    python tests/test_smoke.py            # Tier 1 (+ 자산 있으면 Tier 2)
    PYTHONPATH=src python tests/test_smoke.py

종료 코드: 0 = 통과, 1 = 실패.
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# .env 로드 (GLM_BASE_URL / GLM_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


def _ok(msg):  print(f"  ✅ {msg}")
def _info(msg): print(f"  ·  {msg}")
def _fail(msg):
    print(f"  🚨 {msg}")
    sys.exit(1)


# ============================================================
# Tier 1 — 패키지 무결성
# ============================================================
def tier1_integrity():
    print("[Tier 1] 패키지 무결성 검사")

    # 1. modules 패키지 import → 레지스트리 자동 등록
    import modules  # noqa: F401
    from modules.registry import REGISTRY
    _ok("modules import + registry bootstrap")

    # 2. M4 anchor 6개 모듈이 등록되었는지
    expected = {
        "builder":   "EnrichedHeteroGraphBuilder",
        "encoder":   "LocalPLMEncoder",
        "selector":  "EnsembleSelector",
        "extractor": "MSTPCSTUnionExtractor",
        "filter":    "BidirectionalFilter",
        "generator": "LLMSQLGenerator",
    }
    for cat, name in expected.items():
        if name not in REGISTRY.get(cat, {}):
            _fail(f"레지스트리에 [{cat}] '{name}' 미등록 — available={list(REGISTRY.get(cat, {}))}")
    _ok(f"M4 anchor 6개 모듈 등록 확인: {list(expected.values())}")

    # 3. config 병합이 M4 구성을 산출하는지
    from utils.config_parser import load_and_merge_config
    cfg = load_and_merge_config("m4")
    checks = [
        ("graph_builder", "EnrichedHeteroGraphBuilder"),
        ("nlq_encoder", "LocalPLMEncoder"),
        ("seed_selector", "EnsembleSelector"),
        ("connectivity_extractor", "MSTPCSTUnionExtractor"),
        ("filter", "BidirectionalFilter"),
        ("sql_generator", "LLMSQLGenerator"),
    ]
    for key, name in checks:
        got = cfg.get(key, {}).get("name")
        if got != name:
            _fail(f"config 병합 결과 {key}.name={got} (기대 {name})")
    # selector 핵심 하이퍼파라미터
    sp = cfg["seed_selector"]["params"]
    assert abs(sp["alpha"] - 0.5) < 1e-9 and sp["top_k"] == 20 and sp["query_conditioned"] is True, sp
    _ok("m4.yaml + base_config.yaml 병합 = M4 anchor 구성 (α=0.5, top_k=20, QCond)")
    print("[Tier 1] 통과\n")
    return cfg


# ============================================================
# Tier 2 — 단일 질의 end-to-end (자산 있을 때만)
# ============================================================
def tier2_single_query(cfg):
    print("[Tier 2] 단일 질의 end-to-end 검증")

    ckpt = ROOT / "checkpoints" / "best_gat_qcond_nl3.pt"
    dev_json = ROOT / "data" / "raw" / "BIRD_dev" / "dev.json"
    has_glm = bool(os.getenv("GLM_API_KEY"))

    missing = []
    if not ckpt.exists():     missing.append(f"checkpoint ({ckpt})")
    if not dev_json.exists(): missing.append(f"BIRD dev ({dev_json})")
    if not has_glm:           missing.append("GLM_API_KEY (.env)")
    if missing:
        _info("Tier 2 skip — 다음 자산이 없습니다: " + "; ".join(missing))
        _info("(자산을 갖춘 뒤 다시 실행하면 단일 질의 R/P/F1 을 검증합니다)")
        return

    # CWD 를 ROOT 로 — main.py / config 의 상대 경로 기준 일치
    os.chdir(ROOT)
    from pipeline import SchemaLinkingPipeline
    from utils.evaluator import parse_sql_elements, calculate_schema_metrics

    with open(dev_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    item = dataset[0]
    _info(f"query[0]: db_id={item['db_id']} | {item['question'][:70]}...")

    pipeline = SchemaLinkingPipeline(cfg)
    result = pipeline.run(
        db_id=item["db_id"], query=item["question"],
        evidence=item.get("evidence", ""),
    )

    gold_tables, gold_cols = parse_sql_elements(item.get("SQL", item.get("query", "")))
    gold_cols = set(c.lower() for c in gold_cols)
    pred_cols = set()
    for node in result.get("final_nodes", []):
        if "->" in node:
            continue
        if "." in node:
            t, c = node.split(".", 1)
            tc = f"{t.lower()}.{c.lower()}"
            pred_cols.add(tc if tc in gold_cols else c.lower())
    recall, precision, _, _ = calculate_schema_metrics(pred_cols, gold_cols)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    print(f"  · status     = {result.get('status')}")
    print(f"  · final_nodes= {len(result.get('final_nodes', []))}")
    print(f"  · Recall     = {recall:.4f}")
    print(f"  · Precision  = {precision:.4f}")
    print(f"  · F1         = {f1:.4f}")
    print(f"  · generated_sql[:80] = {result.get('generated_sql','')[:80]}")

    # 단일 질의이므로 정확한 수치는 검증하지 않고, 파이프라인이 비정상(전부 0/빈 결과)이
    # 아닌지만 sanity check 한다.
    if not result.get("final_nodes"):
        _fail("final_nodes 가 비었습니다 — 파이프라인 이상")
    if recall <= 0.0:
        _fail(f"Recall=0 — gold column 을 하나도 회수하지 못함 (이상)")
    _ok("단일 질의 end-to-end 정상 (final_nodes 비어있지 않고 Recall>0)")
    print("[Tier 2] 통과\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" M4 reproducibility package — smoke test")
    print("=" * 60)
    cfg = tier1_integrity()
    tier2_single_query(cfg)
    print("🎉 smoke test 완료.")
