"""§7.3 서버 통합 — Spider/BIRD dev 에서 kappa_diag.csv 생성 (이 코드베이스 결선판)

작업지시서 rfp_HGAT_to_kkapa_v_2026-06-12.md 의 [TODO 1]~[TODO 5] 를 thesis_refactored
실제 파이프라인 객체로 결선한 버전. (work_instructions_P0 §B.1 비교지표 dirichlet/mad 는
kappa_hook.KappaDiagnostics 가 finish_example 에서 같은 혼동집합 기준으로 함께 산출 → CSV 열 병합.)

  - 모델:   models.gat_network_v2.SchemaHeteroGATv2 (query_conditioned, v6w5_variant='a'
            = column self-loop). modules.selectors.direct_gatv2_selector.DirectGATv2Selector 로 로드
            (auto_config_from_ckpt=True → ckpt config['model'] 에서 구조 자동 복원)
  - 그래프: modules.builders.graph_builder.EnrichedHeteroGraphBuilder
  - gold:   utils.evaluator.parse_sql_elements + metadata['col_to_id'] (대소문자 무시 매칭)

체크포인트: best_gat_v6w5_a_s11.pt — column self-loop 포함 학습본 (비-FK 컬럼 conv 붕괴 완화).

궤적: GATv2Conv 출력(활성화 후) 3개 층(X1,X2,X3, 1024-dim). KappaDiagnostics(slope_lo=0) 3점 회귀.
점수/Recall@15: Direct classifier head per-column sigmoid(selector.latest_scores) 의 컬럼 구간 top-15.

실행:
  CUDA_VISIBLE_DEVICES=0,1 python diagnostic/integrate_kappa_spider.py [--limit 20] [--out_csv ...]
  python diagnostic/analyze_p1_p2.py diagnostic/kappa_diag.csv diagnostic/p1p2_result.json
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for p in (str(HERE), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kappa_hook import KappaDiagnostics  # noqa: E402

from modules.builders.graph_builder import EnrichedHeteroGraphBuilder  # noqa: E402
from modules.selectors.direct_gatv2_selector import DirectGATv2Selector  # noqa: E402
from utils.evaluator import parse_sql_elements  # noqa: E402


ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ════════════════════════════════════════════════════════════════════
# [TODO 1] 모델·데이터 로드
# ════════════════════════════════════════════════════════════════════
def load_model_and_data(args):
    selector = DirectGATv2Selector(
        weight_path=args.ckpt,
        query_conditioned=True,
        encoder_type="plm",
        auto_config_from_ckpt=True,
    )
    builder = EnrichedHeteroGraphBuilder(
        plm_model_name=ENCODER_MODEL,
        tables_json_path=args.dev_tables_json,
    )
    with open(args.dev_json, "r", encoding="utf-8") as f:
        dev = json.load(f)
    if args.limit > 0:
        dev = dev[: args.limit]
    return selector, builder, dev


# ════════════════════════════════════════════════════════════════════
# [TODO 2] 스키마 → 혼동 집합 (동일 테이블 컬럼 묶음, 컬럼 타입 로컬 인덱스)
# ════════════════════════════════════════════════════════════════════
def build_confusion_sets(metadata):
    col_to_id = metadata.get("col_to_id", {})
    by_tbl = defaultdict(list)
    for full_name, local_idx in col_to_id.items():
        tbl = full_name.split(".", 1)[0]
        by_tbl[tbl].append(int(local_idx))
    return {f"tbl_{t}": sorted(idxs) for t, idxs in by_tbl.items() if len(idxs) >= 2}


# ════════════════════════════════════════════════════════════════════
# [TODO 3] 컬럼 차수
# ════════════════════════════════════════════════════════════════════
def column_degrees(graph_data):
    deg = defaultdict(int)
    for (src_t, _rel, dst_t), ei in graph_data.edge_index_dict.items():
        if ei.numel() == 0:
            continue
        if src_t == "column":
            for i in ei[0].tolist():
                deg[i] += 1
        if dst_t == "column":
            for i in ei[1].tolist():
                deg[i] += 1
    return dict(deg)


# ════════════════════════════════════════════════════════════════════
# [TODO 4] 층별 궤적 + 점수 (한 번의 forward 로 동시 수집)
# ════════════════════════════════════════════════════════════════════
class _ColumnTrajCapture:
    def __init__(self):
        self.layers = []

    def reset(self):
        self.layers = []

    def hook(self, _module, _inp, out):
        col = out.get("column") if isinstance(out, dict) else None
        if col is not None and col.numel() > 0:
            self.layers.append(F.elu(col).detach().cpu().numpy().astype(float))


def forward_with_traj(selector, graph_data, metadata, question, capture):
    """capture(hook 부착) reset 후 selector.select() 1회 호출 → 컬럼 궤적 수집 +
    Direct classifier per-node sigmoid score 가 latest_scores 에 저장.
    Returns: (traj[list of np[N_col, d]], scores[list, 전체 flat 노드])."""
    capture.reset()
    n_nodes = len(metadata.get("node_metadata", {}))
    selector.select(
        scores=None, candidates=list(range(n_nodes)),
        question=question, graph_data=graph_data, metadata=metadata,
    )
    return list(capture.layers), list(selector.latest_scores)


# ════════════════════════════════════════════════════════════════════
# [TODO 5] 평가 결과 (top-15 컬럼 set, gold 컬럼 set) — 컬럼 로컬 인덱스
# ════════════════════════════════════════════════════════════════════
def topk_and_gold(graph_data, metadata, scores, gold_sql, k=15):
    """대소문자 무시 매칭 (bird_dataset.py case-sensitive 버그를 진단 범위에서 보정)."""
    num_t = graph_data["table"].num_nodes
    num_c = graph_data["column"].num_nodes
    col_to_id = metadata.get("col_to_id", {})

    col_scores = torch.tensor(scores[num_t:num_t + num_c], dtype=torch.float32)
    kk = min(k, num_c)
    topk = set(torch.topk(col_scores, kk).indices.tolist()) if kk > 0 else set()

    gold_tables, gold_cols = parse_sql_elements(gold_sql)  # 이미 소문자
    gold_tables_l = {t.lower() for t in gold_tables}
    gold = set()
    for full_name, idx in col_to_id.items():
        tbl, _, col = full_name.partition(".")
        if col.lower() in gold_cols and tbl.lower() in gold_tables_l:
            gold.add(int(idx))
    return topk, gold


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="§7.3 kappa_v 진단 → kappa_diag.csv")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/checkpoints/best_gat_v6w5_a_s11.pt"))
    ap.add_argument("--dev_json", default=str(ROOT / "data/raw/BIRD_dev/dev.json"))
    ap.add_argument("--dev_db_dir", default=str(ROOT / "data/raw/BIRD_dev/dev_databases"))
    ap.add_argument("--dev_tables_json", default=str(ROOT / "data/raw/BIRD_dev/dev_tables.json"))
    ap.add_argument("--out_csv", default=str(HERE / "kappa_diag.csv"))
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    selector, builder, dev = load_model_and_data(args)

    capture = _ColumnTrajCapture()
    handles = [conv.register_forward_hook(capture.hook) for conv in selector.gat_model.convs]

    db_cache = {}
    all_rows = []
    n_ok = n_skip_noset = n_skip_nogold = n_err = 0
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            for qi, ex in enumerate(dev):
                db_id = ex["db_id"]
                question = ex["question"]
                gold_sql = ex.get("SQL", ex.get("query", "")) or ""
                ex_id = str(ex.get("question_id", qi))
                try:
                    if db_id not in db_cache:
                        g, meta = builder.build(db_id=db_id, db_dir=args.dev_db_dir)
                        csets = build_confusion_sets(meta)
                        degs = column_degrees(g)
                        db_cache[db_id] = (g, meta, csets, degs)
                    graph_data, metadata, csets, degs = db_cache[db_id]

                    if not csets:
                        n_skip_noset += 1
                        continue
                    _, gold_cols = parse_sql_elements(gold_sql)
                    if not gold_cols:
                        n_skip_nogold += 1
                        continue

                    diag = KappaDiagnostics(csets, degrees=degs, slope_lo=0)
                    diag.start_example(ex_id)

                    traj, scores = forward_with_traj(
                        selector, graph_data, metadata, question, capture)
                    if len(traj) < 2:
                        n_err += 1
                        continue
                    for H in traj:
                        diag.record(H)

                    topk, gold = topk_and_gold(
                        graph_data, metadata, scores, gold_sql, k=args.k)
                    rows = diag.finish_example()
                    cr = int(set(gold) <= set(topk))
                    for r in rows:
                        r["is_gold"] = int(r["node_id"] in gold)
                        r["recalled"] = int(r["node_id"] in topk)
                        r["complete_recall"] = cr
                        r["db_id"] = db_id
                    all_rows.extend(rows)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"  [warn] q{qi} ({db_id}) 실패: {e}", file=sys.stderr)

                if (qi + 1) % args.log_every == 0 or (qi + 1) == len(dev):
                    el = time.perf_counter() - t0
                    rate = (qi + 1) / max(el, 1e-6)
                    eta = (len(dev) - qi - 1) / max(rate, 1e-6)
                    print(f"  [{qi+1}/{len(dev)}] ok={n_ok} skip(noset={n_skip_noset},"
                          f"nogold={n_skip_nogold}) err={n_err} | {1.0/rate:.3f}s/q "
                          f"eta={eta/60:.1f}min")
    finally:
        for h in handles:
            h.remove()

    if not all_rows:
        print("저장할 행이 없습니다.")
        return

    keys = sorted({kk for r in all_rows for kk in r})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    el = time.perf_counter() - t0
    print(f"saved -> {args.out_csv} ({len(all_rows)} rows, {n_ok} examples, wall={el:.1f}s)")
    print(f"열: {keys}")


if __name__ == "__main__":
    main()
