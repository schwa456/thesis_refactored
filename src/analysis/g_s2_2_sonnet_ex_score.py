#!/usr/bin/env python3
"""G-S2-2 Sonnet cross-dataset EX 채점 (Spider2-lite local 123).

공식 evaluate.py / evaluate_utils.py 의 채점 로직을 그대로 복제 (bigquery/snowflake/duckdb
의존성 미설치로 import 불가 → 순수 sqlite 경로 함수만 faithful copy).
- predicted_sql: outputs/experiments/g_s2_2_spider2/predicted_sql.jsonl (instance_id, db, predicted_sql)
- sqlite: data/Spider2/spider2-lite/resource/databases/spider2-localdb/{db}.sqlite
- gold exec_result: .../evaluation_suite/gold/exec_result/{id}(_x).csv
- eval standard: .../gold/spider2lite_eval.jsonl (condition_cols, ignore_order)
EX = score==1 비율. exec 실패/dialect 불일치/timeout → score 0.
"""
import json, os, re, math, sqlite3, signal
from pathlib import Path
import pandas as pd

ROOT = "/home/hyeonjin/thesis_refactored"
SP = os.path.join(ROOT, "data/Spider2/spider2-lite")
PRED = os.path.join(ROOT, "outputs/experiments/g_s2_2_spider2/predicted_sql.jsonl")
SQLITE_DIR = os.path.join(SP, "resource/databases/spider2-localdb")
GOLD_EXEC = os.path.join(SP, "evaluation_suite/gold/exec_result")
EVAL_STD = os.path.join(SP, "evaluation_suite/gold/spider2lite_eval.jsonl")
OUT = os.path.join(ROOT, "outputs/analysis")
TIMEOUT_S = 60


# ── 공식 채점 함수 faithful copy (evaluate.py L67-130, L200-223) ──
def extract_sql_query(s):
    m = re.search(r"```sql\n(.*?)\n```", s, re.DOTALL)
    return m.group(1).strip() if m else s


def compare_pandas_table(pred, gold, condition_cols=None, ignore_order=False):
    tol = 1e-2

    def normalize(v):
        return 0 if pd.isna(v) else v

    def vectors_match(v1, v2, ignore_order_=False):
        v1 = [normalize(x) for x in v1]; v2 = [normalize(x) for x in v2]
        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    t_gold = gold_cols.transpose().values.tolist()
    t_pred = pred.transpose().values.tolist()
    for gv in t_gold:
        if not any(vectors_match(gv, pv, ignore_order) for pv in t_pred):
            return 0
    return 1


def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=None, multi_ignore_order=False):
    if not multi_gold:
        return 0
    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(s, list) for s in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]
    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order):
            return 1
    return 0


def resolve_gold_paths(instance_id, gold_dir):
    base = Path(gold_dir) / f"{instance_id}.csv"
    if base.exists():
        return [base], True
    pat = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    files = sorted(f for f in os.listdir(gold_dir) if pat.match(f))
    return [Path(gold_dir) / f for f in files], False


class _TO(Exception):
    pass


def _alarm(signum, frame):
    raise _TO()


def get_sqlite_result(db_path, query):
    """timeout 추가(공식은 무제한) — 안전. 실패시 (False, err)."""
    conn = sqlite3.connect(db_path); mem = sqlite3.connect(":memory:")
    conn.backup(mem)
    old = signal.signal(signal.SIGALRM, _alarm); signal.alarm(TIMEOUT_S)
    try:
        df = pd.read_sql_query(query, mem)
        return True, df
    except _TO:
        return False, f"timeout>{TIMEOUT_S}s"
    except Exception as e:
        return False, str(e)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old); mem.close(); conn.close()


def main():
    preds = [json.loads(l) for l in open(PRED)]
    std = {json.loads(l)["instance_id"]: json.loads(l) for l in open(EVAL_STD)}
    gold_ids = set(std)
    results = []
    n_exec_fail = n_no_gold = n_correct = 0
    fail_reasons = []
    for r in preds:
        iid = r["instance_id"]; db = r["db"]
        sql = extract_sql_query(r["predicted_sql"])
        if iid not in gold_ids:
            results.append(dict(id=iid, db=db, score=None, note="no_gold_std")); n_no_gold += 1
            continue
        dbp = os.path.join(SQLITE_DIR, f"{db}.sqlite")
        if not os.path.exists(dbp):
            results.append(dict(id=iid, db=db, score=0, note="no_sqlite")); fail_reasons.append((iid, "no_sqlite")); continue
        ok, out = get_sqlite_result(dbp, sql)
        if not ok:
            results.append(dict(id=iid, db=db, score=0, note=f"exec_fail:{str(out)[:80]}"))
            n_exec_fail += 1; fail_reasons.append((iid, str(out)[:120])); continue
        gold_paths, is_single = resolve_gold_paths(iid, GOLD_EXEC)
        cc = std[iid].get("condition_cols"); ig = std[iid].get("ignore_order", False)
        if not gold_paths:
            results.append(dict(id=iid, db=db, score=0, note="no_gold_csv")); n_no_gold += 1; continue
        try:
            if is_single:
                sc = compare_pandas_table(out, pd.read_csv(gold_paths[0]), cc, ig)
            else:
                sc = compare_multi_pandas_table(out, [pd.read_csv(p) for p in gold_paths], cc, ig)
        except Exception as e:
            sc = 0; fail_reasons.append((iid, f"compare_err:{e}"))
        results.append(dict(id=iid, db=db, score=sc, note="ok"))
        n_correct += sc

    scored = [x for x in results if x["score"] is not None]
    ex = n_correct / len(scored) if scored else float("nan")
    print("=" * 70)
    print(f"G-S2-2 Sonnet cross-dataset EX (Spider2-lite local)")
    print("=" * 70)
    print(f"  predicted instances     : {len(preds)}")
    print(f"  scored (gold 존재)       : {len(scored)}")
    print(f"  no_gold (std/csv 없음)   : {n_no_gold}")
    print(f"  exec_fail (실행 불가)     : {n_exec_fail}")
    print(f"  correct (score==1)       : {n_correct}")
    print(f"  EX = correct/scored      : {ex:.4f}")
    print(f"  EX = correct/123(all pred): {n_correct/len(preds):.4f}")
    # per-db breakdown
    from collections import defaultdict
    bydb = defaultdict(lambda: [0, 0])
    for x in scored:
        bydb[x["db"]][0] += x["score"]; bydb[x["db"]][1] += 1
    print("\n  per-db (correct/total):")
    for db in sorted(bydb):
        c, t = bydb[db]
        print(f"    {db:28s} {c}/{t}")
    print(f"\n  exec_fail/compare 실패 사례 (상위 15):")
    for iid, why in fail_reasons[:15]:
        print(f"    {iid}: {why}")

    with open(os.path.join(OUT, "g_s2_2_sonnet_ex_score_2026-06-11.json"), "w") as f:
        json.dump(dict(n_pred=len(preds), n_scored=len(scored), n_no_gold=n_no_gold,
                       n_exec_fail=n_exec_fail, n_correct=n_correct, ex=ex,
                       per_db={db: bydb[db] for db in bydb}, results=results,
                       fail_reasons=fail_reasons), f, indent=2)
    print(f"\nWrote {OUT}/g_s2_2_sonnet_ex_score_2026-06-11.json")


if __name__ == "__main__":
    main()
