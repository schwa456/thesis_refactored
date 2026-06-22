#!/usr/bin/env python3
"""G-S2-1 Spider2.0-Lite cross-dataset generalization R/P/F1 + degradation (DECISIONS 2026-06-10).

table-level (main, gold-tables): raw + shard-collapse 두 버전
column-level (secondary, sqlglot partial): column-name-level (alias 해석 불가 → bare col)
stratification: per backend + per scale(n_total_columns bucket)
M4 anchor (R 0.8438/P 0.8329/F1 0.8383) 대비 Δ degradation
caveat: zero-shot (external_knowledge 미사용), over-extract θ=0.1 → P 붕괴 예상
"""
import json, os, re, ast, statistics
from collections import defaultdict, Counter

ROOT = "/home/hyeonjin/thesis_refactored"
PRED = os.path.join(ROOT, "outputs/experiments/g_s2_1_spider2/predicted_schema.jsonl")
GOLD_T = os.path.join(ROOT, "data/Spider2/methods/gold-tables/spider2-lite-gold-tables.jsonl")
GOLD_SQL = os.path.join(ROOT, "data/Spider2/spider2-lite/evaluation_suite/gold/sql")
OUT = os.path.join(ROOT, "outputs/analysis")
M4 = dict(R=0.8438, P=0.8329, F1=0.8383)

SHARD_RE = re.compile(r"^(.*?)_(\d{6,8})$")  # events_20210101 → events_*


def last_comp(t):
    """full-path table → bare last component, lowercase."""
    return t.split(".")[-1].strip().lower()


def collapse_shard(t):
    """events_20210101 → events_*  (date/number-suffixed shard collapse)."""
    m = SHARD_RE.match(t)
    return f"{m.group(1)}_*" if m else t


def prf(pred, gold):
    if not gold:
        return None
    inter = len(pred & gold)
    R = inter / len(gold)
    P = inter / len(pred) if pred else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return R, P, F


def load():
    pred = {}
    for l in open(PRED):
        r = json.loads(l); pred[r["instance_id"]] = r
    gold = {}
    for l in open(GOLD_T):
        r = json.loads(l); gt = r["gold_tables"]
        if isinstance(gt, str):
            gt = ast.literal_eval(gt)
        gold[r["instance_id"]] = gt
    return pred, gold


def scale_bucket(ncol):
    if ncol < 100:
        return "small(<100C)"
    if ncol < 500:
        return "medium(100-500)"
    if ncol < 1500:
        return "large(500-1500)"
    return "enterprise(>=1500)"


def agg(rows, keyset):
    """rows: list of (R,P,F). macro mean."""
    if not rows:
        return None
    return dict(R=statistics.mean(r[0] for r in rows), P=statistics.mean(r[1] for r in rows),
                F1=statistics.mean(r[2] for r in rows), n=len(rows))


def col_from_sql(sql_path, dialect):
    """sqlglot best-effort: gold SQL → column 이름 set (bare col, alias 해석 불가)."""
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(open(sql_path).read(), dialect=dialect)
        cols = set()
        for c in tree.find_all(exp.Column):
            if c.name and c.name != "*":
                cols.add(c.name.strip().lower())
        return cols, True
    except Exception:
        return set(), False


def main():
    pred, gold = load()
    ids = sorted(set(pred) & set(gold))
    print(f"matched instances: {len(ids)} (pred {len(pred)} / gold {len(gold)})")

    # ── table-level: raw + shard-collapse ──
    raw_rows, col_rows = [], []  # col=shard-collapse
    by_backend = defaultdict(lambda: dict(raw=[], shard=[]))
    by_scale = defaultdict(lambda: dict(raw=[], shard=[]))
    n_sharded_instances = 0
    per_inst = {}
    for iid in ids:
        p = pred[iid]
        backend = p["backend"]; ncol = p.get("n_total_columns", 0)
        bucket = scale_bucket(ncol)
        gt = set(last_comp(t) for t in gold[iid])
        pt = set(t.strip().lower() for t in p.get("predicted_tables", []))
        # shard-collapse
        gt_s = set(collapse_shard(t) for t in gt)
        pt_s = set(collapse_shard(t) for t in pt)
        if gt != gt_s or pt != pt_s:
            n_sharded_instances += 1
        rraw = prf(pt, gt); rshd = prf(pt_s, gt_s)
        if rraw:
            raw_rows.append(rraw); by_backend[backend]["raw"].append(rraw); by_scale[bucket]["raw"].append(rraw)
        if rshd:
            col_rows.append(rshd); by_backend[backend]["shard"].append(rshd); by_scale[bucket]["shard"].append(rshd)
        per_inst[iid] = dict(backend=backend, ncol=ncol, bucket=bucket,
                             raw=rraw, shard=rshd, n_pred_t=len(pt), n_gold_t=len(gt))

    print("\n=== TABLE-LEVEL R/P/F1 (gold-tables, macro) ===")
    for name, rows in [("raw (bare)", raw_rows), ("shard-collapse", col_rows)]:
        a = agg(rows, None)
        print(f"  {name:16s}: R={a['R']:.4f} P={a['P']:.4f} F1={a['F1']:.4f} (n={a['n']})  "
              f"ΔF1 vs M4={a['F1']-M4['F1']:+.4f}")
    print(f"  (shard 영향 instance: {n_sharded_instances})")

    print("\n=== per BACKEND (shard-collapse) ===")
    for be in ["bigquery", "snowflake", "sqlite"]:
        a = agg(by_backend[be]["shard"], None); ar = agg(by_backend[be]["raw"], None)
        print(f"  {be:10s}: raw R={ar['R']:.4f} P={ar['P']:.4f} F1={ar['F1']:.4f} | "
              f"shard R={a['R']:.4f} P={a['P']:.4f} F1={a['F1']:.4f} (n={a['n']})")

    print("\n=== per SCALE bucket (shard-collapse, n_total_columns) ===")
    for bk in ["small(<100C)", "medium(100-500)", "large(500-1500)", "enterprise(>=1500)"]:
        rows = by_scale[bk]["shard"]
        if rows:
            a = agg(rows, None)
            print(f"  {bk:18s}: R={a['R']:.4f} P={a['P']:.4f} F1={a['F1']:.4f} (n={a['n']})")

    # ── column-level (secondary, sqlglot best-effort) ──
    DIALECT = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}
    col_rows2 = []; n_parse_ok = n_parse_fail = n_no_sql = 0
    by_be_col = defaultdict(list)
    for iid in ids:
        p = pred[iid]
        sqlp = os.path.join(GOLD_SQL, f"{iid}.sql")
        if not os.path.exists(sqlp):
            n_no_sql += 1; continue
        gcols, ok = col_from_sql(sqlp, DIALECT.get(p["backend"], None))
        if not ok or not gcols:
            n_parse_fail += 1; continue
        n_parse_ok += 1
        pcols = set(c.split(".", 1)[1].strip().lower() if "." in c else c.strip().lower()
                    for c in p.get("predicted_columns", []))
        res = prf(pcols, gcols)
        if res:
            col_rows2.append(res); by_be_col[p["backend"]].append(res)
    print("\n=== COLUMN-LEVEL R/P/F1 (sqlglot best-effort, column-name level) ===")
    a = agg(col_rows2, None)
    if a:
        print(f"  overall: R={a['R']:.4f} P={a['P']:.4f} F1={a['F1']:.4f} (n={a['n']})")
    cov = n_parse_ok / max(len(ids), 1)
    print(f"  coverage: parse_ok={n_parse_ok} / parse_fail={n_parse_fail} / no_sql={n_no_sql} "
          f"({100*cov:.1f}% of {len(ids)} instances)")
    for be in ["bigquery", "snowflake", "sqlite"]:
        if by_be_col[be]:
            a = agg(by_be_col[be], None)
            print(f"    {be:10s}: R={a['R']:.4f} P={a['P']:.4f} F1={a['F1']:.4f} (n={a['n']})")

    # save
    out = dict(m4=M4, n_matched=len(ids), n_sharded=n_sharded_instances,
               table_raw=agg(raw_rows, None), table_shard=agg(col_rows, None),
               by_backend={be: dict(raw=agg(by_backend[be]["raw"], None), shard=agg(by_backend[be]["shard"], None))
                           for be in ["bigquery", "snowflake", "sqlite"]},
               by_scale={bk: agg(by_scale[bk]["shard"], None) for bk in by_scale},
               column=dict(overall=agg(col_rows2, None), coverage=cov, parse_ok=n_parse_ok,
                           parse_fail=n_parse_fail, no_sql=n_no_sql,
                           by_backend={be: agg(by_be_col[be], None) for be in by_be_col}))
    with open(os.path.join(OUT, "g_s2_1_spider2_generalization_2026-06-10.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/g_s2_1_spider2_generalization_2026-06-10.json")


if __name__ == "__main__":
    main()
