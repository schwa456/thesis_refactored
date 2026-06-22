"""§7.5 2차 — 단계 0: case/공백 명명 DB 제외 강건성 재판정 (재학습 불필요)

kappa_diag.csv를 DB 단위 clean/dirty로 나눠 analyze_p1_p2.analyze를 부분집합별 재실행.
dirty(DB) := 그 DB 컬럼명 중 하나라도 (공백 포함) 또는 (원본 != 소문자화) 이면 1.  [정의 고정]
판정 기준은 1차와 동일(analyze_p1_p2를 그대로 import — 변경 금지).

DB·컬럼명 소스(우선순위):
 1) --names db_colnames.json : {db_id: [컬럼명, ...]}   <- 권장(지시서 §0.1 덤프 스니펫)
 2) CSV에 이름 열이 있으면 자동탐지(col_name/col/column/name/node_name)
 3) --dirty dirty_dbs.txt    : 수동 dirty DB 목록(한 줄 1개)
DB 열 자동탐지: db_id/db/database/dataset; 없으면 --db-from-set-id (set_id의 '.' 앞부분).

사용:
  python robust_filter_rejudge.py kappa_diag.csv --names db_colnames.json -o p1p2_robust.json
  python robust_filter_rejudge.py --selftest
"""
import argparse
import csv
import json
from analyze_p1_p2 import analyze

DB_CANDS = ["db_id", "db", "database", "dataset"]
NAME_CANDS = ["col_name", "col", "column", "name", "node_name", "colname"]


def is_dirty_name(s):
    s = str(s)
    return (s != s.lower()) or (" " in s)


def detect(fields, cands):
    for c in cands:
        if c in fields:
            return c
    return None


def split_rows(rows, args):
    fields = list(rows[0].keys())
    dbcol = args.db_col or detect(fields, DB_CANDS)

    def db_of(r):
        if dbcol:
            return str(r[dbcol])
        if args.db_from_set_id:
            return str(r["set_id"]).split(args.sep)[0]
        raise SystemExit(f"DB 열 미발견. 보유 열: {fields} — --db-col 또는 --db-from-set-id 지정")

    if args.names:
        meta = json.load(open(args.names))
        dirty = {db for db, cols in meta.items() if any(is_dirty_name(c) for c in cols)}
    elif args.dirty:
        dirty = {l.strip() for l in open(args.dirty) if l.strip()}
    else:
        namecol = args.name_col or detect(fields, NAME_CANDS)
        if not namecol:
            raise SystemExit("컬럼명 소스 없음: --names(권장) / --dirty / CSV 이름 열 중 하나 필요")
        dirty = set()
        for r in rows:
            if is_dirty_name(r[namecol]):
                dirty.add(db_of(r))
    clean = [r for r in rows if db_of(r) not in dirty]
    dirt = [r for r in rows if db_of(r) in dirty]
    return clean, dirt, sorted(dirty)


def run(rows, args):
    clean, dirt, dlist = split_rows(rows, args)
    out = {
        "n_rows": len(rows), "n_clean_rows": len(clean), "n_dirty_rows": len(dirt),
        "dirty_dbs": dlist,
        "전체(1차 재현)": analyze(rows),
        "clean_only(R1·R2 판정)": analyze(clean) if clean else "행 없음",
        "dirty_only(참고)": analyze(dirt) if dirt else "행 없음",
    }
    co = out["clean_only(R1·R2 판정)"]
    if isinstance(co, dict):
        out["R1(clean P1)"] = co.get("P1_verdict")
        out["R2(clean P2)"] = co.get("P2_verdict")
    return out


def _selftest():
    """진짜 κ 효과는 전 DB 공통으로 심고, dirty DB에만 '버그성' 폭-상관 실패를 추가.
    기대: 전체 P1 δ > clean P1 δ (버그 인플레 분리), P2 δ는 양쪽 유사, R1·R2 지지."""
    import random
    random.seed(7)
    rows = []
    for db, bug in [("clean_a", 0), ("clean_b", 0), ("Dirty C", 1), ("messy_D", 1)]:
        for i in range(600):
            w = random.choice([3, 5, 8, 12, 20])
            k = random.gauss(1.0 + 0.05 * w, 0.5)
            pf = 0.10 + 0.20 * (k - 1.6)
            if bug:
                pf += 0.02 * (w - 3)
            pf = min(max(pf, 0.02), 0.92)
            rows.append(dict(node_id=f"{db}:{i}", db_id=db, set_id=f"{db}.t{w}",
                             col_name=("County Name" if bug and i % 3 == 0 else f"col{i}"),
                             set_size=w, kappa_rel=round(k, 4), is_gold=1,
                             recalled=0 if random.random() < pf else 1))
    args = argparse.Namespace(db_col=None, db_from_set_id=False, sep=".",
                              names=None, dirty=None, name_col=None)
    out = run(rows, args)
    brief = {
        "dirty_dbs": out["dirty_dbs"],
        "전체 P1 δ": out["전체(1차 재현)"]["P1_table_width"]["cliffs_delta"],
        "clean P1 δ": out["clean_only(R1·R2 판정)"]["P1_table_width"]["cliffs_delta"],
        "전체 P2 δ": out["전체(1차 재현)"]["P2_kappa_rel"]["cliffs_delta"],
        "clean P2 δ": out["clean_only(R1·R2 판정)"]["P2_kappa_rel"]["cliffs_delta"],
        "R1": out["R1(clean P1)"], "R2": out["R2(clean P2)"],
    }
    print(json.dumps(brief, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile", nargs="?")
    ap.add_argument("--names")
    ap.add_argument("--dirty")
    ap.add_argument("--db-col")
    ap.add_argument("--name-col")
    ap.add_argument("--db-from-set-id", action="store_true")
    ap.add_argument("--sep", default=".")
    ap.add_argument("-o", "--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.csvfile:
        print(__doc__)
        return
    rows = list(csv.DictReader(open(args.csvfile, newline="")))
    out = run(rows, args)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()