"""§7.3 P1·P2 자동 판정 — kappa_diag.csv 분석 (표준 라이브러리만 사용, scipy 불필요)

입력 CSV 필수 열: node_id, set_id, set_size, kappa_rel, is_gold(0/1), recalled(0/1)
  (integrate_kappa_spider.py가 생성; kappa_abs/kappa_pred 등 추가 열 허용)

사전등록 판정(골격 §6.3):
  P1: 실패(gold & ~recalled) 컬럼의 테이블 폭(set_size) > 성공 컬럼  [+방향, 단측]
  P2: 실패 컬럼의 kappa_rel > 성공 컬럼, 그리고 kappa_rel 사분위 실패율 단조

사용:
  python analyze_p1_p2.py kappa_diag.csv [결과.json]
  python analyze_p1_p2.py --selftest
"""
import csv
import json
import math
import sys
from collections import Counter


def _floats(xs):
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v:  # NaN 제외
            out.append(v)
    return out


def mann_whitney(a, b):
    """단측(a > b 예측) Mann–Whitney U: midrank, 타이 보정 정규근사 + Cliff's delta."""
    a, b = _floats(a), _floats(b)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return dict(n1=n1, n2=n2, p_one_sided=float("nan"), cliffs_delta=float("nan"))
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    vals = [v for v, _ in allv]
    ranks = [0.0] * (n1 + n2)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    R1 = sum(r for r, (_, grp) in zip(ranks, allv) if grp == 0)
    U = R1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    ties = sum(t ** 3 - t for t in Counter(vals).values())
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    z = (U - n1 * n2 / 2) / math.sqrt(var) if var > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return dict(n1=n1, n2=n2, U=round(U, 1), z=round(z, 3),
                p_one_sided=round(p, 6), cliffs_delta=round(2 * U / (n1 * n2) - 1, 3))


def quartile_failrates(vals, fails):
    """정렬된 kappa 사분위별 실패율과 단조 역전 수."""
    pairs = sorted((v, f) for v, f in zip(vals, fails) if v == v)
    n = len(pairs)
    if n < 16:
        return None
    qs = []
    for qi in range(4):
        seg = pairs[qi * n // 4:(qi + 1) * n // 4]
        qs.append(dict(q=qi + 1, n=len(seg),
                       fail_rate=round(sum(f for _, f in seg) / len(seg), 4)))
    rates = [q["fail_rate"] for q in qs]
    inv = sum(1 for i in range(3) if rates[i + 1] < rates[i] - 1e-12)
    return dict(quartiles=qs, monotone_inversions=inv)


def analyze(rows):
    gold = [r for r in rows if int(float(r["is_gold"])) == 1]
    fail = [r for r in gold if int(float(r["recalled"])) == 0]
    succ = [r for r in gold if int(float(r["recalled"])) == 1]
    out = {"n_gold": len(gold), "n_fail": len(fail), "n_succ": len(succ)}

    # ---- P1: 테이블 폭 ----
    p1 = mann_whitney([r["set_size"] for r in fail], [r["set_size"] for r in succ])
    out["P1_table_width"] = p1
    if p1["p_one_sided"] != p1["p_one_sided"]:
        out["P1_verdict"] = "판정불가(표본부족)"
    elif p1["cliffs_delta"] <= 0:
        out["P1_verdict"] = "방향불일치"
    else:
        out["P1_verdict"] = "지지" if p1["p_one_sided"] < 0.05 else "판정불가(p>=.05)"

    # ---- P2: kappa_rel ----
    p2 = mann_whitney([r.get("kappa_rel") for r in fail], [r.get("kappa_rel") for r in succ])
    out["P2_kappa_rel"] = p2
    kv, kf = [], []
    for r in gold:
        try:
            v = float(r.get("kappa_rel"))
        except (TypeError, ValueError):
            continue
        if v == v:
            kv.append(v)
            kf.append(1 - int(float(r["recalled"])))
    qa = quartile_failrates(kv, kf)
    out["P2_quartiles"] = qa
    mono = qa is not None and qa["monotone_inversions"] <= 1
    if p2["p_one_sided"] != p2["p_one_sided"]:
        out["P2_verdict"] = "판정불가(표본부족)"
    elif p2["cliffs_delta"] <= 0:
        out["P2_verdict"] = "방향불일치"
    elif p2["p_one_sided"] < 0.05 and mono:
        out["P2_verdict"] = "지지"
    else:
        out["P2_verdict"] = "부분지지/판정불가"

    # ---- 교란 점검: 테이블 폭 3분위 내 P2 방향 ----
    ws = sorted(float(r["set_size"]) for r in gold)
    if len(ws) >= 30:
        c1, c2 = ws[len(ws) // 3], ws[2 * len(ws) // 3]
        strata = {}
        for name, pred in (("저폭", lambda w: w <= c1),
                           ("중폭", lambda w: c1 < w <= c2),
                           ("고폭", lambda w: w > c2)):
            f = [r.get("kappa_rel") for r in fail if pred(float(r["set_size"]))]
            s = [r.get("kappa_rel") for r in succ if pred(float(r["set_size"]))]
            strata[name] = mann_whitney(f, s).get("cliffs_delta")
        out["P2_width_strata_delta(교란점검)"] = strata
    return out


def _selftest():
    import random
    random.seed(0)

    def make(planted):
        rows = []
        for i in range(900):
            w = random.choice([3, 5, 8, 12, 20])
            k = random.gauss(1.0 + 0.08 * w, 0.5) if planted else random.gauss(1.6, 0.5)
            pf = 0.08 + (0.22 * (k - 1.6) + 0.015 * (w - 8) if planted else 0.0)
            pf = min(max(pf, 0.02), 0.9)
            rec = 0 if random.random() < pf else 1
            rows.append(dict(node_id=i, set_id=f"t{w}", set_size=w,
                             kappa_rel=round(k, 4), is_gold=1, recalled=rec))
        return rows

    for tag, planted in (("효과 심은 데이터", True), ("귀무 데이터", False)):
        r = analyze(make(planted))
        brief = {k: v for k, v in r.items() if "verdict" in k or k.startswith("n_")}
        print(f"[{tag}]", json.dumps(brief, ensure_ascii=False))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        _selftest()
        return
    if len(sys.argv) < 2:
        print(__doc__)
        return
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    out = analyze(rows)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()