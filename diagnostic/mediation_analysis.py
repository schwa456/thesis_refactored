"""§7.5 매개 분석 — "테이블 폭은 κ를 통해서만 실패에 작용한다"의 통제-없는 봉인.
R1(통제 없는 인과)·R2(3-DB 과소표본) 동시 방어. 재학습 불필요, kappa_diag.csv만.

판정 일관성: analyze_p1_p2의 mann_whitney를 import 재사용.
표준 라이브러리만(로지스틱은 IRLS 자체 구현, 부트스트랩 자체 구현).

가설(실행 전 고정):
  M1 (κ 조건부 폭 무효): κ_rel 사분위로 층화 후 각 빈의 폭 효과 δ_q ≈ 0 (예측 0/음수).
  M2 (폭 조건부 κ 생존): set_size 삼분위로 층화 후 각 층의 κ 효과 δ_t > 0, 전 층 동부호.
  M3 (매개 비율): 폭→실패 총효과 중 κ 매개분 ≈ 1 (폭 직접효과 ≈ 0).
                  로지스틱 c(폭) vs c'(폭|κ), 매개비율 = (c-c')/c.
  CI (R2 직격): clean P1·P2의 Cliff's δ에 DB클러스터 + 노드 부트스트랩.
                clean P1 CI가 0을 포함함을 정직 제시 + M1이 그 부재를 메커니즘으로 설명.

사용:
  python mediation_analysis.py kappa_diag.csv --names db_colnames.json [-o med.json]
  python mediation_analysis.py --selftest
"""
import argparse
import csv
import json
import math
import random

from analyze_p1_p2 import mann_whitney


# ---------- 공통 ----------
def fails_succ(rows):
    gold = [r for r in rows if int(float(r["is_gold"])) == 1]
    f = [r for r in gold if int(float(r["recalled"])) == 0]
    s = [r for r in gold if int(float(r["recalled"])) == 1]
    return gold, f, s


def quantile_edges(vals, k):
    v = sorted(vals)
    if not v:
        return []
    return [v[min(len(v) * i // k, len(v) - 1)] for i in range(1, k)]


# ---------- M1: κ 사분위 내 폭 효과 ----------
def m1_kappa_stratified_width(rows):
    gold, _, _ = fails_succ(rows)
    ks = [float(r["kappa_rel"]) for r in gold
          if r.get("kappa_rel") not in (None, "") and math.isfinite(float(r["kappa_rel"]))]
    if len(ks) < 40:
        return {"note": "표본부족", "n": len(ks)}
    e = quantile_edges(ks, 4)

    def qbin(x):
        x = float(x)
        for i, c in enumerate(e):
            if x <= c:
                return i
        return len(e)
    out = {}
    for q in range(4):
        sub = [r for r in gold
               if r.get("kappa_rel") not in (None, "") and math.isfinite(float(r["kappa_rel"]))
               and qbin(r["kappa_rel"]) == q]
        f = [float(r["set_size"]) for r in sub if int(float(r["recalled"])) == 0]
        s = [float(r["set_size"]) for r in sub if int(float(r["recalled"])) == 1]
        mw = mann_whitney(f, s)
        out[f"kappa_Q{q+1}"] = {"n_fail": len(f), "n_succ": len(s),
                                "width_delta": mw["cliffs_delta"], "p": mw["p_one_sided"]}
    deltas = [v["width_delta"] for v in out.values()
              if isinstance(v, dict) and v["width_delta"] == v["width_delta"]]
    out["summary"] = {
        "mean_within_width_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "max_abs": round(max((abs(d) for d in deltas), default=0), 4),
        "예측": "≈0 (폭은 κ 매개 → κ 고정 시 폭 잔여효과 소멸)",
    }
    return out


# ---------- M2: 폭 삼분위 내 κ 효과 ----------
def m2_width_stratified_kappa(rows):
    gold, _, _ = fails_succ(rows)
    ws = sorted(float(r["set_size"]) for r in gold)
    if len(ws) < 30:
        return {"note": "표본부족", "n": len(ws)}
    c1, c2 = ws[len(ws) // 3], ws[2 * len(ws) // 3]
    out = {}
    for name, pred in (("저폭", lambda w: w <= c1),
                       ("중폭", lambda w: c1 < w <= c2),
                       ("고폭", lambda w: w > c2)):
        sub = [r for r in gold if pred(float(r["set_size"]))]
        f = [r.get("kappa_rel") for r in sub if int(float(r["recalled"])) == 0]
        s = [r.get("kappa_rel") for r in sub if int(float(r["recalled"])) == 1]
        mw = mann_whitney(f, s)
        out[name] = {"n_fail": len([x for x in f if x not in (None, "")]),
                     "kappa_delta": mw["cliffs_delta"], "p": mw["p_one_sided"]}
    ds = [v["kappa_delta"] for v in out.values() if v["kappa_delta"] == v["kappa_delta"]]
    out["summary"] = {"all_positive": all(d > 0 for d in ds) if ds else None,
                      "min_delta": round(min(ds), 4) if ds else None,
                      "예측": ">0 전 층 동부호 (κ는 폭과 독립으로 작동)"}
    return out


# ---------- M3: 이진 매개비율 (IRLS 로지스틱) ----------
def _standardize(col):
    m = sum(col) / len(col)
    sd = (sum((x - m) ** 2 for x in col) / len(col)) ** 0.5 or 1.0
    return [(x - m) / sd for x in col], m, sd


def logistic_irls(X, y, iters=50, ridge=1e-6):
    """X: 행렬(열=특징, 절편 미포함), y: 0/1. 절편 자동 추가. 반환: 계수(절편 제외 dict 아님, 리스트)."""
    n = len(y)
    p = len(X[0])
    Xc = [[1.0] + list(row) for row in X]  # 절편
    d = p + 1
    beta = [0.0] * d
    for _ in range(iters):
        # eta, mu
        g = [0.0] * d
        H = [[0.0] * d for _ in range(d)]
        for i in range(n):
            eta = sum(beta[j] * Xc[i][j] for j in range(d))
            eta = max(min(eta, 30), -30)
            mu = 1.0 / (1.0 + math.exp(-eta))
            w = max(mu * (1 - mu), 1e-6)
            r = y[i] - mu
            for a in range(d):
                g[a] += r * Xc[i][a]
                for b in range(d):
                    H[a][b] += w * Xc[i][a] * Xc[i][b]
        for a in range(d):
            H[a][a] += ridge
        # solve H delta = g (가우스 소거)
        delta = _solve(H, g)
        if delta is None:
            break
        beta = [beta[j] + delta[j] for j in range(d)]
        if max(abs(x) for x in delta) < 1e-8:
            break
    return beta  # beta[0]=절편, beta[1:]=특징


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def m3_mediation_ratio(rows):
    gold, _, _ = fails_succ(rows)
    data = [(float(r["set_size"]), float(r["kappa_rel"]), 1 - int(float(r["recalled"])))
            for r in gold
            if r.get("kappa_rel") not in (None, "") and math.isfinite(float(r["kappa_rel"]))]
    if len(data) < 50:
        return {"note": "표본부족", "n": len(data)}
    W = [d[0] for d in data]
    K = [d[1] for d in data]
    Y = [d[2] for d in data]
    Wz, _, _ = _standardize(W)
    Kz, _, _ = _standardize(K)
    # total: Y ~ W
    c = logistic_irls([[w] for w in Wz], Y)[1]
    # direct: Y ~ W + K
    bd = logistic_irls([[w, k] for w, k in zip(Wz, Kz)], Y)
    cprime, ck = bd[1], bd[2]
    ratio = (c - cprime) / c if abs(c) > 1e-9 else float("nan")
    return {
        "total_effect_width(c)": round(c, 4),
        "direct_effect_width|kappa(c')": round(cprime, 4),
        "kappa_coef(c_k)": round(ck, 4),
        "mediation_ratio_(c-c')/c": round(ratio, 4),
        "예측": "≈1 (폭 직접효과 c'≈0; 폭→κ→실패)",
    }


# ---------- 부트스트랩 CI (R2) ----------
def _delta_p1(rows):
    _, f, s = fails_succ(rows)
    return mann_whitney([r["set_size"] for r in f], [r["set_size"] for r in s])["cliffs_delta"]


def _delta_p2(rows):
    _, f, s = fails_succ(rows)
    return mann_whitney([r.get("kappa_rel") for r in f], [r.get("kappa_rel") for r in s])["cliffs_delta"]


def _db_of(r, dbcol):
    return str(r[dbcol]) if dbcol else str(r["set_id"]).split(".")[0]


def bootstrap_ci(rows, dbcol, B=2000, seed=0):
    rng = random.Random(seed)
    # 노드 단위
    def node_boot(metric):
        ds = []
        n = len(rows)
        for _ in range(B):
            samp = [rows[rng.randrange(n)] for _ in range(n)]
            try:
                d = metric(samp)
                if d == d:
                    ds.append(d)
            except Exception:
                pass
        ds.sort()
        return ds
    # DB 클러스터 단위
    dbs = {}
    for r in rows:
        dbs.setdefault(_db_of(r, dbcol), []).append(r)
    db_keys = list(dbs)
    def db_boot(metric):
        ds = []
        for _ in range(B):
            samp = []
            for _ in range(len(db_keys)):
                samp += dbs[db_keys[rng.randrange(len(db_keys))]]
            try:
                d = metric(samp)
                if d == d:
                    ds.append(d)
            except Exception:
                pass
        ds.sort()
        return ds

    def ci(ds):
        if not ds:
            return None
        lo = ds[int(0.025 * len(ds))]
        hi = ds[min(int(0.975 * len(ds)), len(ds) - 1)]
        return {"lo": round(lo, 4), "hi": round(hi, 4),
                "includes_0": lo <= 0 <= hi, "median": round(ds[len(ds) // 2], 4)}
    return {
        "n_dbs": len(db_keys),
        "P1_width": {"node_CI": ci(node_boot(_delta_p1)), "db_cluster_CI": ci(db_boot(_delta_p1))},
        "P2_kappa": {"node_CI": ci(node_boot(_delta_p2)), "db_cluster_CI": ci(db_boot(_delta_p2))},
    }


# ---------- clean 필터 (robust_filter 재사용 로직) ----------
def is_dirty_name(s):
    s = str(s)
    return (s != s.lower()) or (" " in s)


def clean_rows(rows, names_path, dbcol):
    if not names_path:
        return rows, []  # 필터 없이 전체
    meta = json.load(open(names_path))
    dirty = {db for db, cols in meta.items() if any(is_dirty_name(c) for c in cols)}
    cl = [r for r in rows if _db_of(r, dbcol) not in dirty]
    return cl, sorted(dirty)


# ---------- 실행 ----------
def run(rows, names_path=None, dbcol=None, B=2000):
    cl, dirty = clean_rows(rows, names_path, dbcol)
    target = cl if names_path else rows
    return {
        "scope": "clean_only" if names_path else "all_rows",
        "dirty_dbs": dirty,
        "n_target_rows": len(target),
        "M1_kappa_stratified_width(R1: 폭 매개 검정)": m1_kappa_stratified_width(target),
        "M2_width_stratified_kappa(R1: κ 독립 생존)": m2_width_stratified_kappa(target),
        "M3_binary_mediation(R1: 매개비율)": m3_mediation_ratio(target),
        "bootstrap_CI(R2: clean P1 0 포함 정직제시)": bootstrap_ci(target, dbcol, B=B),
    }


# ---------- 자체시험 ----------
def _make_synth(direct_width, seed=0):
    """폭→κ→실패 매개 구조. direct_width=0이면 순수매개, >0이면 폭 직접효과 주입."""
    rng = random.Random(seed)
    rows = []
    for db in ["a", "b", "Dirty C", "messy_D"]:
        for i in range(700):
            w = rng.choice([3, 5, 8, 12, 20])
            # κ는 폭에 단조 의존 (폭→κ)
            k = 1.0 + 0.06 * w + rng.gauss(0, 0.4)
            # 실패확률은 κ를 통해서만 (+ direct_width 만큼만 폭 직접)
            logit = -3.0 + 1.4 * (k - 1.6) + direct_width * (w - 8) / 10.0
            pf = 1 / (1 + math.exp(-logit))
            rows.append(dict(node_id=f"{db}:{i}", set_id=f"{db}.t{w}", db_id=db,
                             col_name=f"col{i}", set_size=w, kappa_rel=round(k, 4),
                             is_gold=1, recalled=0 if rng.random() < pf else 1))
    return rows


def _selftest():
    for label, dw in [("순수매개(direct=0)", 0.0), ("직접효과주입(direct=2)", 2.0)]:
        out = run(_make_synth(dw), names_path=None, dbcol="db_id", B=400)
        m1 = out["M1_kappa_stratified_width(R1: 폭 매개 검정)"]["summary"]
        m2 = out["M2_width_stratified_kappa(R1: κ 독립 생존)"]["summary"]
        m3 = out["M3_binary_mediation(R1: 매개비율)"]
        ratio = m3["mediation_ratio_(c-c')/c"]
        cc = m3["total_effect_width(c)"]
        cp = m3["direct_effect_width|kappa(c')"]
        print(f"\n[{label}]")
        print(f"  M1 평균 폭효과(κ층화 후): {m1['mean_within_width_delta']}  (순수매개면 ≈0)")
        print(f"  M2 최소 κ효과(폭층화 후): {m2['min_delta']}  all_positive={m2['all_positive']}")
        print(f"  M3 매개비율: {ratio}  (c={cc}, c'={cp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile", nargs="?")
    ap.add_argument("--names")
    ap.add_argument("--db-col")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("-o", "--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.csvfile:
        print(__doc__)
        return
    rows = list(csv.DictReader(open(a.csvfile, newline="")))
    dbcol = a.db_col or ("db_id" if "db_id" in rows[0] else None)
    out = run(rows, names_path=a.names, dbcol=dbcol, B=a.B)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()