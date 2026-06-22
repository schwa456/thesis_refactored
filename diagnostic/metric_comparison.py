"""§7.2 측정 지표 비교 — κ_rel vs Dirichlet(정규화/비정규화) vs MAD 실패예측력 (R5 방어).
재학습 불필요. 동일 BIRD CSV·동일 노드집합·동일 판정 논리(mann_whitney 재사용).

전제: CSV에 비교 지표 열이 있어야 함. 표준 kappa_diag.csv는 kappa_rel만 보유 →
  integrate/kappa_hook 단계에서 다음 열을 추가 산출해야 비교 가능(없으면 이 스크립트가 안내):
    - dirichlet_norm : 노드별 정규화 Dirichlet 에너지(층 평균 또는 최종층)
    - dirichlet_unnorm : 비정규화 Dirichlet
    - mad : 노드별 MAD(이웃 평균 코사인 거리)
  (열 이름은 --cols로 매핑 가능)

지표 방향 규약: 모두 "값이 클수록 붕괴 큼 = 실패 가능성↑" 방향으로 통일(MAD는 거리이므로
  작을수록 붕괴 → 비교 시 부호 반전 옵션 --invert mad).

사용:
  python metric_comparison.py kappa_diag.csv [--names db_colnames.json]
  python metric_comparison.py --selftest
"""
import argparse
import csv
import json
import math
import random

from analyze_p1_p2 import mann_whitney, quartile_failrates

DEFAULT_METRICS = {
    "kappa_rel": "kappa_rel",
    "dirichlet_norm": "dirichlet_norm",
    "dirichlet_unnorm": "dirichlet_unnorm",
    "mad": "mad",
}


def auc_from_mw(fail_vals, succ_vals):
    """실패=양성. AUC = P(지표_fail > 지표_succ). mann_whitney의 U로부터 직접."""
    mw = mann_whitney(fail_vals, succ_vals)
    d = mw["cliffs_delta"]
    if d != d:
        return float("nan"), mw
    return (d + 1) / 2.0, mw  # AUC = (δ+1)/2


def evaluate_metric(rows, col, invert=False):
    gold = [r for r in rows if int(float(r["is_gold"])) == 1]
    def val(r):
        x = r.get(col)
        if x in (None, ""):
            return None
        x = float(x)
        if not math.isfinite(x):
            return None
        return -x if invert else x
    fail = [val(r) for r in gold if int(float(r["recalled"])) == 0]
    succ = [val(r) for r in gold if int(float(r["recalled"])) == 1]
    fail = [x for x in fail if x is not None]
    succ = [x for x in succ if x is not None]
    if len(fail) < 10 or len(succ) < 10:
        return {"note": "표본부족", "n_fail": len(fail), "n_succ": len(succ)}
    auc, mw = auc_from_mw(fail, succ)
    # 사분위 단조성
    allv, fl = [], []
    for r in gold:
        v = val(r)
        if v is not None:
            allv.append(v)
            fl.append(1 - int(float(r["recalled"])))
    q = quartile_failrates(allv, fl)
    return {"AUC": round(auc, 4), "cliffs_delta": mw["cliffs_delta"],
            "p_one_sided": mw["p_one_sided"],
            "quartile_failrates": q.get("rates") if isinstance(q, dict) else q,
            "inversions": q.get("inversions") if isinstance(q, dict) else None}


def run(rows, cols=None, invert=("mad",)):
    cols = cols or DEFAULT_METRICS
    present = {name: c for name, c in cols.items() if c in rows[0]}
    missing = {name: c for name, c in cols.items() if c not in rows[0]}
    out = {"present_metrics": list(present), "missing_columns": missing}
    if missing:
        out["note"] = ("비교 지표 열 누락 — integrate/kappa_hook에서 산출 필요: "
                       + ", ".join(missing.values())
                       + ". kappa_rel만으로는 단독 평가만 가능.")
    res = {}
    for name, c in present.items():
        res[name] = evaluate_metric(rows, c, invert=(name in invert))
    out["metrics"] = res
    # 순위(예측력 AUC 기준)
    ranked = sorted(((n, m.get("AUC")) for n, m in res.items() if isinstance(m, dict) and "AUC" in m),
                    key=lambda x: -(x[1] or 0))
    out["ranking_by_AUC"] = ranked
    if ranked:
        top = ranked[0][0]
        out["headline"] = (f"최고 예측력: {top} (AUC={ranked[0][1]})"
                           + ("" if top == "kappa_rel" else
                              " — κ_rel이 1위 아님: C3을 '닫힌형·차수불변·처방연결' 우위로 재정의 검토"))
    return out


def _selftest():
    """κ가 실패를 더 잘 예측하고, Dirichlet은 잎-허브 갭만 재 노이즈, MAD는 약하게."""
    rng = random.Random(0)
    rows = []
    for i in range(1500):
        w = rng.choice([3, 5, 8, 12, 20])
        k = 1.0 + 0.06 * w + rng.gauss(0, 0.4)
        fail = 1 if rng.random() < 1 / (1 + math.exp(-(-3 + 1.4 * (k - 1.6)))) else 0
        rows.append(dict(
            node_id=str(i), set_id=f"db.t{w}", db_id="db", set_size=w, is_gold=1,
            recalled=1 - fail, kappa_rel=round(k, 4),
            dirichlet_norm=round(k * 0.3 + rng.gauss(0, 0.8), 4),       # κ와 약상관 + 노이즈
            dirichlet_unnorm=round(k * 0.5 + rng.gauss(0, 0.6), 4),
            mad=round(2.0 - 0.2 * (k - 1.6) + rng.gauss(0, 0.5), 4),    # 붕괴 클수록 작아짐 → invert
        ))
    out = run(rows)
    print(json.dumps({"ranking": out["ranking_by_AUC"],
                      "kappa_AUC": out["metrics"]["kappa_rel"]["AUC"],
                      "dir_norm_AUC": out["metrics"]["dirichlet_norm"]["AUC"],
                      "mad_AUC": out["metrics"]["mad"]["AUC"],
                      "headline": out["headline"]}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile", nargs="?")
    ap.add_argument("--names")  # 호환용(미사용)
    ap.add_argument("--cols")   # JSON: {"mad":"my_mad_col",...}
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
    cols = dict(DEFAULT_METRICS)
    if a.cols:
        cols.update(json.load(open(a.cols)))
    out = run(rows, cols=cols)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()