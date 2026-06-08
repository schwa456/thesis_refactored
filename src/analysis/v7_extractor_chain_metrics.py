#!/usr/bin/env python3
"""V7 Extractor chain (W0 baseline + W2 FKP + W3 STE) Extractor-stage R/P/F1 측정.

방법론 (참조 결과 재현 검증 완료):
  - R/P = per-query recall/precision 평균 (macro)
  - F1  = per-query F1 의 평균 (macro F1; R/P 로 계산한 F1 아님)
  - n_avg = per-query Extractor 출력 column 수 평균
  - Extractor stage = predictions.jsonl 의 extractor_info.extractor_selected_nodes
      (dict {table: [col,...]}) flatten → lowercase → FK arrow 'col->table.col' split('->')[0].strip() → dedupe
  - gold = output_v7_{cell}.jsonl 의 gold_cols (이미 lowercase), question_id join
  - Selector stage = selector_info.selected_nodes_top_k 의 'table.col' col part lowercase
  - NoneFilter (W2/W3) 위 Extractor = final, baseline(W0, BidirectionalFilter) 위 post-filter 별도.

산출:
  notebooks/analysis_results/ 리포트용 CSV/JSON + stdout 요약.
"""
import json, os, statistics, csv
from collections import OrderedDict

ROOT = "/home/hyeonjin/thesis_refactored"
V7DIR = os.path.join(ROOT, "outputs/experiments/abl/v7_extractor_redesign")
OUTDIR = os.path.join(ROOT, "outputs/analysis")
os.makedirs(OUTDIR, exist_ok=True)

# cell -> (wave, output_filename, has_filter)
CELLS = OrderedDict([
    # V7-W0 baseline (BidirectionalFilter active → extractor != final)
    ("baseline_seed42",  ("W0", "output_v7_w0_baseline_seed42.jsonl",  True)),
    ("baseline_seed123", ("W0", "output_v7_w0_baseline_seed123.jsonl", True)),
    ("baseline_seed7",   ("W0", "output_v7_w0_baseline_seed7.jsonl",   True)),
    # V7-W2 FKP (NoneFilter)
    ("fkp_k005", ("W2", "output_v7_fkp_k005.jsonl", False)),
    ("fkp_k010", ("W2", "output_v7_fkp_k010.jsonl", False)),
    ("fkp_k015", ("W2", "output_v7_fkp_k015.jsonl", False)),
    ("fkp_k020", ("W2", "output_v7_fkp_k020.jsonl", False)),
    ("fkp_k030", ("W2", "output_v7_fkp_k030.jsonl", False)),
    ("fkp_k050", ("W2", "output_v7_fkp_k050.jsonl", False)),
    ("fkp_k100", ("W2", "output_v7_fkp_k100.jsonl", False)),
    ("fkp_ax_coltbl", ("W2", "output_v7_fkp_ax_coltbl.jsonl", False)),
    ("fkp_ax_thr05",  ("W2", "output_v7_fkp_ax_thr05.jsonl",  False)),
    ("fkp_ax_nofk",   ("W2", "output_v7_fkp_ax_nofk.jsonl",   False)),
    # V7-W3 STE (NoneFilter)
    ("ste_k005", ("W3", "output_v7_ste_k005.jsonl", False)),
    ("ste_k010", ("W3", "output_v7_ste_k010.jsonl", False)),
    ("ste_k015", ("W3", "output_v7_ste_k015.jsonl", False)),
    ("ste_k020", ("W3", "output_v7_ste_k020.jsonl", False)),
    ("ste_k030", ("W3", "output_v7_ste_k030.jsonl", False)),
    ("ste_k050", ("W3", "output_v7_ste_k050.jsonl", False)),
    ("ste_k100", ("W3", "output_v7_ste_k100.jsonl", False)),
    ("ste_ax_thr05",   ("W3", "output_v7_ste_ax_thr05.jsonl",   False)),
    ("ste_ax_thr03",   ("W3", "output_v7_ste_ax_thr03.jsonl",   False)),
    ("ste_ax_colonly", ("W3", "output_v7_ste_ax_colonly.jsonl", False)),
    ("ste_ax_nocap",   ("W3", "output_v7_ste_ax_nocap.jsonl",   False)),
])

# M4 anchor reference (Extractor stage, c01_01 MSTPCSTUnion)
M4_ANCHOR = {"R": 0.9927, "P": 0.1267, "F1": 0.2073, "n_avg": 54.53}


def norm_col(c):
    """FK arrow 'col->table.col' → col part, lowercase strip."""
    c = str(c)
    if "->" in c:
        c = c.split("->")[0]
    return c.strip().lower()


def flatten_extractor_cols(selected):
    """extractor_selected_nodes dict {table: [cols]} → set of lowercase col names."""
    out = set()
    if isinstance(selected, dict):
        for tbl, cols in selected.items():
            if isinstance(cols, list):
                for c in cols:
                    cc = norm_col(c)
                    if cc:
                        out.add(cc)
    elif isinstance(selected, list):
        for c in selected:
            cc = norm_col(c)
            if cc:
                out.add(cc)
    return out


def selector_cols(top_k):
    """selected_nodes_top_k ['table.col' | 'table'] → col part set lowercase."""
    out = set()
    if not isinstance(top_k, list):
        return out
    for e in top_k:
        e = str(e)
        if "." in e:
            col = e.split(".", 1)[1].strip().lower()
            if col:
                out.add(col)
    return out


def rpf(pred, gold):
    if not gold:
        return None  # gold 없는 query 제외 (분모 0)
    inter = len(pred & gold)
    R = inter / len(gold)
    P = inter / len(pred) if pred else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return R, P, F


def measure_cell(cell, meta):
    wave, ofname, has_filter = meta
    cdir = os.path.join(V7DIR, cell)
    preds = {}
    with open(os.path.join(cdir, "predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            preds[r["question_id"]] = r
    # extractor + selector stage
    ext_R, ext_P, ext_F, ext_n = [], [], [], []
    sel_R, sel_P, sel_F = [], [], []
    post_R, post_P, post_F = [], [], []  # post-filter (final_nodes via output file)
    with open(os.path.join(cdir, ofname)) as f:
        for line in f:
            o = json.loads(line)
            qid = o["question_id"]
            gold = set(g.strip().lower() for g in o.get("gold_cols", []) if g.strip())
            pr = preds.get(qid)
            if pr is None or not gold:
                continue
            # extractor stage (recompute)
            ext_cols = flatten_extractor_cols(pr.get("extractor_info", {}).get("extractor_selected_nodes"))
            res = rpf(ext_cols, gold)
            if res:
                ext_R.append(res[0]); ext_P.append(res[1]); ext_F.append(res[2])
                ext_n.append(len(ext_cols))
            # selector stage
            scols = selector_cols(pr.get("selector_info", {}).get("selected_nodes_top_k"))
            sres = rpf(scols, gold)
            if sres:
                sel_R.append(sres[0]); sel_P.append(sres[1]); sel_F.append(sres[2])
            # post-filter (output file recall/precision)
            R = o.get("recall", 0.0); P = o.get("precision", 0.0)
            post_R.append(R); post_P.append(P)
            post_F.append(2*P*R/(P+R) if (P+R) > 0 else 0.0)

    def agg(xs):
        return statistics.mean(xs) if xs else 0.0
    out = {
        "cell": cell, "wave": wave, "has_filter": has_filter, "n_q": len(ext_R),
        "ext_R": agg(ext_R), "ext_P": agg(ext_P), "ext_F1": agg(ext_F), "ext_n_avg": agg(ext_n),
        "sel_R": agg(sel_R), "sel_P": agg(sel_P), "sel_F1": agg(sel_F),
        "post_R": agg(post_R), "post_P": agg(post_P), "post_F1": agg(post_F),
    }
    return out


def main():
    rows = []
    for cell, meta in CELLS.items():
        row = measure_cell(cell, meta)
        rows.append(row)
        print(f"{cell:18s} {row['wave']} | EXT R={row['ext_R']:.4f} P={row['ext_P']:.4f} "
              f"F1={row['ext_F1']:.4f} n={row['ext_n_avg']:.2f} | SEL R={row['sel_R']:.4f} "
              f"P={row['sel_P']:.4f} F1={row['sel_F1']:.4f}")

    # write CSV
    csvp = os.path.join(OUTDIR, "v7_extractor_chain_2026-06-05.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # write JSON
    jsonp = os.path.join(OUTDIR, "v7_extractor_chain_2026-06-05.json")
    with open(jsonp, "w") as f:
        json.dump({"m4_anchor": M4_ANCHOR, "cells": rows}, f, indent=2)
    print(f"\nWrote {csvp}\nWrote {jsonp}")


if __name__ == "__main__":
    main()
