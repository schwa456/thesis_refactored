"""Wave 16 — Encoder Backbone Stage-wise R/P/F1/EX 분해 (post-measurement analyzer).

DECISIONS 2026-05-21 (Wave 16) §7.2 / §7.5 정합. Selector only → +Extractor (no filter) →
+Filter (final) 의 cumulative R/P/F1 + EX 분해. 비교 base = Wave 6 P2 M4 anchor (all-MiniLM).

Spec (Wave 13 patch f67fa65 evaluator + Wave 10 Phase B Spec A col-only 통일):
  - Gold:   parse_sql_elements(gold_sql) (alias-aware), col-only lowercase
  - Pred:   main.py:101-125 path (FK arrow excluded, table.col 또는 col lowercase)
  - R/P:    calculate_schema_metrics
  - F1:     F1_harm = 2·R·P/(R+P) (overall R, P 위 harmonic)
  - EX:     Wave 16 predictions 의 ex_score_selector_only / ex_score_extractor_only / ex_score
            Wave 6 anchor 는 final EX 만 (stage-별 EX 로깅 미존재)

Stage-wise node 추출:
  - Selector only:    selector_info.selected_nodes_top_k (Wave 16) 또는 score_analysis top-20 (Wave 6)
  - +Extractor:       extractor_info.extractor_selected_nodes dict {table: [cols]} (Wave 16 only)
                      Wave 6 는 동일 pipeline 의 Wave 15 no_filter cell 위 proxy 사용
  - +Filter (final):  predictions.jsonl 의 final_nodes

산출:
  - notebooks/analysis_results/wave16_encoder_backbone_m4_2026-05-22.md (§1~§9)
  - outputs/analysis/wave16_encoder_backbone_stagewise_2026-05-22.csv
  - outputs/analysis/wave16_encoder_backbone_stagewise_2026-05-22.json
  - outputs/experiments/abl/wave16_encoder_backbone/m16_qwen3_0.6b_m4/output_stagewise_rpf1.jsonl
  - outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional/output_stagewise_rpf1.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path("/home/hyeonjin/thesis_refactored")
sys.path.insert(0, str(ROOT / "src"))

from utils.evaluator import parse_sql_elements, calculate_schema_metrics  # noqa: E402

CELLS = [
    ("wave16_qwen3", ROOT / "outputs/experiments/abl/wave16_encoder_backbone/m16_qwen3_0.6b_m4",
     "score_analysis_m16_qwen3_0.6b_m4.jsonl"),
    ("wave6_minilm", ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional",
     "score_analysis_w6_p2_m4_bidirectional.jsonl"),
]

# Wave 15 no_filter cell — Wave 6 anchor 의 "+Extractor (no filter)" stage proxy
# (Enriched + QCond + MSTPCSTUnion + no filter, identical to Wave 6 stack minus M4 Filter)
WAVE15_NO_FILTER_DIR = ROOT / "outputs/experiments/abl/wave15_module_ablation/m15_no_filter_enriched_qcond_mst_pcst"

TOPK = 20  # Selector top_k (config retain)


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _harm(R: float, P: float) -> float:
    return 2 * R * P / (R + P) if (R + P) > 0 else 0.0


def load_gold_lookup() -> Dict[int, Dict]:
    """qid → {gold_sql, db_id, difficulty}."""
    with (ROOT / "data/raw/BIRD_dev/dev.json").open() as f:
        items = json.load(f)
    return {
        int(it["question_id"]): {
            "gold_sql": it.get("SQL", it.get("query", "")),
            "db_id": it.get("db_id"),
            "difficulty": it.get("difficulty", "unknown"),
        }
        for it in items
    }


def pred_cols_from_nodes(nodes: List[str], gold_cols_lower: Set[str]) -> Set[str]:
    """main.py:101-125 path — FK arrow 제외 + col-only lowercase + table.col disambiguation."""
    pred: Set[str] = set()
    for node in nodes or []:
        if not isinstance(node, str):
            continue
        if "->" in node:
            continue
        if "." in node:
            tbl, col = node.split(".", 1)
            col_lower = col.lower()
            tbl_col_lower = f"{tbl.lower()}.{col_lower}"
            if tbl_col_lower in gold_cols_lower:
                pred.add(tbl_col_lower)
            else:
                pred.add(col_lower)
        # pure table (no dot) → main.py pred_tables only, not pred_cols
    return pred


def extractor_dict_to_nodes(ext_dict: Dict[str, List[str]]) -> List[str]:
    """extractor_info.extractor_selected_nodes ({table: [cols]}) → flat node list ["table.col", ...]."""
    out: List[str] = []
    for tbl, cols in (ext_dict or {}).items():
        for c in cols:
            # FK arrow elements like "CDSCode->schools.CDSCode" remain; '->' detection later skips them
            out.append(f"{tbl}.{c}")
    return out


def reconstruct_topk_from_scores(score_rows: List[Dict], k: int = TOPK) -> List[str]:
    """score_analysis_*.jsonl 의 한 query 위 top-k node_name (desc score sort)."""
    sorted_rows = sorted(score_rows, key=lambda x: -x.get("score", 0.0))
    return [r["node_name"] for r in sorted_rows[:k]]


def group_scores_by_query(path: Path) -> Dict[int, List[Dict]]:
    groups: Dict[int, List[Dict]] = defaultdict(list)
    with path.open() as f:
        for ln in f:
            d = json.loads(ln)
            groups[int(d["query_id"])].append(d)
    return groups


def measure_per_query(
    cell_tag: str,
    cell_dir: Path,
    score_filename: str,
    gold_lookup: Dict[int, Dict],
) -> Tuple[List[Dict], Dict]:
    """Per-query R/P/F1 (selector / extractor / final) + 집계 summary."""

    pred_rows = _read_jsonl(cell_dir / "predictions.jsonl")
    score_groups = group_scores_by_query(cell_dir / score_filename)

    is_wave16 = cell_tag == "wave16_qwen3"

    per_q_rows: List[Dict] = []
    Rs_sel, Ps_sel, F1s_sel, EXs_sel = [], [], [], []
    Rs_ext, Ps_ext, F1s_ext, EXs_ext = [], [], [], []
    Rs_fin, Ps_fin, F1s_fin, EXs_fin = [], [], [], []

    by_diff_sel: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    by_diff_ext: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    by_diff_fin: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)

    n_missing_extractor = 0
    n_missing_selector = 0
    n_total = 0

    for r in pred_rows:
        qid = r.get("question_id")
        if qid is None:
            continue
        meta = gold_lookup.get(int(qid))
        if not meta or not meta["gold_sql"]:
            continue
        gold_sql = meta["gold_sql"]
        difficulty = meta["difficulty"]

        _gold_tables, gold_cols = parse_sql_elements(gold_sql)
        gold_cols_lower = {c.lower() for c in gold_cols}

        # --- Selector only stage ---
        if is_wave16:
            sel_nodes = (r.get("selector_info") or {}).get("selected_nodes_top_k") or []
        else:
            # reconstruct top-k from score_analysis
            sg = score_groups.get(int(qid), [])
            if not sg:
                n_missing_selector += 1
                sel_nodes = []
            else:
                sel_nodes = reconstruct_topk_from_scores(sg, TOPK)
        pred_sel = pred_cols_from_nodes(sel_nodes, gold_cols_lower)
        R_sel, P_sel, _m, _x = calculate_schema_metrics(pred_sel, gold_cols_lower)
        F1_sel = _harm(R_sel, P_sel)
        if is_wave16:
            EX_sel = int(r.get("ex_score_selector_only", 0) or 0)
        else:
            EX_sel = -1  # Wave 6 anchor: not logged

        # --- +Extractor (no filter) stage ---
        if is_wave16:
            ext_dict = (r.get("extractor_info") or {}).get("extractor_selected_nodes")
            if ext_dict is None:
                n_missing_extractor += 1
                ext_nodes = []
            else:
                ext_nodes = extractor_dict_to_nodes(ext_dict)
            pred_ext = pred_cols_from_nodes(ext_nodes, gold_cols_lower)
            R_ext, P_ext, _m, _x = calculate_schema_metrics(pred_ext, gold_cols_lower)
            F1_ext = _harm(R_ext, P_ext)
            EX_ext = int(r.get("ex_score_extractor_only", 0) or 0)
        else:
            # Wave 6: no per-query extractor node logging — sentinel (skipped at agg)
            R_ext = P_ext = F1_ext = -1.0
            EX_ext = -1

        # --- +Filter (final) stage ---
        final_nodes = r.get("final_nodes", []) or []
        pred_fin = pred_cols_from_nodes(final_nodes, gold_cols_lower)
        R_fin, P_fin, _m, _x = calculate_schema_metrics(pred_fin, gold_cols_lower)
        F1_fin = _harm(R_fin, P_fin)
        EX_fin = int(r.get("ex_score", 0) or 0)

        per_q_rows.append({
            "qid": int(qid),
            "db_id": r.get("db_id"),
            "difficulty": difficulty,
            "sel_R": round(R_sel, 4), "sel_P": round(P_sel, 4),
            "sel_F1": round(F1_sel, 4), "sel_EX": EX_sel,
            "ext_R": round(R_ext, 4) if R_ext >= 0 else None,
            "ext_P": round(P_ext, 4) if P_ext >= 0 else None,
            "ext_F1": round(F1_ext, 4) if F1_ext >= 0 else None,
            "ext_EX": EX_ext if EX_ext >= 0 else None,
            "fin_R": round(R_fin, 4), "fin_P": round(P_fin, 4),
            "fin_F1": round(F1_fin, 4), "fin_EX": EX_fin,
            "gold_cols_count": len(gold_cols_lower),
            "sel_pred_cols_count": len(pred_sel),
            "fin_pred_cols_count": len(pred_fin),
        })

        Rs_sel.append(R_sel); Ps_sel.append(P_sel); F1s_sel.append(F1_sel)
        if EX_sel >= 0:
            EXs_sel.append(EX_sel)
            by_diff_sel[difficulty].append((R_sel, P_sel, F1_sel, EX_sel))
        else:
            by_diff_sel[difficulty].append((R_sel, P_sel, F1_sel, -1))

        if R_ext >= 0:
            Rs_ext.append(R_ext); Ps_ext.append(P_ext); F1s_ext.append(F1_ext); EXs_ext.append(EX_ext)
            by_diff_ext[difficulty].append((R_ext, P_ext, F1_ext, EX_ext))

        Rs_fin.append(R_fin); Ps_fin.append(P_fin); F1s_fin.append(F1_fin); EXs_fin.append(EX_fin)
        by_diff_fin[difficulty].append((R_fin, P_fin, F1_fin, EX_fin))

        n_total += 1

    def _agg(Rs, Ps, F1s, EXs):
        if not Rs:
            return {"n": 0}
        n = len(Rs)
        R = sum(Rs) / n
        P = sum(Ps) / n
        out = {
            "n": n,
            "R": R, "P": P,
            "F1_perq_mean": sum(F1s) / n,
            "F1_harm": _harm(R, P),
        }
        if EXs:
            out["EX"] = sum(EXs) / len(EXs)
        return out

    def _agg_per_diff(by_diff):
        out = {}
        for d, recs in by_diff.items():
            if not recs:
                continue
            Rs = [x[0] for x in recs if x[0] >= 0]
            Ps = [x[1] for x in recs if x[1] >= 0]
            F1s = [x[2] for x in recs if x[2] >= 0]
            EXs = [x[3] for x in recs if x[3] >= 0]
            if not Rs:
                continue
            out[d] = {
                "n": len(Rs),
                "R": sum(Rs) / len(Rs),
                "P": sum(Ps) / len(Ps),
                "F1_perq_mean": sum(F1s) / len(F1s),
                "F1_harm": _harm(sum(Rs) / len(Rs), sum(Ps) / len(Ps)),
            }
            if EXs:
                out[d]["EX"] = sum(EXs) / len(EXs)
        return out

    summary = {
        "cell_tag": cell_tag,
        "n_total": n_total,
        "n_missing_extractor": n_missing_extractor,
        "n_missing_selector": n_missing_selector,
        "selector": {
            "overall": _agg(Rs_sel, Ps_sel, F1s_sel, EXs_sel if is_wave16 else []),
            "per_difficulty": _agg_per_diff(by_diff_sel),
        },
        "extractor": {
            "overall": _agg(Rs_ext, Ps_ext, F1s_ext, EXs_ext),
            "per_difficulty": _agg_per_diff(by_diff_ext),
        } if is_wave16 else {"overall": {"n": 0}, "per_difficulty": {}, "note": "not logged for Wave 6 anchor; see Wave 15 no_filter proxy"},
        "final": {
            "overall": _agg(Rs_fin, Ps_fin, F1s_fin, EXs_fin),
            "per_difficulty": _agg_per_diff(by_diff_fin),
        },
    }

    return per_q_rows, summary


def measure_wave15_no_filter_proxy(gold_lookup: Dict[int, Dict]) -> Dict:
    """Wave 15 no_filter cell — Wave 6 anchor 의 '+Extractor (no filter)' stage proxy."""
    pred_rows = _read_jsonl(WAVE15_NO_FILTER_DIR / "predictions.jsonl")
    Rs, Ps, F1s, EXs = [], [], [], []
    by_diff: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)

    for r in pred_rows:
        qid = r.get("question_id")
        if qid is None:
            continue
        meta = gold_lookup.get(int(qid))
        if not meta or not meta["gold_sql"]:
            continue
        gold_sql = meta["gold_sql"]
        difficulty = meta["difficulty"]

        _gt, gold_cols = parse_sql_elements(gold_sql)
        gold_cols_lower = {c.lower() for c in gold_cols}
        final_nodes = r.get("final_nodes", []) or []
        pred = pred_cols_from_nodes(final_nodes, gold_cols_lower)
        R, P, _m, _x = calculate_schema_metrics(pred, gold_cols_lower)
        F1 = _harm(R, P)
        EX = int(r.get("ex_score", 0) or 0)

        Rs.append(R); Ps.append(P); F1s.append(F1); EXs.append(EX)
        by_diff[difficulty].append((R, P, F1, EX))

    n = len(Rs)
    R = sum(Rs) / n
    P = sum(Ps) / n
    pd_out = {}
    for d, recs in by_diff.items():
        Rs_d = [x[0] for x in recs]
        Ps_d = [x[1] for x in recs]
        F1s_d = [x[2] for x in recs]
        EXs_d = [x[3] for x in recs]
        Rm = sum(Rs_d) / len(Rs_d)
        Pm = sum(Ps_d) / len(Ps_d)
        pd_out[d] = {
            "n": len(Rs_d), "R": Rm, "P": Pm,
            "F1_perq_mean": sum(F1s_d) / len(F1s_d),
            "F1_harm": _harm(Rm, Pm),
            "EX": sum(EXs_d) / len(EXs_d),
        }

    return {
        "cell_tag": "wave15_no_filter_proxy",
        "overall": {
            "n": n, "R": R, "P": P,
            "F1_perq_mean": sum(F1s) / n,
            "F1_harm": _harm(R, P),
            "EX": sum(EXs) / n,
        },
        "per_difficulty": pd_out,
    }


def main():
    gold_lookup = load_gold_lookup()
    print(f"Loaded {len(gold_lookup)} gold records from dev.json")
    print()

    summaries: Dict[str, Dict] = {}
    for cell_tag, cell_dir, score_fname in CELLS:
        print(f"=== {cell_tag} === ({cell_dir.name})")
        per_q, summary = measure_per_query(cell_tag, cell_dir, score_fname, gold_lookup)
        summaries[cell_tag] = summary

        out_path = cell_dir / "output_stagewise_rpf1.jsonl"
        with out_path.open("w") as f:
            for row in per_q:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  → per-query → {out_path}  (n={len(per_q)})")

        for stage in ("selector", "extractor", "final"):
            s = summary[stage]["overall"]
            if s.get("n", 0) == 0:
                print(f"    {stage:10s}  (skipped)")
                continue
            ex_str = f"  EX={s['EX']:.4f}" if "EX" in s else "  EX=N/A"
            print(f"    {stage:10s}  n={s['n']}  R={s['R']:.4f}  P={s['P']:.4f}  F1_perq={s['F1_perq_mean']:.4f}  F1_harm={s['F1_harm']:.4f}{ex_str}")
        print()

    # Wave 15 no_filter proxy for Wave 6 anchor's "+Extractor" stage
    print("=== wave15_no_filter_proxy (Wave 6 anchor +Extractor proxy) ===")
    proxy = measure_wave15_no_filter_proxy(gold_lookup)
    ov = proxy["overall"]
    print(f"    overall    n={ov['n']}  R={ov['R']:.4f}  P={ov['P']:.4f}  F1_harm={ov['F1_harm']:.4f}  EX={ov['EX']:.4f}")
    summaries["wave15_no_filter_proxy"] = proxy
    print()

    # Sanity check: Wave 16 final R/P/F1/EX against HISTORY (R=0.9337, P=0.7563, F1=0.8358, EX=0.5124)
    # Wave 6 anchor: R=0.9357, P=0.7593, F1=0.8383, EX=0.5300
    print("=" * 100)
    print("Sanity Check (vs HISTORY metrics.txt)")
    print("=" * 100)
    expected = {
        "wave16_qwen3":  {"R": 0.9337, "P": 0.7563, "F1": 0.8358, "EX": 0.5124},
        "wave6_minilm":  {"R": 0.9325, "P": 0.7593, "F1": 0.8383, "EX": 0.5300},  # metrics.txt R=0.9325 (pre/post-patch nuance)
        "wave15_no_filter_proxy": {"R": 0.9959, "P": 0.1268, "F1": 0.2250, "EX": 0.5137},
    }
    for tag, exp in expected.items():
        if tag == "wave15_no_filter_proxy":
            ov = summaries[tag]["overall"]
        else:
            ov = summaries[tag]["final"]["overall"]
        dR = ov["R"] - exp["R"]; dP = ov["P"] - exp["P"]; dF = ov["F1_harm"] - exp["F1"]; dE = ov["EX"] - exp["EX"]
        flag = "✅" if all(abs(x) < 5e-4 for x in [dR, dP, dF, dE]) else "⚠"
        print(f"  {tag}: R={ov['R']:.4f}(Δ{dR:+.4f}) P={ov['P']:.4f}(Δ{dP:+.4f}) F1={ov['F1_harm']:.4f}(Δ{dF:+.4f}) EX={ov['EX']:.4f}(Δ{dE:+.4f}) {flag}")
    print()

    # CSV
    csv_path = ROOT / "outputs/analysis/wave16_encoder_backbone_stagewise_2026-05-22.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_tag", "stage", "scope", "n", "R", "P", "F1_perq_mean", "F1_harm", "EX"])
        for tag in ("wave16_qwen3", "wave6_minilm"):
            for stage in ("selector", "extractor", "final"):
                s = summaries[tag][stage]
                ov = s["overall"]
                if ov.get("n", 0) == 0:
                    w.writerow([tag, stage, "overall", 0, "", "", "", "", ""])
                    continue
                w.writerow([
                    tag, stage, "overall", ov["n"],
                    round(ov["R"], 4), round(ov["P"], 4),
                    round(ov["F1_perq_mean"], 4), round(ov["F1_harm"], 4),
                    round(ov.get("EX", -1), 4) if "EX" in ov else "",
                ])
                for d in ("simple", "moderate", "challenging"):
                    pd = s.get("per_difficulty", {}).get(d)
                    if not pd:
                        continue
                    w.writerow([
                        tag, stage, d, pd["n"],
                        round(pd["R"], 4), round(pd["P"], 4),
                        round(pd["F1_perq_mean"], 4), round(pd["F1_harm"], 4),
                        round(pd.get("EX", -1), 4) if "EX" in pd else "",
                    ])
        # Wave 15 proxy row
        proxy_ov = summaries["wave15_no_filter_proxy"]["overall"]
        w.writerow([
            "wave15_no_filter_proxy", "extractor_no_filter", "overall", proxy_ov["n"],
            round(proxy_ov["R"], 4), round(proxy_ov["P"], 4),
            round(proxy_ov["F1_perq_mean"], 4), round(proxy_ov["F1_harm"], 4),
            round(proxy_ov["EX"], 4),
        ])
        for d in ("simple", "moderate", "challenging"):
            pd = summaries["wave15_no_filter_proxy"]["per_difficulty"].get(d)
            if pd:
                w.writerow([
                    "wave15_no_filter_proxy", "extractor_no_filter", d, pd["n"],
                    round(pd["R"], 4), round(pd["P"], 4),
                    round(pd["F1_perq_mean"], 4), round(pd["F1_harm"], 4),
                    round(pd["EX"], 4),
                ])
    print(f"→ csv:  {csv_path}")

    # JSON summary
    json_path = ROOT / "outputs/analysis/wave16_encoder_backbone_stagewise_2026-05-22.json"
    with json_path.open("w") as f:
        json.dump({"summaries": summaries, "expected_sanity": expected}, f, indent=2, ensure_ascii=False)
    print(f"→ json: {json_path}")


if __name__ == "__main__":
    main()
