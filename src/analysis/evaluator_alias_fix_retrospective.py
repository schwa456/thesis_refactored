"""Wave 13 Phase B — Retrospective R/P/F1 재측정 (evaluator alias resolution patch 영향 정합).

Patch commit f67fa65 (src/utils/evaluator.py:9-37) 후 의 새 `parse_sql_elements` 로
~60 cells × per-query gold_cols 재계산 + 본 framework 의 모든 cell × R_new / P_new / F1_new
정량 정정. R_old / P_old 는 출력 logged value (output_*.jsonl 또는 oracle prior) 또는
inline old-evaluator path 로 재계산 — 본 script 는 양 path 모두 inline 으로 정합.

산출:
  - outputs/analysis/evaluator_alias_fix_retrospective_2026-05-20.csv (cells × Δ matrix)
  - outputs/analysis/evaluator_alias_fix_retrospective_2026-05-20.json (full summary)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sqlglot
from sqlglot.expressions import Alias, Column, Table

ROOT = Path("/home/hyeonjin/thesis_refactored")
sys.path.insert(0, str(ROOT / "src"))

# ---- inline old + new evaluator (so we can compare side-by-side) ------

def parse_sql_elements_old(sql: str) -> Tuple[Set[str], Set[str]]:
    """Pre-Wave-13 evaluator (alias 포함, 본 patch 이전)."""
    if not sql:
        return set(), set()
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        tables = set(node.name.lower() for node in parsed.find_all(Table) if node.name)
        columns = set(node.name.lower() for node in parsed.find_all(Column) if node.name)
        return tables, columns
    except Exception:
        return set(), set()


def parse_sql_elements_new(sql: str) -> Tuple[Set[str], Set[str]]:
    """Post-Wave-13 evaluator (alias 제외, src/utils/evaluator.py:9-37 patch)."""
    if not sql:
        return set(), set()
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        tables = set(node.name.lower() for node in parsed.find_all(Table) if node.name)
        alias_names = set()
        for alias_node in parsed.find_all(Alias):
            if alias_node.alias:
                alias_names.add(alias_node.alias.lower())
        columns = set(
            node.name.lower() for node in parsed.find_all(Column)
            if node.name and node.name.lower() not in alias_names
        )
        return tables, columns
    except Exception:
        return set(), set()


def calc_metrics(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    inter = pred & gold
    R = (len(inter) / len(gold)) if gold else 0.0
    P = (len(inter) / len(pred)) if pred else 0.0
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return R, P, F1


# ---- pred_cols extractors (cell-type specific, all return col-only lowercase set) -----

def pred_cols_from_final_nodes(final_nodes: List[str], gold_cols_lower: Set[str]) -> Set[str]:
    """main.py:101-125 정합 — final_nodes 의 col only (FK arrow 제외, lowercase)."""
    pred = set()
    for node in final_nodes or []:
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
        # else: pure table — pred_cols 에 미포함 (main.py: pred_tables 만)
    return pred


def pred_cols_b1_full(db_id: str, db_lookup: Dict) -> Set[str]:
    return set((db_lookup.get(db_id) or {}).get("cols", set()))


def pred_cols_b2_gold_table(gold_tables: Set[str], db_id: str, db_lookup: Dict) -> Set[str]:
    cols_by_table = (db_lookup.get(db_id) or {}).get("cols_by_table", {})
    pred = set()
    for t in gold_tables:
        pred |= cols_by_table.get(t.lower(), set())
    return pred


def pred_cols_b3_gold_column(gold_cols: Set[str]) -> Set[str]:
    return set(gold_cols)


# ---- db schema + difficulty lookups -----------------------------------

def load_db_schema_lookup() -> Dict[str, Dict[str, Set[str]]]:
    with (ROOT / "data/raw/BIRD_dev/dev_tables.json").open() as f:
        items = json.load(f)
    out: Dict[str, Dict] = {}
    for db in items:
        db_id = db["db_id"]
        table_names = [t.lower() for t in db.get("table_names_original", [])]
        all_cols: Set[str] = set()
        cols_by_table: Dict[str, Set[str]] = {t: set() for t in table_names}
        for table_idx, col_name in db.get("column_names_original", []):
            if table_idx == -1:
                continue
            col_lower = col_name.lower()
            all_cols.add(col_lower)
            if 0 <= table_idx < len(table_names):
                cols_by_table[table_names[table_idx]].add(col_lower)
        out[db_id] = {"cols": all_cols, "cols_by_table": cols_by_table}
    return out


def load_dev_lookup() -> Dict[int, Dict]:
    """qid → {gold_sql, db_id, difficulty}."""
    dev_path = ROOT / "data/raw/BIRD_dev/dev.json"
    if not dev_path.exists():
        return {}
    with dev_path.open() as f:
        items = json.load(f)
    out = {}
    for it in items:
        qid = it["question_id"]
        out[qid] = {
            "gold_sql": it.get("SQL", it.get("query", "")),
            "db_id": it.get("db_id"),
            "difficulty": it.get("difficulty", "unknown"),
        }
    return out


# ---- cell measurement ------------------------------------------------

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


def measure_standard_cell(
    cell_dir: Path,
    cell_label: str,
    db_lookup: Dict,
    dev_lookup: Dict,
) -> Dict:
    """Wave 5/6/8 (Comb-A 포함) + c01/c02/c03 cells.
       predictions.jsonl 의 final_nodes + output_*.jsonl 의 gold_sql 사용 (output 미존재 시 dev_lookup fallback).
    """
    pred_path = cell_dir / "predictions.jsonl"
    if not pred_path.exists():
        return {"cell": cell_label, "missing": True, "reason": "predictions.jsonl not found"}
    output_path = next(cell_dir.glob("output_*.jsonl"), None)
    out_rows = {r["question_id"]: r for r in _read_jsonl(output_path)} if output_path and output_path.exists() else {}
    pred_rows = _read_jsonl(pred_path)

    # qid 정의: predictions.jsonl 의 question_id (integer)
    Rs_old, Ps_old, F1s_old = [], [], []
    Rs_new, Ps_new, F1s_new = [], [], []
    delta_R_per_q: List[Tuple[int, float]] = []  # (qid, ΔR)
    nq_aliased = 0  # gold 에 alias 가 있었던 query count

    for r in pred_rows:
        qid = r.get("question_id")
        if qid is None:
            continue
        gold_sql = (out_rows.get(qid) or {}).get("gold_sql") or (dev_lookup.get(qid) or {}).get("gold_sql") or ""
        if not gold_sql:
            continue
        final_nodes = r.get("final_nodes", []) or []

        gold_tables_old, gold_cols_old = parse_sql_elements_old(gold_sql)
        gold_cols_old = {c.lower() for c in gold_cols_old}
        gold_tables_new, gold_cols_new = parse_sql_elements_new(gold_sql)
        gold_cols_new = {c.lower() for c in gold_cols_new}

        if gold_cols_old != gold_cols_new:
            nq_aliased += 1

        # pred_cols (main.py logic) — gold_cols_lower 는 새 spec 위 (table.col disambiguation)
        # main.py 는 OLD spec 으로 했음 단 disambig branch 결과는 거의 동일 (gold_cols 가 col only 일 때 false branch)
        pred_cols_for_old = pred_cols_from_final_nodes(final_nodes, gold_cols_old)
        pred_cols_for_new = pred_cols_from_final_nodes(final_nodes, gold_cols_new)

        R_old, P_old, F1_old = calc_metrics(pred_cols_for_old, gold_cols_old)
        R_new, P_new, F1_new = calc_metrics(pred_cols_for_new, gold_cols_new)

        Rs_old.append(R_old); Ps_old.append(P_old); F1s_old.append(F1_old)
        Rs_new.append(R_new); Ps_new.append(P_new); F1s_new.append(F1_new)
        if R_new > R_old + 1e-9:
            delta_R_per_q.append((qid, R_new - R_old))

    if not Rs_old:
        return {"cell": cell_label, "missing": True, "reason": "no qualifying records"}

    n = len(Rs_old)
    R_old_mean = sum(Rs_old) / n
    P_old_mean = sum(Ps_old) / n
    F1_old_mean = sum(F1s_old) / n
    R_new_mean = sum(Rs_new) / n
    P_new_mean = sum(Ps_new) / n
    F1_new_mean = sum(F1s_new) / n

    return {
        "cell": cell_label,
        "n": n,
        "R_old": R_old_mean, "R_new": R_new_mean, "dR": R_new_mean - R_old_mean,
        "P_old": P_old_mean, "P_new": P_new_mean, "dP": P_new_mean - P_old_mean,
        "F1_old": F1_old_mean, "F1_new": F1_new_mean, "dF1": F1_new_mean - F1_old_mean,
        "n_aliased_queries": nq_aliased,
        "n_queries_with_R_lift": len(delta_R_per_q),
    }


def measure_oracle_cell(
    cell_dir: Path,
    cell_label: str,
    pred_kind: str,  # "B1_full" | "B2_gold_table" | "B3_gold_column"
    db_lookup: Dict,
    dev_lookup: Dict,
) -> Dict:
    """Wave 12 oracle (B1/B2/B3) — predictions.jsonl 의 qid + gold_sql 사용 (final_nodes 미존재)."""
    pred_path = cell_dir / "predictions.jsonl"
    rows = _read_jsonl(pred_path)
    if not rows:
        return {"cell": cell_label, "missing": True, "reason": "predictions.jsonl empty"}

    Rs_old, Ps_old, F1s_old = [], [], []
    Rs_new, Ps_new, F1s_new = [], [], []
    nq_aliased = 0

    for r in rows:
        qid_str = r.get("qid")
        db_id = r.get("db_id")
        gold_sql = r.get("gold_sql") or ""
        if not gold_sql or db_id is None:
            continue

        gold_tables_old, gold_cols_old = parse_sql_elements_old(gold_sql)
        gold_cols_old = {c.lower() for c in gold_cols_old}
        gold_tables_new, gold_cols_new = parse_sql_elements_new(gold_sql)
        gold_cols_new = {c.lower() for c in gold_cols_new}
        if gold_cols_old != gold_cols_new:
            nq_aliased += 1

        if pred_kind == "B1_full":
            pred = pred_cols_b1_full(db_id, db_lookup)
        elif pred_kind == "B2_gold_table":
            pred = pred_cols_b2_gold_table({t.lower() for t in gold_tables_new}, db_id, db_lookup)
        elif pred_kind == "B3_gold_column":
            # B3 : pred = gold_cols (each spec) — paired with same gold
            pred_old = pred_cols_b3_gold_column(gold_cols_old)
            pred_new = pred_cols_b3_gold_column(gold_cols_new)
            R_old, P_old, F1_old = calc_metrics(pred_old, gold_cols_old)
            R_new, P_new, F1_new = calc_metrics(pred_new, gold_cols_new)
            Rs_old.append(R_old); Ps_old.append(P_old); F1s_old.append(F1_old)
            Rs_new.append(R_new); Ps_new.append(P_new); F1s_new.append(F1_new)
            continue
        else:
            raise ValueError(f"Unknown pred_kind: {pred_kind}")

        # B1, B2: pred 자체는 alias 영향 없음 (DB col 또는 gold table col set)
        R_old, P_old, F1_old = calc_metrics(pred, gold_cols_old)
        R_new, P_new, F1_new = calc_metrics(pred, gold_cols_new)
        Rs_old.append(R_old); Ps_old.append(P_old); F1s_old.append(F1_old)
        Rs_new.append(R_new); Ps_new.append(P_new); F1s_new.append(F1_new)

    n = len(Rs_old)
    return {
        "cell": cell_label,
        "n": n,
        "R_old": sum(Rs_old) / n, "R_new": sum(Rs_new) / n, "dR": (sum(Rs_new) - sum(Rs_old)) / n,
        "P_old": sum(Ps_old) / n, "P_new": sum(Ps_new) / n, "dP": (sum(Ps_new) - sum(Ps_old)) / n,
        "F1_old": sum(F1s_old) / n, "F1_new": sum(F1s_new) / n, "dF1": (sum(F1s_new) - sum(F1s_old)) / n,
        "n_aliased_queries": nq_aliased,
    }


# ---- cell registry ----------------------------------------------------

def all_cells():
    """Return list of (cell_label, cell_dir, kind, opt_extra)."""
    out = []
    # Wave 5 anchor
    out.append(("Wave5_c01_01_anchor_5_14", ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_theta_0.1", "standard"))
    out.append(("Wave7_c01_01_relog",       ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog", "standard"))

    # Wave 6 9 cells
    w6 = ROOT / "outputs/experiments/abl/wave6_recall_biased"
    for sub, label in [
        ("wave6_p1_recall_biased_mild",         "Wave6_M1A_mild"),
        ("wave6_p1_recall_biased_strong",       "Wave6_M1B_strong"),
        ("wave6_p1_recall_biased_exclusion_rule","Wave6_M1C_exclusion"),
        ("w6_p2a_m2cot_strong",                  "Wave6_M2_CoT_Gated"),
        ("w6_p2_m3_voting",                      "Wave6_M3_voting"),
        ("w6_p2_m4_bidirectional",               "Wave6_M4_Bidirectional"),
        ("w6_p2_m5_two_stage",                   "Wave6_M5_two_stage"),
        ("w6_p4_c1_m4_strong",                   "Wave6_C1_M4_strong"),
        ("w6_p5_c2_m4_majority",                 "Wave6_C2_M4_majority"),
    ]:
        out.append((label, w6 / sub, "standard"))

    # Wave 8 8 cells + Comb-A
    w8 = ROOT / "outputs/experiments/abl/wave8_m4_extensions"
    for d, sub, label in [
        ("d1_decompose", "abl_wave8_d1v1_multi_backward",   "Wave8_D1v1_multi_backward"),
        ("d1_decompose", "abl_wave8_d1v2_full_decompose",   "Wave8_D1v2_full_decompose"),
        ("d2_steiner",   "abl_wave8_d2v1_direct_fk",        "Wave8_D2v1_direct_fk"),
        ("d2_steiner",   "abl_wave8_d2v2_bridge_1hop",      "Wave8_D2v2_bridge_1hop"),
        ("d3_verify",    "abl_wave8_d3v1_verify1round",     "Wave8_D3v1_verify1round"),
        ("d3_verify",    "abl_wave8_d3v2_verify2round",     "Wave8_D3v2_verify2round"),
        ("d4_value_hint","abl_wave8_d4v1_value_hint_forward","Wave8_D4v1_value_hint"),
        ("d4_value_hint","abl_wave8_d4v3_forced_include",   "Wave8_D4v3_forced_include"),
        ("comb_a",       "abl_wave8_comb_a_value_hint_verify2round", "Wave8_CombA_D4v1_D3v2"),
    ]:
        out.append((label, w8 / d / sub, "standard"))

    # Wave 9 baseline relog
    w9 = ROOT / "outputs/baselines/wave9_relog"
    for sub, label in [
        ("g_retriever_relog", "Wave9_G_Retriever_relog"),
        ("linkalign_relog",   "Wave9_LinkAlign_relog"),
        ("xiyansql_relog",    "Wave9_XiYanSQL_relog"),
    ]:
        out.append((label, w9 / sub, "standard"))

    # Wave 12 oracle cells
    out.append(("Wave12_B1_full",        ROOT / "outputs/analysis/glm_baseline/b1_full",       "oracle:B1_full"))
    out.append(("Wave12_B2_gold_table",  ROOT / "outputs/analysis/glm_baseline/b2_gold_table", "oracle:B2_gold_table"))
    out.append(("Wave12_B3_gold_column", ROOT / "outputs/analysis/glm_baseline/b3_gold_column","oracle:B3_gold_column"))

    # Wave 11 Schema Serialization Direction C (1st run + rerun, post-Wave 13 patch)
    # 1st run (pre-patch evaluator) — retrospective recompute with new evaluator
    w11 = ROOT / "outputs/experiments/abl/wave11_schema_serialization"
    out.append(("Wave11_c_v0_baseline_1strun",        w11 / "c_v0_baseline_run1",            "standard"))
    out.append(("Wave11_c_v1_source_tagged",          w11 / "c_v1_source_tagged",            "standard"))
    out.append(("Wave11_c_v2_question_enrichment",    w11 / "c_v2_question_enrichment",      "standard"))
    out.append(("Wave11_c_v3a_flat_merged_fk_1strun", w11 / "c_v3a_flat_merged_fk_run1",     "standard"))
    out.append(("Wave11_c_v3b_flat_merged_no_fk_1strun",w11 / "c_v3b_flat_merged_no_fk_run1","standard"))
    out.append(("Wave11_comb_c_tagged_enriched",      w11 / "comb_c_tagged_enriched",        "standard"))
    # Rerun (post-patch — LLM re-executed with patched evaluator)
    out.append(("Wave11_c_v0_baseline_rerun",         w11 / "c_v0_baseline",                 "standard"))
    out.append(("Wave11_c_v3a_flat_merged_fk_rerun",  w11 / "c_v3a_flat_merged_fk",          "standard"))
    out.append(("Wave11_c_v3b_flat_merged_no_fk_rerun",w11 / "c_v3b_flat_merged_no_fk",      "standard"))

    # c01 6 cells (Phase 1.1 θ sweep)
    c01 = ROOT / "outputs/experiments/abl/c01_threshold_sweep"
    for theta in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]:
        out.append((f"c01_theta_{theta}", c01 / f"c01_{['','01','02','03','04','05','06']['0.1 0.2 0.3 0.4 0.5 0.6'.split().index(theta)+1]}_theta_{theta}", "standard"))
    # safer construction
    # (Override above with explicit list to avoid index errors)
    out = [t for t in out if not t[0].startswith("c01_theta_")]
    for i, theta in zip(range(1, 7), ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]):
        out.append((f"c01_theta_{theta}", c01 / f"c01_{i:02d}_theta_{theta}", "standard"))

    # c02 7 cells (Phase 1.2 K sweep)
    c02 = ROOT / "outputs/experiments/abl/c02_topk_sweep"
    for i, K in zip(range(1, 8), [15, 20, 30, 40, 50, 70, 100]):
        out.append((f"c02_topk_{K}", c02 / f"c02_{i:02d}_topk_{K}", "standard"))

    # c03 25 cells (Phase 2 grid)
    c03 = ROOT / "outputs/experiments/abl/c03_phase2_grid"
    cells_25 = sorted([p for p in c03.iterdir() if p.is_dir() and p.name.startswith("p2_")], key=lambda p: p.name)
    for p in cells_25:
        out.append((f"c03_{p.name}", p, "standard"))

    return out


def main():
    db_lookup = load_db_schema_lookup()
    dev_lookup = load_dev_lookup()
    print(f"Loaded {len(db_lookup)} DBs, {len(dev_lookup)} dev queries")
    print()

    results = []
    cells = all_cells()
    print(f"Measuring {len(cells)} cells...")
    print()

    for label, cell_dir, kind in cells:
        if kind == "standard":
            res = measure_standard_cell(cell_dir, label, db_lookup, dev_lookup)
        elif kind.startswith("oracle:"):
            pred_kind = kind.split(":", 1)[1]
            res = measure_oracle_cell(cell_dir, label, pred_kind, db_lookup, dev_lookup)
        else:
            continue

        if res.get("missing"):
            print(f"  [SKIP] {label}: {res.get('reason')}")
            continue

        print(
            f"  {label:42s} n={res['n']:4d}  R: {res['R_old']:.4f} → {res['R_new']:.4f} "
            f"(ΔR={res['dR']:+.4f})  P: {res['P_old']:.4f} → {res['P_new']:.4f} (ΔP={res['dP']:+.4f})  "
            f"F1: {res['F1_old']:.4f} → {res['F1_new']:.4f} (ΔF1={res['dF1']:+.4f})  "
            f"n_alias_q={res['n_aliased_queries']}"
        )
        results.append(res)

    # Aggregate summary
    print()
    print("=" * 110)
    print(f"Summary across {len(results)} cells")
    print("=" * 110)
    dRs = [r["dR"] for r in results]
    dPs = [r["dP"] for r in results]
    dF1s = [r["dF1"] for r in results]
    print(f"  Mean ΔR  across cells: {sum(dRs)/len(dRs):+.6f} (min={min(dRs):+.6f}, max={max(dRs):+.6f})")
    print(f"  Mean ΔP  across cells: {sum(dPs)/len(dPs):+.6f} (min={min(dPs):+.6f}, max={max(dPs):+.6f})")
    print(f"  Mean ΔF1 across cells: {sum(dF1s)/len(dF1s):+.6f} (min={min(dF1s):+.6f}, max={max(dF1s):+.6f})")

    # Sanity check
    n_neg_dR = sum(1 for r in results if r["dR"] < -1e-9)
    n_nonzero_dP = sum(1 for r in results if abs(r["dP"]) > 1e-9)
    print(f"\n  Sanity check 1 (ΔR ≥ 0 across all cells): {'✅ PASS' if n_neg_dR == 0 else f'❌ FAIL ({n_neg_dR} cells with ΔR < 0)'}")
    print(f"  Sanity check 2 (ΔP ≈ 0 expected, since alias not in pred):")
    print(f"     Non-zero ΔP cells: {n_nonzero_dP} / {len(results)}")

    # Save CSV
    csv_path = ROOT / "outputs/analysis/evaluator_alias_fix_retrospective_2026-05-20.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell", "n", "R_old", "R_new", "dR", "P_old", "P_new", "dP",
            "F1_old", "F1_new", "dF1", "n_aliased_queries",
        ])
        for r in results:
            w.writerow([
                r["cell"], r["n"],
                round(r["R_old"], 4), round(r["R_new"], 4), round(r["dR"], 4),
                round(r["P_old"], 4), round(r["P_new"], 4), round(r["dP"], 4),
                round(r["F1_old"], 4), round(r["F1_new"], 4), round(r["dF1"], 4),
                r.get("n_aliased_queries", 0),
            ])
    print(f"\n→ csv:  {csv_path}")

    # Save full JSON
    json_path = ROOT / "outputs/analysis/evaluator_alias_fix_retrospective_2026-05-20.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"→ json: {json_path}")


if __name__ == "__main__":
    main()
