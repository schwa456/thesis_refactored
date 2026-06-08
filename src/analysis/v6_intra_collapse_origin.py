#!/usr/bin/env python3
"""V6 intra-table collapse 원인 진단 (DECISIONS 2026-06-06 #4).

M4 anchor (best_gat_qcond_nl3.pt, v1 SchemaHeteroGAT NL3) 위:
  (1) L0 raw PLM input intra-table MAD (hi/lo, per-DB, european_football_2) — 가설 A(input) vs B(first-conv) 판별
  (2) self vs neighbor 분해 — L1 hub 컬럼 update 의 self-transform vs aggregated-neighbor (cosine L1↔L0, L1↔table)
  (3) attention attribution — hub 컬럼 L1 수신 메시지 table node vs fk vs sibling 비중

reuse: v1_v5_retrospective_v6_metrics (model load + layer hook)
GPU: CUDA_VISIBLE_DEVICES=0,1
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

ROOT = Path("/home/hyeonjin/thesis_refactored")
for p in (str(ROOT / "src"), str(ROOT / "src/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v1_v5_retrospective_v6_metrics as R
from v1_v5_retrospective_v6_metrics import (
    load_model_for_cell, extract_layerwise_via_hook, load_dev,
    _build_col_table_index, DEVICE, DEV_DB_DIR, TABLES_JSON, ENCODER_MODEL_NAME, COL_TO_TAB_EDGE,
)
from modules.encoders.local_encoder import LocalPLMEncoder
from modules.builders.graph_builder import EnrichedHeteroGraphBuilder

OUT = ROOT / "outputs/analysis"
HI_THR = 30
M4 = dict(tag="M4_anchor", ckpt_path="outputs/checkpoints/best_gat_qcond_nl3.pt",
          model_class="v1", v1_kwargs=dict(in_channels=384, hidden_channels=256, out_channels=256,
                                           num_layers=3, heads=4, query_conditioned=True, query_supernode=False))
# baseline ref (Phase0 §5.2): hi-deg L1=0.0136 L2=0.0092 L3=0.0046 ; lo-deg L1=0.1155 L2=0.0737 L3=0.0407


def intra_mad_hilo(col_emb, cb_edge, hi_thr=HI_THR):
    _, t2c = _build_col_table_index(cb_edge)
    normed = F.normalize(col_emb.float(), dim=-1)
    hi, lo = [], []
    for t, cols in t2c.items():
        if len(cols) < 2:
            continue
        e = normed[torch.tensor(cols, dtype=torch.long)]
        n = len(cols); iu = torch.triu_indices(n, n, 1)
        d = (1.0 - e @ e.T)[iu[0], iu[1]]
        (hi if len(cols) > hi_thr else lo).append(d)
    m = lambda L: float(torch.cat(L).mean()) if L else None
    return m(hi), m(lo)


def main():
    print(f"Device: {DEVICE} | CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    dev = load_dev()
    qids = list(range(len(dev)))
    enc = LocalPLMEncoder(model_name=ENCODER_MODEL_NAME)
    model = load_model_for_cell(M4)
    builder = EnrichedHeteroGraphBuilder(plm_model_name=ENCODER_MODEL_NAME, tables_json_path=str(TABLES_JSON))
    db_cache = {}

    # ── Task 1: L0..L3 intra-MAD hi/lo (per-layer, overall + per-DB) ──
    layer_hi = defaultdict(list); layer_lo = defaultdict(list)   # layer_idx -> [vals]
    db_layer_hi = defaultdict(lambda: defaultdict(list))         # db -> layer -> [vals]
    # ── Task 2/3 accumulators (hub columns only) ──
    # NOTE: L0=384(PLM) vs L1=1024(hidden×heads) — cross-layer cosine 불가. 동일 layer 내만.
    cos_l1_table = []                   # hub col L1 vs table L1 (수렴 to 공유 소스)
    cos_l0_l0_sib = []                  # hub col L0 vs sibling-mean L0 (input similarity)
    cos_l1_l1_sib = []                  # hub col L1 vs sibling-mean L1 (L1 similarity)
    t0 = time.time(); n_skip = 0
    for i, qid in enumerate(qids):
        item = dev[qid]; db = item["db_id"]
        if db not in db_cache:
            try:
                db_cache[db] = builder.build(db_id=db, db_dir=str(DEV_DB_DIR))
            except Exception as e:
                n_skip += 1; continue
        gd, md = db_cache[db]; data = gd.clone()
        try:
            q = enc.encode([item["question"]]); q = q[0] if isinstance(q, tuple) else q
            if q.dim() == 3: q = q.mean(1)
            if q.dim() == 1: q = q.unsqueeze(0)
            q = q.to(DEVICE)
            xd = {nt: x.to(DEVICE) for nt, x in data.x_dict.items()}
            ed = {et: ei.to(DEVICE) for et, ei in data.edge_index_dict.items()}
            cb = ed.get(COL_TO_TAB_EDGE)
            if cb is None:
                n_skip += 1; continue
            cb_cpu = cb.cpu()
            embs, na = extract_layerwise_via_hook(model, xd, ed, query_emb=q)
        except Exception as e:
            n_skip += 1; continue
        # embs = [L0_PLM, L1, L2, L3, L_out]; use 0..3
        for li in range(min(4, len(embs))):
            ce = embs[li].get("column")
            if ce is None:
                continue
            hi, lo = intra_mad_hilo(ce, cb_cpu)
            if hi is not None:
                layer_hi[li].append(hi); db_layer_hi[db][li].append(hi)
            if lo is not None:
                layer_lo[li].append(lo)
        # Task 2/3: hub columns (tables >30 col) — cosine self/neighbor
        col_to_tab, t2c = _build_col_table_index(cb_cpu)
        l0 = embs[0].get("column"); l1 = embs[1].get("column")
        l0t = embs[0].get("table"); l1t = embs[1].get("table")
        if l0 is not None and l1 is not None:
            l0n = F.normalize(l0.float(), dim=-1); l1n = F.normalize(l1.float(), dim=-1)
            for t, cols in t2c.items():
                if len(cols) <= HI_THR or len(cols) < 2:
                    continue
                idx = torch.tensor(cols, dtype=torch.long)
                # L1 col vs table L1 (convergence to table)
                if l1t is not None and t < l1t.size(0):
                    tv = F.normalize(l1t[t:t+1].float(), dim=-1)
                    cos_l1_table.extend((l1n[idx] * tv).sum(-1).tolist())
                # sibling-mean cosine (L0 input sim vs L1 sim)
                for j, c in enumerate(cols):
                    others = [cc for cc in cols if cc != c]
                    om0 = F.normalize(l0.float()[torch.tensor(others)].mean(0, keepdim=True), dim=-1)
                    om1 = F.normalize(l1.float()[torch.tensor(others)].mean(0, keepdim=True), dim=-1)
                    cos_l0_l0_sib.append(float((l0n[c:c+1] * om0).sum()))
                    cos_l1_l1_sib.append(float((l1n[c:c+1] * om1).sum()))
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(qids)} ({time.time()-t0:.0f}s)")

    def mean(L): return sum(L)/len(L) if L else None
    # ── Report ──
    print("\n=== TASK 1: intra-table MAD trajectory L0→L3 (M4 anchor) ===")
    print(f"{'layer':>6} {'hi-deg':>9} {'lo-deg':>9}  (baseline ref: hi L1=0.0136 L3=0.0046 / lo L1=0.1155 L3=0.0407)")
    LN = {0: "L0_PLM", 1: "L1_GAT", 2: "L2_GAT", 3: "L3_GAT"}
    task1 = {}
    for li in range(4):
        h, l = mean(layer_hi[li]), mean(layer_lo[li])
        task1[LN[li]] = dict(hi=h, lo=l)
        print(f"{LN[li]:>6} {h:>9.4f} {l:>9.4f}")
    # european_football_2 per-layer hi-deg
    print("\n  european_football_2 hi-deg per-layer (Player_Attributes/Match 115-col test bed):")
    ef = db_layer_hi.get("european_football_2", {})
    for li in range(4):
        v = mean(ef.get(li, []))
        print(f"    {LN[li]}: {v:.4f}" if v is not None else f"    {LN[li]}: --")

    print("\n=== TASK 2: self vs neighbor (hub columns; 구조: add_self_loops=False + per-layer residual 없음) ===")
    print(f"  cos(L1 col, table L1)         = {mean(cos_l1_table):.4f}   (1=공유 table 소스로 수렴)")
    print(f"  cos(L0 col, sibling-mean L0)  = {mean(cos_l0_l0_sib):.4f}   (input 유사도, PLM 384d)")
    print(f"  cos(L1 col, sibling-mean L1)  = {mean(cos_l1_l1_sib):.4f}   (L1 유사도, GAT 1024d)")
    print(f"  → Δsibling-cos (L1−L0) = {mean(cos_l1_l1_sib)-mean(cos_l0_l0_sib):+.4f} (양수=conv 가 동질화 가속)")

    out = dict(baseline_ref=dict(hi_L1=0.0136, hi_L3=0.0046, lo_L1=0.1155, lo_L3=0.0407),
               task1_trajectory=task1,
               task1_european_football_2={LN[li]: mean(ef.get(li, [])) for li in range(4)},
               task2_self_neighbor=dict(cos_l1_table=mean(cos_l1_table),
                                        cos_l0_sibling=mean(cos_l0_l0_sib), cos_l1_sibling=mean(cos_l1_l1_sib),
                                        delta_sibling_cos=mean(cos_l1_l1_sib)-mean(cos_l0_l0_sib),
                                        arch_note="add_self_loops=False + no per-layer residual → column self-transform 구조적 부재; conv output = pure neighbor (table+fk) message"),
               n_skip=n_skip, n_q=len(qids))
    with open(OUT / "v6_intra_collapse_origin_2026-06-06.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/v6_intra_collapse_origin_2026-06-06.json ({time.time()-t0:.0f}s, skip={n_skip})")


if __name__ == "__main__":
    main()
