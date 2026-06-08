#!/usr/bin/env python3
"""V6-W5 Primary 게이트 — L1 hi-deg intra-table MAD (3 cells × s11).

W5-a column self-loop / W5-b per-layer residual / W5-c 조합. self-loop+residual 은
forward 위 적용 (conv 모듈 밖) → forward_hook 부정확. 모델 내장 capture_layerwise_outputs
(`model._captured_layer_outputs`, post-residual 진짜 layer 출력) 사용.

baseline: M4 anchor (qcond_nl3) L1 hi-deg intra-MAD=0.0136 / L0 PLM=0.4201 (lo-deg L1=0.1155)
  — v6_intra_table_collapse_origin_2026-06-06.md §2 정합 (재측정 0.0136 reproduce).
게이트: W5 cell A/B/C 중 L1 hi-deg intra-MAD 가 0.0136 → L0 0.4201 방향 회복?
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
R.V2_KW_WHITELIST |= {"v6w5_variant", "capture_layerwise_outputs"}
from v1_v5_retrospective_v6_metrics import (
    load_model_for_cell, load_dev, _build_col_table_index,
    DEVICE, DEV_DB_DIR, TABLES_JSON, ENCODER_MODEL_NAME, COL_TO_TAB_EDGE,
)
from modules.encoders.local_encoder import LocalPLMEncoder
from modules.builders.graph_builder import EnrichedHeteroGraphBuilder

OUT = ROOT / "outputs/analysis"
HI = 30
BASE_L1_HI = 0.0136   # M4 anchor L1 hi-deg intra-MAD (collapse base)
BASE_L0_PLM = 0.4201  # L0 raw PLM hi-deg (recovery target direction)
BASE_L1_LO = 0.1155   # lo-deg L1 ref
CELLS = [
    dict(tag="v6w5_a", label="column self-loop", ckpt_path="outputs/checkpoints/best_gat_v6w5_a_s11.pt", model_class="v2", r15=0.5732),
    dict(tag="v6w5_b", label="per-layer residual", ckpt_path="outputs/checkpoints/best_gat_v6w5_b_s11.pt", model_class="v2", r15=0.5715),
    dict(tag="v6w5_c", label="self-loop + residual", ckpt_path="outputs/checkpoints/best_gat_v6w5_c_s11.pt", model_class="v2", r15=0.5723),
]


def intra_mad_hilo(col_emb, cb_edge):
    _, t2c = _build_col_table_index(cb_edge)
    normed = F.normalize(col_emb.float(), dim=-1)
    hi, lo = [], []
    for t, cols in t2c.items():
        if len(cols) < 2:
            continue
        e = normed[torch.tensor(cols, dtype=torch.long)]
        n = len(cols); iu = torch.triu_indices(n, n, 1)
        d = (1.0 - e @ e.T)[iu[0], iu[1]]
        (hi if len(cols) > HI else lo).append(d)
    m = lambda L: float(torch.cat(L).mean()) if L else None
    return m(hi), m(lo)


def analyze(cell, dev, qids, enc, builder):
    print("=" * 76)
    print(f"[{cell['tag']}] {cell['label']} (R@15={cell['r15']:.4f})")
    model = load_model_for_cell(cell)
    model.capture_layerwise_outputs = True  # 내장 capture 강제 (post-residual 진짜 출력)
    db_cache = {}
    over_hi, over_lo = defaultdict(list), defaultdict(list)   # layer_idx -> [vals]
    db_hi = defaultdict(lambda: defaultdict(list))            # db -> layer -> [vals]
    n_skip = 0; t0 = time.time()
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
            with torch.no_grad():
                _ = model(xd, ed, query_emb=q)
            caps = model._captured_layer_outputs  # [L1, L2, L3] post-residual
        except Exception as e:
            if n_skip < 3:
                import traceback; traceback.print_exc()
            n_skip += 1; continue
        # L0 PLM (raw input column) + L1..L3 captured
        l0 = data.x_dict.get("column")
        if l0 is not None:
            h, l = intra_mad_hilo(l0, cb_cpu)
            if h is not None: over_hi[0].append(h); db_hi[db][0].append(h)
            if l is not None: over_lo[0].append(l)
        for li, cap in enumerate(caps, start=1):
            ce = cap.get("column")
            if ce is None: continue
            h, l = intra_mad_hilo(ce.cpu(), cb_cpu)
            if h is not None: over_hi[li].append(h); db_hi[db][li].append(h)
            if l is not None: over_lo[li].append(l)
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(qids)} ({time.time()-t0:.0f}s)")
    mean = lambda L: sum(L)/len(L) if L else None
    LN = {0: "L0_PLM", 1: "L1_GAT", 2: "L2_GAT", 3: "L3_GAT"}
    res = dict(tag=cell["tag"], label=cell["label"], r15=cell["r15"], n_skip=n_skip,
               trajectory={LN[li]: dict(hi=mean(over_hi[li]), lo=mean(over_lo[li])) for li in sorted(over_hi)},
               european_football_2={LN[li]: mean(db_hi.get("european_football_2", {}).get(li, [])) for li in range(4)})
    l1hi = res["trajectory"].get("L1_GAT", {}).get("hi")
    print(f"  → L1 hi-deg intra-MAD = {l1hi:.4f} (base 0.0136, target→L0 0.4201)  Δvs base = {l1hi-BASE_L1_HI:+.4f}")
    print(f"     L0={res['trajectory'].get('L0_PLM',{}).get('hi'):.4f} L1={l1hi:.4f} "
          f"L2={res['trajectory'].get('L2_GAT',{}).get('hi'):.4f} L3={res['trajectory'].get('L3_GAT',{}).get('hi'):.4f} (hi-deg)")
    ef = res["european_football_2"].get("L1_GAT")
    print(f"     european_football_2 L1 hi-deg = {ef:.4f}" if ef is not None else "     EF2 L1: --")
    print(f"     ({time.time()-t0:.0f}s, skip={n_skip})")
    return res


def main():
    print(f"Device: {DEVICE} | CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    dev = load_dev(); qids = list(range(len(dev)))
    enc = LocalPLMEncoder(model_name=ENCODER_MODEL_NAME)
    builder = EnrichedHeteroGraphBuilder(plm_model_name=ENCODER_MODEL_NAME, tables_json_path=str(TABLES_JSON))
    results = [analyze(c, dev, qids, enc, builder) for c in CELLS]
    print("\n=== SUMMARY: L1 hi-deg intra-MAD (base 0.0136 / L0 0.4201 / lo-deg 0.1155) ===")
    print(f"{'cell':10s} {'label':22s} {'L1 hi':>8} {'Δvs base':>9} {'L0':>8} {'L3 hi':>8} {'EF2 L1':>8}")
    for r in results:
        t = r["trajectory"]; l1 = t["L1_GAT"]["hi"]; l0 = t["L0_PLM"]["hi"]; l3 = t["L3_GAT"]["hi"]
        ef = r["european_football_2"]["L1_GAT"]
        print(f"{r['tag']:10s} {r['label']:22s} {l1:>8.4f} {l1-BASE_L1_HI:>+9.4f} {l0:>8.4f} {l3:>8.4f} "
              f"{ef:>8.4f}" if ef is not None else f"{r['tag']:10s} {r['label']:22s} {l1:>8.4f}")
    out = dict(baseline=dict(L1_hi=BASE_L1_HI, L0_PLM=BASE_L0_PLM, L1_lo=BASE_L1_LO), hi_thr=HI, cells=results)
    with open(OUT / "v6_w5_l1_mad_2026-06-07.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/v6_w5_l1_mad_2026-06-07.json")


if __name__ == "__main__":
    main()
