#!/usr/bin/env python3
"""Sonnet vs GLM addendum 2축 (DECISIONS 2026-06-10 #6 §사용자답변):
(a) EX ∝ ext_nodes invariance gate — Sonnet STE k-sweep Pearson, GLM era +0.9732 와 같은 방향인지.
(b) gold-removal taxonomy — matched config(STE) 에서 Sonnet 이 GLM 대비 추가 제거한 gold column 분류
    (PK/FK/numeric/text, table 내 위치) → systematic(prompt 회복) vs diffuse(model-intrinsic).
"""
import json, os, statistics, math
from collections import Counter, defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
SON = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10")
GLM = os.path.join(ROOT, "outputs/experiments/ste_topk_e2e_confirm_2026_06_09")
TABLES = "/SSL_NAS/peoples/khj/thesis/dev/dev_tables.json"
OUT = os.path.join(ROOT, "outputs/analysis")

# ── (a) invariance ──
# STE ext_nodes (deterministic, GLM/Sonnet 공유) + Sonnet batch_gate EX
STE = {"015": dict(ext=15.00), "020": dict(ext=19.81), "025": dict(ext=23.12)}
for k in STE:
    bg = os.path.join(SON, f"ste/ste_k{k}_sonnet/batch_gate_report.json")
    STE[k]["son_ex"] = json.load(open(bg))["EX"]
    gp = os.path.join(GLM, f"m4_ste_k{k}_e2e/predictions.jsonl")
    g = [json.loads(l) for l in open(gp)]
    STE[k]["glm_ex"] = statistics.mean(r["ex_score"] for r in g)


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else float("nan")


ks = ["015", "020", "025"]
ext = [STE[k]["ext"] for k in ks]
sex = [STE[k]["son_ex"] for k in ks]
gex = [STE[k]["glm_ex"] for k in ks]
print("=" * 80)
print("(a) EX ∝ ext_nodes invariance gate (Sonnet STE k-sweep)")
print("=" * 80)
print(f"  {'cell':8s} {'ext_nodes':>10} {'Sonnet EX':>10} {'GLM EX':>10}")
for k in ks:
    print(f"  ste_k{k:5s} {STE[k]['ext']:>10.2f} {STE[k]['son_ex']:>10.4f} {STE[k]['glm_ex']:>10.4f}")
r_son = pearson(ext, sex); r_glm = pearson(ext, gex)
print(f"  Pearson(ext_nodes, EX): Sonnet={r_son:+.4f}  GLM(same cells)={r_glm:+.4f}")
print(f"  GLM era ref (V6-W6 6 cells): Spearman(EX, ext_nodes)=+0.9732")
print(f"  → 방향 {'일치(EX∝width)' if r_son > 0 else '불일치'} : 3rd-backbone(Sonnet) LLM-invariance "
      f"{'성립' if r_son > 0.9 else '부분/약(n=3)'}")

# ── (b) gold-removal taxonomy ──
tabs = {db["db_id"]: db for db in json.load(open(TABLES))}


def col_meta(db_id):
    """display_name(lower) → dict(type,is_pk,is_fk,tbl_idx,ord_in_tbl,n_in_tbl)."""
    db = tabs.get(db_id)
    if not db:
        return {}
    pks = set()
    for p in db["primary_keys"]:
        (pks.update(p) if isinstance(p, list) else pks.add(p))
    fks = set()
    for a, b in db["foreign_keys"]:
        fks.add(a); fks.add(b)
    # ordinal within table
    tbl_counter = defaultdict(int); ordinal = {}
    for gi, (ti, nm) in enumerate(db["column_names"]):
        if ti < 0:
            continue
        ordinal[gi] = tbl_counter[ti]; tbl_counter[ti] += 1
    m = {}
    for gi, (ti, nm) in enumerate(db["column_names"]):
        if ti < 0:
            continue
        typ = db["column_types"][gi]
        cat = ("PK" if gi in pks else "FK" if gi in fks
               else "numeric" if typ in ("integer", "real") else "text" if typ == "text" else typ)
        m.setdefault(nm.strip().lower(), dict(type=typ, cat=cat, tbl=ti,
                                              ord=ordinal[gi], n_in_tbl=tbl_counter[ti]))
    # fix n_in_tbl (final count)
    for v in m.values():
        v["n_in_tbl"] = tbl_counter[v["tbl"]]
    return m


print("\n" + "=" * 80)
print("(b) gold-removal taxonomy — matched config STE (backbone isolated)")
print("=" * 80)
son_extra = Counter()       # Sonnet 가 GLM 대비 추가 제거한 gold (cat 별)
glm_extra = Counter()       # GLM 가 Sonnet 대비 추가 제거 (대칭 참조)
both_removed = Counter()    # 양쪽 다 제거한 gold (baseline)
pos_buckets = Counter()     # son_extra 의 table 내 위치 (early/mid/late)
examples = []
n_son_extra = n_glm_extra = n_both = n_gold = 0
for k in ks:
    sp = {r["question_id"]: r for r in
          (json.loads(l) for l in open(os.path.join(SON, f"ste/ste_k{k}_sonnet/predictions.jsonl")))}
    go = {r["question_id"]: r for r in
          (json.loads(l) for l in open(os.path.join(GLM, f"m4_ste_k{k}_e2e/output_m4_ste_k{k}_e2e.jsonl")))}
    for q in set(sp) & set(go):
        gold = set(c.strip().lower() for c in go[q].get("gold_cols", []))
        if not gold:
            continue
        s_pred = set(c.strip().lower() for c in sp[q].get("pred_cols", []))
        g_pred = set(c.strip().lower() for c in go[q].get("pred_cols", []))
        meta = col_meta(go[q]["db_id"])
        n_gold += len(gold)
        s_rm = gold - s_pred; g_rm = gold - g_pred
        se = s_rm - g_rm       # Sonnet 만 제거 (GLM 은 유지)
        ge = g_rm - s_rm       # GLM 만 제거
        bo = s_rm & g_rm
        n_son_extra += len(se); n_glm_extra += len(ge); n_both += len(bo)
        for c in se:
            mm = meta.get(c, dict(cat="unknown", ord=-1, n_in_tbl=0))
            son_extra[mm["cat"]] += 1
            if mm["n_in_tbl"]:
                frac = mm["ord"] / max(mm["n_in_tbl"] - 1, 1)
                pos_buckets["early(0-33%)" if frac < .33 else "mid(33-66%)" if frac < .66 else "late(66-100%)"] += 1
            if len(examples) < 12 and mm["cat"] != "unknown":
                examples.append((go[q]["db_id"], c, mm["cat"], k))
        for c in ge:
            glm_extra[meta.get(c, dict(cat="unknown"))["cat"]] += 1
        for c in bo:
            both_removed[meta.get(c, dict(cat="unknown"))["cat"]] += 1

print(f"gold col 총계(STE k015+020+025, matched)={n_gold}")
print(f"  Sonnet-만-제거(GLM 유지) = {n_son_extra}  ({100*n_son_extra/n_gold:.2f}% of gold)")
print(f"  GLM-만-제거(Sonnet 유지) = {n_glm_extra}  ({100*n_glm_extra/n_gold:.2f}% of gold)  [대칭 참조]")
print(f"  양쪽-제거(baseline)      = {n_both}  ({100*n_both/n_gold:.2f}% of gold)")
print(f"\n  Sonnet-extra 제거 gold 의 type 분포: {dict(son_extra)}")
print(f"  GLM-extra 제거 gold 의 type 분포   : {dict(glm_extra)}")
print(f"  양쪽-제거 gold 의 type 분포(baseline): {dict(both_removed)}")
print(f"  Sonnet-extra 의 table 내 위치       : {dict(pos_buckets)}")
print(f"\n  대표 Sonnet-extra 제거 사례:")
for db, c, cat, k in examples:
    print(f"    [{db}] '{c}'  ({cat}, ste_k{k})")

out = dict(
    invariance=dict(cells=ks, ext=ext, son_ex=sex, glm_ex=gex,
                    pearson_son=r_son, pearson_glm=r_glm, glm_era_ref=0.9732),
    taxonomy=dict(n_gold=n_gold, n_son_extra=n_son_extra, n_glm_extra=n_glm_extra, n_both=n_both,
                  son_extra_cat=dict(son_extra), glm_extra_cat=dict(glm_extra),
                  both_cat=dict(both_removed), son_extra_pos=dict(pos_buckets),
                  examples=examples),
)
with open(os.path.join(OUT, "sonnet_vs_glm_addendum_2026-06-10.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUT}/sonnet_vs_glm_addendum_2026-06-10.json")
