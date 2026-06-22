import os,sys,json
sys.path.insert(0,"src")
from utils.evaluator import parse_sql_elements
DEV={d["question_id"]:d for d in json.load(open("data/raw/BIRD_dev/dev.json"))}
def nodes_to_pred(fn, gc):
    pt,pc=set(),set()
    for n in fn:
        if not isinstance(n,str) or "->" in n: continue
        if "." in n:
            t,c=n.split(".",1); pt.add(t.lower()); tcl=f"{t.lower()}.{c.lower()}"; pc.add(tcl if tcl in gc else c.lower())
        else: pt.add(n.lower())
    return pt,pc
print("| k | R(전) | P(전) | F1(전) | ext_nodes(전) |")
print("|---|---|---|---|---|")
for k in ["015","020","025","030","040","050","060","070","080","090"]:
    p=f"outputs/experiments/sonnet_rebaseline_2026_06_10/ste_nofilter/ste_k{k}_nofilter/predictions.jsonl"
    if not os.path.exists(p): print(f"| {int(k)} | (없음) |"); continue
    R=P=F=0.0; n=0; extsum=0
    for line in open(p):
        d=json.loads(line); qid=d.get("question_id")
        if qid not in DEV: continue
        gt,gc=parse_sql_elements(DEV[qid].get("SQL",""))
        gt={t.lower() for t in gt}; gc={c.lower() for c in gc}
        fn=d.get("final_nodes",[]); extsum+=len([x for x in fn if isinstance(x,str) and "->" not in x])
        pt,pc=nodes_to_pred(fn,gc); ga=gt|gc; pa=pt|pc
        if not ga: continue
        inter=len(ga&pa); rec=inter/len(ga); prec=inter/len(pa) if pa else 0.0
        R+=rec; P+=prec; F+=(2*rec*prec/(rec+prec) if rec+prec else 0.0); n+=1
    print(f"| {int(k)} | {R/n:.4f} | {P/n:.4f} | {F/n:.4f} | {extsum/n:.2f} |")
