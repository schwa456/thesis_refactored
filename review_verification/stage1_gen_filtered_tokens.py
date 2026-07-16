"""
Stage 1 보조 — '논문 의도 조건 (b)' 재조립: filter 출력(7.4노드) → generator 입력 토큰.

주의: m4_canonical 실행본은 실제로는 filter 출력을 generator 에 넣지 않았다
(generator 는 extractor 출력 ~75노드 수신 — stage1_reconstruct_prompts.py 로 확정).
이 스크립트는 '만약 논문 서술대로 7.4노드를 넣었다면' 의 generator 토큰을 산정한다.

소스: predictions.jsonl 의 pred_tables/pred_cols (= 실제 filter 출력, R/P/F1 보고에 사용된 것).
LLM 호출 없음. tiktoken cl100k 근사.
실행: PYTHONPATH=src python review_verification/stage1_gen_filtered_tokens.py
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")
def ntok(s): return len(ENC.encode(s or "", disallowed_special=()))

from modules.filters.agents import AgentUtils
from prompts.prompt_manager import PromptManager

PRED = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet/predictions.jsonl"
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
DEV_BY_ID = {d.get("question_id", i): d for i, d in enumerate(DEV)}

pm = PromptManager()
rows = []
n_nodes_list = []
for line in open(PRED, encoding="utf-8"):
    r = json.loads(line)
    qid = r["question_id"]
    pt = r.get("pred_tables", []) or []
    pc = r.get("pred_cols", []) or []
    # filter 출력 subgraph 재구성 (pred_cols: "table.col" 또는 "col")
    sg = {t: [] for t in pt}
    for c in pc:
        if "." in c:
            t, col = c.split(".", 1)
            sg.setdefault(t, []).append(col)
        else:
            # bare col → 첫 table 에 귀속 (토큰수는 귀속과 거의 무관)
            (sg[pt[0]].append(c) if pt else sg.setdefault("_", []).append(c))
    n_nodes = sum(len(v) if v else 1 for v in sg.values()) if sg else 0
    n_nodes_list.append(n_nodes)
    ddl = AgentUtils.generate_ddl(sg)
    item = DEV_BY_ID.get(qid, {})
    prompt = pm.load_prompt(file_name="sql_generator", section="sql_generator",
                            schema_str=ddl, evidence=item.get("evidence", "") or "(none)",
                            query=item.get("question", ""))
    rows.append(ntok(prompt))

avg = sum(rows) / len(rows)
print(f"[gen_filtered] {len(rows)} queries")
print(f"  filter 출력 노드수 평균: {sum(n_nodes_list)/len(n_nodes_list):.2f} (논문 7.4)")
print(f"  gen_filtered 입력 토큰 평균 (7.4노드→gen): {avg:.1f} tok  (cl100k 근사)")
