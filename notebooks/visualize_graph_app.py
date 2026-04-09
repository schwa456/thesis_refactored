"""
visualize_graph_app.py — Schema Linking Graph Visualizer (Streamlit)

선택한 (experiment, question_id) 의 로그/스코어를 파싱하여 그래프를 재구성하고,
사이드바의 PCST 하이퍼파라미터를 실시간으로 바꾸면서 Subgraph 추출 결과를
시각적으로 비교할 수 있는 도구.

실행:
    cd /home/hyeonjin/thesis_refactored
    streamlit run notebooks/visualize_graph_app.py --server.port 8501
"""

import os
import sys
import re
import ast
import json
import glob
import tempfile
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

# 프로젝트 루트 / src 경로 등록
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.extractors.pcst import PCSTExtractor, AdaptivePCSTExtractor  # noqa: E402

# ──────────────────────────────────────────────────────────────
# 경로 헬퍼
# ──────────────────────────────────────────────────────────────

OUTPUTS_DIR = os.path.join(ROOT, "outputs")
LOGS_DIR = os.path.join(ROOT, "logs")
BIRD_DEV_JSON = os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")


def discover_experiments() -> List[str]:
    """outputs/experiments + outputs/baselines 하위 디렉토리 리스트 반환."""
    exps = []
    for sub in ("experiments", "baselines"):
        d = os.path.join(OUTPUTS_DIR, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if os.path.isdir(full):
                exps.append(name)
    return exps


def _strip_prefix(exp_name: str) -> str:
    """experiment 디렉토리 안의 파일들은 'experiment_' 접두어가 제거되어 있다."""
    if exp_name.startswith("experiment_"):
        return exp_name[len("experiment_"):]
    return exp_name


def get_score_path(exp_name: str) -> Optional[str]:
    short = _strip_prefix(exp_name)
    if exp_name.startswith("experiment"):
        return os.path.join(OUTPUTS_DIR, "experiments", exp_name, f"score_analysis_{short}.jsonl")
    if exp_name.startswith("baseline") or exp_name.startswith("preliminary"):
        return os.path.join(OUTPUTS_DIR, "baselines", exp_name, f"score_analysis_{short}.jsonl")
    return None


def get_log_files(exp_name: str) -> List[str]:
    short = _strip_prefix(exp_name)
    sub = "experiments" if exp_name.startswith("experiment") else "baselines"
    pattern = os.path.join(LOGS_DIR, sub, exp_name, f"{short}_*.log")
    files = sorted(glob.glob(pattern))
    if files:
        return files
    # fallback: any .log
    return sorted(glob.glob(os.path.join(LOGS_DIR, sub, exp_name, "*.log")))


# ──────────────────────────────────────────────────────────────
# 파싱 (캐시)
# ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_qids_in_experiment(exp_name: str) -> List[int]:
    """score_analysis 파일에 등장하는 모든 question_id 정렬 반환."""
    score_path = get_score_path(exp_name)
    if not score_path or not os.path.exists(score_path):
        return []
    qids = set()
    with open(score_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data.get("query_id")
                if qid is not None:
                    qids.add(int(qid))
            except json.JSONDecodeError:
                continue
    return sorted(qids)


@st.cache_data(show_spinner=False)
def parse_score_analysis(exp_name: str, target_qid: int) -> Dict[str, Any]:
    """score_analysis_*.jsonl에서 target_qid에 해당하는 노드/스코어/gold 추출."""
    score_path = get_score_path(exp_name)
    out = {"node_scores": {}, "gold_schema": []}
    if not score_path or not os.path.exists(score_path):
        return out
    with open(score_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("query_id") != target_qid:
                continue
            name = data["node_name"]
            out["node_scores"][name] = data["score"]
            if data.get("is_gold"):
                out["gold_schema"].append(name)
    return out


_QUESTION_ANY_PAT = re.compile(r"Question\s+\d+:")
_LOG_TIME_PAT = re.compile(r"^\[20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")


def _parse_one_log(log_path: str, target_qid: int) -> Dict[str, Any]:
    """단일 .log 파일에서 target_qid 블록을 파싱."""
    parsed = {
        "question": "",
        "metadata": {},
        "seeds": [],
        "extracted_nodes": [],
        "final_nodes": [],
        "generated_sql": "",
    }
    is_target = False
    in_metadata_block = False
    in_subgraph_block = False
    in_sql_block = False
    metadata_str = ""
    subgraph_str = ""

    qstart_pat = re.compile(rf"Question\s+{target_qid}:")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if qstart_pat.search(line):
                    is_target = True
                    parsed["question"] = line.split(f"Question {target_qid}:")[-1].strip()
                    continue

                if is_target and _QUESTION_ANY_PAT.search(line) and not qstart_pat.search(line):
                    break
                if not is_target:
                    continue

                if in_sql_block:
                    if _LOG_TIME_PAT.match(line):
                        in_sql_block = False
                    else:
                        parsed["generated_sql"] += "\n" + line.strip()
                        continue

                if "metadata: {" in line:
                    metadata_str = line.split("metadata: ")[1].strip()
                    if metadata_str.count("{") == metadata_str.count("}") and metadata_str.count("[") == metadata_str.count("]"):
                        try:
                            parsed["metadata"] = ast.literal_eval(metadata_str)
                        except Exception:
                            pass
                    else:
                        in_metadata_block = True
                elif in_metadata_block:
                    metadata_str += line.strip()
                    if metadata_str.count("{") == metadata_str.count("}") and metadata_str.count("[") == metadata_str.count("]"):
                        in_metadata_block = False
                        try:
                            parsed["metadata"] = ast.literal_eval(metadata_str)
                        except Exception:
                            pass

                if "subgraph_dict: {" in line:
                    subgraph_str = line.split("subgraph_dict: ")[1].strip()
                    if subgraph_str.count("{") == subgraph_str.count("}") and subgraph_str.count("[") == subgraph_str.count("]"):
                        try:
                            parsed["extracted_nodes"] = _flatten_subgraph_dict(ast.literal_eval(subgraph_str))
                        except Exception:
                            pass
                    else:
                        in_subgraph_block = True
                elif in_subgraph_block:
                    subgraph_str += line.strip()
                    if subgraph_str.count("{") == subgraph_str.count("}") and subgraph_str.count("[") == subgraph_str.count("]"):
                        in_subgraph_block = False
                        try:
                            parsed["extracted_nodes"] = _flatten_subgraph_dict(ast.literal_eval(subgraph_str))
                        except Exception:
                            pass

                if "seeds: [" in line:
                    try:
                        parsed["seeds"] = ast.literal_eval(line.split("seeds: ")[1].strip())
                    except Exception:
                        pass

                if "Final Nodes: [" in line:
                    try:
                        parsed["final_nodes"] = ast.literal_eval(line.split("Final Nodes: ")[1].strip())
                    except Exception:
                        pass

                if "Generated SQL:" in line:
                    parsed["generated_sql"] = line.split("Generated SQL:")[1].strip()
                    in_sql_block = True
    except FileNotFoundError:
        pass

    return parsed


def _flatten_subgraph_dict(sg: Dict[str, List[str]]) -> List[str]:
    flat = []
    for tbl, cols in sg.items():
        flat.append(tbl)
        for c in cols:
            if "->" in str(c) or "." in str(c):
                flat.append(str(c))
            else:
                flat.append(f"{tbl}.{c}")
    return flat


@st.cache_data(show_spinner=False)
def parse_log(exp_name: str, target_qid: int) -> Dict[str, Any]:
    """experiment의 모든 log 파일을 훑어 가장 정보가 풍부한 결과를 반환."""
    logs = get_log_files(exp_name)
    best = None
    for lp in logs:
        p = _parse_one_log(lp, target_qid)
        if p["metadata"]:
            # metadata가 들어있는 첫 결과를 채택
            return p
        if best is None and p.get("question"):
            best = p
    return best or {"question": "", "metadata": {}, "seeds": [], "extracted_nodes": [], "final_nodes": [], "generated_sql": ""}


@st.cache_data(show_spinner=False)
def load_dev_meta() -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(BIRD_DEV_JSON):
        return {}
    with open(BIRD_DEV_JSON, "r", encoding="utf-8") as f:
        dev = json.load(f)
    return {int(d["question_id"]): d for d in dev}


# ──────────────────────────────────────────────────────────────
# 그래프 재구성 + Extractor 재실행
# ──────────────────────────────────────────────────────────────

def reconstruct_graph(metadata: Dict[str, Any], scores_by_name: Dict[str, float]) -> nx.Graph:
    g = nx.Graph()
    node_meta = metadata.get("node_metadata", {})
    edges = metadata.get("edges", [])
    edge_types = metadata.get("edge_types", [])

    if not node_meta:
        return g

    for idx, name in node_meta.items():
        idx_int = int(idx)
        n_type = "table" if "." not in str(name) and "->" not in str(name) else "column"
        score = round(float(scores_by_name.get(name, 0.0)), 4)
        g.add_node(name, idx=idx_int, name=name, type=n_type, similarity_score=score)

    for i, (u_idx, v_idx) in enumerate(edges):
        u_name = node_meta.get(u_idx) or node_meta.get(int(u_idx))
        v_name = node_meta.get(v_idx) or node_meta.get(int(v_idx))
        if u_name and v_name:
            etype = edge_types[i] if i < len(edge_types) else "relation"
            g.add_edge(u_name, v_name, type=etype)
    return g


def run_extractor(metadata: Dict[str, Any],
                  scores_by_name: Dict[str, float],
                  seeds_idx: List[int],
                  extractor_type: str,
                  hp: Dict[str, float]) -> List[str]:
    """선택한 PCST 변형을 현재 하이퍼파라미터로 실행하여 노드 이름 리스트 반환."""
    node_meta = metadata.get("node_metadata", {})
    if not node_meta:
        return []

    # idx → name 정렬 (graph_data 기준 인덱스 순서)
    n = max(int(k) for k in node_meta.keys()) + 1
    idx_to_name = {int(k): v for k, v in node_meta.items()}
    score_list = [float(scores_by_name.get(idx_to_name.get(i, ""), 0.0)) for i in range(n)]

    if extractor_type == "Basic PCST":
        extractor = PCSTExtractor(
            base_cost=hp["base_cost"],
            belongs_to_cost=hp["belongs_to_cost"],
            fk_cost=hp["fk_cost"],
            macro_cost=hp["macro_cost"],
            node_threshold=hp["node_threshold"],
        )
    else:
        extractor = AdaptivePCSTExtractor(
            base_cost=hp["base_cost"],
            belongs_to_cost=hp["belongs_to_cost"],
            fk_cost=hp["fk_cost"],
            macro_cost=hp["macro_cost"],
            node_threshold=0.0,
            percentile=hp["percentile"],
            min_prize_nodes=int(hp["min_prize_nodes"]),
            max_prize_nodes=int(hp["max_prize_nodes"]),
        )

    selected_idx, _ = extractor.extract(
        graph_data=metadata,
        node_scores=score_list,
        seed_nodes=seeds_idx,
    )
    return [idx_to_name.get(int(i), str(i)) for i in selected_idx]


# ──────────────────────────────────────────────────────────────
# pyvis 렌더링 (graph_visualizer.py 와 동일 색상 규칙)
# ──────────────────────────────────────────────────────────────

def render_pyvis(graph: nx.Graph,
                 question: str,
                 seeds: set,
                 extracted: set,
                 final: set,
                 gold: set) -> str:
    net = Network(height="750px", width="100%", bgcolor="#111827",
                  font_color="white", directed=True)

    for node_id, data in graph.nodes(data=True):
        nid = str(node_id).strip()
        node_type = data.get("type", "column")
        shape = "box" if node_type == "table" else "dot"
        size = 30 if node_type == "table" else 15

        if nid in final and nid in gold:
            bg = "#10B981"; shape = "star"; size = 50; cat = "TP"
        elif nid in final:
            bg = "#EF4444"; size = 15; cat = "FP"
        elif nid in gold:
            bg = "#3B82F6"; shape = "diamond"; size = 40; cat = "FN"
        elif nid in seeds:
            bg = "#F59E0B"; cat = "Seed (filtered out)"
        else:
            bg = "#4B5563"; cat = "Unselected"

        in_extracted = nid in extracted
        if in_extracted:
            color = {
                "background": bg,
                "border": "#22D3EE",
                "highlight": {"background": bg, "border": "#67E8F9"},
            }
            border_width = 5
            if size < 20:
                size = 22
        else:
            color = bg
            border_width = 1

        title = (
            f"Name: {data.get('name', nid)}\n"
            f"Type: {node_type.upper()}\n"
            f"Score: {data.get('similarity_score', 0.0)}\n"
            f"In PCST: {in_extracted}\n"
            f"Category: {cat}\n"
        )

        net.add_node(node_id, label=str(node_id), title=title, color=color,
                     shape=shape, size=size, borderWidth=border_width)

    for u, v, edata in graph.edges(data=True):
        su, sv = str(u).strip(), str(v).strip()
        if su in final and sv in final:
            ec, w = "#10B981", 3
        elif su in extracted and sv in extracted:
            ec, w = "#22D3EE", 2
        elif su in seeds and sv in seeds:
            ec, w = "#F59E0B", 1
        else:
            ec, w = "#374151", 1
        net.add_edge(u, v, color=ec, width=w, title=edata.get("type", ""))

    if question:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(question, width=70))
        net.add_node(
            "__QUESTION__", label=f"Q: {wrapped}",
            shape="text", font={"size": 24, "color": "#E5E7EB", "align": "center"},
            x=0, y=-900, physics=False, fixed=True,
        )

    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 150
        }
      },
      "interaction": { "zoomView": true, "dragView": true }
    }
    """)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        tmp_path = f.name
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)

    custom_css = """
    <style>
    .vis-tooltip {
      white-space: pre-wrap !important;
      max-width: 400px !important;
      word-wrap: break-word !important;
      padding: 10px !important;
      font-family: monospace !important;
      font-size: 13px !important;
      background-color: #1e293b !important;
      color: #e2e8f0 !important;
      border: 1px solid #475569 !important;
      border-radius: 6px !important;
    }
    </style>
    </head>
    """
    return html.replace("</head>", custom_css)


# ──────────────────────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────────────────────

def metrics(pred: set, gold: set) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return r, p, f


def main():
    st.set_page_config(page_title="Schema Linking Graph Visualizer", layout="wide")
    st.title("Schema Linking Graph Visualizer")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Experiment")
        experiments = discover_experiments()
        if not experiments:
            st.error("No experiments found under outputs/")
            return
        default_idx = experiments.index("experiment_b4_xiyan_filter") if "experiment_b4_xiyan_filter" in experiments else 0
        exp_name = st.selectbox("Experiment / Baseline", experiments, index=default_idx)

        qids = list_qids_in_experiment(exp_name)
        if not qids:
            st.error(f"No question_ids found in {exp_name}")
            return

        st.header("Question")
        qid = st.selectbox(f"Question ID ({len(qids)} total)", qids, index=0)

        st.header("Extractor")
        extractor_type = st.radio("Type", ["Basic PCST", "Adaptive PCST"], index=1)

        st.header("Hyperparameters")
        hp = {}
        hp["base_cost"] = st.slider("base_cost", 0.0, 2.0, 0.05, 0.01)
        hp["belongs_to_cost"] = st.slider("belongs_to_cost", 0.0, 1.0, 0.01, 0.005)
        hp["fk_cost"] = st.slider("fk_cost", 0.0, 1.0, 0.05, 0.01)
        hp["macro_cost"] = st.slider("macro_cost", 0.0, 2.0, 0.5, 0.05)

        if extractor_type == "Basic PCST":
            hp["node_threshold"] = st.slider("node_threshold", 0.0, 1.0, 0.1, 0.01)
        else:
            hp["percentile"] = st.slider("percentile", 50.0, 99.0, 80.0, 1.0)
            hp["min_prize_nodes"] = st.slider("min_prize_nodes", 1, 20, 3, 1)
            hp["max_prize_nodes"] = st.slider("max_prize_nodes", 5, 100, 25, 1)

    # ── 데이터 로드 ──
    score_data = parse_score_analysis(exp_name, qid)
    log_data = parse_log(exp_name, qid)
    dev_meta = load_dev_meta().get(qid, {})

    metadata = log_data.get("metadata", {})
    if not metadata:
        st.error(f"`metadata: {{...}}` 를 로그에서 찾을 수 없습니다. ({exp_name} / qid={qid})\n"
                 f"해당 실험은 디버그 로그가 없을 수 있습니다.")
        return

    scores_by_name = score_data["node_scores"]
    gold_set = {str(x).strip() for x in score_data["gold_schema"]}

    # 원본(로그) seeds / extracted / final
    node_meta = metadata.get("node_metadata", {})
    seed_idx_list = log_data.get("seeds", []) or []
    seeds_text = [node_meta.get(int(i), str(i)) for i in seed_idx_list]
    seeds_set = {str(x).strip() for x in seeds_text}
    final_set = {str(x).strip() for x in (log_data.get("final_nodes") or [])}
    original_extracted_set = {str(x).strip() for x in (log_data.get("extracted_nodes") or [])}

    # 라이브 PCST 재실행
    live_extracted = run_extractor(metadata, scores_by_name, seed_idx_list, extractor_type, hp)
    live_extracted_set = {str(x).strip() for x in live_extracted}

    # ── 상단 정보 ──
    col_q, col_db = st.columns([3, 1])
    with col_q:
        st.markdown(f"**Question:** {log_data.get('question') or dev_meta.get('question', '(unknown)')}")
        if dev_meta.get("SQL"):
            st.code(dev_meta["SQL"], language="sql")
    with col_db:
        st.metric("DB", dev_meta.get("db_id", "?"))
        st.metric("Difficulty", dev_meta.get("difficulty", "?"))

    # ── 메트릭 비교 ──
    st.subheader("Metrics — Live PCST vs. Original Run")
    r_live, p_live, f_live = metrics(live_extracted_set, gold_set)
    r_orig, p_orig, f_orig = metrics(original_extracted_set, gold_set)
    r_final, p_final, f_final = metrics(final_set, gold_set)

    cols = st.columns(4)
    cols[0].metric("Graph", f"{len(node_meta)} nodes")
    cols[1].metric("Gold", f"{len(gold_set)}")
    cols[2].metric("Seeds", f"{len(seeds_set)}")
    cols[3].metric("Final (after Filter)", f"{len(final_set)}")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Live PCST (current sliders)**")
        st.metric("|extracted|", len(live_extracted_set))
        st.metric("Recall", f"{r_live:.3f}")
        st.metric("Precision", f"{p_live:.3f}")
        st.metric("F1", f"{f_live:.3f}")
    with cols[1]:
        st.markdown("**Original PCST (from log)**")
        st.metric("|extracted|", len(original_extracted_set))
        st.metric("Recall", f"{r_orig:.3f}")
        st.metric("Precision", f"{p_orig:.3f}")
        st.metric("F1", f"{f_orig:.3f}")
    with cols[2]:
        st.markdown("**Final (after Filter)**")
        st.metric("|final|", len(final_set))
        st.metric("Recall", f"{r_final:.3f}")
        st.metric("Precision", f"{p_final:.3f}")
        st.metric("F1", f"{f_final:.3f}")

    # ── 그래프 렌더 ──
    st.subheader("Graph (Live PCST highlighted with cyan border)")
    nx_graph = reconstruct_graph(metadata, scores_by_name)
    if nx_graph.number_of_nodes() == 0:
        st.warning("그래프 메타데이터를 재구성할 수 없습니다.")
        return

    html = render_pyvis(nx_graph,
                        question=log_data.get("question", ""),
                        seeds=seeds_set,
                        extracted=live_extracted_set,
                        final=final_set,
                        gold=gold_set)
    components.html(html, height=820, scrolling=True)

    # ── 범례 ──
    with st.expander("Legend", expanded=False):
        st.markdown("""
        - **🟢 Green Star** — TP (Final ∩ Gold)
        - **🔴 Red** — FP (Final, not Gold)
        - **🔷 Blue Diamond** — FN (Gold, missed)
        - **🟡 Yellow** — Seed only (filtered out)
        - **⚫ Gray** — Unselected
        - **🟦 Cyan border (width=5)** — Live PCST 가 선택한 Subgraph
        - Edge colors: Green (Final) > Cyan (PCST) > Yellow (Seeds) > Gray (default)
        """)

    # ── 노드 테이블 ──
    with st.expander("Node Details", expanded=False):
        try:
            import pandas as pd
            rows = []
            for nid, data in nx_graph.nodes(data=True):
                snid = str(nid).strip()
                rows.append({
                    "name": snid,
                    "type": data.get("type"),
                    "score": data.get("similarity_score"),
                    "gold": snid in gold_set,
                    "seed": snid in seeds_set,
                    "extracted_live": snid in live_extracted_set,
                    "extracted_orig": snid in original_extracted_set,
                    "final": snid in final_set,
                })
            df = pd.DataFrame(rows).sort_values("score", ascending=False)
            st.dataframe(df, use_container_width=True, height=420)
        except ImportError:
            st.info("pandas 가 없어 테이블 표시를 건너뜁니다.")


if __name__ == "__main__":
    main()
