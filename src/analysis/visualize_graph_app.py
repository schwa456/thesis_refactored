"""
visualize_graph_app.py — Schema Linking Graph Visualizer (Streamlit)

선택한 (experiment, question_id) 의 로그/스코어를 파싱하여 그래프를 재구성하고,
사이드바의 Extractor / 하이퍼파라미터를 실시간으로 바꾸면서 Subgraph 추출 결과를
시각적으로 비교할 수 있는 도구.

주요 기능:
  - 모든 experiment 자동 탐색 (outputs/experiments/, baselines/, root)
  - 모든 등록된 Extractor 라이브 재실행 (PCST 변형 8+, MST/Steiner 3종, TopK)
  - 임의 N개 실험을 골라 비교 (구 8-cell ablation 하드코딩 제거)
  - 실험 config 자동 로드 → Selector / Filter / Builder 설정 caption 표시

실행:
    cd /home/hyeonjin/thesis_refactored
    streamlit run src/analysis/visualize_graph_app.py --server.port 8501
"""

import os
import sys
import re
import ast
import json
import glob
import tempfile
import types as _types
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

# ──────────────────────────────────────────────────────────────
# 프로젝트 루트 / src 경로 등록
# ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# modules/__init__.py 가 builders→encoders→… 를 전부 import 하여
# torch_geometric, sentence_transformers 등 무거운 의존성을 끌어오므로,
# modules 패키지를 빈 껍데기로 선점한 뒤 pcst/mst 만 직접 로드한다.
for _mod_name in ["modules", "modules.builders", "modules.encoders",
                  "modules.projectors", "modules.selectors",
                  "modules.filters", "modules.generators"]:
    if _mod_name not in sys.modules:
        _stub = _types.ModuleType(_mod_name)
        if _mod_name == "modules":
            _stub.__path__ = [os.path.join(SRC, "modules")]
            _stub.__package__ = "modules"
        sys.modules[_mod_name] = _stub

# Lightweight imports (no heavy deps)
from modules.registry import register  # noqa: E402

# Extractors — 모두 한 번에 import (torch / pcst_fast / networkx 만 필요)
from modules.extractors.pcst import (  # noqa: E402
    PCSTExtractor,
    AdaptivePCSTExtractor,
    ProductCostPCSTExtractor,
    ScoreDrivenPCSTExtractor,
    ComponentAwareAdaptivePCSTExtractor,
    ComponentAwareProductCostPCSTExtractor,
    TopologyCostPCSTExtractor,
    ComponentAwareTopologyCostPCSTExtractor,
    DynamicPCSTExtractor,
)
from modules.extractors.mst import MSTExtractor  # noqa: E402
from modules.extractors.mst_kruskal import MSTKruskalExtractor  # noqa: E402
from modules.extractors.mst_pcst_union import MSTPCSTUnionExtractor  # noqa: E402
from modules.extractors.baseline import TopKExtractor  # noqa: E402

# ──────────────────────────────────────────────────────────────
# 경로 헬퍼
# ──────────────────────────────────────────────────────────────

OUTPUTS_DIR = os.path.join(ROOT, "outputs")
LOGS_DIR = os.path.join(ROOT, "logs")
CONFIGS_DIR = os.path.join(ROOT, "configs")
BIRD_DEV_JSON = os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")
BIRD_DEV_TABLES_JSON = os.path.join(ROOT, "data/raw/BIRD_dev/dev_tables.json")


# ──────────────────────────────────────────────────────────────
# Curated comparison presets (선택만 하면 자동으로 묶여서 비교됨)
# 새 실험을 추가하려면 여기에 표시명 + 실험 디렉토리만 등록하면 된다.
# ──────────────────────────────────────────────────────────────

COMPARISON_PRESETS: Dict[str, List[Tuple[str, str]]] = {
    "Paper main pipeline (Wave 3)": [
        ("Main: Enriched + QCond + MST Kruskal + XiYan(GLM)",
         "s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm"),
        ("Sub: Plain Ens + MST Kruskal + XiYan(GLM)",
         "s04_ablation/extractor/plain_ens_a05_mst_kruskal_glm"),
        ("Selector only (no Extractor, no Filter)",
         "s04_ablation/pipeline/enriched_qcond_a05_selector_only"),
        ("No Extractor + Filter only",
         "s04_ablation/pipeline/enriched_qcond_a05_no_extractor_glm"),
    ],
    "Extractor ablation (Plain Ens α=0.5 + GLM)": [
        ("Adaptive PCST",
         "s04_ablation/extractor/plain_ens_a05_adaptive_glm"),
        ("MST Kruskal",
         "s04_ablation/extractor/plain_ens_a05_mst_kruskal_glm"),
        ("MST ∪ PCST (Union)",
         "s04_ablation/extractor/plain_ens_a05_mst_pcst_union_glm"),
        ("Steiner 2-approx (topk seed)",
         "s04_ablation/extractor/plain_ens_a05_steiner_glm"),
        ("Steiner 2-approx (threshold seed)",
         "s04_ablation/extractor/plain_ens_a05_steiner_threshold_glm"),
        ("MST (true Kruskal of induced)",
         "s04_ablation/extractor/plain_ens_a05_mst_glm"),
    ],
    "Stagewise α-sweep (Plain GLM)": [
        ("α=0 (cosine only)",
         "s04_ablation/stagewise/plain_cos_a1_glm"),
        ("α=0.5 (Plain ensemble)",
         "s04_ablation/stagewise/plain_ens_a05_glm"),
        ("α=1 (GAT only)",
         "s04_ablation/stagewise/plain_gat_a0_glm"),
    ],
    "Old 2x2x2 (vLLM era reference)": [
        ("#1 C+B+N — Cosine · Basic PCST · No Filter",
         "s01_vector_only/a01_basic_pcst/s01_a01_02_raw_pcst_baseline"),
        ("#2 C+B+X — Cosine · Basic PCST · XiYan",
         "abl/a01_2x2x2_selector_extractor_filter/abl_a01_05_cos_basic_xiyan"),
        ("#3 C+A+N — Cosine · Adaptive PCST · No Filter",
         "s01_vector_only/a02_adaptive_pcst/s01_a02_01_adaptive_pcst"),
        ("#4 C+A+X — Cosine · Adaptive PCST · XiYan",
         "abl/a01_2x2x2_selector_extractor_filter/abl_a01_07_cos_adaptive_xiyan"),
        ("#5 E+B+N — Ensemble · Basic PCST · No Filter",
         "s03_gat_ensemble/a01_basic_pcst/s03_a01_01_ensemble_basic"),
        ("#6 E+B+X — Ensemble · Basic PCST · XiYan",
         "abl/a01_2x2x2_selector_extractor_filter/abl_a01_06_ens_basic_xiyan"),
        ("#7 E+A+N — Ensemble · Adaptive PCST · No Filter",
         "s03_gat_ensemble/a02_adaptive_pcst/s03_a02_01_combined"),
        ("#8 E+A+X — Ensemble · Adaptive PCST · XiYan",
         "s03_gat_ensemble/a02_adaptive_pcst/s03_a02_03_xiyan_filter"),
    ],
}


# ──────────────────────────────────────────────────────────────
# 실험 / 로그 / config 탐색
# ──────────────────────────────────────────────────────────────

def _find_score_analysis(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    for f in os.listdir(directory):
        if f.startswith("score_analysis_") and f.endswith(".jsonl"):
            return os.path.join(directory, f)
    return None


def _walk_experiment_dirs(base: str, max_depth: int = 5) -> List[Tuple[str, str]]:
    results = []
    base = os.path.abspath(base)

    def _recurse(current: str, depth: int, prefix: str):
        if depth > max_depth or not os.path.isdir(current):
            return
        if _find_score_analysis(current):
            name = prefix or os.path.basename(current)
            results.append((name, current))
            return
        for entry in sorted(os.listdir(current)):
            child = os.path.join(current, entry)
            if os.path.isdir(child) and not entry.startswith(".") and entry not in (
                "analysis", "archive", "checkpoints", "configs", "logs", "no_filter"
            ):
                child_prefix = f"{prefix}/{entry}" if prefix else entry
                _recurse(child, depth + 1, child_prefix)

    _recurse(base, 0, "")
    return results


@st.cache_data(show_spinner=False, ttl=300)
def discover_all_experiments() -> Dict[str, str]:
    """모든 실험 디렉토리를 탐색하여 {display_name: abs_path} dict 반환."""
    found: Dict[str, str] = {}
    skip_root = {"experiments", "baselines", "analysis", "archive",
                 "checkpoints", "configs", "logs", "summary_all.csv"}

    for name, path in _walk_experiment_dirs(os.path.join(OUTPUTS_DIR, "experiments")):
        found[name] = path

    for name, path in _walk_experiment_dirs(os.path.join(OUTPUTS_DIR, "baselines")):
        found[f"baselines/{name}"] = path

    if os.path.isdir(OUTPUTS_DIR):
        for entry in sorted(os.listdir(OUTPUTS_DIR)):
            if entry in skip_root:
                continue
            full = os.path.join(OUTPUTS_DIR, entry)
            if os.path.isdir(full) and _find_score_analysis(full):
                if entry not in found:
                    found[entry] = full

    return dict(sorted(found.items()))


def get_score_path(exp_name: str) -> Optional[str]:
    all_exps = discover_all_experiments()
    if exp_name in all_exps:
        return _find_score_analysis(all_exps[exp_name])

    short = exp_name[len("experiment_"):] if exp_name.startswith("experiment_") else exp_name
    for candidate_dir in [
        os.path.join(OUTPUTS_DIR, "experiments", exp_name),
        os.path.join(OUTPUTS_DIR, "baselines", exp_name),
        os.path.join(OUTPUTS_DIR, exp_name),
    ]:
        sa = _find_score_analysis(candidate_dir)
        if sa:
            return sa
    _ = short
    return None


def _resolve_log_dir(exp_name: str) -> Optional[str]:
    all_exps = discover_all_experiments()
    if exp_name in all_exps:
        abs_path = all_exps[exp_name]
        rel = os.path.relpath(abs_path, OUTPUTS_DIR)
        log_candidate = os.path.join(LOGS_DIR, rel)
        if os.path.isdir(log_candidate):
            return log_candidate
        flat_candidate = os.path.join(LOGS_DIR, os.path.basename(abs_path))
        if os.path.isdir(flat_candidate):
            return flat_candidate
    for sub in ("experiments", "baselines", ""):
        d = os.path.join(LOGS_DIR, sub, exp_name) if sub else os.path.join(LOGS_DIR, exp_name)
        if os.path.isdir(d):
            return d
    return None


def get_log_files(exp_name: str) -> List[str]:
    log_dir = _resolve_log_dir(exp_name)
    if not log_dir:
        return []
    return sorted(glob.glob(os.path.join(log_dir, "*.log")))


@st.cache_data(show_spinner=False, ttl=300)
def get_metrics_summary(exp_name: str) -> Dict[str, Any]:
    """metrics.txt 의 R/P/F1/EX/시간 요약 반환."""
    all_exps = discover_all_experiments()
    if exp_name not in all_exps:
        return {}
    metrics_path = os.path.join(all_exps[exp_name], "metrics.txt")
    if not os.path.exists(metrics_path):
        return {}
    out: Dict[str, Any] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


@st.cache_data(show_spinner=False, ttl=300)
def get_config_for_experiment(exp_name: str) -> Dict[str, Any]:
    """configs/<exp_name>.yaml 을 찾아 dict 로 반환 (yaml 미설치시 raw text)."""
    candidates = [
        os.path.join(CONFIGS_DIR, f"{exp_name}.yaml"),
        os.path.join(CONFIGS_DIR, "experiments", f"{exp_name.replace('experiments/', '', 1)}.yaml")
        if exp_name.startswith("experiments/") else None,
    ]
    candidates = [c for c in candidates if c]

    # 일반 케이스: outputs path 의 'experiments/...' 구조와 동일
    if not exp_name.startswith("experiments/") and not exp_name.startswith("baselines/"):
        candidates.append(os.path.join(CONFIGS_DIR, "experiments", f"{exp_name}.yaml"))

    for path in candidates:
        if path and os.path.exists(path):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                with open(path, "r", encoding="utf-8") as f:
                    return {"_raw_text": f.read()}
            except Exception as e:
                return {"_error": str(e)}
    return {}


# ──────────────────────────────────────────────────────────────
# JSONL 파싱 (캐시)
# ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_qids_in_experiment(exp_name: str) -> List[int]:
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
    score_path = get_score_path(exp_name)
    out: Dict[str, Any] = {"node_scores": {}, "gold_schema": []}
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


def _strip_array_literals(s: str) -> str:
    """Replace `array(...)` (numpy repr) with `None`, tracking nested parens.

    파이프라인 로그가 numpy.ndarray repr 을 metadata 안에 포함하면
    ast.literal_eval 이 실패하므로, parsing 전에 None 으로 치환한다.
    """
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i:i+6] == "array(":
            depth = 1
            j = i + 6
            while j < n and depth > 0:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            out.append("None")
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _safe_literal_eval(s: str) -> Optional[Any]:
    """numpy 등을 stub 처리한 뒤 ast.literal_eval. 실패 시 None."""
    try:
        return ast.literal_eval(_strip_array_literals(s))
    except Exception:
        return None


def _parse_one_log(log_path: str, target_qid: int) -> Dict[str, Any]:
    """단일 .log 파일에서 target_qid 블록 파싱.

    metadata / subgraph_dict 가 numpy array 등 multi-line repr 을 가질 수 있어,
    timestamp 라인이 다시 나오기 전까지 continuation 으로 누적한 뒤 파싱한다.
    """
    parsed: Dict[str, Any] = {
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
    metadata_buf = ""
    subgraph_buf = ""

    qstart_pat = re.compile(rf"Question\s+{target_qid}:")

    def _try_finalize_metadata():
        if not metadata_buf:
            return
        d = _safe_literal_eval(metadata_buf)
        if isinstance(d, dict):
            parsed["metadata"] = d

    def _try_finalize_subgraph():
        if not subgraph_buf:
            return
        d = _safe_literal_eval(subgraph_buf)
        if isinstance(d, dict):
            parsed["extracted_nodes"] = _flatten_subgraph_dict(d)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # 새 timestamp 라인이 시작되면 continuation block 종료
                if _LOG_TIME_PAT.match(line):
                    if in_metadata_block:
                        in_metadata_block = False
                        _try_finalize_metadata()
                        metadata_buf = ""
                    if in_subgraph_block:
                        in_subgraph_block = False
                        _try_finalize_subgraph()
                        subgraph_buf = ""
                    if in_sql_block:
                        in_sql_block = False

                if qstart_pat.search(line):
                    is_target = True
                    parsed["question"] = line.split(f"Question {target_qid}:")[-1].strip()
                    continue

                if is_target and _QUESTION_ANY_PAT.search(line) and not qstart_pat.search(line):
                    # 다음 question 시작 → 현재 진행중인 블록 마무리
                    if in_metadata_block:
                        _try_finalize_metadata()
                    if in_subgraph_block:
                        _try_finalize_subgraph()
                    break
                if not is_target:
                    continue

                if in_sql_block:
                    parsed["generated_sql"] += "\n" + line.rstrip("\n")
                    continue

                if in_metadata_block:
                    metadata_buf += " " + line.rstrip("\n").strip()
                    continue

                if in_subgraph_block:
                    subgraph_buf += " " + line.rstrip("\n").strip()
                    continue

                if "metadata: {" in line:
                    metadata_buf = line.split("metadata: ", 1)[1].rstrip("\n")
                    in_metadata_block = True
                    continue

                if "subgraph_dict: {" in line:
                    subgraph_buf = line.split("subgraph_dict: ", 1)[1].rstrip("\n")
                    in_subgraph_block = True
                    continue

                if "seeds: [" in line:
                    val = _safe_literal_eval(line.split("seeds: ", 1)[1].strip())
                    if isinstance(val, list):
                        parsed["seeds"] = val

                if "Final Nodes: [" in line:
                    val = _safe_literal_eval(line.split("Final Nodes: ", 1)[1].strip())
                    if isinstance(val, list):
                        parsed["final_nodes"] = val

                if "Generated SQL:" in line:
                    parsed["generated_sql"] = line.split("Generated SQL:", 1)[1].strip()
                    in_sql_block = True

            # EOF: continuation 진행중이었다면 마무리
            if in_metadata_block:
                _try_finalize_metadata()
            if in_subgraph_block:
                _try_finalize_subgraph()
    except FileNotFoundError:
        pass

    return parsed


@st.cache_data(show_spinner=False)
def parse_log(exp_name: str, target_qid: int) -> Dict[str, Any]:
    logs = get_log_files(exp_name)
    best = None
    for lp in logs:
        p = _parse_one_log(lp, target_qid)
        if p["metadata"]:
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
# Plain Schema Graph (실험 무관, dev_tables.json 만 사용)
# ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dev_tables() -> Dict[str, Dict[str, Any]]:
    """dev_tables.json → {db_id: schema_dict}."""
    if not os.path.exists(BIRD_DEV_TABLES_JSON):
        return {}
    with open(BIRD_DEV_TABLES_JSON, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def build_plain_schema_graph(db_id: str,
                             include_fk_node: bool = True,
                             include_t2t: bool = True,
                             use_natural_names: bool = False) -> nx.Graph:
    """dev_tables.json 의 스키마로부터 plain NetworkX 그래프 생성.

    노드:
      - table  (table_names_original 또는 table_names)
      - column (table.col 형식)
      - fk_node (FK pair 당 1개, "src_table.src_col->dst_table.dst_col")  — optional

    엣지:
      - belongs_to: column → table
      - is_source_of: src_column → fk_node
      - points_to: fk_node → dst_column
      - table_to_table: src_table ↔ dst_table  — optional
    """
    g = nx.Graph()
    schemas = load_dev_tables()
    if db_id not in schemas:
        return g

    s = schemas[db_id]
    table_names_orig = s["table_names_original"]
    table_names_nl = s.get("table_names", table_names_orig)
    column_pairs_orig = s["column_names_original"]  # [(table_idx, col), ...]
    column_pairs_nl = s.get("column_names", column_pairs_orig)
    foreign_keys = s.get("foreign_keys", [])
    primary_keys_raw = s.get("primary_keys", [])
    primary_keys: set = set()
    for pk in primary_keys_raw:
        if isinstance(pk, list):
            primary_keys.update(pk)
        else:
            primary_keys.add(pk)

    table_disp = table_names_nl if use_natural_names else table_names_orig
    col_disp = column_pairs_nl if use_natural_names else column_pairs_orig

    # tables
    for tidx, tname in enumerate(table_disp):
        g.add_node(str(tname), name=str(tname), type="table",
                   table_idx=tidx, similarity_score=0.0,
                   nl_name=table_names_nl[tidx])

    # columns (skip * pseudo-column at index 0)
    col_id_to_name: Dict[int, str] = {}
    for cidx, (tidx, cname) in enumerate(col_disp):
        if tidx < 0:
            col_id_to_name[cidx] = "*"
            continue
        full = f"{table_disp[tidx]}.{cname}"
        col_id_to_name[cidx] = full
        is_pk = cidx in primary_keys
        g.add_node(full, name=full, type="column",
                   col_idx=cidx, similarity_score=0.0,
                   is_pk=is_pk,
                   nl_name=str(column_pairs_nl[cidx][1]) if cidx < len(column_pairs_nl) else cname)
        # belongs_to edge
        g.add_edge(full, str(table_disp[tidx]), type="belongs_to")

    # FK edges + optional fk_node + table_to_table
    seen_t2t = set()
    for fk in foreign_keys:
        if not isinstance(fk, list) or len(fk) != 2:
            continue
        src_cid, dst_cid = fk
        if src_cid not in col_id_to_name or dst_cid not in col_id_to_name:
            continue
        src_col = col_id_to_name[src_cid]
        dst_col = col_id_to_name[dst_cid]
        # src/dst tables (skip * column)
        src_tidx = column_pairs_orig[src_cid][0]
        dst_tidx = column_pairs_orig[dst_cid][0]
        if src_tidx < 0 or dst_tidx < 0:
            continue
        src_t = str(table_disp[src_tidx])
        dst_t = str(table_disp[dst_tidx])

        if include_fk_node:
            fk_name = f"{src_col}->{dst_col}"
            g.add_node(fk_name, name=fk_name, type="fk_node",
                       similarity_score=0.0)
            g.add_edge(src_col, fk_name, type="is_source_of")
            g.add_edge(fk_name, dst_col, type="points_to")

        if include_t2t and src_t != dst_t:
            key = tuple(sorted([src_t, dst_t]))
            if key not in seen_t2t:
                g.add_edge(src_t, dst_t, type="table_to_table")
                seen_t2t.add(key)

    return g


def render_plain_pyvis(graph: nx.Graph, title: str = "") -> str:
    """실험 annotation 없이 plain 한 색상으로 schema graph 렌더링."""
    net = Network(height="800px", width="100%", bgcolor="#111827",
                  font_color="white", directed=True)

    # 색상: type 별로 구분 (gold/seed 등 없음)
    type_color = {
        "table": "#3B82F6",   # 파랑 — table
        "column": "#9CA3AF",  # 회색 — column
        "fk_node": "#A855F7", # 보라 — fk node
    }
    type_shape = {
        "table": "box",
        "column": "dot",
        "fk_node": "diamond",
    }
    type_size = {
        "table": 35,
        "column": 14,
        "fk_node": 18,
    }

    for node_id, data in graph.nodes(data=True):
        ntype = data.get("type", "column")
        bg = type_color.get(ntype, "#9CA3AF")
        shape = type_shape.get(ntype, "dot")
        size = type_size.get(ntype, 14)

        # PK 강조
        if data.get("is_pk"):
            bg = "#F59E0B"
            size = 20

        nl_name = data.get("nl_name", "")
        title_text = (
            f"Name: {data.get('name', node_id)}\n"
            f"Type: {ntype.upper()}\n"
            + (f"Natural name: {nl_name}\n" if nl_name and nl_name != data.get('name') else "")
            + (f"Primary key: yes\n" if data.get("is_pk") else "")
        )
        net.add_node(node_id, label=str(node_id), title=title_text,
                     color=bg, shape=shape, size=size, borderWidth=1)

    edge_color = {
        "belongs_to": "#374151",
        "is_source_of": "#A855F7",
        "points_to": "#A855F7",
        "table_to_table": "#10B981",
    }
    edge_width = {
        "belongs_to": 1,
        "is_source_of": 2,
        "points_to": 2,
        "table_to_table": 3,
    }
    for u, v, edata in graph.edges(data=True):
        et = edata.get("type", "")
        net.add_edge(u, v, color=edge_color.get(et, "#374151"),
                     width=edge_width.get(et, 1), title=et)

    if title:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(title, width=70))
        net.add_node(
            "__TITLE__", label=wrapped,
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
# 그래프 재구성 + Extractor 라이브 재실행
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


# ──────────────────────────────────────────────────────────────
# Extractor 레지스트리: (display_name, factory, hyperparameter_spec)
#   factory: hp dict → Extractor instance
#   hyperparameter_spec: list of (name, type, low, high, default, step, label, help)
# ──────────────────────────────────────────────────────────────

PCST_COMMON_HP = [
    ("base_cost", float, 0.0, 2.0, 0.05, 0.01, "base_cost", "기본 edge cost"),
    ("belongs_to_cost", float, 0.0, 1.0, 0.01, 0.005, "belongs_to_cost", "table→column edge"),
    ("fk_cost", float, 0.0, 1.0, 0.05, 0.01, "fk_cost", "FK 노드 경유 edge"),
    ("macro_cost", float, 0.0, 2.0, 0.5, 0.05, "macro_cost", "table↔table edge"),
    ("hub_discount", float, 0.0, 1.0, 0.2, 0.05, "hub_discount", "허브 할인 (Dynamic 만)"),
]

ADAPTIVE_HP = [
    ("percentile", float, 50.0, 99.0, 80.0, 1.0, "percentile", "score 분포 상위 P%"),
    ("min_prize_nodes", int, 1, 30, 3, 1, "min_prize_nodes", ""),
    ("max_prize_nodes", int, 5, 100, 25, 1, "max_prize_nodes", ""),
]

PRODUCT_COST_HP = [
    ("bt_weight", float, 0.0, 1.0, 0.1, 0.01, "bt_weight", "table→column 무게"),
    ("fk_weight", float, 0.0, 1.0, 0.2, 0.01, "fk_weight", "FK 무게"),
    ("macro_weight", float, 0.0, 2.0, 0.5, 0.05, "macro_weight", "table↔table 무게"),
    ("min_cost", float, 1e-5, 1e-2, 1e-4, 1e-5, "min_cost", "최소 cost (양수 보장)"),
]

SCORE_DRIVEN_HP = [
    ("belongs_to_weight", float, 0.0, 1.0, 0.3, 0.05, "belongs_to_weight", ""),
    ("fk_weight", float, 0.0, 1.0, 0.5, 0.05, "fk_weight", ""),
    ("macro_weight", float, 0.0, 2.0, 1.5, 0.1, "macro_weight", ""),
    ("epsilon", float, 1e-5, 1e-2, 1e-4, 1e-5, "epsilon", ""),
]

TOPOLOGY_HP = [
    ("gamma", float, 0.0, 5.0, 1.0, 0.1, "gamma", "log-degree 영향력"),
    ("lambda_prize", float, 0.0, 2.0, 0.3, 0.05, "lambda_prize", "prize tiebreaker 가중"),
    ("cost_scale", float, 0.0, 1.0, 0.1, 0.01, "cost_scale", "전체 cost 배율"),
]


def _make_basic_pcst(hp):
    return PCSTExtractor(
        base_cost=hp["base_cost"],
        belongs_to_cost=hp["belongs_to_cost"],
        fk_cost=hp["fk_cost"],
        macro_cost=hp["macro_cost"],
        hub_discount=hp.get("hub_discount", 0.2),
        node_threshold=hp["node_threshold"],
    )


def _make_adaptive_pcst(hp):
    return AdaptivePCSTExtractor(
        base_cost=hp["base_cost"],
        belongs_to_cost=hp["belongs_to_cost"],
        fk_cost=hp["fk_cost"],
        macro_cost=hp["macro_cost"],
        hub_discount=hp.get("hub_discount", 0.2),
        node_threshold=0.0,
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_product_cost(hp):
    return ProductCostPCSTExtractor(
        bt_weight=hp["bt_weight"],
        fk_weight=hp["fk_weight"],
        macro_weight=hp["macro_weight"],
        min_cost=hp["min_cost"],
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_ca_product_cost(hp):
    return ComponentAwareProductCostPCSTExtractor(
        bt_weight=hp["bt_weight"],
        fk_weight=hp["fk_weight"],
        macro_weight=hp["macro_weight"],
        min_cost=hp["min_cost"],
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_ca_adaptive(hp):
    return ComponentAwareAdaptivePCSTExtractor(
        base_cost=hp["base_cost"],
        belongs_to_cost=hp["belongs_to_cost"],
        fk_cost=hp["fk_cost"],
        macro_cost=hp["macro_cost"],
        hub_discount=hp.get("hub_discount", 0.2),
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_score_driven(hp):
    return ScoreDrivenPCSTExtractor(
        belongs_to_weight=hp["belongs_to_weight"],
        fk_weight=hp["fk_weight"],
        macro_weight=hp["macro_weight"],
        epsilon=hp["epsilon"],
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_topology(hp):
    return TopologyCostPCSTExtractor(
        gamma=hp["gamma"],
        lambda_prize=hp["lambda_prize"],
        cost_scale=hp["cost_scale"],
        degree_combination=hp.get("degree_combination", "max"),
        percentile=hp["percentile"],
        min_prize_nodes=int(hp["min_prize_nodes"]),
        max_prize_nodes=int(hp["max_prize_nodes"]),
    )


def _make_dynamic(hp):
    return DynamicPCSTExtractor(
        base_cost=hp["base_cost"],
        belongs_to_cost=hp["belongs_to_cost"],
        fk_cost=hp["fk_cost"],
        macro_cost=hp["macro_cost"],
        hub_discount=hp["hub_discount"],
        node_threshold=hp["node_threshold"],
    )


def _make_mst_steiner(hp):
    return MSTExtractor(seed_mode=hp["seed_mode"], score_threshold=hp["score_threshold"])


def _make_mst_kruskal(hp):
    return MSTKruskalExtractor(score_threshold=hp["score_threshold"])


def _make_mst_pcst_union(hp):
    return MSTPCSTUnionExtractor(
        score_threshold=hp["score_threshold"],
        base_cost=hp["base_cost"],
        belongs_to_cost=hp["belongs_to_cost"],
        fk_cost=hp["fk_cost"],
        macro_cost=hp["macro_cost"],
    )


def _make_topk(hp):
    return TopKExtractor(top_k=int(hp["top_k"]))


# (display_name, factory, hp_specs, extra_options)
EXTRACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Basic PCST": {
        "factory": _make_basic_pcst,
        "hp_specs": PCST_COMMON_HP + [
            ("node_threshold", float, 0.0, 1.0, 0.1, 0.01, "node_threshold", "고정 score threshold"),
        ],
        "needs_seeds": False,
    },
    "Adaptive PCST (Per-query Pn)": {
        "factory": _make_adaptive_pcst,
        "hp_specs": PCST_COMMON_HP + ADAPTIVE_HP,
        "needs_seeds": False,
    },
    "ComponentAware Adaptive PCST": {
        "factory": _make_ca_adaptive,
        "hp_specs": PCST_COMMON_HP + ADAPTIVE_HP,
        "needs_seeds": False,
    },
    "ProductCost PCST": {
        "factory": _make_product_cost,
        "hp_specs": ADAPTIVE_HP + PRODUCT_COST_HP,
        "needs_seeds": False,
    },
    "ComponentAware ProductCost PCST": {
        "factory": _make_ca_product_cost,
        "hp_specs": ADAPTIVE_HP + PRODUCT_COST_HP,
        "needs_seeds": False,
    },
    "ScoreDriven PCST (방안 A)": {
        "factory": _make_score_driven,
        "hp_specs": ADAPTIVE_HP + SCORE_DRIVEN_HP,
        "needs_seeds": False,
    },
    "Topology Cost PCST": {
        "factory": _make_topology,
        "hp_specs": ADAPTIVE_HP + TOPOLOGY_HP,
        "needs_seeds": False,
        "extra_select": [("degree_combination", ["max", "min", "mean", "product"], "max")],
    },
    "Dynamic PCST (Hub-discount)": {
        "factory": _make_dynamic,
        "hp_specs": PCST_COMMON_HP + [
            ("node_threshold", float, 0.0, 1.0, 0.1, 0.01, "node_threshold", ""),
        ],
        "needs_seeds": False,
    },
    "MST Steiner 2-approx": {
        "factory": _make_mst_steiner,
        "hp_specs": [
            ("score_threshold", float, 0.0, 1.0, 0.1, 0.01, "score_threshold (threshold-mode 만 사용)", ""),
        ],
        "needs_seeds": True,
        "extra_select": [("seed_mode", ["topk", "threshold"], "topk")],
    },
    "MST Kruskal (induced subgraph)": {
        "factory": _make_mst_kruskal,
        "hp_specs": [
            ("score_threshold", float, 0.0, 1.0, 0.1, 0.01, "score_threshold", "induced subgraph 정의"),
        ],
        "needs_seeds": False,
    },
    "MST ∪ PCST (Union)": {
        "factory": _make_mst_pcst_union,
        "hp_specs": [
            ("score_threshold", float, 0.0, 1.0, 0.1, 0.01, "score_threshold", ""),
            ("base_cost", float, 0.0, 2.0, 1.0, 0.05, "base_cost", ""),
            ("belongs_to_cost", float, 0.0, 1.0, 0.01, 0.005, "belongs_to_cost", ""),
            ("fk_cost", float, 0.0, 1.0, 0.05, 0.01, "fk_cost", ""),
            ("macro_cost", float, 0.0, 2.0, 0.5, 0.05, "macro_cost", ""),
        ],
        "needs_seeds": False,
    },
    "TopK (no edges)": {
        "factory": _make_topk,
        "hp_specs": [
            ("top_k", int, 1, 50, 15, 1, "top_k", ""),
        ],
        "needs_seeds": False,
    },
}


def render_hp_sliders(extractor_name: str, key_prefix: str = "") -> Dict[str, Any]:
    spec = EXTRACTOR_REGISTRY[extractor_name]
    hp: Dict[str, Any] = {}
    for name, ttype, lo, hi, default, step, label, helptxt in spec["hp_specs"]:
        widget_key = f"{key_prefix}_{extractor_name}_{name}"
        if ttype is int:
            hp[name] = st.slider(label, int(lo), int(hi), int(default), int(step), key=widget_key,
                                  help=helptxt or None)
        else:
            hp[name] = st.slider(label, float(lo), float(hi), float(default), float(step),
                                  key=widget_key, help=helptxt or None)
    for name, choices, default in spec.get("extra_select", []):
        widget_key = f"{key_prefix}_{extractor_name}_{name}"
        hp[name] = st.selectbox(name, choices, index=choices.index(default), key=widget_key)
    return hp


def run_live_extractor(metadata: Dict[str, Any],
                       scores_by_name: Dict[str, float],
                       seeds_idx: List[int],
                       extractor_name: str,
                       hp: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    node_meta = metadata.get("node_metadata", {})
    if not node_meta:
        return [], {}

    n = max(int(k) for k in node_meta.keys()) + 1
    idx_to_name = {int(k): v for k, v in node_meta.items()}
    score_list = [float(scores_by_name.get(idx_to_name.get(i, ""), 0.0)) for i in range(n)]

    import numpy as _np
    scores_arr = _np.array(score_list, dtype=_np.float64)

    extractor = EXTRACTOR_REGISTRY[extractor_name]["factory"](hp)
    needs_seeds = EXTRACTOR_REGISTRY[extractor_name].get("needs_seeds", False)

    seed_arg: Optional[List[int]] = list(seeds_idx) if needs_seeds and seeds_idx else None
    selected_idx, _ = extractor.extract(
        graph_data=metadata,
        node_scores=score_list,
        seed_nodes=seed_arg,
    )

    info = dict(getattr(extractor, "last_info", {}) or {})
    info.setdefault("score_min", float(scores_arr.min()) if scores_arr.size else 0.0)
    info.setdefault("score_max", float(scores_arr.max()) if scores_arr.size else 0.0)
    info.setdefault("score_mean", float(scores_arr.mean()) if scores_arr.size else 0.0)
    info.setdefault("score_std", float(scores_arr.std()) if scores_arr.size else 0.0)

    return [idx_to_name.get(int(i), str(i)) for i in selected_idx], info


# ──────────────────────────────────────────────────────────────
# pyvis 렌더링
# ──────────────────────────────────────────────────────────────

def render_pyvis(graph: nx.Graph,
                 question: str,
                 seeds: set,
                 extracted: set,
                 final: set,
                 gold: set,
                 pcst_threshold: float = 0.0) -> str:
    net = Network(height="750px", width="100%", bgcolor="#111827",
                  font_color="white", directed=True)

    for node_id, data in graph.nodes(data=True):
        nid = str(node_id).strip()
        node_type = data.get("type", "column")
        shape = "box" if node_type == "table" else "dot"
        size = 30 if node_type == "table" else 15

        in_seed = nid in seeds
        in_extracted = nid in extracted
        in_final = nid in final
        is_gold = nid in gold
        score = data.get("similarity_score", 0.0)

        if in_final and is_gold:
            bg = "#10B981"; shape = "star"; size = 50; cat = "TP"
        elif in_final:
            bg = "#EF4444"; size = 15; cat = "FP"
        elif is_gold and in_seed:
            bg = "#60A5FA"; shape = "diamond"; size = 40; cat = "FN (was Seed)"
        elif is_gold:
            bg = "#2563EB"; shape = "triangle"; size = 35; cat = "FN (missed from Seed)"
        elif in_seed:
            bg = "#F59E0B"; cat = "Seed (filtered out)"
        else:
            bg = "#4B5563"; cat = "Unselected"

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

        selection_status = "SELECTED" if in_final else "NOT SELECTED"
        if in_final:
            stage_detail = "Seed -> Extractor -> Filter : Passed all stages" if in_extracted else "Filter : Directly included"
        else:
            if not in_seed and not in_extracted:
                if score < pcst_threshold and pcst_threshold > 0:
                    stage_detail = f"Dropped at: Seed Selection (score {score:.4f} < threshold {pcst_threshold:.4f})"
                else:
                    stage_detail = "Dropped at: Seed Selection (not in top-k seeds)"
            elif in_seed and not in_extracted:
                stage_detail = "Dropped at: Extractor (seed but not in subgraph)"
            elif in_extracted and not in_final:
                stage_detail = "Dropped at: LLM Filter (Extractor selected but filter removed)"
            else:
                stage_detail = "Dropped at: Unknown stage"

        title = (
            f"Name: {data.get('name', nid)}\n"
            f"Type: {node_type.upper()}\n"
            f"Score: {score}\n"
            f"Gold: {'Yes' if is_gold else 'No'}\n"
            f"---\n"
            f"Result: {selection_status}\n"
            f"  Seed: {'Yes' if in_seed else 'No'}\n"
            f"  Extractor (live): {'Yes' if in_extracted else 'No'}\n"
            f"  Final: {'Yes' if in_final else 'No'}\n"
            f"---\n"
            f"{stage_detail}\n"
            f"Category: {cat}"
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
# Helpers
# ──────────────────────────────────────────────────────────────

def metrics(pred: set, gold: set) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return r, p, f


def _load_model_data(exp_name: str, qid: int) -> Optional[Dict[str, Any]]:
    score_data = parse_score_analysis(exp_name, qid)
    log_data = parse_log(exp_name, qid)
    metadata = log_data.get("metadata", {})
    if not metadata:
        return None

    scores_by_name = score_data["node_scores"]
    gold_set = {str(x).strip() for x in score_data["gold_schema"]}
    node_meta = metadata.get("node_metadata", {})
    seed_idx_list = log_data.get("seeds", []) or []
    seeds_text = [node_meta.get(int(i), str(i)) for i in seed_idx_list]

    return {
        "exp_name": exp_name,
        "score_data": score_data,
        "log_data": log_data,
        "metadata": metadata,
        "scores_by_name": scores_by_name,
        "gold_set": gold_set,
        "node_meta": node_meta,
        "seed_idx_list": seed_idx_list,
        "seeds_set": {str(x).strip() for x in seeds_text},
        "extracted_set": {str(x).strip() for x in (log_data.get("extracted_nodes") or [])},
        "final_set": {str(x).strip() for x in (log_data.get("final_nodes") or [])},
    }


def _config_caption(cfg: Dict[str, Any]) -> str:
    """config dict → 한 줄 summary (Builder/Selector/Extractor/Filter)."""
    if not cfg:
        return "(config 없음)"
    if "_raw_text" in cfg:
        return "(yaml 미설치, raw text 만 가능)"
    if "_error" in cfg:
        return f"(config 로딩 실패: {cfg['_error']})"

    bits = []
    gb = cfg.get("graph_builder", {})
    if gb:
        bits.append(f"B={gb.get('name', '?')}")
    ss = cfg.get("seed_selector") or cfg.get("selector") or {}
    if ss:
        sname = ss.get("name", "?")
        sparams = ss.get("params", {}) or {}
        alpha = sparams.get("alpha")
        qcond = sparams.get("query_conditioned")
        topk = sparams.get("top_k")
        sel_summary = sname
        if alpha is not None:
            sel_summary += f" α={alpha}"
        if qcond:
            sel_summary += " QCond"
        if topk:
            sel_summary += f" k={topk}"
        bits.append(f"S={sel_summary}")
    ce = cfg.get("connectivity_extractor") or cfg.get("extractor") or {}
    if ce:
        bits.append(f"E={ce.get('name', '?')}")
    fl = cfg.get("filter", {}) or {}
    if fl:
        fname = fl.get("name", "?")
        fparams = fl.get("params", {}) or {}
        provider = fparams.get("provider")
        model = fparams.get("model_name")
        ftxt = fname
        if provider or model:
            ftxt += f" ({provider or '?'}/{(model or '?').split('/')[-1]})"
        bits.append(f"F={ftxt}")
    return " · ".join(bits) if bits else "(empty)"


def _format_metric(m: Dict[str, Any], key: str) -> str:
    if key not in m:
        return "-"
    v = m[key]
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


# ──────────────────────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────────────────────

def _render_schema_only_mode():
    """실험 무관 plain schema graph view."""
    schemas = load_dev_tables()
    if not schemas:
        with st.sidebar:
            st.error(f"{BIRD_DEV_TABLES_JSON} 를 찾을 수 없습니다.")
        return

    with st.sidebar:
        st.header("Database")
        db_id = st.selectbox(
            f"DB ({len(schemas)} total)",
            sorted(schemas.keys()),
            index=0,
        )

        st.header("Schema graph options")
        include_fk_node = st.checkbox(
            "Show FK nodes (`A.col->B.col`)", value=True,
            help="OFF: column ↔ column 직결, ON: 중간에 fk_node 노드 삽입 (실제 builder 가 만드는 형태)",
        )
        include_t2t = st.checkbox(
            "Show table↔table macro edges", value=True,
        )
        use_natural_names = st.checkbox(
            "Use natural names instead of original", value=False,
            help="dev_tables.json 의 column_names (자연어) vs column_names_original (원본)",
        )

    s = schemas[db_id]
    nx_graph = build_plain_schema_graph(
        db_id,
        include_fk_node=include_fk_node,
        include_t2t=include_t2t,
        use_natural_names=use_natural_names,
    )

    n_t = sum(1 for _, d in nx_graph.nodes(data=True) if d.get("type") == "table")
    n_c = sum(1 for _, d in nx_graph.nodes(data=True) if d.get("type") == "column")
    n_fk = sum(1 for _, d in nx_graph.nodes(data=True) if d.get("type") == "fk_node")
    n_pk = sum(1 for _, d in nx_graph.nodes(data=True) if d.get("is_pk"))

    st.subheader(f"DB: {db_id}")
    st.caption(
        f"Source: dev_tables.json · #tables={len(s['table_names_original'])} · "
        f"#columns={sum(1 for ti, _ in s['column_names_original'] if ti >= 0)} · "
        f"#FKs={len(s.get('foreign_keys', []))}"
    )
    cols = st.columns(5)
    cols[0].metric("Total nodes", nx_graph.number_of_nodes())
    cols[1].metric("Tables", n_t)
    cols[2].metric("Columns", n_c)
    cols[3].metric("FK nodes", n_fk)
    cols[4].metric("PK columns", n_pk)
    st.metric("Total edges", nx_graph.number_of_edges())

    if nx_graph.number_of_nodes() == 0:
        st.warning("그래프가 비어 있습니다.")
        return

    html = render_plain_pyvis(nx_graph, title=f"Schema: {db_id}")
    components.html(html, height=820, scrolling=True)

    with st.expander("Legend", expanded=False):
        st.markdown("""
        - **Blue Box** — Table
        - **Gray Dot** — Column
        - **Yellow** — Primary Key column
        - **Purple Diamond** — FK node (`src.col->dst.col`)
        - **Edge colors**: gray=belongs_to · purple=FK 경유 (is_source_of/points_to) · green=table↔table macro
        """)

    with st.expander("Schema (raw dev_tables.json entry)", expanded=False):
        st.json(s)


def main():
    st.set_page_config(page_title="Schema Linking Graph Visualizer", layout="wide")
    st.title("Schema Linking Graph Visualizer")

    # ── Mode selection (사이드바 첫 항목) ──
    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "View Mode",
            ["Single Experiment", "Multi-Experiment Compare", "Curated Preset",
             "Schema Only (no experiment)"],
            index=2,
        )

    # ── Schema Only: 실험 무관 ──
    if mode == "Schema Only (no experiment)":
        _render_schema_only_mode()
        return

    all_exps = discover_all_experiments()
    if not all_exps:
        st.error("outputs/ 아래에서 score_analysis_*.jsonl 을 가진 실험을 찾을 수 없습니다.")
        return

    exp_names = list(all_exps.keys())

    # ── Sidebar (실험 모드) ──
    with st.sidebar:
        if mode == "Curated Preset":
            st.header("Preset")
            preset_name = st.selectbox("Preset", list(COMPARISON_PRESETS.keys()), index=0)
            preset_items = COMPARISON_PRESETS[preset_name]

            # 표시용 라벨 → 실제 exp_name 매핑
            label_to_dir = {lbl: d for lbl, d in preset_items}
            available_labels = [lbl for lbl, d in preset_items if d in all_exps]
            missing_labels = [lbl for lbl, d in preset_items if d not in all_exps]
            if missing_labels:
                st.caption(f"⚠️ 누락된 셀 ({len(missing_labels)}): " + ", ".join(missing_labels[:3]))

            selected_labels = st.multiselect(
                "Cells", available_labels, default=available_labels,
            )
            selected_dirs = [label_to_dir[lbl] for lbl in selected_labels]

        elif mode == "Multi-Experiment Compare":
            st.header("Experiments")
            categories = sorted(set(e.split("/")[0] for e in exp_names))
            cat_filter = st.multiselect("Category filter", categories, default=categories)
            filtered = [e for e in exp_names if e.split("/")[0] in cat_filter]

            search = st.text_input("Substring search (optional)")
            if search:
                filtered = [e for e in filtered if search.lower() in e.lower()]

            selected_dirs = st.multiselect(
                f"Experiments to compare ({len(filtered)} match)",
                filtered, default=filtered[:2] if len(filtered) >= 2 else filtered,
            )

        else:  # Single Experiment
            st.header("Experiment")
            categories = sorted(set(e.split("/")[0] for e in exp_names))
            cat_filter = st.selectbox("Category", ["(all)"] + categories, index=0)
            filtered = [e for e in exp_names if cat_filter == "(all)" or e.startswith(cat_filter)]

            search = st.text_input("Substring search (optional)")
            if search:
                filtered = [e for e in filtered if search.lower() in e.lower()]

            if not filtered:
                st.warning("일치하는 실험이 없습니다.")
                return
            exp_name = st.selectbox(f"Experiment ({len(filtered)} match)", filtered, index=0)
            selected_dirs = [exp_name]

        if not selected_dirs:
            st.warning("실험을 1개 이상 선택하세요.")
            return

        qids = list_qids_in_experiment(selected_dirs[0])
        if not qids:
            st.error(f"{selected_dirs[0]} 에서 question_id 를 찾지 못함.")
            return

        st.header("Question")
        qid = st.selectbox(f"Question ID ({len(qids)} total)", qids, index=0)

        st.header("Live Extractor")
        extractor_name = st.selectbox(
            "Extractor type",
            list(EXTRACTOR_REGISTRY.keys()),
            index=list(EXTRACTOR_REGISTRY.keys()).index("Adaptive PCST (Per-query Pn)"),
        )
        st.caption(f"needs seeds: {EXTRACTOR_REGISTRY[extractor_name].get('needs_seeds', False)}")

        st.header("Hyperparameters")
        hp = render_hp_sliders(extractor_name, key_prefix="live")

    # ── Main ──
    dev_meta = load_dev_meta().get(qid, {})

    col_q, col_db = st.columns([3, 1])
    with col_q:
        st.markdown(f"**Question {qid}:** {dev_meta.get('question', '(loading from log...)')}")
        if dev_meta.get("evidence"):
            st.caption(f"Evidence: {dev_meta['evidence']}")
        if dev_meta.get("SQL"):
            st.code(dev_meta["SQL"], language="sql")
    with col_db:
        st.metric("DB", dev_meta.get("db_id", "?"))
        st.metric("Difficulty", dev_meta.get("difficulty", "?"))

    # ── Compare 모드 (2개 이상 선택) ──
    if len(selected_dirs) > 1:
        st.subheader(f"Comparison ({len(selected_dirs)} experiments)")

        import pandas as pd
        rows = []
        loaded: Dict[str, Dict[str, Any]] = {}

        for d in selected_dirs:
            cfg = get_config_for_experiment(d)
            run_metrics = get_metrics_summary(d)
            mdata = _load_model_data(d, qid)
            row: Dict[str, Any] = {
                "Experiment": d,
                "Config": _config_caption(cfg),
                "Run R": _format_metric(run_metrics, "recall"),
                "Run P": _format_metric(run_metrics, "precision"),
                "Run EX": _format_metric(run_metrics, "ex"),
            }
            if mdata is None:
                row.update({
                    "Status": "no log",
                    "Seeds": "-", "|Extracted|": "-", "|Final|": "-",
                    "R(E)": "-", "P(E)": "-", "F1(E)": "-",
                    "R(F)": "-", "P(F)": "-", "F1(F)": "-",
                })
                rows.append(row)
                continue

            loaded[d] = mdata
            gold = mdata["gold_set"]
            ext = mdata["extracted_set"]
            fin = mdata["final_set"]
            r_e, p_e, f_e = metrics(ext, gold)
            r_f, p_f, f_f = metrics(fin, gold)

            row.update({
                "Status": "ok",
                "Seeds": len(mdata["seeds_set"]),
                "|Extracted|": len(ext),
                "|Final|": len(fin) if fin else len(ext),
                "R(E)": f"{r_e:.4f}",
                "P(E)": f"{p_e:.4f}",
                "F1(E)": f"{f_e:.4f}",
                "R(F)": f"{r_f:.4f}" if fin else "-",
                "P(F)": f"{p_f:.4f}" if fin else "-",
                "F1(F)": f"{f_f:.4f}" if fin else "-",
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if loaded:
            example_gold = next(iter(loaded.values()))["gold_set"]
            st.markdown(f"**Gold nodes:** {len(example_gold)} — {sorted(example_gold)}")

        # 각 실험 탭별 상세
        if loaded:
            st.subheader("Per-experiment detail")
            tabs = st.tabs([d.split("/")[-1] for d in loaded.keys()])
            for tab, d in zip(tabs, loaded):
                with tab:
                    mdata = loaded[d]
                    cfg = get_config_for_experiment(d)
                    run_metrics = get_metrics_summary(d)

                    st.markdown(f"**{d}**")
                    st.caption(f"Config: {_config_caption(cfg)}")
                    if run_metrics:
                        st.caption(
                            f"Run-level: R={_format_metric(run_metrics, 'recall')} · "
                            f"P={_format_metric(run_metrics, 'precision')} · "
                            f"EX={_format_metric(run_metrics, 'ex')}"
                        )

                    live_extracted, live_info = run_live_extractor(
                        mdata["metadata"], mdata["scores_by_name"],
                        mdata["seed_idx_list"], extractor_name, hp)
                    live_set = {str(x).strip() for x in live_extracted}

                    gold = mdata["gold_set"]
                    r_l, p_l, f_l = metrics(live_set, gold)
                    r_e, p_e, f_e = metrics(mdata["extracted_set"], gold)
                    r_f, p_f, f_f = metrics(mdata["final_set"], gold)

                    if live_info:
                        thr = live_info.get("extractor_threshold")
                        sel_n = live_info.get("extractor_num_selected_nodes", "?")
                        cap_bits = [f"|live|={sel_n}"]
                        if thr is not None:
                            cap_bits.append(f"threshold={thr:.4f}")
                        if "score_min" in live_info:
                            cap_bits.append(
                                f"scores=[{live_info['score_min']:.3f}, {live_info['score_max']:.3f}]"
                            )
                        st.caption(" · ".join(cap_bits))

                    mc = st.columns(3)
                    with mc[0]:
                        st.markdown("**Live Extractor (current sliders)**")
                        st.write(f"|nodes|={len(live_set)}  R={r_l:.4f}  P={p_l:.4f}  F1={f_l:.4f}")
                    with mc[1]:
                        st.markdown("**Original Extractor (from log)**")
                        st.write(
                            f"|nodes|={len(mdata['extracted_set'])}  "
                            f"R={r_e:.4f}  P={p_e:.4f}  F1={f_e:.4f}"
                        )
                    with mc[2]:
                        st.markdown("**Final (after Filter)**")
                        if mdata["final_set"]:
                            st.write(
                                f"|nodes|={len(mdata['final_set'])}  "
                                f"R={r_f:.4f}  P={p_f:.4f}  F1={f_f:.4f}"
                            )
                        else:
                            st.write("(no filter applied)")

                    nx_graph = reconstruct_graph(mdata["metadata"], mdata["scores_by_name"])
                    if nx_graph.number_of_nodes() > 0:
                        html = render_pyvis(
                            nx_graph,
                            question="",
                            seeds=mdata["seeds_set"],
                            extracted=live_set,
                            final=mdata["final_set"],
                            gold=gold,
                            pcst_threshold=float(live_info.get("extractor_threshold") or 0.0),
                        )
                        components.html(html, height=700, scrolling=True)

                    with st.expander("Node details", expanded=False):
                        nd_rows = []
                        for nid, ndata in nx_graph.nodes(data=True):
                            snid = str(nid).strip()
                            nd_rows.append({
                                "name": snid,
                                "type": ndata.get("type"),
                                "score": ndata.get("similarity_score"),
                                "gold": snid in gold,
                                "seed": snid in mdata["seeds_set"],
                                "extracted_live": snid in live_set,
                                "extracted_orig": snid in mdata["extracted_set"],
                                "final": snid in mdata["final_set"],
                            })
                        if nd_rows:
                            ndf = pd.DataFrame(nd_rows).sort_values("score", ascending=False)
                            st.dataframe(ndf, use_container_width=True, height=350)

                    with st.expander("Generated SQL", expanded=False):
                        sql = mdata["log_data"].get("generated_sql", "")
                        if sql:
                            st.code(sql, language="sql")
                        else:
                            st.write("(no SQL generated)")

                    with st.expander("Config (raw)", expanded=False):
                        st.json(cfg)

        with st.expander("Legend", expanded=False):
            st.markdown("""
            - **Green Star** — TP (Final ∩ Gold)
            - **Red** — FP (Final, not Gold)
            - **Light Blue Diamond** — FN: Seed까지는 선택됨 (Extractor/Filter 탈락)
            - **Dark Blue Triangle** — FN: Seed부터 선택 안 됨 (가장 심각한 누락)
            - **Yellow** — Seed only (filtered out)
            - **Gray** — Unselected
            - **Cyan border** — Live Extractor selected subgraph
            - Edge colors: Green (Final) > Cyan (Extractor) > Yellow (Seeds) > Gray
            """)
        return

    # ── Single Experiment ──
    exp_name = selected_dirs[0]
    cfg = get_config_for_experiment(exp_name)
    run_metrics = get_metrics_summary(exp_name)

    st.caption(f"Config: {_config_caption(cfg)}")
    if run_metrics:
        st.caption(
            f"Run-level: R={_format_metric(run_metrics, 'recall')} · "
            f"P={_format_metric(run_metrics, 'precision')} · "
            f"EX={_format_metric(run_metrics, 'ex')} · "
            f"filter_time_mean={_format_metric(run_metrics, 'filter_time_mean_s')}s"
        )

    score_data = parse_score_analysis(exp_name, qid)
    log_data = parse_log(exp_name, qid)
    metadata = log_data.get("metadata", {})
    if not metadata:
        st.error(
            f"`metadata: {{...}}` 를 로그에서 찾을 수 없습니다. ({exp_name} / qid={qid})\n"
            f"해당 실험은 디버그 로그가 없을 수 있습니다."
        )
        return

    scores_by_name = score_data["node_scores"]
    gold_set = {str(x).strip() for x in score_data["gold_schema"]}
    node_meta = metadata.get("node_metadata", {})
    seed_idx_list = log_data.get("seeds", []) or []
    seeds_text = [node_meta.get(int(i), str(i)) for i in seed_idx_list]
    seeds_set = {str(x).strip() for x in seeds_text}
    final_set = {str(x).strip() for x in (log_data.get("final_nodes") or [])}
    original_extracted_set = {str(x).strip() for x in (log_data.get("extracted_nodes") or [])}

    live_extracted, live_info = run_live_extractor(
        metadata, scores_by_name, seed_idx_list, extractor_name, hp)
    live_extracted_set = {str(x).strip() for x in live_extracted}

    st.subheader("Metrics — Live Extractor vs. Original Run")

    if live_info:
        thr = live_info.get("extractor_threshold")
        sel_n = live_info.get("extractor_num_selected_nodes", "?")
        thr_line_bits = [f"**{extractor_name}**", f"|selected| = {sel_n}"]
        if thr is not None:
            thr_line_bits.append(f"threshold = {thr:.4f}")
        if "score_min" in live_info:
            thr_line_bits.append(
                f"score range = [{live_info['score_min']:.4f}, {live_info['score_max']:.4f}]"
            )
            thr_line_bits.append(
                f"mean = {live_info['score_mean']:.4f}, std = {live_info['score_std']:.4f}"
            )
        st.markdown(" | ".join(thr_line_bits))

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
        st.markdown("**Live Extractor (current sliders)**")
        st.metric("|extracted|", len(live_extracted_set))
        st.metric("Recall", f"{r_live:.4f}")
        st.metric("Precision", f"{p_live:.4f}")
        st.metric("F1", f"{f_live:.4f}")
    with cols[1]:
        st.markdown("**Original Extractor (from log)**")
        st.metric("|extracted|", len(original_extracted_set))
        st.metric("Recall", f"{r_orig:.4f}")
        st.metric("Precision", f"{p_orig:.4f}")
        st.metric("F1", f"{f_orig:.4f}")
    with cols[2]:
        st.markdown("**Final (after Filter)**")
        st.metric("|final|", len(final_set))
        st.metric("Recall", f"{r_final:.4f}")
        st.metric("Precision", f"{p_final:.4f}")
        st.metric("F1", f"{f_final:.4f}")

    st.subheader("Graph (Live Extractor highlighted with cyan border)")
    nx_graph = reconstruct_graph(metadata, scores_by_name)
    if nx_graph.number_of_nodes() == 0:
        st.warning("그래프 메타데이터를 재구성할 수 없습니다.")
        return

    html = render_pyvis(
        nx_graph,
        question=log_data.get("question", ""),
        seeds=seeds_set,
        extracted=live_extracted_set,
        final=final_set,
        gold=gold_set,
        pcst_threshold=float(live_info.get("extractor_threshold") or 0.0),
    )
    components.html(html, height=820, scrolling=True)

    with st.expander("Legend", expanded=False):
        st.markdown("""
        - **Green Star** — TP (Final ∩ Gold)
        - **Red** — FP (Final, not Gold)
        - **Light Blue Diamond** — FN: Seed까지는 선택됨 (Extractor/Filter 탈락)
        - **Dark Blue Triangle** — FN: Seed부터 선택 안 됨 (가장 심각한 누락)
        - **Yellow** — Seed only (filtered out)
        - **Gray** — Unselected
        - **Cyan border** — Live Extractor selected subgraph
        - Edge colors: Green (Final) > Cyan (Extractor) > Yellow (Seeds) > Gray
        """)

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

    with st.expander("Generated SQL", expanded=False):
        sql = log_data.get("generated_sql", "")
        if sql:
            st.code(sql, language="sql")
        else:
            st.write("(no SQL generated)")

    with st.expander("Live Extractor info (raw)", expanded=False):
        st.json(live_info)

    with st.expander("Config (raw)", expanded=False):
        st.json(cfg)


if __name__ == "__main__":
    main()
