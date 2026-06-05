"""V6-W3 (Phase 3 hub 차수 축소) — 그래프 구조 3 variants.

DECISIONS 2026-06-06 + planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 3
정합. RFP H1 (hub-accelerated over-smoothing) + H2 (hub 테이블 컬럼 collapse,
hi-deg L3 MAD=0.0046) 의 직접 검증 wave. **단순 disconnect 재확인 아니라
GAT-necessity 인과 검증** (hub 차수 축소로 GAT 가 살아나는가).

게이트 (analyzer 측 측정):
- L1 mechanism: hi-deg 테이블 intra-MAD 0.0046 → ~0.04 회복 방향
- L2 GAT-necessity: (a) GAT-only (α=0) hub-table 품질 ↑, (b) GAT 순기여 % ↑
  (현 +2.1%), (c) hub-heavy DB (european_football_2 등) e2e EX ↑

3 variants (한 wave 단위 비교):
- A `V6W3VirtualSummaryBuilder`  — table→summary→column 2-hop 구조
- B `V6W3ColumnPoolingBuilder`   — table feature = column embedding 의 weighted pool
- C `V6W3HubLocalVNBuilder`      — degree > median table 위 Local VN_G 부착

base = `EnrichedHeteroGraphBuilder` (M4 anchor 정합). 기존 metadata dict keys
모두 유지 — downstream selector/extractor/filter 인터페이스 보존. 신규 keys 만
추가. cache 충돌 방지 위 `extra_cache_suffix` 속성 노출 (bird_dataset.py 가
파일명 생성 시 합성).

per-DB stratification 필수: european_football_2 (115-col, complete collapse 0.0000)
natural test bed — variant 별 hub 식별 / pooling 결과를 metadata['hub_*'] 키 위
노출하여 analyzer 가 db-stratified 측정 가능.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.builders.graph_builder import EnrichedHeteroGraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 공통 helper
# --------------------------------------------------------------------------- #

def _column_means_per_table(
    data: HeteroData,
    table_to_id: Dict[str, int],
    col_to_id: Dict[str, int],
    schema_columns: Dict[str, List[Dict[str, Any]]],
) -> torch.Tensor:
    """각 table 위 속한 column embedding 의 mean.

    빈 table (column 0개) → zeros vector (PLM dim).
    반환 shape: (T, D).
    """
    T = len(table_to_id)
    D = data['column'].x.shape[1] if data['column'].x.numel() > 0 else 0
    if T == 0 or D == 0:
        return torch.zeros((T, D), dtype=torch.float32)

    pooled = torch.zeros((T, D), dtype=data['column'].x.dtype)
    for table, cols in schema_columns.items():
        if table not in table_to_id:
            continue
        t_id = table_to_id[table]
        col_ids = [col_to_id[f"{table}.{c['name']}"] for c in cols
                   if f"{table}.{c['name']}" in col_to_id]
        if not col_ids:
            continue
        pooled[t_id] = data['column'].x[col_ids].mean(dim=0)
    return pooled


def _identify_hub_tables(
    table_to_id: Dict[str, int],
    schema_columns: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: str = "median",
    min_columns: int = 0,
) -> Tuple[List[int], float, Dict[int, int]]:
    """hub table 식별 — column count > median (per-DB).

    Args:
        table_to_id: {table_name -> flat_idx}
        schema_columns: {table_name -> List[col dict]}
        strategy: "median" (현 default) — 추후 "mean+std" 등 확장 여지
        min_columns: hub 최소 column count 절댓값 (median + min_columns 둘 다 통과)

    Returns:
        (hub_table_indices, threshold_value, table_col_count)
        - hub_table_indices: median 초과 + min_columns 초과 위 table flat idx list
        - threshold_value: 사용된 임계값 (median)
        - table_col_count: {table_idx -> column count}
    """
    if not table_to_id:
        return [], 0.0, {}
    counts: Dict[int, int] = {}
    for table, t_idx in table_to_id.items():
        counts[t_idx] = len(schema_columns.get(table, []))

    counts_arr = np.array(list(counts.values()), dtype=np.int32)
    if strategy == "median":
        threshold = float(np.median(counts_arr))
    elif strategy == "mean":
        threshold = float(np.mean(counts_arr))
    else:
        raise ValueError(f"Unknown hub strategy: {strategy}")

    hubs = sorted([
        t_idx for t_idx, c in counts.items()
        if c > threshold and c > min_columns
    ])
    return hubs, threshold, counts


# --------------------------------------------------------------------------- #
# Variant A — 테이블-요약 (virtual) 노드
# --------------------------------------------------------------------------- #

@register("builder", "V6W3VirtualSummaryBuilder")
class V6W3VirtualSummaryBuilder(EnrichedHeteroGraphBuilder):
    """V6-W3 variant A — `table_summary` virtual node.

    table → table_summary → column 의 2-hop alternative path 도입. 기존
    table → column 의 1-hop 직접 message passing 은 retain — selector 측에서
    summary path 만 사용 / 두 path 모두 사용 / 등 선택 가능.

    신규 node type: `table_summary` (수 = # tables).
    신규 edge types (4종):
      - ('table', 'has_summary', 'table_summary')
      - ('table_summary', 'summary_of', 'table')
      - ('table_summary', 'summarizes', 'column')
      - ('column', 'aggregated_by', 'table_summary')

    summary node feature init: 해당 table 위 column embedding mean
    (PLM-encoded `data['column'].x` 의 row mean).

    PCST flat 인덱싱: 기존 table → column → fk_node → **table_summary**
    (마지막 블록 추가). 신규 metadata keys:
      - `summary_to_id: Dict[str, int]` (table_name -> summary local idx)
      - `summary_flat_offset: int` (PCST flat 위 summary 블록 시작 idx)

    Hub 차수 직접 축소 아니라, hub 의 message passing 위 1-hop 직접 평균을
    summary 경유로 우회 (1/d 희석 완화).
    """

    extra_cache_suffix = "_v6w3_a"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("V6W3VirtualSummaryBuilder — table_summary virtual node 활성")

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        t_total = time.perf_counter()
        data, metadata = super().build(db_id, db_dir)

        table_to_id: Dict[str, int] = metadata['table_to_id']
        col_to_id: Dict[str, int] = metadata['col_to_id']
        fk_to_id: Dict[str, int] = metadata['fk_to_id']
        T = len(table_to_id)
        C = len(col_to_id)
        F = len(fk_to_id)

        # schema_info 위 column 정보 재수집 (super().build() 위 컴포넌트 별 schema_info 변수 재사용 가능)
        schema_info = self._get_schema_info(self._resolve_db_path(db_id, db_dir))
        schema_cols = schema_info["columns"]

        # ----- summary node features (mean pool of columns per table) -----
        t = time.perf_counter()
        summary_x = _column_means_per_table(data, table_to_id, col_to_id, schema_cols)
        data['table_summary'].x = summary_x
        t_summary_feat = time.perf_counter() - t

        # ----- summary edges -----
        ts_src, ts_dst = [], []     # table -> summary
        st_src, st_dst = [], []     # summary -> table
        sc_src, sc_dst = [], []     # summary -> column
        cs_src, cs_dst = [], []     # column -> summary

        summary_to_id: Dict[str, int] = {}
        for table, t_idx in table_to_id.items():
            s_idx = t_idx  # summary local idx = table local idx (1대1)
            summary_to_id[table] = s_idx
            ts_src.append(t_idx); ts_dst.append(s_idx)
            st_src.append(s_idx); st_dst.append(t_idx)
            for col in schema_cols.get(table, []):
                full_name = f"{table}.{col['name']}"
                if full_name in col_to_id:
                    c_idx = col_to_id[full_name]
                    sc_src.append(s_idx); sc_dst.append(c_idx)
                    cs_src.append(c_idx); cs_dst.append(s_idx)

        data['table', 'has_summary', 'table_summary'].edge_index = torch.tensor(
            [ts_src, ts_dst], dtype=torch.long)
        data['table_summary', 'summary_of', 'table'].edge_index = torch.tensor(
            [st_src, st_dst], dtype=torch.long)
        data['table_summary', 'summarizes', 'column'].edge_index = torch.tensor(
            [sc_src, sc_dst], dtype=torch.long)
        data['column', 'aggregated_by', 'table_summary'].edge_index = torch.tensor(
            [cs_src, cs_dst], dtype=torch.long)

        # ----- PCST flat 인덱싱: summary 블록은 table/column/fk_node 다음 위 -----
        summary_flat_offset = T + C + F  # 기존 (T,C,F) 블록 다음
        for tbl, s_idx in summary_to_id.items():
            metadata['node_metadata'][summary_flat_offset + s_idx] = f"__summary__({tbl})"

        existing_edges = list(metadata['edges'])
        existing_types = list(metadata['edge_types'])

        new_edges: List[Tuple[int, int]] = []
        new_types: List[str] = []
        # table -> summary  (table flat idx 는 [0, T) 그대로, summary 는 +offset)
        for s, d in zip(ts_src, ts_dst):
            new_edges.append((s, summary_flat_offset + d))
            new_types.append('has_summary')
        # summary -> table
        for s, d in zip(st_src, st_dst):
            new_edges.append((summary_flat_offset + s, d))
            new_types.append('summary_of')
        # summary -> column (column flat idx = T + col_local)
        for s, d in zip(sc_src, sc_dst):
            new_edges.append((summary_flat_offset + s, T + d))
            new_types.append('summarizes')
        # column -> summary
        for s, d in zip(cs_src, cs_dst):
            new_edges.append((T + s, summary_flat_offset + d))
            new_types.append('aggregated_by')

        metadata['edges'] = existing_edges + new_edges
        metadata['edge_types'] = existing_types + new_types

        # ----- 신규 metadata keys (downstream optional consumption) -----
        metadata['summary_to_id'] = summary_to_id
        metadata['summary_flat_offset'] = int(summary_flat_offset)
        metadata['v6w3_variant'] = 'A'

        # ----- builder_info extra -----
        info = dict(self.last_info or {})
        info['builder_type'] = "V6W3VirtualSummaryBuilder"
        info['builder_v6w3_variant'] = 'A'
        info['builder_num_summary_nodes'] = int(T)
        info['builder_num_summary_edges'] = int(len(new_edges))
        timings = info.get('builder_timings', {}) or {}
        timings['v6w3_summary_feat_s'] = float(t_summary_feat)
        timings['v6w3_total_s'] = float(time.perf_counter() - t_total)
        info['builder_timings'] = timings
        self.last_info = info
        metadata['builder_info'] = dict(info)
        return data, metadata

    @staticmethod
    def _resolve_db_path(db_id: str, db_dir: str) -> str:
        import os
        path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(path):
            path = os.path.join(db_dir, "dev_databases", db_id, f"{db_id}.sqlite")
        return path


# --------------------------------------------------------------------------- #
# Variant B — 계층적 column→table 풀링 (attention pooling init)
# --------------------------------------------------------------------------- #

@register("builder", "V6W3ColumnPoolingBuilder")
class V6W3ColumnPoolingBuilder(EnrichedHeteroGraphBuilder):
    """V6-W3 variant B — table feature override (column 위 weighted pooling).

    기존 EnrichedHeteroGraphBuilder 는 table feature = PLM("Table: {name} (nl)").
    이 variant 는 table feature 를 해당 table 위 column embedding 의 **weighted
    pooling 결과** 로 override — table → column 의 redundancy 완화 + table 노드
    가 column 의 평균을 직접 표현 (1/d 희석 의 구조적 보완).

    pool_mode:
      - "uniform" (default): w_i = 1/n (mean)
      - "cosine_softmax": w_i = softmax_j(sum_k!=i cos(col_i, col_k)) — table 내
        다른 column 들과 평균 cosine 위 softmax (semantic centrality init).
        column 1개 table → uniform fallback.

    학습 시 selector 측이 attention weight 를 갱신할 수 있도록 metadata 위
    `table_pool_weights: Dict[table_idx, np.ndarray]` 노출. weights shape 은
    각 table 위 column 수.

    node/edge structure 동일 (table_summary 추가 X). table.x 만 override.
    """

    extra_cache_suffix = "_v6w3_b"

    def __init__(self, pool_mode: str = "uniform", **kwargs):
        super().__init__(**kwargs)
        valid = {"uniform", "cosine_softmax"}
        if pool_mode not in valid:
            raise ValueError(f"pool_mode must be in {valid}, got '{pool_mode}'")
        self.pool_mode = pool_mode
        logger.info(f"V6W3ColumnPoolingBuilder — pool_mode='{pool_mode}'")

    @staticmethod
    def _cosine_softmax_weights(col_emb: torch.Tensor) -> np.ndarray:
        """column embedding 끼리 cosine similarity 위 mean → softmax.

        col_emb shape (n, D). 반환 (n,) numpy weights (sum=1).
        n=1 → [1.0]. n=0 → empty.
        """
        n = col_emb.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        if n == 1:
            return np.array([1.0], dtype=np.float32)
        normed = torch.nn.functional.normalize(col_emb, dim=-1)
        sim = normed @ normed.T  # (n, n)
        # exclude self-similarity (diagonal)
        sim = sim - torch.eye(n, dtype=sim.dtype) * sim.diagonal()
        centrality = sim.sum(dim=-1) / max(n - 1, 1)  # (n,)
        # softmax over n
        w = torch.softmax(centrality, dim=-1).numpy().astype(np.float32)
        return w

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        t_total = time.perf_counter()
        data, metadata = super().build(db_id, db_dir)

        table_to_id = metadata['table_to_id']
        col_to_id = metadata['col_to_id']
        T = len(table_to_id)
        D = data['column'].x.shape[1] if data['column'].x.numel() > 0 else 0

        # schema_info 재로드 (table -> columns 매핑)
        schema_info = self._get_schema_info(
            V6W3VirtualSummaryBuilder._resolve_db_path(db_id, db_dir))
        schema_cols = schema_info["columns"]

        t = time.perf_counter()
        new_table_x = torch.zeros((T, D), dtype=data['table'].x.dtype) if D > 0 else data['table'].x.clone()
        pool_weights: Dict[int, np.ndarray] = {}

        for table, t_idx in table_to_id.items():
            col_ids = [col_to_id[f"{table}.{c['name']}"] for c in schema_cols.get(table, [])
                       if f"{table}.{c['name']}" in col_to_id]
            if not col_ids:
                # empty table — retain original (PLM "Table: name") feature
                new_table_x[t_idx] = data['table'].x[t_idx]
                pool_weights[t_idx] = np.zeros((0,), dtype=np.float32)
                continue
            col_emb = data['column'].x[col_ids]  # (n, D)
            if self.pool_mode == "uniform":
                w = np.full(len(col_ids), 1.0 / len(col_ids), dtype=np.float32)
            else:  # cosine_softmax
                w = self._cosine_softmax_weights(col_emb)
            pool_weights[t_idx] = w
            w_tensor = torch.tensor(w, dtype=col_emb.dtype)
            new_table_x[t_idx] = (col_emb * w_tensor.unsqueeze(-1)).sum(dim=0)

        data['table'].x = new_table_x
        t_pool = time.perf_counter() - t

        # ----- metadata 위 신규 keys -----
        metadata['table_pool_weights'] = pool_weights
        metadata['table_pool_mode'] = self.pool_mode
        metadata['v6w3_variant'] = 'B'

        # ----- builder_info extra -----
        info = dict(self.last_info or {})
        info['builder_type'] = "V6W3ColumnPoolingBuilder"
        info['builder_v6w3_variant'] = 'B'
        info['builder_pool_mode'] = self.pool_mode
        info['builder_num_pooled_tables'] = int(sum(1 for w in pool_weights.values() if w.size > 0))
        timings = info.get('builder_timings', {}) or {}
        timings['v6w3_pool_s'] = float(t_pool)
        timings['v6w3_total_s'] = float(time.perf_counter() - t_total)
        info['builder_timings'] = timings
        self.last_info = info
        metadata['builder_info'] = dict(info)
        return data, metadata


# --------------------------------------------------------------------------- #
# Variant C — 고차수 테이블 Local VN_G (hub-only Virtual Node)
# --------------------------------------------------------------------------- #

@register("builder", "V6W3HubLocalVNBuilder")
class V6W3HubLocalVNBuilder(EnrichedHeteroGraphBuilder):
    """V6-W3 variant C — hub-table only Local Virtual Node.

    naive 전역 VN 은 smoothing 가속 — RFP H1 정합 위 **이질성 인지** (degree
    > median) hub table 만 Local VN_G 부착. hub table 의 column 들이 VN_G
    경유 short-circuit 위 message passing → hub-collapse mechanism 완화.

    hub identification (per-DB):
      - strategy="median" (default): column count > median → hub
      - min_columns: 절대 임계 (median + min_columns 모두 통과)

    신규 node type: `local_vn` (수 = # hub tables only).
    신규 edge types (4종):
      - ('table', 'has_local_vn', 'local_vn')   — hub table 만
      - ('local_vn', 'serves_table', 'table')
      - ('local_vn', 'aggregates', 'column')    — hub table 위 column 모두
      - ('column', 'feeds_into', 'local_vn')

    Local VN feature init: hub table 위 column embedding mean.

    PCST flat 인덱싱: table → column → fk_node → **local_vn** (마지막 블록).
    신규 metadata keys:
      - `hub_tables: List[int]` (hub table flat idx)
      - `hub_threshold: float` (median value)
      - `hub_strategy: str`
      - `local_vn_to_id: Dict[str, int]` (hub_table_name -> local_vn local idx)
      - `local_vn_flat_offset: int`
      - `table_col_count: Dict[int, int]` (per-table column 수, debugging)
    """

    extra_cache_suffix = "_v6w3_c"

    def __init__(
        self,
        hub_strategy: str = "median",
        hub_min_columns: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hub_strategy = hub_strategy
        self.hub_min_columns = int(hub_min_columns)
        logger.info(
            f"V6W3HubLocalVNBuilder — hub_strategy='{hub_strategy}' "
            f"hub_min_columns={hub_min_columns}"
        )

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        t_total = time.perf_counter()
        data, metadata = super().build(db_id, db_dir)

        table_to_id = metadata['table_to_id']
        col_to_id = metadata['col_to_id']
        fk_to_id = metadata['fk_to_id']
        T = len(table_to_id)
        C = len(col_to_id)
        F = len(fk_to_id)
        D = data['column'].x.shape[1] if data['column'].x.numel() > 0 else 0

        schema_info = self._get_schema_info(
            V6W3VirtualSummaryBuilder._resolve_db_path(db_id, db_dir))
        schema_cols = schema_info["columns"]

        # ----- Hub identification -----
        t = time.perf_counter()
        hubs, threshold, counts = _identify_hub_tables(
            table_to_id, schema_cols,
            strategy=self.hub_strategy, min_columns=self.hub_min_columns,
        )
        H = len(hubs)
        t_hub_id = time.perf_counter() - t

        # ----- Local VN features (per hub table) -----
        # idx_in_table 순서 (hubs sorted by table flat idx)
        id_to_table = {v: k for k, v in table_to_id.items()}
        local_vn_to_id: Dict[str, int] = {}
        if H > 0 and D > 0:
            vn_x = torch.zeros((H, D), dtype=data['column'].x.dtype)
            for vn_local_idx, hub_t_idx in enumerate(hubs):
                hub_tbl_name = id_to_table[hub_t_idx]
                local_vn_to_id[hub_tbl_name] = vn_local_idx
                col_ids = [
                    col_to_id[f"{hub_tbl_name}.{c['name']}"]
                    for c in schema_cols.get(hub_tbl_name, [])
                    if f"{hub_tbl_name}.{c['name']}" in col_to_id
                ]
                if col_ids:
                    vn_x[vn_local_idx] = data['column'].x[col_ids].mean(dim=0)
        else:
            vn_x = torch.zeros((H, D if D > 0 else 1), dtype=torch.float32)

        data['local_vn'].x = vn_x

        # ----- Local VN edges (hub-only) -----
        ht_src, ht_dst = [], []     # table -> local_vn
        th_src, th_dst = [], []     # local_vn -> table
        hc_src, hc_dst = [], []     # local_vn -> column
        ch_src, ch_dst = [], []     # column -> local_vn

        for hub_t_idx in hubs:
            hub_tbl_name = id_to_table[hub_t_idx]
            vn_idx = local_vn_to_id[hub_tbl_name]
            ht_src.append(hub_t_idx); ht_dst.append(vn_idx)
            th_src.append(vn_idx); th_dst.append(hub_t_idx)
            for col in schema_cols.get(hub_tbl_name, []):
                full_name = f"{hub_tbl_name}.{col['name']}"
                if full_name in col_to_id:
                    c_idx = col_to_id[full_name]
                    hc_src.append(vn_idx); hc_dst.append(c_idx)
                    ch_src.append(c_idx); ch_dst.append(vn_idx)

        if H > 0:
            data['table', 'has_local_vn', 'local_vn'].edge_index = torch.tensor(
                [ht_src, ht_dst], dtype=torch.long)
            data['local_vn', 'serves_table', 'table'].edge_index = torch.tensor(
                [th_src, th_dst], dtype=torch.long)
            if hc_src:
                data['local_vn', 'aggregates', 'column'].edge_index = torch.tensor(
                    [hc_src, hc_dst], dtype=torch.long)
                data['column', 'feeds_into', 'local_vn'].edge_index = torch.tensor(
                    [ch_src, ch_dst], dtype=torch.long)

        # ----- PCST flat 인덱싱: local_vn 블록 마지막 -----
        local_vn_flat_offset = T + C + F
        for hub_tbl_name, vn_idx in local_vn_to_id.items():
            metadata['node_metadata'][local_vn_flat_offset + vn_idx] = f"__local_vn__({hub_tbl_name})"

        existing_edges = list(metadata['edges'])
        existing_types = list(metadata['edge_types'])

        new_edges: List[Tuple[int, int]] = []
        new_types: List[str] = []
        for s, d in zip(ht_src, ht_dst):
            new_edges.append((s, local_vn_flat_offset + d))
            new_types.append('has_local_vn')
        for s, d in zip(th_src, th_dst):
            new_edges.append((local_vn_flat_offset + s, d))
            new_types.append('serves_table')
        for s, d in zip(hc_src, hc_dst):
            new_edges.append((local_vn_flat_offset + s, T + d))
            new_types.append('aggregates')
        for s, d in zip(ch_src, ch_dst):
            new_edges.append((T + s, local_vn_flat_offset + d))
            new_types.append('feeds_into')

        metadata['edges'] = existing_edges + new_edges
        metadata['edge_types'] = existing_types + new_types

        # ----- 신규 metadata keys -----
        metadata['hub_tables'] = list(hubs)
        metadata['hub_threshold'] = float(threshold)
        metadata['hub_strategy'] = self.hub_strategy
        metadata['hub_min_columns'] = int(self.hub_min_columns)
        metadata['local_vn_to_id'] = local_vn_to_id
        metadata['local_vn_flat_offset'] = int(local_vn_flat_offset)
        metadata['table_col_count'] = counts
        metadata['v6w3_variant'] = 'C'

        # ----- builder_info extra -----
        info = dict(self.last_info or {})
        info['builder_type'] = "V6W3HubLocalVNBuilder"
        info['builder_v6w3_variant'] = 'C'
        info['builder_hub_strategy'] = self.hub_strategy
        info['builder_hub_threshold'] = float(threshold)
        info['builder_num_hub_tables'] = int(H)
        info['builder_num_local_vn_edges'] = int(len(new_edges))
        info['builder_hub_table_names'] = [id_to_table[i] for i in hubs]
        timings = info.get('builder_timings', {}) or {}
        timings['v6w3_hub_id_s'] = float(t_hub_id)
        timings['v6w3_total_s'] = float(time.perf_counter() - t_total)
        info['builder_timings'] = timings
        self.last_info = info
        metadata['builder_info'] = dict(info)
        return data, metadata
