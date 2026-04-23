import os
import csv
import json
import sqlite3
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Tuple, List, Optional

from modules.registry import register
from modules.base import BaseGraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

@register("builder", "HeteroGraphBuilder")
class HeteroGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        plm_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        add_t2t_edges: bool = True,
        **kwargs,
    ):
        super().__init__()
        logger.info(f"Loading PLM for initial node features: {plm_model_name}...")
        self.encoder = SentenceTransformer(plm_model_name)
        self.add_t2t_edges = add_t2t_edges
        self.last_info: Dict[str, Any] = {}

    @staticmethod
    def _text_stats(texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {"count": 0, "mean_chars": 0.0, "max_chars": 0, "min_chars": 0}
        lens = [len(t) for t in texts]
        return {
            "count": len(lens),
            "mean_chars": float(sum(lens) / len(lens)),
            "max_chars": int(max(lens)),
            "min_chars": int(min(lens)),
        }

    def _build_builder_info(
        self,
        *,
        builder_type: str,
        db_id: str,
        schema_info: Dict[str, Any],
        table_to_id: Dict[str, int],
        col_to_id: Dict[str, int],
        fk_to_id: Dict[str, int],
        table_texts: List[str],
        col_texts: List[str],
        fk_texts: List[str],
        pcst_edges: List[Tuple[int, int]],
        pcst_edge_types: List[str],
        metadata: Dict[str, Any],
        timings: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        edge_type_counts: Dict[str, int] = {}
        for et in pcst_edge_types:
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        fk_distance = metadata.get("fk_distance")
        finite_fk_distances: List[float] = []
        if isinstance(fk_distance, np.ndarray) and fk_distance.size > 0:
            mask = np.isfinite(fk_distance) & (fk_distance > 0)
            if mask.any():
                finite_fk_distances = fk_distance[mask].tolist()

        info = {
            "builder_type": builder_type,
            "builder_db_id": db_id,
            "builder_num_tables": len(table_to_id),
            "builder_num_columns": len(col_to_id),
            "builder_num_fk_nodes": len(fk_to_id),
            "builder_num_fks_raw": len(schema_info.get("foreign_keys", []) or []),
            "builder_num_edges_total": len(pcst_edges),
            "builder_edge_type_counts": edge_type_counts,
            "builder_table_text_stats": self._text_stats(table_texts),
            "builder_col_text_stats": self._text_stats(col_texts),
            "builder_fk_text_stats": self._text_stats(fk_texts),
            "builder_fk_num_components": int(metadata.get("fk_num_components", 0) or 0),
            "builder_fk_mean_distance": (
                float(sum(finite_fk_distances) / len(finite_fk_distances))
                if finite_fk_distances else 0.0
            ),
            "builder_fk_max_distance": (
                int(max(finite_fk_distances)) if finite_fk_distances else 0
            ),
            "builder_fk_pair_reachable": int(
                np.count_nonzero(metadata["fk_reachability"])
                - len(table_to_id)
            ) if isinstance(metadata.get("fk_reachability"), np.ndarray) and metadata["fk_reachability"].size else 0,
            "builder_timings": timings,
        }
        if extra:
            info.update({f"builder_{k}": v for k, v in extra.items()})
        return info

    def _get_schema_info(self, db_path: str) -> Dict[str, Any]:
        """SQLite 파일에서 직접 스키마 정보를 추출합니다."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # NFS-hosted large SQLite 상에서 per-column DISTINCT 쿼리가 N번 full-scan
        # 을 유발. 컬럼 샘플은 enrichment 용도이므로 단일 top-K scan 으로 대체.
        try:
            cursor.execute("PRAGMA cache_size = -524288;")  # 512MB page cache
            cursor.execute("PRAGMA temp_store = MEMORY;")
        except sqlite3.Error:
            pass

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']

        columns_dict = {}
        all_fks = []

        SAMPLE_SCAN_LIMIT = 50
        MAX_UNIQUE = 3

        for table in tables:
            safe_table = table.replace("'", "''")

            cursor.execute(f"PRAGMA table_info('{safe_table}');")
            columns_data = cursor.fetchall()
            col_names = [row[1] for row in columns_data]
            col_types = [row[2] for row in columns_data]

            sample_rows: List[tuple] = []
            if col_names:
                try:
                    quoted_cols = ", ".join(f'"{c}"' for c in col_names)
                    cursor.execute(
                        f'SELECT {quoted_cols} FROM "{table}" LIMIT {SAMPLE_SCAN_LIMIT};'
                    )
                    sample_rows = cursor.fetchall()
                except sqlite3.Error as e:
                    logger.debug(f"Failed to scan {table}: {e}")

            col_list = []
            for i, (col_name, col_type) in enumerate(zip(col_names, col_types)):
                seen: List[str] = []
                for row in sample_rows:
                    v = row[i]
                    if v is None:
                        continue
                    sv = str(v)[:50]
                    if sv in seen:
                        continue
                    seen.append(sv)
                    if len(seen) >= MAX_UNIQUE:
                        break
                col_list.append({
                    "name": col_name,
                    "type": col_type,
                    "samples": seen,
                })
            columns_dict[table] = col_list

            cursor.execute(f"PRAGMA foreign_key_list('{safe_table}');")
            for row in cursor.fetchall():
                all_fks.append({
                    "from_table": table,
                    "from_column": row[3],
                    "to_table": row[2],
                    "to_column": row[4]
                })

        conn.close()
        return {"tables": tables, "columns": columns_dict, "foreign_keys": all_fks}

    def _generate_fk_descriptions(self, foreign_keys: List[Dict]) -> Dict[str, str]:
        """FK 관계를 자연어 설명으로 변환합니다 (GAT 학습용)."""
        descriptions = {}
        for fk in foreign_keys:
            edge_id = f"{fk['from_table']}.{fk['from_column']}->{fk['to_table']}.{fk['to_column']}"
            desc = f"Foreign key relationship connecting {fk['from_table']}'s {fk['from_column']} to {fk['to_table']}'s {fk['to_column']}."
            descriptions[edge_id] = desc
        return descriptions

    def _compute_fk_reachability(
        self,
        schema_info: Dict[str, Any],
        table_to_id: Dict[str, int],
    ) -> Dict[str, Any]:
        """Symbolic Layer 1 (proposal #9 — Neurosymbolic) precompute.

        Builds the table-level FK graph and exposes:
          - fk_adjacency            (T,T) int8  directed (from -> to)
          - fk_adjacency_undirected (T,T) int8  symmetric (join-reachable)
          - fk_reachability         (T,T) bool  transitive closure (undirected)
          - fk_distance             (T,T) f32   BFS distance (inf if unreachable)
          - fk_shortest_paths       Dict[(i,j) -> {distance, table_path,
                                                   edge_path, fk_edge_ids}]
          - fk_components           Dict[table_idx -> component_id]
          - fk_num_components       int
          - fk_edge_lookup          Dict[(i,j) -> List[fk_edge_id]] (directed)

        Cost: BIRD schemas have T < 20 → BFS/Floyd-style work is microseconds.
        Downstream consumers (Selector S-V / Extractor E-III / Filter FL-III)
        may freely ignore these keys; nothing existing depends on them.
        """
        T = len(table_to_id)
        if T == 0:
            return {
                "fk_adjacency": np.zeros((0, 0), dtype=np.int8),
                "fk_adjacency_undirected": np.zeros((0, 0), dtype=np.int8),
                "fk_reachability": np.zeros((0, 0), dtype=bool),
                "fk_distance": np.zeros((0, 0), dtype=np.float32),
                "fk_shortest_paths": {},
                "fk_components": {},
                "fk_num_components": 0,
                "fk_edge_lookup": {},
            }

        adj_dir = np.zeros((T, T), dtype=np.int8)
        edge_lookup: Dict[Tuple[int, int], List[str]] = {}
        for fk in schema_info.get("foreign_keys", []):
            f_t, t_t = fk["from_table"], fk["to_table"]
            if f_t not in table_to_id or t_t not in table_to_id:
                continue
            i, j = table_to_id[f_t], table_to_id[t_t]
            if i == j:
                continue  # self-FK: ignore for join-path purposes
            adj_dir[i, j] = 1
            edge_id = f"{f_t}.{fk['from_column']}->{t_t}.{fk['to_column']}"
            edge_lookup.setdefault((i, j), []).append(edge_id)

        adj_undir = np.maximum(adj_dir, adj_dir.T)

        sparse_undir = csr_matrix(adj_undir.astype(np.float32))
        if sparse_undir.nnz == 0:
            distances = np.full((T, T), np.inf, dtype=np.float32)
            np.fill_diagonal(distances, 0.0)
            predecessors = np.full((T, T), -9999, dtype=np.int32)
        else:
            distances, predecessors = shortest_path(
                sparse_undir, directed=False,
                return_predecessors=True, unweighted=True,
            )
            distances = distances.astype(np.float32)

        reachability = np.isfinite(distances)

        shortest_paths: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for i in range(T):
            for j in range(T):
                if i == j or not reachability[i, j]:
                    continue
                edges_path: List[Tuple[int, int]] = []
                cur = j
                visited = {cur}
                while cur != i:
                    prev = int(predecessors[i, cur])
                    if prev < 0 or prev in visited:
                        break
                    edges_path.append((prev, cur))
                    visited.add(prev)
                    cur = prev
                if cur != i:
                    continue
                edges_path.reverse()
                hop_edge_ids: List[List[str]] = []
                for src_t, dst_t in edges_path:
                    ids = (edge_lookup.get((src_t, dst_t)) or
                           edge_lookup.get((dst_t, src_t)) or [])
                    hop_edge_ids.append(list(ids))
                shortest_paths[(i, j)] = {
                    "distance": int(distances[i, j]),
                    "table_path": [i] + [d for _, d in edges_path],
                    "edge_path": edges_path,
                    "fk_edge_ids": hop_edge_ids,
                }

        if sparse_undir.nnz == 0:
            n_comp, labels = T, np.arange(T, dtype=np.int32)
        else:
            n_comp, labels = connected_components(sparse_undir, directed=False)
        components = {int(idx): int(labels[idx]) for idx in range(T)}

        return {
            "fk_adjacency": adj_dir,
            "fk_adjacency_undirected": adj_undir,
            "fk_reachability": reachability,
            "fk_distance": distances,
            "fk_shortest_paths": shortest_paths,
            "fk_components": components,
            "fk_num_components": int(n_comp),
            "fk_edge_lookup": edge_lookup,
        }

    def _compute_schema_diameter(
        self,
        table_to_id: Dict[str, int],
        col_to_id: Dict[str, int],
        fk_to_id: Dict[str, int],
        pcst_edges: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """B-III.b — full-hetero schema graph diameter (advisor 2026-04-21 의견 2).

        Computes the **full heterograph** (table + column + fk_node) diameter
        treating all edges as undirected (message passing is bidirectional).
        Disconnected components: per-component diameter, take the max.
        Eccentricity per flat node id is exposed for finer-grained Selector
        depth control.

        Co-execution with `_compute_fk_reachability` is the design intent —
        both run at end of build() with negligible cost on BIRD-sized schemas.
        Table-only FK subgraph diameter is intentionally **not** computed
        here; it is tracked as a separate sub-task (different semantics).
        """
        N = len(table_to_id) + len(col_to_id) + len(fk_to_id)
        if N == 0:
            return {"schema_diameter": 0, "schema_eccentricity": {}}
        if not pcst_edges:
            return {
                "schema_diameter": 0,
                "schema_eccentricity": {int(i): 0 for i in range(N)},
            }

        rows = [s for s, _ in pcst_edges] + [d for _, d in pcst_edges]
        cols = [d for _, d in pcst_edges] + [s for s, _ in pcst_edges]
        weights = [1.0] * (2 * len(pcst_edges))
        sparse = csr_matrix((weights, (rows, cols)), shape=(N, N))

        distances = shortest_path(sparse, directed=False, unweighted=True).astype(np.float32)

        ecc: Dict[int, int] = {}
        for i in range(N):
            row = distances[i]
            finite = row[np.isfinite(row)]
            ecc[int(i)] = int(finite.max()) if finite.size else 0
        diameter = int(max(ecc.values())) if ecc else 0
        return {
            "schema_diameter": diameter,
            "schema_eccentricity": ecc,
        }

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        """
        db_id와 db_dir을 받아 스키마를 로드하고 그래프를 빌드합니다.
        """
        t_total = time.perf_counter()
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            # BIRD 구조상 db_id 폴더 안에 db_id.sqlite가 있음
            db_path = os.path.join(db_dir, "dev_databases", db_id, f"{db_id}.sqlite")

        t = time.perf_counter()
        schema_info = self._get_schema_info(db_path)
        t_schema = time.perf_counter() - t
        fk_descriptions = self._generate_fk_descriptions(schema_info["foreign_keys"])
        
        data = HeteroData()
        table_to_id, col_to_id, fk_to_id = {}, {}, {}
        table_texts, col_texts, fk_texts = [], [], []

        # ---------------------------------------------------------
        # Step 1: Nodes (기존 로직 유지)
        # ---------------------------------------------------------
        for idx, t in enumerate(schema_info["tables"]):
            table_to_id[t] = idx
            table_texts.append(f"Table: {t}")

        c_idx = 0
        for table, cols in schema_info["columns"].items():
            for col in cols:
                full_name = f"{table}.{col['name']}"
                col_to_id[full_name] = c_idx
                sample_str = f" Example values: {', '.join(col['samples'])}." if col['samples'] else ""
                col_texts.append(f"Column: {col['name']} in table {table}, type {col['type']}.{sample_str}")
                c_idx += 1

        for idx, (edge_id, desc) in enumerate(fk_descriptions.items()):
            fk_to_id[edge_id] = idx
            fk_texts.append(desc)

        # Step 2: Encoding
        t = time.perf_counter()
        data['table'].x = self.encoder.encode(table_texts, convert_to_tensor=True).cpu()
        data['column'].x = self.encoder.encode(col_texts, convert_to_tensor=True).cpu()
        if fk_texts:
            data['fk_node'].x = self.encoder.encode(fk_texts, convert_to_tensor=True).cpu()
        else:
            data['fk_node'].x = torch.empty((0, self.encoder.get_sentence_embedding_dimension())).cpu()
        t_encode = time.perf_counter() - t

        # Step 3: Edges
        h_src, h_dst = [], [] # Table -> Column
        f_src, f_dst = [], [] # Col -> FK
        r_src, r_dst = [], [] # FK -> Col
        t_fk_src, t_fk_dst = [], [] # Table <-> Table (Macro Edge)

        for table, cols in schema_info["columns"].items():
            t_id = table_to_id[table]
            for col in cols:
                c_id = col_to_id[f"{table}.{col['name']}"]
                h_src.append(t_id); h_dst.append(c_id)

        for fk in schema_info["foreign_keys"]:
            f_col = f"{fk['from_table']}.{fk['from_column']}"
            t_col = f"{fk['to_table']}.{fk['to_column']}"
            edge_id = f"{f_col}->{t_col}"

            if edge_id in fk_to_id and f_col in col_to_id and t_col in col_to_id:
                fid, cid1, cid2 = fk_to_id[edge_id], col_to_id[f_col], col_to_id[t_col]
                f_src.append(cid1); f_dst.append(fid)
                r_src.append(fid); r_dst.append(cid2)

            f_t = fk['from_table']
            t_t = fk['to_table']
            if self.add_t2t_edges and f_t in table_to_id and t_t in table_to_id:
                t_fk_src.extend([table_to_id[f_t], table_to_id[t_t]])
                t_fk_dst.extend([table_to_id[t_t], table_to_id[f_t]])

        data['table', 'has_column', 'column'].edge_index = torch.tensor([h_src, h_dst], dtype=torch.long)
        data['column', 'belongs_to', 'table'].edge_index = torch.tensor([h_dst, h_src], dtype=torch.long)

        if f_src:
            data['column', 'is_source_of', 'fk_node'].edge_index = torch.tensor([f_src, f_dst], dtype=torch.long)
            data['fk_node', 'points_to', 'column'].edge_index = torch.tensor([r_src, r_dst], dtype=torch.long)

        if t_fk_src:
            data['table', 'table_to_table', 'table'].edge_index = torch.tensor([t_fk_src, t_fk_dst], dtype=torch.long)

        # Step 4: Metadata for PCST
        num_t, num_c = len(table_to_id), len(col_to_id)
        node_meta = {}
        for k, v in table_to_id.items(): node_meta[v] = k
        for k, v in col_to_id.items(): node_meta[v + num_t] = k
        for k, v in fk_to_id.items(): node_meta[v + num_t + num_c] = k

        pcst_edges = (
            [(s, d + num_t) for s, d in zip(h_src, h_dst)] +
            [(s + num_t, d + num_t + num_c) for s, d in zip(f_src, f_dst)] +
            [(s + num_t + num_c, d + num_t) for s, d in zip(r_src, r_dst)] +
            [(s, d) for s, d in zip(t_fk_src, t_fk_dst)]
        )

        pcst_edge_types = (
            ['belongs_to'] * len(h_src) +
            ['is_source_of'] * len(f_src) +
            ['points_to'] * len(r_src) +
            ['table_to_table'] * len(t_fk_src) # Macro Edge 타입
        )

        metadata = {
            'table_to_id': table_to_id, 'col_to_id': col_to_id, 'fk_to_id': fk_to_id,
            'node_metadata': node_meta,
            'edges': pcst_edges,
            'edge_types': pcst_edge_types,
            'add_t2t_edges': bool(self.add_t2t_edges),
        }
        t = time.perf_counter()
        metadata.update(self._compute_fk_reachability(schema_info, table_to_id))
        t_reach = time.perf_counter() - t
        t = time.perf_counter()
        metadata.update(self._compute_schema_diameter(table_to_id, col_to_id, fk_to_id, pcst_edges))
        t_diam = time.perf_counter() - t

        self.last_info = self._build_builder_info(
            builder_type="HeteroGraphBuilder",
            db_id=db_id,
            schema_info=schema_info,
            table_to_id=table_to_id,
            col_to_id=col_to_id,
            fk_to_id=fk_to_id,
            table_texts=table_texts,
            col_texts=col_texts,
            fk_texts=fk_texts,
            pcst_edges=pcst_edges,
            pcst_edge_types=pcst_edge_types,
            metadata=metadata,
            timings={
                "schema_load_s": float(t_schema),
                "encode_s": float(t_encode),
                "reachability_s": float(t_reach),
                "diameter_s": float(t_diam),
                "total_s": float(time.perf_counter() - t_total),
            },
            extra={
                "add_t2t_edges": bool(self.add_t2t_edges),
                "schema_diameter": int(metadata.get("schema_diameter", 0) or 0),
            },
        )
        metadata['builder_info'] = dict(self.last_info)
        return data, metadata


@register("builder", "EnrichedHeteroGraphBuilder")
class EnrichedHeteroGraphBuilder(HeteroGraphBuilder):
    """
    HeteroGraphBuilder를 확장하여 노드 텍스트에 추가 메타정보를 주입합니다.
    - Table 노드: tables.json의 자연어 테이블명
    - Column 노드: database_description/*.csv의 column_description, value_description
    """
    def __init__(self, plm_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 tables_json_path: Optional[str] = None, **kwargs):
        super().__init__(plm_model_name=plm_model_name, **kwargs)
        self.table_nl_names: Dict[str, Dict[str, str]] = {}  # db_id -> {original_name: nl_name}
        self.column_nl_names: Dict[str, Dict[str, str]] = {}  # db_id -> {table.col: nl_name}
        if tables_json_path and os.path.exists(tables_json_path):
            self._load_tables_json(tables_json_path)
            logger.info(f"Loaded NL names for {len(self.table_nl_names)} databases from {tables_json_path}")

    def _load_tables_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        for db in tables_data:
            db_id = db['db_id']
            # Table NL names
            originals = db.get('table_names_original', [])
            nl_names = db.get('table_names', [])
            self.table_nl_names[db_id] = {}
            for orig, nl in zip(originals, nl_names):
                if orig.lower() != nl.lower():
                    self.table_nl_names[db_id][orig] = nl
            # Column NL names
            col_originals = db.get('column_names_original', [])
            col_nl = db.get('column_names', [])
            self.column_nl_names[db_id] = {}
            for (tid, orig_col), (_, nl_col) in zip(col_originals, col_nl):
                if tid < 0:
                    continue
                table_name = originals[tid] if tid < len(originals) else ""
                if orig_col.lower() != nl_col.lower():
                    self.column_nl_names[db_id][f"{table_name}.{orig_col}"] = nl_col

    def _load_column_descriptions(self, db_path: str, db_id: str) -> Dict[str, Dict[str, str]]:
        """database_description/*.csv에서 컬럼 설명을 로드합니다."""
        desc_dir = os.path.join(os.path.dirname(db_path), "database_description")
        if not os.path.isdir(desc_dir):
            return {}

        result = {}  # "table.column" -> {"description": ..., "value_description": ...}
        for csv_file in os.listdir(desc_dir):
            if not csv_file.endswith('.csv'):
                continue
            table_name = csv_file[:-4]  # strip .csv
            csv_path = os.path.join(desc_dir, csv_file)
            try:
                with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        orig_col = row.get('original_column_name', '').strip()
                        if not orig_col:
                            continue
                        key = f"{table_name}.{orig_col}"
                        col_desc = row.get('column_description', '').strip()
                        val_desc = row.get('value_description', '').strip()
                        if col_desc or val_desc:
                            result[key] = {
                                "description": col_desc,
                                "value_description": val_desc
                            }
            except Exception as e:
                logger.debug(f"Failed to read {csv_path}: {e}")
        return result

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        t_total = time.perf_counter()
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(db_dir, "dev_databases", db_id, f"{db_id}.sqlite")

        t = time.perf_counter()
        schema_info = self._get_schema_info(db_path)
        t_schema = time.perf_counter() - t
        fk_descriptions = self._generate_fk_descriptions(schema_info["foreign_keys"])

        # Enrichment sources
        t = time.perf_counter()
        col_descriptions = self._load_column_descriptions(db_path, db_id)
        t_enrich_load = time.perf_counter() - t
        table_nl = self.table_nl_names.get(db_id, {})
        col_nl = self.column_nl_names.get(db_id, {})
        _enrich_tbl_nl_applied = 0
        _enrich_col_nl_applied = 0
        _enrich_col_desc_applied = 0
        _enrich_col_val_desc_applied = 0

        data = HeteroData()
        table_to_id, col_to_id, fk_to_id = {}, {}, {}
        table_texts, col_texts, fk_texts = [], [], []

        # --- Table Nodes (enriched) ---
        for idx, t in enumerate(schema_info["tables"]):
            table_to_id[t] = idx
            nl_name = table_nl.get(t, "")
            if nl_name:
                table_texts.append(f"Table: {t} ({nl_name})")
                _enrich_tbl_nl_applied += 1
            else:
                table_texts.append(f"Table: {t}")

        # --- Column Nodes (enriched) ---
        c_idx = 0
        for table, cols in schema_info["columns"].items():
            for col in cols:
                full_name = f"{table}.{col['name']}"
                col_to_id[full_name] = c_idx

                # Base text
                parts = [f"Column: {col['name']} in table {table}, type {col['type']}."]

                # NL column name from tables.json
                nl_col_name = col_nl.get(full_name, "")
                if nl_col_name:
                    parts.append(f"Meaning: {nl_col_name}.")
                    _enrich_col_nl_applied += 1

                # Column description from CSV
                desc_info = col_descriptions.get(full_name, {})
                col_desc = desc_info.get("description", "")
                if col_desc and col_desc.lower() != col['name'].lower():
                    parts.append(f"Description: {col_desc}.")
                    _enrich_col_desc_applied += 1

                # Value description from CSV
                val_desc = desc_info.get("value_description", "")
                if val_desc:
                    # Truncate very long value descriptions
                    if len(val_desc) > 200:
                        val_desc = val_desc[:200] + "..."
                    parts.append(f"Values info: {val_desc}.")
                    _enrich_col_val_desc_applied += 1

                # Example values from DB
                if col['samples']:
                    parts.append(f"Example values: {', '.join(col['samples'])}.")

                col_texts.append(" ".join(parts))
                c_idx += 1

        # --- FK Nodes (unchanged) ---
        for idx, (edge_id, desc) in enumerate(fk_descriptions.items()):
            fk_to_id[edge_id] = idx
            fk_texts.append(desc)

        # --- Encoding ---
        t = time.perf_counter()
        data['table'].x = self.encoder.encode(table_texts, convert_to_tensor=True).cpu()
        data['column'].x = self.encoder.encode(col_texts, convert_to_tensor=True).cpu()
        if fk_texts:
            data['fk_node'].x = self.encoder.encode(fk_texts, convert_to_tensor=True).cpu()
        else:
            data['fk_node'].x = torch.empty((0, self.encoder.get_sentence_embedding_dimension())).cpu()
        t_encode = time.perf_counter() - t

        # --- Edges (동일 로직) ---
        h_src, h_dst = [], []
        f_src, f_dst = [], []
        r_src, r_dst = [], []
        t_fk_src, t_fk_dst = [], []

        for table, cols in schema_info["columns"].items():
            t_id = table_to_id[table]
            for col in cols:
                c_id = col_to_id[f"{table}.{col['name']}"]
                h_src.append(t_id); h_dst.append(c_id)

        for fk in schema_info["foreign_keys"]:
            f_col = f"{fk['from_table']}.{fk['from_column']}"
            t_col = f"{fk['to_table']}.{fk['to_column']}"
            edge_id = f"{f_col}->{t_col}"
            if edge_id in fk_to_id and f_col in col_to_id and t_col in col_to_id:
                fid, cid1, cid2 = fk_to_id[edge_id], col_to_id[f_col], col_to_id[t_col]
                f_src.append(cid1); f_dst.append(fid)
                r_src.append(fid); r_dst.append(cid2)
            f_t = fk['from_table']
            t_t = fk['to_table']
            if self.add_t2t_edges and f_t in table_to_id and t_t in table_to_id:
                t_fk_src.extend([table_to_id[f_t], table_to_id[t_t]])
                t_fk_dst.extend([table_to_id[t_t], table_to_id[f_t]])

        data['table', 'has_column', 'column'].edge_index = torch.tensor([h_src, h_dst], dtype=torch.long)
        data['column', 'belongs_to', 'table'].edge_index = torch.tensor([h_dst, h_src], dtype=torch.long)
        if f_src:
            data['column', 'is_source_of', 'fk_node'].edge_index = torch.tensor([f_src, f_dst], dtype=torch.long)
            data['fk_node', 'points_to', 'column'].edge_index = torch.tensor([r_src, r_dst], dtype=torch.long)
        if t_fk_src:
            data['table', 'table_to_table', 'table'].edge_index = torch.tensor([t_fk_src, t_fk_dst], dtype=torch.long)

        # --- Metadata ---
        num_t, num_c = len(table_to_id), len(col_to_id)
        node_meta = {}
        for k, v in table_to_id.items(): node_meta[v] = k
        for k, v in col_to_id.items(): node_meta[v + num_t] = k
        for k, v in fk_to_id.items(): node_meta[v + num_t + num_c] = k

        pcst_edges = (
            [(s, d + num_t) for s, d in zip(h_src, h_dst)] +
            [(s + num_t, d + num_t + num_c) for s, d in zip(f_src, f_dst)] +
            [(s + num_t + num_c, d + num_t) for s, d in zip(r_src, r_dst)] +
            [(s, d) for s, d in zip(t_fk_src, t_fk_dst)]
        )
        pcst_edge_types = (
            ['belongs_to'] * len(h_src) +
            ['is_source_of'] * len(f_src) +
            ['points_to'] * len(r_src) +
            ['table_to_table'] * len(t_fk_src)
        )

        metadata = {
            'table_to_id': table_to_id, 'col_to_id': col_to_id, 'fk_to_id': fk_to_id,
            'node_metadata': node_meta,
            'edges': pcst_edges,
            'edge_types': pcst_edge_types,
            'add_t2t_edges': bool(self.add_t2t_edges),
        }
        t = time.perf_counter()
        metadata.update(self._compute_fk_reachability(schema_info, table_to_id))
        t_reach = time.perf_counter() - t
        t = time.perf_counter()
        metadata.update(self._compute_schema_diameter(table_to_id, col_to_id, fk_to_id, pcst_edges))
        t_diam = time.perf_counter() - t

        self.last_info = self._build_builder_info(
            builder_type="EnrichedHeteroGraphBuilder",
            db_id=db_id,
            schema_info=schema_info,
            table_to_id=table_to_id,
            col_to_id=col_to_id,
            fk_to_id=fk_to_id,
            table_texts=table_texts,
            col_texts=col_texts,
            fk_texts=fk_texts,
            pcst_edges=pcst_edges,
            pcst_edge_types=pcst_edge_types,
            metadata=metadata,
            timings={
                "schema_load_s": float(t_schema),
                "enrich_load_s": float(t_enrich_load),
                "encode_s": float(t_encode),
                "reachability_s": float(t_reach),
                "diameter_s": float(t_diam),
                "total_s": float(time.perf_counter() - t_total),
            },
            extra={
                "enrich_tbl_nl_applied": _enrich_tbl_nl_applied,
                "enrich_col_nl_applied": _enrich_col_nl_applied,
                "enrich_col_desc_applied": _enrich_col_desc_applied,
                "enrich_col_val_desc_applied": _enrich_col_val_desc_applied,
                "enrich_col_desc_rows": len(col_descriptions),
                "enrich_tbl_nl_available": len(table_nl),
                "enrich_col_nl_available": len(col_nl),
                "add_t2t_edges": bool(self.add_t2t_edges),
                "schema_diameter": int(metadata.get("schema_diameter", 0) or 0),
            },
        )
        metadata['builder_info'] = dict(self.last_info)
        return data, metadata


@register("builder", "RFMCompatibleBuilder")
class RFMCompatibleBuilder(EnrichedHeteroGraphBuilder):
    """Zero-shot RFM serializer (proposal #2 — Schema/Table-Llama family).

    Adds a structural-token serialization (`[TAB] ... [COL] ... [FK→] ...`)
    on top of the EnrichedHeteroGraphBuilder graph, and exposes both the
    serialized string and its whitespace-token list via metadata.

    `serialize(db_id, db_dir)` is callable standalone for offline encoding /
    profiling without paying the full graph-build cost.
    """

    SPECIAL_TOKENS = {
        "db": "[DB]",
        "tab": "[TAB]",
        "tab_end": "[/TAB]",
        "col": "[COL]",
        "type": "[TYPE]",
        "pk": "[PK]",
        "desc": "[DESC]",
        "val": "[VAL]",
        "fk_block": "[FKS]",
        "fk_arrow": "[FK→]",
    }

    def __init__(
        self,
        include_values: bool = True,
        max_values: int = 3,
        value_max_chars: int = 50,
        max_desc_chars: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_values = include_values
        self.max_values = max_values
        self.value_max_chars = value_max_chars
        self.max_desc_chars = max_desc_chars

    def _resolve_db_path(self, db_id: str, db_dir: str) -> str:
        path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(path):
            path = os.path.join(db_dir, "dev_databases", db_id, f"{db_id}.sqlite")
        return path

    def _serialize_from_schema(
        self,
        db_id: str,
        schema_info: Dict[str, Any],
        col_descriptions: Dict[str, Dict[str, str]],
        table_nl: Dict[str, str],
        col_nl: Dict[str, str],
        include_values: bool,
        max_values: int,
    ) -> Tuple[str, List[str]]:
        S = self.SPECIAL_TOKENS
        parts: List[str] = [S["db"], db_id]

        for table in schema_info["tables"]:
            parts.append(S["tab"])
            parts.append(table)
            nl = table_nl.get(table)
            if nl:
                parts.append(f"({nl})")
            for col in schema_info["columns"].get(table, []):
                parts.append(S["col"])
                parts.append(col["name"])
                cnl = col_nl.get(f"{table}.{col['name']}")
                if cnl:
                    parts.append(f"({cnl})")
                parts.append(S["type"])
                parts.append(str(col.get("type") or "TEXT"))
                desc = col_descriptions.get(f"{table}.{col['name']}", {})
                cd = (desc.get("description") or "").strip()
                if cd and cd.lower() != col["name"].lower():
                    if len(cd) > self.max_desc_chars:
                        cd = cd[: self.max_desc_chars] + "..."
                    parts.append(S["desc"])
                    parts.append(cd)
                if include_values and col.get("samples"):
                    samples = [
                        (s[: self.value_max_chars] + "…") if len(s) > self.value_max_chars else s
                        for s in col["samples"][:max_values]
                    ]
                    parts.append(S["val"])
                    parts.append(", ".join(samples))
            parts.append(S["tab_end"])

        if schema_info["foreign_keys"]:
            parts.append(S["fk_block"])
            for fk in schema_info["foreign_keys"]:
                parts.append(f"{fk['from_table']}.{fk['from_column']}")
                parts.append(S["fk_arrow"])
                parts.append(f"{fk['to_table']}.{fk['to_column']}")

        text = " ".join(parts)
        tokens = text.split()
        return text, tokens

    def serialize(
        self,
        db_id: str,
        db_dir: str,
        include_values: Optional[bool] = None,
        max_values: Optional[int] = None,
    ) -> str:
        db_path = self._resolve_db_path(db_id, db_dir)
        schema_info = self._get_schema_info(db_path)
        col_descriptions = self._load_column_descriptions(db_path, db_id)
        table_nl = self.table_nl_names.get(db_id, {})
        col_nl = self.column_nl_names.get(db_id, {})
        text, _ = self._serialize_from_schema(
            db_id, schema_info, col_descriptions, table_nl, col_nl,
            self.include_values if include_values is None else include_values,
            self.max_values if max_values is None else max_values,
        )
        return text

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        data, meta = super().build(db_id, db_dir)
        # Re-extract for serialization (cheap SQLite read; avoids cross-call coupling).
        db_path = self._resolve_db_path(db_id, db_dir)
        schema_info = self._get_schema_info(db_path)
        col_descriptions = self._load_column_descriptions(db_path, db_id)
        table_nl = self.table_nl_names.get(db_id, {})
        col_nl = self.column_nl_names.get(db_id, {})
        t_ser = time.perf_counter()
        text, tokens = self._serialize_from_schema(
            db_id, schema_info, col_descriptions, table_nl, col_nl,
            self.include_values, self.max_values,
        )
        ser_s = time.perf_counter() - t_ser
        meta["rfm_text"] = text
        meta["rfm_tokens"] = tokens
        meta["rfm_special_tokens"] = list(self.SPECIAL_TOKENS.values())

        info = dict(getattr(self, "last_info", {}) or meta.get("builder_info", {}) or {})
        info["builder_type"] = "RFMCompatibleBuilder"
        info["builder_rfm_char_count"] = len(text)
        info["builder_rfm_token_count"] = len(tokens)
        info["builder_rfm_include_values"] = bool(self.include_values)
        info["builder_rfm_max_values"] = int(self.max_values)
        timings = info.get("builder_timings", {}) or {}
        timings["rfm_serialize_s"] = float(ser_s)
        info["builder_timings"] = timings
        self.last_info = info
        meta["builder_info"] = dict(info)
        return data, meta


@register("builder", "TripletGraphBuilder")
class TripletGraphBuilder(EnrichedHeteroGraphBuilder):
    """Extends EnrichedHeteroGraphBuilder with triplet relation edge embeddings."""

    def __init__(self, triplet_path='data/processed/triplet_relations.json', **kwargs):
        super().__init__(**kwargs)
        self.triplet_data = {}
        if os.path.exists(triplet_path):
            with open(triplet_path, 'r', encoding='utf-8') as f:
                self.triplet_data = json.load(f)
            logger.info(f"Loaded triplet relations for {len(self.triplet_data)} databases")

    def _build_triplet_lookup(self, db_id):
        db_data = self.triplet_data.get(db_id, {})
        lookup = {}
        for edge in db_data.get('edges', []):
            key = (edge['subject'], edge['object'])
            lookup[key] = edge['relation']
            rev_key = (edge['object'], edge['subject'])
            if rev_key not in lookup:
                lookup[rev_key] = edge['relation']
        return lookup

    def build(self, db_id, db_dir):
        data, metadata = super().build(db_id, db_dir)
        triplet_lookup = self._build_triplet_lookup(db_id)

        node_meta = metadata['node_metadata']
        edges = metadata['edges']
        edge_types = metadata['edge_types']

        triplet_texts = []
        triplet_matched = 0
        triplet_fk_fallback = 0
        triplet_edge_type_fallback = 0
        for (src_id, dst_id), et in zip(edges, edge_types):
            src_name = node_meta.get(src_id, str(src_id))
            dst_name = node_meta.get(dst_id, str(dst_id))

            relation = triplet_lookup.get((src_name, dst_name))
            if relation is not None:
                triplet_matched += 1
            else:
                # Try FK node format: "table.col->table.col"
                if '->' in src_name:
                    parts = src_name.split('->')
                    resolved = triplet_lookup.get(
                        (parts[0].strip(), parts[1].strip()))
                    if resolved is not None:
                        relation = resolved
                        triplet_fk_fallback += 1
                    else:
                        relation = et
                        triplet_edge_type_fallback += 1
                elif '->' in dst_name:
                    parts = dst_name.split('->')
                    resolved = triplet_lookup.get(
                        (parts[0].strip(), parts[1].strip()))
                    if resolved is not None:
                        relation = resolved
                        triplet_fk_fallback += 1
                    else:
                        relation = et
                        triplet_edge_type_fallback += 1
                else:
                    relation = et
                    triplet_edge_type_fallback += 1

            src_short = src_name.split('.')[-1] if '.' in src_name and '->' not in src_name else src_name
            dst_short = dst_name.split('.')[-1] if '.' in dst_name and '->' not in dst_name else dst_name
            triplet_texts.append(f"{src_short} {relation} {dst_short}")

        t_emb = time.perf_counter()
        if triplet_texts:
            edge_embeddings = self.encoder.encode(
                triplet_texts, convert_to_tensor=True).cpu()
        else:
            edge_embeddings = torch.empty(
                (0, self.encoder.get_sentence_embedding_dimension()))
        t_emb = time.perf_counter() - t_emb

        metadata['edge_embeddings'] = edge_embeddings
        metadata['triplet_texts'] = triplet_texts

        info = dict(getattr(self, "last_info", {}) or metadata.get("builder_info", {}) or {})
        info["builder_type"] = "TripletGraphBuilder"
        info["builder_triplet_edges_total"] = len(triplet_texts)
        info["builder_triplet_matched"] = triplet_matched
        info["builder_triplet_fk_fallback"] = triplet_fk_fallback
        info["builder_triplet_edge_type_fallback"] = triplet_edge_type_fallback
        info["builder_triplet_lookup_size"] = len(triplet_lookup)
        info["builder_triplet_loaded_dbs"] = len(self.triplet_data)
        info["builder_triplet_edge_emb_shape"] = list(edge_embeddings.shape)
        timings = info.get("builder_timings", {}) or {}
        timings["triplet_embed_s"] = float(t_emb)
        info["builder_timings"] = timings
        self.last_info = info
        metadata["builder_info"] = dict(info)
        return data, metadata