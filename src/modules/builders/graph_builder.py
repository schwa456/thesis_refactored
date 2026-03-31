import os
import sqlite3
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Tuple, List

from modules.registry import register
from modules.base import BaseGraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

@register("builder", "HeteroGraphBuilder")
class HeteroGraphBuilder(BaseGraphBuilder):
    def __init__(self, plm_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', **kwargs):
        super().__init__()
        logger.info(f"Loading PLM for initial node features: {plm_model_name}...")
        self.encoder = SentenceTransformer(plm_model_name)

    def _get_schema_info(self, db_path: str) -> Dict[str, Any]:
        """SQLite 파일에서 직접 스키마 정보를 추출합니다."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Tables 추출
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
        
        # 2. Columns & Foreign Keys 추출
        columns_dict = {}
        all_fks = []

        for table in tables:
            safe_table = table.replace("'", "''")

            # Columns
            cursor.execute(f"PRAGMA table_info('{safe_table}');")
            columns_data = cursor.fetchall()

            col_list = []
            for raw in columns_data:
                col_name = raw[1]
                col_type = row[2]

                samples = []
                try:
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 3;'
                    cursor.execute(query)
                    samples = [str(val[0])[:50] for val in cursor.fetchall()]
                except sqlite3.Error as e:
                    logger.debug(f"Failed to fetch samples for {table}.{col_name}: {e}")

                col_list.append({
                    "name": col_name,
                    "type": col_type,
                    "samaples": samples
                })
            columns_dict[table] = col_list
            
            # Foreign Keys
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

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict]:
        """
        db_id와 db_dir을 받아 스키마를 로드하고 그래프를 빌드합니다.
        """
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            # BIRD 구조상 db_id 폴더 안에 db_id.sqlite가 있음
            db_path = os.path.join(db_dir, "dev_databases", db_id, f"{db_id}.sqlite")
            
        schema_info = self._get_schema_info(db_path)
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
        data['table'].x = self.encoder.encode(table_texts, convert_to_tensor=True).cpu()
        data['column'].x = self.encoder.encode(col_texts, convert_to_tensor=True).cpu()
        if fk_texts:
            data['fk_node'].x = self.encoder.encode(fk_texts, convert_to_tensor=True).cpu()
        else:
            data['fk_node'].x = torch.empty((0, self.encoder.get_sentence_embedding_dimension())).cpu()

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
            if f_t in table_to_id and t_t in table_to_id:
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
            'edge_types': pcst_edge_types
        }
        
        return data, metadata