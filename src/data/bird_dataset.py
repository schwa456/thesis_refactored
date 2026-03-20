import os
import json
import torch
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm
from typing import List, Dict, Tuple

from modules.builders.graph_builder import HeteroGraphBuilder
from modules.encoders.api_encoder import APIEncoder
from utils.evaluator import parse_sql_elements
from utils.logger import get_logger

logger = get_logger(__name__)

class BIRDGraphDataset(Dataset):
    def __init__(self, json_path: str, db_dir: str, builder: HeteroGraphBuilder, encoder, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.db_dir = db_dir
        self.json_path = json_path
        self.builder = builder
        self.encoder = encoder

        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        logger.info(f"Loaded {len(self.raw_data)} queries from {json_path}")

        file_name = os.path.basename(json_path).split('.')[0]
        self.cache_path = f"/home/hyeonjin/thesis_refactored/data/processed/{file_name}_graphs.pt"

        self.data_list = self._process_data()
    
    def _process_data(self) -> List[HeteroData]:
        # 1. 로컬 캐시가 존재하는지 확인
        if os.path.exists(self.cache_path):
            logger.info(f"♻️ Found pre-processed graphs at {self.cache_path}. Loading...")
            try:
                return torch.load(self.cache_path, weights_only=False)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing...")

        # 2. 캐시가 없으면 원래 로직대로 생성
        logger.info(f"🚀 No cache found. Processing graphs from {self.json_path} (this may take a while)...")
        processed = []
        db_cache = {}

        for item in tqdm(self.raw_data, desc="Processing Dataset"):
            db_id = item['db_id']
            question = item['question']
            gold_sql = item.get('SQL', item.get('query', ''))

            # Graph Build (Caching)
            if db_id not in db_cache:
                graph_data, metadata = self.builder.build(db_id=db_id, db_dir=self.db_dir)
                db_cache[db_id] = (graph_data, metadata)
            else:
                graph_data, metadata = db_cache[db_id]

            data = graph_data.clone()

            # NLQ Embedding
            q_emb_tensor, q_mask_tensor = self.encoder.encode([question])
            
            data['query'] = q_emb_tensor.squeeze(0).cpu()
            data['query_mask'] = q_mask_tensor.squeeze(0).cpu()

            # Gold Schema Parsing and Lebeling
            gold_tables, gold_cols = parse_sql_elements(gold_sql)

            num_tables = data['table'].x.size(0)
            num_cols = data['column'].x.size(0)

            y_table = torch.zeros(num_tables, dtype=torch.float).cpu()
            y_col = torch.zeros(num_cols, dtype=torch.float).cpu()

            table_to_id = metadata.get('table_to_id', {})
            col_to_id = metadata.get('col_to_id', {})

            for t in gold_tables:
                if t in table_to_id:
                    y_table[table_to_id[t]] = 1.0
            
            for c in gold_cols:
                if c in col_to_id:
                    y_col[col_to_id[c]] = 1.0
            
            data['table'].y = y_table
            data['column'].y = y_col

            if 'fk_node' in data and data['fk_node'].x.size(0) > 0:
                num_fk = data['fk_node'].x.size(0)
                y_fk = torch.zeros(num_fk, dtype=torch.float)

                if num_fk > 0:
                    fk_to_id = metadata.get('fk_to_id', {})
                    for fk_edge_str, f_idx in fk_to_id.items():
                        try:
                            src_str, dst_str = fk_edge_str.split("->")
                            src_table = src_str.split(".")[0]
                            dst_table = dst_str.split(".")[0]

                            # 두 테이블이 모두 정답 쿼리에 포함되었다면, 이 FK는 Implicit Bridge로 간주
                            if src_table in gold_tables and dst_table in gold_tables:
                                y_fk[f_idx] = 1.0
                        except Exception as e:
                            logger.debug(f"FK String parsing error: {fk_edge_str} - {e}")
                            continue
                data['fk_node'].y = y_fk

            processed.append(data)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        logger.info(f"💾 Saving processed graphs to {self.cache_path} for future use...")
        torch.save(processed, self.cache_path)

        return processed
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
