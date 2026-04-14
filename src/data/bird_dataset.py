import os
import json
import torch
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm
from typing import List, Dict, Tuple

from utils.evaluator import parse_sql_elements
from utils.logger import get_logger

logger = get_logger(__name__)

class BIRDGraphDataset(Dataset):
    def __init__(self, json_path: str, db_dir: str, builder, encoder, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.db_dir = db_dir
        self.json_path = json_path
        self.builder = builder
        self.encoder = encoder

        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        file_name = os.path.basename(json_path).split('.')[0]
        # Enriched builder uses a separate cache on NAS to save local disk
        builder_suffix = "_enriched" if type(builder).__name__ == "EnrichedHeteroGraphBuilder" else ""
        # PLMEncoder produces sentence-level [1, 384], TokenEncoder produces token-level [seq_len, 384]
        encoder_name = type(encoder).__name__
        encoder_suffix = "_plm" if encoder_name == "LocalPLMEncoder" else ""
        if builder_suffix:
            cache_dir = os.path.dirname(json_path)  # NAS: same dir as train.json
            self.cache_path = os.path.join(cache_dir, f"{file_name}{builder_suffix}{encoder_suffix}_graphs.pt")
        else:
            self.cache_path = f"/home/hyeonjin/thesis_refactored/data/processed/{file_name}{encoder_suffix}_graphs.pt"

        # _process_data에서 데이터 리스트를 생성하거나 로드함
        self.data_list = self._get_or_create_data()
    
    def _get_or_create_data(self):
        # 1. 캐시가 있으면 로드
        if os.path.exists(self.cache_path):
            logger.info(f"♻️  Loading cached graphs from {self.cache_path}...")
            # HeteroData 객체이므로 weights_only=False 필수
            return torch.load(self.cache_path, weights_only=False)

        # 2. 캐시가 없으면 전체 빌드 (최초 1회만 실행됨)
        logger.info(f"🏗️  No cache found. Building graphs for {len(self.raw_data)} queries...")
        processed = []
        db_cache = {}

        for item in tqdm(self.raw_data, desc="Building Cache"):
            db_id = item['db_id']
            # 그래프 빌드
            if db_id not in db_cache:
                graph_data, metadata = self.builder.build(db_id=db_id, db_dir=self.db_dir)
                db_cache[db_id] = (graph_data, metadata)
            else:
                graph_data, metadata = db_cache[db_id]

            data = graph_data.clone()
            
            # 메타데이터 주입 (String 데이터도 저장 가능)
            data.db_id = db_id
            data.gold_sql = item.get('SQL', item.get('query', ''))
            
            # 질문 임베딩 (Encoding도 미리 해서 저장!)
            enc_result = self.encoder.encode([item['question']])
            # TokenEncoder returns (embeddings, mask) tuple; PLMEncoder returns tensor
            if isinstance(enc_result, tuple):
                q_emb = enc_result[0]
            else:
                q_emb = enc_result
            q_emb = q_emb.cpu()
            # token-level [1, seq_len, 384] → sentence-level [1, 384]
            if q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)  # [1, 384]
            # 반드시 [1, 384] 형태로 저장 (PyG DataLoader가 dim=0으로 cat → [B, 384])
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)  # [384] → [1, 384]
            data['query'] = q_emb

            # 정답 라벨링 (수정된 로직 적용)
            gold_tables, gold_cols = parse_sql_elements(data.gold_sql)
            table_to_id = metadata.get('table_to_id', {})
            col_to_id = metadata.get('col_to_id', {})

            y_table = torch.zeros(data['table'].x.size(0))
            y_col = torch.zeros(data['column'].x.size(0))

            for t in gold_tables:
                if t in table_to_id: y_table[table_to_id[t]] = 1.0
            
            for c in gold_cols:
                # table.column 형태와 column 단독 형태 모두 체크
                if c in col_to_id: 
                    y_col[col_to_id[c]] = 1.0
                else:
                    for full_name, idx in col_to_id.items():
                        if full_name.endswith(f".{c}"):
                            # 해당 컬럼의 테이블이 정답 테이블 목록에 있을 때만 1로 세팅
                            if full_name.split('.')[0] in gold_tables:
                                y_col[idx] = 1.0

            data['table'].y = y_table
            data['column'].y = y_col

            # FK 노드 label: 양 끝 테이블이 모두 gold일 때 y_fk = 1.0
            y_fk = torch.zeros(data['fk_node'].x.size(0))
            fk_to_id = metadata.get('fk_to_id', {})
            for fk_key, fk_idx in fk_to_id.items():
                # fk_key: "from_table.from_col->to_table.to_col"
                parts = fk_key.split('->')
                if len(parts) == 2:
                    from_table = parts[0].split('.')[0].lower()
                    to_table = parts[1].split('.')[0].lower()
                    if from_table in gold_tables and to_table in gold_tables:
                        y_fk[fk_idx] = 1.0
            data['fk_node'].y = y_fk

            processed.append(data)

        # 3. 빌드 완료 후 파일로 저장
        logger.info(f"💾 Saving processed graphs to {self.cache_path}")
        torch.save(processed, self.cache_path)
        return processed
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class BIRDSuperNodeDataset(Dataset):
    """
    BIRDGraphDataset을 래핑하여 Query Super Node를 그래프에 주입하는 데이터셋.
    query_node가 모든 table, column, fk_node에 양방향 edge를 가진다.
    """
    def __init__(self, base_dataset: BIRDGraphDataset, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.base_dataset = base_dataset
        self.data_list = self._inject_supernode()

    def _inject_supernode(self):
        logger.info("🔗 Injecting query super nodes into all graphs...")
        result = []
        for data in tqdm(self.base_dataset.data_list, desc="Injecting SuperNode"):
            data = data.clone()
            q_emb = data['query']  # [1, 384]

            # query_node feature 설정
            data['query_node'].x = q_emb  # [1, 384]

            # 모든 schema 노드에 양방향 edge 생성
            for schema_nt in ['table', 'column', 'fk_node']:
                num_nodes = data[schema_nt].x.size(0)
                if num_nodes == 0:
                    # edge가 없어도 빈 텐서로 설정 (DataLoader collate 호환)
                    data['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = \
                        torch.zeros((2, 0), dtype=torch.long)
                    data[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = \
                        torch.zeros((2, 0), dtype=torch.long)
                    continue

                # query_node(0) → schema_node(0..N-1)
                src = torch.zeros(num_nodes, dtype=torch.long)  # query_node idx always 0
                dst = torch.arange(num_nodes, dtype=torch.long)

                data['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = \
                    torch.stack([src, dst], dim=0)
                data[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = \
                    torch.stack([dst, src], dim=0)

            result.append(data)
        return result

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]