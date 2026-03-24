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
        self.cache_path = f"/home/hyeonjin/thesis_refactored/data/processed/{file_name}_graphs.pt"

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
            q_emb, _ = self.encoder.encode([item['question']])
            data['query'] = q_emb.squeeze(0).cpu()

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
            processed.append(data)

        # 3. 빌드 완료 후 파일로 저장
        logger.info(f"💾 Saving processed graphs to {self.cache_path}")
        torch.save(processed, self.cache_path)
        return processed
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        # 이미 data_list에 로드되어 있으므로 바로 반환
        return self.data_list[idx]