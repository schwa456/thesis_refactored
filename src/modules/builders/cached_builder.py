import os
import faiss
import pickle
from typing import Dict, Any, Tuple

from modules.registry import register
from modules.base import BaseGraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

@register("builder", "CachedGraphBuilder")
class CachedGraphBuilder(BaseGraphBuilder):
    """
    오프라인에서 미리 구축된 FAISS 인덱스와 그래프 토폴로지(메타데이터)를
    메모리로 빠르게 로드하여 파이프라인에 공급하는 '온라인 전용' 빌더입니다.
    무거운 GNN 연산을 피하고 Retrieval 속도를 극대화합니다.
    """
    def __init__(self, cache_dir: str = "./data/processed", **kwargs):
        super().__init__()
        self.cache_dir = cache_dir
        logger.info(f"Initialized CachedGraphBuilder (Cache Dir: {self.cache_dir})")
    
    def build(self, db_id: str=None, **kwargs) -> Tuple[Any, Dict]:
        if db_id is None:
            raise ValueError("🚨 CachedGraphBuilder requires 'db_id' to load the correct FAISS index.")
        
        index_path = os.path.join(self.cache_dir, f"{db_id}.faiss")
        meta_path = os.path.join(self.cache_dir, f"{db_id}_metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.error(f"Cache files missing for DB: {db_id}")
            raise FileNotFoundError(f"🚨 오프라인 캐시를 찾을 수 없습니다: {index_path} 또는 {meta_path}")

        logger.debug(f"Loading pre-computed FAISS index and graph topology for [{db_id}]...")

        # 1. FAISS Vector DB 로드 (Node Embeddings)
        # 이 인덱스 객체 자체가 그래프 노드들의 잠재 공간(Latent Space)을 나타냅니다.
        faiss_index = faiss.read_index(index_path)

        # 2. 토폴로지 메타데이터 로드 (Edges, Node Names, Types)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)

        logger.info(f"✅ Successfully loaded [{db_id}] | Nodes: {faiss_index.ntotal}, Edges: {len(metadata.get('edges', []))}")

        # 파이프라인의 다음 단계(Selector 등)가 사용할 수 있도록 반환
        # graph_data 자리에 FAISS 인덱스를 넘겨 검색에 활용합니다.
        return faiss_index, metadata