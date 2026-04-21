import os
import time
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
        self.last_info: Dict[str, Any] = {}
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

        t = time.perf_counter()
        faiss_index = faiss.read_index(index_path)
        t_faiss = time.perf_counter() - t

        t = time.perf_counter()
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        t_meta = time.perf_counter() - t

        logger.info(f"✅ Successfully loaded [{db_id}] | Nodes: {faiss_index.ntotal}, Edges: {len(metadata.get('edges', []))}")

        self.last_info = {
            "builder_type": "CachedGraphBuilder",
            "builder_db_id": db_id,
            "builder_cache_dir": self.cache_dir,
            "builder_faiss_ntotal": int(faiss_index.ntotal),
            "builder_faiss_d": int(getattr(faiss_index, "d", 0) or 0),
            "builder_num_edges_total": len(metadata.get("edges", []) or []),
            "builder_num_tables": len(metadata.get("table_to_id", {}) or {}),
            "builder_num_columns": len(metadata.get("col_to_id", {}) or {}),
            "builder_num_fk_nodes": len(metadata.get("fk_to_id", {}) or {}),
            "builder_timings": {
                "faiss_load_s": float(t_faiss),
                "metadata_load_s": float(t_meta),
                "total_s": float(t_faiss + t_meta),
            },
        }
        metadata["builder_info"] = dict(self.last_info)
        return faiss_index, metadata