import os
import torch
import sqlite3
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from thefuzz import process

from modules.registry import register
from modules.base import BaseSelector
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "XiYanSelector")
class XiYanSelector(BaseSelector):
    """
    XiYanSQL의 Multi-path Retrieval 로직 + Value Retrieval (DB 직접 조회)를 모사한 Selector.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", top_k: int = 20, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", db_dir: str = "./data/raw/BIRD_dev/dev_databases", **kwargs):
        self.top_k = top_k
        self.db_dir = db_dir
        self.client = APIClient()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"Initialized XiYanSelector with Value Retrieval (DB Access Enabled)")

    def _extract_keywords(self, question: str) -> List[str]:
        prompt = f"Extract critical keywords from the following question for database schema retrieval. Give me the keywords in comma separated format.\nQuestion: {question}"
        response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=0.0)
        keywords_str = response.split(":")[-1] if ":" in response else response
        keywords = [k.strip().lower() for k in keywords_str.split(',') if k.strip()]
        return keywords

    def select(self, question: str, candidates: List[Any], db_id: str = None, metadata: Dict = None, **kwargs) -> Dict[str, float]:
        # 정수형 인덱스로 들어온 candidates를 텍스트(테이블명.컬럼명)로 번역
        if metadata and 'node_metadata' in metadata:
            text_candidates = [metadata['node_metadata'].get(idx, str(idx)) for idx in candidates]
        else:
            text_candidates = candidates # 이미 텍스트 리스트라면 그대로 사용
            
        keywords = self._extract_keywords(question)
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        kw_embs = self.embedder.encode(keywords, convert_to_tensor=True) if keywords else q_emb.unsqueeze(0)
        
        tables = [c for c in text_candidates if "." not in c]
        columns = [c for c in text_candidates if "." in c]
        
        table_embs = {t: self.embedder.encode(f"Table {t}", convert_to_tensor=True) for t in tables}
        col_embs = {c: self.embedder.encode(f"Column {c.split('.')[1]} in table {c.split('.')[0]}", convert_to_tensor=True) for c in columns}
        
        # 1차 스코어링 (Semantic Similarity)
        column_scores = {}
        for col in columns:
            tbl_name, col_name = col.split(".")
            t_emb = table_embs.get(tbl_name, self.embedder.encode(tbl_name, convert_to_tensor=True))
            score_qe_table = util.cos_sim(q_emb, t_emb).item()
            c_emb = col_embs[col]
            scores_kw_col = util.cos_sim(kw_embs, c_emb)
            max_score_kw_col = torch.max(scores_kw_col).item()
            
            final_score = max(score_qe_table * max_score_kw_col, 0.0)
            column_scores[col] = final_score

        # 상위 2배수(top_k * 2) 추출 후 Value Retrieval 적용하여 순위 미세조정 (Reranking)
        k_candidate = min(self.top_k * 2, len(column_scores))
        top_candidates = dict(sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:k_candidate])

        # 2차 스코어링 (Value Retrieval via DB Engine)
        if db_id and keywords:
            db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    for col in list(top_candidates.keys()):
                        tbl_name, col_name = col.split(".")
                        try:
                            # DB에서 최대 100개의 Distinct Value 샘플링
                            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{tbl_name}" WHERE "{col_name}" IS NOT NULL LIMIT 100')
                            values = [str(row[0]) for row in cursor.fetchall()]
                            
                            # 추출된 질의 키워드와 실제 DB 값 간의 Fuzzy Matching
                            for kw in keywords:
                                matches = process.extractBests(kw, values, score_cutoff=85, limit=1)
                                if matches:
                                    logger.debug(f"[Value Match!] Kw: '{kw}' -> Col: '{col}' (Value: {matches[0][0]})")
                                    # 매칭 성공 시 해당 컬럼의 스코어를 대폭 상승 (가중치 부여)
                                    top_candidates[col] += 0.5 
                        except Exception:
                            continue
                    conn.close()
                except Exception as e:
                    logger.warning(f"Failed to access DB {db_id} for Value Retrieval: {e}")

        # 최종 상위 K개 정렬 및 반환
        final_top_k = sorted(top_candidates.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return {k: v for k, v in final_top_k}