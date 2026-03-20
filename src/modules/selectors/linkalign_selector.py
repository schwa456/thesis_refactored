import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

from modules.registry import register
from modules.base import BaseSelector
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "LinkAlignSelector")
class LinkAlignSelector(BaseSelector):
    """
    LinkAlign (EMNLP 2025) Baseline 모사:
    초기 검색 결과를 바탕으로 LLM이 누락된 스키마를 추론하여 질의를 재작성(Query Rewriting)한 뒤,
    2차 검색(Multi-round Semantic Enhanced Retrieval)을 수행하는 Selector입니다.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", top_k: int = 20, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        self.top_k = top_k
        self.model_name = model_name
        self.client = APIClient()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"Initialized LinkAlignSelector (Query Rewriting Enabled, k={self.top_k})")

    def select(self, question: str, candidates: List[Any], db_id: str = None, metadata: Dict = None, **kwargs) -> Dict[str, float]:
        # 1. 메타데이터 변환 (Index -> Text)
        if metadata and 'node_metadata' in metadata:
            text_candidates = [metadata['node_metadata'].get(idx, str(idx)) for idx in candidates]
        else:
            text_candidates = candidates
            
        cand_embs = self.embedder.encode(text_candidates, convert_to_tensor=True)
        
        # 2. 1차 검색 (Initial Retrieval)
        q_emb_initial = self.embedder.encode(question, convert_to_tensor=True)
        initial_scores = util.cos_sim(q_emb_initial, cand_embs)[0]
        
        # 상위 K개 임시 추출 (Auditor에게 보여줄 Context)
        top_k_initial_idx = torch.topk(initial_scores, k=min(self.top_k, len(text_candidates))).indices.tolist()
        initial_schema = [text_candidates[i] for i in top_k_initial_idx]
        
        # 3. Schema Auditor & Query Rewriter (LLM 개입)
        prompt = f"""You are a database Schema Auditor for Text-to-SQL.
Original Question: "{question}"
Initially Retrieved Schema (Top-{self.top_k}): {initial_schema}

Task:
1. Identify if any essential tables/columns (e.g., implicit bridge tables for JOINs) are missing from the Initial Schema to fully answer the question.
2. Rewrite the Original Question to explicitly include the inferred missing schema keywords.
3. Return ONLY the rewritten question string, without any other text.
"""
        rewritten_question = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=0.0).strip()
        logger.debug(f"[LinkAlign] Original Q: {question}")
        logger.debug(f"[LinkAlign] Rewritten Q: {rewritten_question}")
        
        # 4. 2차 검색 (Semantic Enhanced Retrieval with Rewritten Query)
        q_emb_rewritten = self.embedder.encode(rewritten_question, convert_to_tensor=True)
        
        # 앙상블: 원본 질의와 재작성된 질의의 벡터를 평균내어 최종 질의 벡터 생성
        q_emb_final = (q_emb_initial + q_emb_rewritten) / 2.0
        final_scores = util.cos_sim(q_emb_final, cand_embs)[0]
        
        # 5. 최종 상위 K개 정렬 및 반환
        top_k_final_idx = torch.topk(final_scores, k=min(self.top_k, len(text_candidates))).indices.tolist()
        
        selected_seeds = {}
        for idx in top_k_final_idx:
            selected_seeds[text_candidates[idx]] = final_scores[idx].item()
            
        return selected_seeds