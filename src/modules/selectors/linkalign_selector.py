import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

from modules.registry import register
from modules.base import BaseSelector
from prompts.prompt_manager import PromptManager
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
    def __init__(self, model_name: str, top_k: int, embedding_model: str, **kwargs):
        self.top_k = top_k
        self.model_name = model_name
        self.prompt_manager = PromptManager()
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
        prompt = self.prompt_manager.load_prompt(
            file_name='selector',
            section='link_align_selector',
            question=question,
            top_k=self.top_k,
            candidates=candidates,
            initial_schema=initial_schema
        )
        rewritten_question = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=0.0).strip()
        logger.debug(f"[LinkAlign] Original Q: {question}")
        logger.debug(f"[LinkAlign] Rewritten Q: {rewritten_question}")
        
        # 4. 2차 검색 (Semantic Enhanced Retrieval with Rewritten Query)
        q_emb_rewritten = self.embedder.encode(rewritten_question, convert_to_tensor=True)
        
        # 앙상블: 원본 질의와 재작성된 질의의 벡터를 평균내어 최종 질의 벡터 생성
        q_emb_final = (q_emb_initial + q_emb_rewritten) / 2.0
        final_scores = util.cos_sim(q_emb_final, cand_embs)[0]

        if metadata and 'node_metadata' in metadata:
            total_nodes = len(metadata['node_metadata'])
            all_scores = [0.0] * total_nodes
            
            # final_scores는 candidates 리스트와 순서가 1:1로 매칭됨
            for i, cand_idx in enumerate(candidates):
                # 인덱스(정수) 형태인 경우 바로 매핑
                if isinstance(cand_idx, int) and cand_idx < total_nodes:
                    all_scores[cand_idx] = final_scores[i].item()
                # 텍스트 형태인 경우 역탐색하여 인덱스 매핑
                elif isinstance(cand_idx, str):
                    for idx, name in metadata['node_metadata'].items():
                        if name == cand_idx:
                            all_scores[int(idx)] = final_scores[i].item()
                            break
                            
            self.latest_scores = all_scores
        else:
            self.latest_scores = final_scores.tolist()
        
        # 5. 최종 상위 K개 정렬 및 반환
        top_k_final_idx = torch.topk(final_scores, k=min(self.top_k, len(text_candidates))).indices.tolist()
        
        selected_seeds = {}
        for idx in top_k_final_idx:
            selected_seeds[text_candidates[idx]] = final_scores[idx].item()
            
        return selected_seeds