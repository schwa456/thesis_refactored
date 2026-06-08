import torch
import spacy
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModel
from modules.registry import register
from modules.base import BaseEncoder
from utils.logger import get_logger

logger = get_logger(__name__)

@register("encoder", "TokenEncoder")
class TokenEncoder(BaseEncoder):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_wem_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
        logger.info(f"TokenEncoder initialized with spaCy filtering.")

    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. spaCy를 이용한 불용어 및 품사 기반 필터링
        # 의미 있는 품사(명사, 고유명사, 숫자, 동사, 형용사)만 추출 후보로 선정
        valid_pos = {"NOUN", "PROPN", "NUM", "VERB", "ADJ"}
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state # (Batch, Seq_Len, Hidden_Dim)

        # 2. Mask 생성 (Batch, Seq_Len)
        mask = torch.zeros(inputs.input_ids.shape, dtype=torch.bool, device=self.device)
        
        for i, text in enumerate(texts):
            doc = self.nlp(text)
            # spaCy 토큰과 Transformer 토큰 간의 정렬(Alignment) 수행
            # 여기서는 단순화를 위해 tokenizer가 만든 각 토큰의 텍스트를 spaCy로 다시 검사
            for j, token_id in enumerate(inputs.input_ids[i]):
                if token_id in self.tokenizer.all_special_ids:
                    continue
                
                token_text = self.tokenizer.decode([token_id]).strip().lower()
                # spaCy를 통해 해당 토큰이 불용어인지 혹은 의미 없는 품사인지 검사
                spacy_token = self.nlp(token_text)
                if len(spacy_token) > 0:
                    t = spacy_token[0]
                    if not t.is_stop and t.pos_ in valid_pos:
                        mask[i, j] = True
                    
        return token_embeddings, mask