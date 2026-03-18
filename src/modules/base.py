from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional, Union
import torch.nn as nn

try:
    import torch
    Tensor = torch.Tensor
except ImportError:
    Tensor = Any

# ==========================================
# 1. Graph Builder Interface
# ==========================================
class BaseGraphBuilder(ABC):
    """
    DB 스키마를 읽어들여 그래프 데이터 구조(PyG HeteroData 등)로 변환하는 규격.
    """
    @abstractmethod
    def build(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Args:
            오프라인 빌더: build(schema_info=..., fk_descriptions=...)
            온라인 빌더(Cache): build(db_id="spider_california_schools")
        Returns:
            Tuple[Any, Dict]: (그래프 데이터 객체, 메타데이터 매핑 딕셔너리)
        """
        pass

# ==========================================
# 2. NLQ Encoder Interface
# ==========================================
class BaseEncoder(ABC):
    """
    자연어 질문(NLQ)을 Dense Vector(임베딩)로 변환하는 규격.
    """
    @abstractmethod
    def encode(self, texts: List[str]) -> Tensor:
        """
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
        Returns:
            Tensor: (Batch, Hidden_Dim) 형태의 임베딩 텐서
        """
        pass

# ==========================================
# 3. Projector Interface (선택적 모듈)
# ==========================================
class BaseProjector(ABC, nn.Module):
    """
    서로 다른 차원의 임베딩을 동일한 잠재 공간으로 투영하는 규격.
    nn.Module을 상속받아 파이토치 모델의 특성을 가집니다.
    """
    def __init__(self):
        super(BaseProjector, self).__init__()

    @abstractmethod
    def forward(self, query_emb: Tensor, node_embs: Tensor) -> Tuple[Tensor, Tensor]:
        """투영된 두 텐서를 반환해야 합니다."""
        pass
        
    @abstractmethod
    def compute_similarity(self, z_query: Tensor, z_nodes: Tensor) -> Tensor:
        """투영된 텐서 간의 유사도 점수(예: MaxSim)를 계산하여 반환합니다."""
        pass

# ==========================================
# 4. Seed Selector Interface
# ==========================================
class BaseSelector(ABC):
    @abstractmethod
    def select(self, **kwargs) -> Union[List[Any], Dict[str, float]]:
        """
        입력: 모델에 따라 scores, candidates, question 등을 kwargs로 받음.
        출력: 선택된 노드 리스트, 또는 {노드명: 점수} 형태의 딕셔너리
        """
        pass

# ==========================================
# 5. Connectivity Extractor Interface
# ==========================================
class BaseExtractor(ABC):
    """
    노드 점수와 그래프 구조(비용)를 바탕으로, 구조적으로 연결된 최적의 서브그래프를 추출하는 규격.
    (예: MST, PCST 등)
    """
    @abstractmethod
    def extract(self, graph_data: Any, node_scores: List[float], seed_nodes: Optional[List[int]] = None) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Args:
            graph_data (Any): 전체 스키마 그래프
            node_scores (List[float]): Selector가 예측한 각 노드의 점수(Prize)
            seed_nodes (Optional[List[int]]): 반드시 포함해야 할 터미널 노드들 (Top-K 등)
        Returns:
            Tuple[List[int], List[Tuple]]: (최종 선택된 노드 인덱스 목록, 최종 선택된 엣지 목록)
        """
        pass

# ==========================================
# 6. Filter / Agent Interface
# ==========================================
class BaseFilter(ABC):
    @abstractmethod
    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        입력: 자연어 질의(query)와 Extractor가 추출한 서브그래프 딕셔너리
        출력: 최종 정제된 노드 리스트와 에이전트의 추론(reasoning), 상태(Answerable/Unanswerable) 등을 담은 Dict
        """
        pass

# ==========================================
# 7. SQL Generator
# ==========================================
class BaseGenerator(ABC):
    """최종 스키마와 질문을 바탕으로 SQL 쿼리를 생성하는 규격"""
    @abstractmethod
    def generate(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> str:
        pass