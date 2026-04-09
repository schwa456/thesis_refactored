import math
import torch
import numpy as np
try:
    import pcst_fast
except ImportError:
    pcst_fast = None

from collections import defaultdict
from torch_geometric.utils import degree
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

@register("extractor", "PCSTExtractor")
class PCSTExtractor(BaseExtractor):
    def __init__(self, 
                 base_cost: float = 1.0, 
                 belongs_to_cost: float = 0.01, 
                 fk_cost: float = 0.05,        # [추가] FK 이동 비용 (매우 낮음)
                 macro_cost: float = 0.5,      # [추가] Table-Table 거시적 연결 비용
                 hub_discount: float = 0.2,    # [추가] 동적 할당을 위한 할인율 (gamma)
                 node_threshold: float = 0.5, 
                 **kwargs):
        
        self.base_cost = base_cost
        self.belongs_to_cost = belongs_to_cost
        self.fk_cost = fk_cost
        self.macro_cost = macro_cost
        self.hub_discount = hub_discount
        self.node_threshold = node_threshold
        logger.info(f"Initialized PCST Extractor (Cost: {base_cost}, Threshold: {node_threshold})")

    def _compute_cost(self, edge_type: str) -> float:
        if edge_type == 'belongs_to':
            return self.belongs_to_cost
        elif edge_type in ['is_source_of', 'points_to']:
            return self.fk_cost
        elif edge_type == 'table_to_table':
            return self.macro_cost
        return self.base_cost

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed. Please install it to use PCSTExtractor.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        
        prizes = np.maximum(np.array(node_scores, dtype=np.float64) - self.node_threshold, 0.0)
        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]
        
        logger.debug(f"[PCST] Extracted {len(selected_nodes)} nodes and {len(selected_edges)} edges.")
        return selected_nodes.tolist(), selected_edges

@register("extractor", "DynamicPCSTExtractor")
class DynamicPCSTExtractor(PCSTExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info(f"Initialized Dynamic PCST Extractor (Macro: {self.macro_cost}, Hub Discount: {self.hub_discount})")

    def _compute_dynamic_cost(self, edge_type: str, u: int, v: int, degrees: Dict[int, int]) -> float:
        """
        엣지 타입과 노드의 연결 차수(Degree)를 기반으로 동적 비용을 계산합니다.
        """
        # 1. 엣지 타입별 Base Cost 설정
        if edge_type == 'belongs_to':
            c_type = self.belongs_to_cost
        elif edge_type in ['is_source_of', 'points_to']:
            c_type = self.fk_cost
        elif edge_type == 'table_to_table':
            c_type = self.macro_cost
        else:
            c_type = self.base_cost

        # 2. 동적 비용 할인 적용 (Hub-Discount)
        # u(Source)에서 v(Target)로 갈 때, 도착지 v가 중요한 허브(Degree가 높음)라면 비용을 낮춤
        deg_v = degrees.get(v, 0)
        
        # 공식: C / (1 + gamma * log(1 + degree(v)))
        discount_factor = 1.0 + (self.hub_discount * math.log1p(deg_v))
        final_cost = c_type / discount_factor
        
        return final_cost

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed. Please install it to use PCSTExtractor.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        
        # [핵심 로직] 각 노드의 Degree(차수) 계산
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1

        # 노드 Prize 계산 (Threshold 적용)
        prizes = np.maximum(np.array(node_scores, dtype=np.float64) - self.node_threshold, 0.0)
        
        # [핵심 로직] 엣지마다 동적으로 Cost 계산
        costs = []
        for i, (u, v) in enumerate(edges):
            e_type = edge_types[i] if i < len(edge_types) else 'default'
            dyn_cost = self._compute_dynamic_cost(e_type, u, v, degrees)
            costs.append(dyn_cost)
            
        costs_arr = np.array(costs, dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        # PCST-fast 실행
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs_arr, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]
        
        logger.debug(f"[Dynamic PCST] Extracted {len(selected_nodes)} nodes and {len(selected_edges)} edges.")
        return selected_nodes.tolist(), selected_edges

@register("extractor", "PPRPCSTExtractor")
class PPRPCSTExtractor(PCSTExtractor):
    """(기존 코드 유지: PPR 로직을 동적 PCST 위에 그대로 상속하여 사용)"""
    def __init__(self, belongs_to_cost: float = 0.01, ppr_alpha: float = 0.15, ppr_max_iter: int = 50, node_threshold: float = 0.5, **kwargs):
        super().__init__(belongs_to_cost=belongs_to_cost, node_threshold=node_threshold, **kwargs)
        self.ppr_alpha = ppr_alpha
        self.ppr_max_iter = ppr_max_iter
        logger.info(f"Initialized Advanced PCST Extractor with PPR (Alpha: {ppr_alpha})")

    def _compute_ppr_prizes(self, num_nodes: int, edges: List[Tuple[int, int]], initial_prizes: List[float]) -> List[float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p = torch.tensor(initial_prizes, dtype=torch.float32, device=device)
        p = torch.relu(p - self.node_threshold) 
        
        if p.sum() == 0 or not edges:
            return initial_prizes
        
        p = p / p.sum()
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        row, col = edge_index_undirected
        deg = degree(col, num_nodes, dtype=torch.float32)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        val = deg_inv[col]

        adj = torch.sparse_coo_tensor(edge_index_undirected, val, (num_nodes, num_nodes), device=device)

        r = p.clone()
        for _ in range(self.ppr_max_iter):
            r_next = (1 - self.ppr_alpha) * torch.sparse.mm(adj, r.unsqueeze(1)).squeeze(1) + self.ppr_alpha * p
            if torch.norm(r_next - r) < 1e-5:
                r = r_next
                break
            r = r_next
        
        max_ppr = r.max().item()
        if max_ppr > 0:
            r = r / max_ppr * max(initial_prizes)
            
        return r.cpu().tolist()

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        edges = graph_data.get('edges', [])
        ppr_prizes = self._compute_ppr_prizes(len(node_scores), edges, node_scores)
        # 부모 클래스(동적 PCST)의 extract 메서드 호출
        return super().extract(graph_data, ppr_prizes, seed_nodes, **kwargs)

@register("extractor", "UncertaintyPCSTExtractor")
class UncertaintyPCSTExtractor(PCSTExtractor):
    def __init__(self, alpha: float = 2.0, uncertainty_margin: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.uncertainty_margin = uncertainty_margin
    
    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed. Please install it to use PCSTExtractor.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])

        scores_arr = np.array(node_scores, dtype=np.float64)

        # 1. Deadzone 컷 오프
        effective_threshold = max(self.node_threshold, 0.5 + self.uncertainty_margin)
        raw_prizes = np.maximum(scores_arr - effective_threshold, 0.0)

        # 2. 비선형 Power Scaling
        prizes = np.power(raw_prizes, self.alpha)
        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]
        
        logger.debug(f"[PCST] Extracted {len(selected_nodes)} nodes and {len(selected_edges)} edges.")
        return selected_nodes.tolist(), selected_edges

@register("extractor", "DynamicUncertaintyPCSTExtractor")
class DynamicUncertaintyPCSTExtractor(DynamicPCSTExtractor):
    """
    [최종 제안 방법론]
    Dynamic Cost를 통해 브릿지 테이블(Hub)로의 연결성을 강화하고 (Recall 상승),
    Uncertainty Scaling을 통해 확신도 낮은 노드의 Prize를 깎아 노이즈를 방어합니다 (Precision 방어).
    """
    def __init__(self, alpha: float = 2.0, uncertainty_margin: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.uncertainty_margin = uncertainty_margin
        logger.info(f"Initialized Dynamic+Uncertainty PCST Extractor")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        
        # 1. Degree 계산 (Dynamic Cost 용)
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1

        # 2. Uncertainty Scaling (Prize 고도화)
        scores_arr = np.array(node_scores, dtype=np.float64)
        effective_threshold = max(self.node_threshold, 0.5 + self.uncertainty_margin)
        raw_prizes = np.maximum(scores_arr - effective_threshold, 0.0)
        prizes = np.power(raw_prizes, self.alpha)
        
        # 3. Dynamic Cost 계산
        costs = []
        for i, (u, v) in enumerate(edges):
            e_type = edge_types[i] if i < len(edge_types) else 'default'
            dyn_cost = self._compute_dynamic_cost(e_type, u, v, degrees)
            costs.append(dyn_cost)
            
        costs_arr = np.array(costs, dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs_arr, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]
        
        logger.debug(f"[Dynamic+Uncertainty] Extracted {len(selected_nodes)} nodes.")
        return selected_nodes.tolist(), selected_edges

@register("extractor", "GATAwarePCSTExtractor")
class GATAwarePCSTExtractor(PCSTExtractor):
    """
    본 논문의 제안 모델:
    PPR과 같은 정적 확산 대신, 학습된 GAT가 뱉어낸 '구조-문맥 융합 점수'를
    PCST의 Prize로 직접 사용하여 Implicit Bridge Table을 확정적으로 포착합니다.
    """
    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        return super().extract(graph_data, node_scores, **kwargs)


@register("extractor", "AdaptivePCSTExtractor")
class AdaptivePCSTExtractor(PCSTExtractor):
    """
    [Phase B-1] Per-query adaptive threshold를 사용하는 PCST.
    고정 node_threshold 대신 각 query의 score 분포 상위 percentile을 기준으로
    threshold를 동적으로 설정하여 subgraph 크기를 일정하게 유지한다.
    """
    def __init__(self,
                 percentile: float = 80.0,
                 min_prize_nodes: int = 3,
                 max_prize_nodes: int = 25,
                 **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile
        self.min_prize_nodes = min_prize_nodes
        self.max_prize_nodes = max_prize_nodes
        logger.info(f"Initialized AdaptivePCSTExtractor (percentile={percentile}, "
                     f"min_nodes={min_prize_nodes}, max_nodes={max_prize_nodes})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # Adaptive threshold: score 분포의 상위 percentile 기준
        adaptive_threshold = np.percentile(scores_arr, self.percentile)

        # Fallback: threshold가 너무 높거나 낮으면 조정
        prize_count = np.sum(scores_arr > adaptive_threshold)
        if prize_count < self.min_prize_nodes:
            # prize 노드가 너무 적으면 threshold 하향
            sorted_scores = np.sort(scores_arr)[::-1]
            adaptive_threshold = sorted_scores[min(self.min_prize_nodes - 1, len(sorted_scores) - 1)]
        elif prize_count > self.max_prize_nodes:
            # prize 노드가 너무 많으면 threshold 상향
            sorted_scores = np.sort(scores_arr)[::-1]
            adaptive_threshold = sorted_scores[min(self.max_prize_nodes - 1, len(sorted_scores) - 1)]

        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)
        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[AdaptivePCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, selected={len(selected_nodes)} nodes")
        return selected_nodes.tolist(), selected_edges