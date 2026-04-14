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

    def _compute_adaptive_threshold(self, scores_arr: np.ndarray) -> float:
        """Per-query adaptive threshold 계산 (하위 클래스에서도 재사용)."""
        adaptive_threshold = np.percentile(scores_arr, self.percentile)
        prize_count = np.sum(scores_arr > adaptive_threshold)
        if prize_count < self.min_prize_nodes:
            sorted_scores = np.sort(scores_arr)[::-1]
            adaptive_threshold = sorted_scores[min(self.min_prize_nodes - 1, len(sorted_scores) - 1)]
        elif prize_count > self.max_prize_nodes:
            sorted_scores = np.sort(scores_arr)[::-1]
            adaptive_threshold = sorted_scores[min(self.max_prize_nodes - 1, len(sorted_scores) - 1)]
        return adaptive_threshold

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)

        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Edge-less 케이스 (isolated component): 점수 기반으로 prize 노드만 반환
        if len(edges) == 0:
            selected = np.where(prizes > 0)[0].tolist()
            return selected, []

        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64).reshape(-1, 2)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[AdaptivePCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, selected={len(selected_nodes)} nodes")
        return selected_nodes.tolist(), selected_edges


@register("extractor", "ScoreDrivenPCSTExtractor")
class ScoreDrivenPCSTExtractor(AdaptivePCSTExtractor):
    """
    [방안 A] Score-Driven Adaptive Cost PCST.
    Edge cost를 고정값 대신 endpoint 노드의 score로부터 유도하여,
    Prize와 Cost가 같은 스케일에서 자연스럽게 균형을 이루도록 한다.

    핵심 아이디어: cost(u→v) = type_weight × max(threshold - score_v, epsilon)
    - 도착 노드(v)의 score가 threshold 이상이면 cost ≈ epsilon (거의 무료)
    - score가 낮을수록 cost가 높아져 연결을 억제
    - type_weight로 edge 종류별 상대적 비용 차등 유지
    """
    def __init__(self,
                 belongs_to_weight: float = 0.3,
                 fk_weight: float = 0.5,
                 macro_weight: float = 1.5,
                 epsilon: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.belongs_to_weight = belongs_to_weight
        self.fk_weight = fk_weight
        self.macro_weight = macro_weight
        self.epsilon = epsilon
        logger.info(f"Initialized ScoreDrivenPCSTExtractor "
                     f"(weights: bt={belongs_to_weight}, fk={fk_weight}, macro={macro_weight})")

    def _get_type_weight(self, edge_type: str) -> float:
        if edge_type == 'belongs_to':
            return self.belongs_to_weight
        elif edge_type in ['is_source_of', 'points_to']:
            return self.fk_weight
        elif edge_type == 'table_to_table':
            return self.macro_weight
        return 1.0

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # Adaptive threshold (부모 클래스 로직 재사용)
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)

        # Prize 계산 (동일)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Score-Driven Cost: cost(u->v) = type_weight * max(threshold - score_v, epsilon)
        costs = np.empty(len(edges), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            e_type = edge_types[i] if i < len(edge_types) else 'default'
            w = self._get_type_weight(e_type)
            score_v = scores_arr[v] if v < len(scores_arr) else 0.0
            costs[i] = w * max(adaptive_threshold - score_v, self.epsilon)

        edges_arr = np.array(edges, dtype=np.int64)
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[ScoreDrivenPCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, "
                     f"cost range=[{costs.min():.4f}, {costs.max():.4f}], "
                     f"selected={len(selected_nodes)} nodes")
        return selected_nodes.tolist(), selected_edges


@register("extractor", "ProductCostPCSTExtractor")
class ProductCostPCSTExtractor(AdaptivePCSTExtractor):
    """
    [아이디어 2] Score-Driven Product Cost PCST.
    Edge cost를 양 endpoint 노드의 정규화 점수의 곱으로 설정하여,
    Prize와 Cost가 같은 스케일에서 자연스럽게 균형을 이루도록 한다.

    핵심: cost(u,v) = type_base × (1 - norm_score_u) × (1 - norm_score_v)
    - 양 끝 점수 높음 → cost ≈ 0 (연결 장려)
    - 한쪽 낮음 → cost 적당 (신중한 연결)
    - 양 끝 낮음 → cost ≈ type_base (연결 억제)
    """
    def __init__(self,
                 bt_weight: float = 0.1,
                 fk_weight: float = 0.2,
                 macro_weight: float = 0.5,
                 min_cost: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.bt_weight = bt_weight
        self.fk_weight = fk_weight
        self.macro_weight = macro_weight
        self.min_cost = min_cost
        logger.info(f"Initialized ProductCostPCSTExtractor "
                     f"(weights: bt={bt_weight}, fk={fk_weight}, macro={macro_weight}, "
                     f"min_cost={min_cost})")

    def _get_type_base(self, edge_type: str) -> float:
        if edge_type == 'belongs_to':
            return self.bt_weight
        elif edge_type in ['is_source_of', 'points_to']:
            return self.fk_weight
        elif edge_type == 'table_to_table':
            return self.macro_weight
        return 1.0

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # 1. Adaptive threshold & prizes (부모 클래스 로직 재사용)
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Edge-less 케이스 (isolated component): 점수 기반으로 prize 노드만 반환
        if len(edges) == 0:
            selected = np.where(prizes > 0)[0].tolist()
            return selected, []

        # 2. Score 정규화 (min-max → [0, 1])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max - s_min > 1e-8:
            norm_scores = (scores_arr - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores_arr)

        # 3. Edge별 product cost 계산
        costs = np.empty(len(edges), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            e_type = edge_types[i] if i < len(edge_types) else 'default'
            type_base = self._get_type_base(e_type)
            product = (1.0 - norm_scores[u]) * (1.0 - norm_scores[v])
            costs[i] = max(type_base * product, self.min_cost)

        edges_arr = np.array(edges, dtype=np.int64).reshape(-1, 2)

        # 4. pcst_fast 실행
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[ProductCostPCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, "
                     f"cost range=[{costs.min():.4f}, {costs.max():.4f}], "
                     f"selected={len(selected_nodes)} nodes")
        return selected_nodes.tolist(), selected_edges


class ComponentAwareMixin:
    """
    [아이디어 4] Connected Component 분리 후 독립 실행 Mixin.
    Union-Find로 connected components를 분해한 뒤, 각 component에서
    독립적으로 adaptive threshold를 계산하고 PCST를 실행한다.

    PCST의 목적함수는 connected components에 대해 가법적으로 분해 가능하므로,
    component별 독립 실행이 근사 손실 없이 전체 최적해와 동등하다.
    """
    def _decompose(self, edges: List[Tuple[int, int]], num_nodes: int
                   ) -> List[Tuple[List[int], List[int]]]:
        """Union-Find로 connected components 분리. O(V + E)."""
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        for u, v in edges:
            union(u, v)

        # Group nodes and edges by component
        from collections import defaultdict
        comp_nodes = defaultdict(list)
        for n in range(num_nodes):
            comp_nodes[find(n)].append(n)

        comp_edges = defaultdict(list)
        for ei, (u, v) in enumerate(edges):
            comp_edges[find(u)].append(ei)

        return [(nodes, comp_edges.get(root, []))
                for root, nodes in comp_nodes.items()]

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        num_nodes = len(node_scores)
        scores_arr = np.array(node_scores, dtype=np.float64)

        components = self._decompose(edges, num_nodes)

        # Single component → delegate directly (most common case in BIRD)
        if len(components) <= 1:
            return super().extract(graph_data, node_scores, seed_nodes, **kwargs)

        all_selected_nodes = []
        all_selected_edges = []

        for comp_node_ids, comp_edge_indices in components:
            # Skip components with no meaningful scores
            comp_scores = scores_arr[comp_node_ids]
            if comp_scores.max() < 0.01:
                continue

            # Build local graph data with remapped node indices
            global_to_local = {g: l for l, g in enumerate(comp_node_ids)}
            comp_edges_local = []
            comp_edge_types = []
            for ei in comp_edge_indices:
                u, v = edges[ei]
                comp_edges_local.append((global_to_local[u], global_to_local[v]))
                comp_edge_types.append(edge_types[ei] if ei < len(edge_types) else 'default')

            comp_graph_data = {
                'edges': comp_edges_local,
                'edge_types': comp_edge_types,
            }
            comp_scores_list = comp_scores.tolist()

            # Component 내 독립 실행
            sel_nodes_local, sel_edges_local = super().extract(
                comp_graph_data, comp_scores_list, seed_nodes=None, **kwargs)

            # Remap back to global indices
            for ln in sel_nodes_local:
                all_selected_nodes.append(comp_node_ids[ln])
            for lu, lv in sel_edges_local:
                all_selected_edges.append((comp_node_ids[lu], comp_node_ids[lv]))

        logger.debug(f"[ComponentAware] {len(components)} components, "
                     f"selected {len(all_selected_nodes)} nodes total")
        return all_selected_nodes, all_selected_edges


@register("extractor", "ComponentAwareAdaptivePCSTExtractor")
class ComponentAwareAdaptivePCSTExtractor(ComponentAwareMixin, AdaptivePCSTExtractor):
    """ComponentAwareMixin + AdaptivePCSTExtractor 결합."""
    pass


@register("extractor", "ComponentAwareProductCostPCSTExtractor")
class ComponentAwareProductCostPCSTExtractor(ComponentAwareMixin, ProductCostPCSTExtractor):
    """ComponentAwareMixin + ProductCostPCSTExtractor 결합 (아이디어 2+4)."""
    pass


@register("extractor", "EdgePrizePCSTExtractor")
class EdgePrizePCSTExtractor(AdaptivePCSTExtractor):
    """
    G-Retriever-style edge prize mechanism on top of AdaptivePCST.

    Uses triplet relation edge embeddings (from TripletGraphBuilder) to compute
    cosine similarity with the query embedding, then assigns prizes to top-k edges.
    High-prize edges get cost reduction; if prize > cost, a virtual node is created
    to guarantee the edge is selected.
    """
    def __init__(self,
                 topk_e: int = 5,
                 edge_cost: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.topk_e = topk_e
        self.edge_cost = edge_cost
        logger.info(f"Initialized EdgePrizePCSTExtractor (topk_e={topk_e}, edge_cost={edge_cost})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        edge_embeddings = graph_data.get('edge_embeddings', None)
        query_embedding = graph_data.get('query_embedding', None)
        scores_arr = np.array(node_scores, dtype=np.float64)
        num_nodes = len(scores_arr)

        # --- Node prizes (same as AdaptivePCST) ---
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        node_prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # --- Edge prizes (G-Retriever style) ---
        c = 0.01  # small constant for tie-breaking
        e_prizes = np.zeros(len(edges), dtype=np.float64)

        if edge_embeddings is not None and query_embedding is not None:
            edge_emb_t = torch.as_tensor(edge_embeddings, dtype=torch.float32)
            query_emb_t = torch.as_tensor(query_embedding, dtype=torch.float32)
            if query_emb_t.dim() == 1:
                query_emb_t = query_emb_t.unsqueeze(0)

            cos_sim = torch.nn.functional.cosine_similarity(
                query_emb_t, edge_emb_t, dim=-1)

            topk_e = min(self.topk_e, cos_sim.unique().size(0))
            if topk_e > 0:
                topk_values, _ = torch.topk(cos_sim.unique(), topk_e, largest=True)
                # Zero out below threshold
                cos_sim[cos_sim < topk_values[-1]] = 0.0
                # Assign ranked prizes (higher rank → higher prize)
                last_val = topk_e
                for k in range(topk_e):
                    indices = cos_sim == topk_values[k]
                    value = min((topk_e - k) / indices.sum().item(), last_val)
                    cos_sim[indices] = value
                    last_val = value * (1 - c)

            e_prizes = cos_sim.numpy().astype(np.float64)

        # Ensure at least one edge can be selected
        cost_e = self.edge_cost
        if e_prizes.max() > 0:
            cost_e = min(cost_e, e_prizes.max() * (1 - c / 2))

        # --- Build PCST input with virtual nodes for high-prize edges ---
        pcst_edges = []
        pcst_costs = []
        edge_mapping = {}  # pcst_edge_idx -> original_edge_idx
        virtual_node_prizes = []
        virtual_node_to_edge = {}  # virtual_node_id -> original_edge_idx

        for i, (src, dst) in enumerate(edges):
            prize_e = e_prizes[i]
            base_cost = self._compute_cost(edge_types[i] if i < len(edge_types) else 'default')

            if prize_e <= cost_e:
                edge_mapping[len(pcst_edges)] = i
                pcst_edges.append((src, dst))
                pcst_costs.append(max(base_cost - prize_e, 0.0))
            else:
                # Virtual node: guarantees this edge is included
                virtual_id = num_nodes + len(virtual_node_prizes)
                virtual_node_to_edge[virtual_id] = i
                pcst_edges.append((src, virtual_id))
                pcst_edges.append((virtual_id, dst))
                pcst_costs.append(0.0)
                pcst_costs.append(0.0)
                virtual_node_prizes.append(prize_e - cost_e)

        all_prizes = np.concatenate([node_prizes, np.array(virtual_node_prizes)]) \
            if virtual_node_prizes else node_prizes

        edges_arr = np.array(pcst_edges, dtype=np.int64)
        costs_arr = np.array(pcst_costs, dtype=np.float64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(
            edges_arr, all_prizes, costs_arr, -1, 1, 'gw', 0)

        # Map back: real nodes only, reconstruct original edges
        real_nodes = [n for n in selected_nodes if n < num_nodes]
        selected_original_edges = []
        for ei in selected_edges_idx:
            if ei in edge_mapping:
                selected_original_edges.append(edges[edge_mapping[ei]])
        # Virtual nodes → add their corresponding original edges
        for n in selected_nodes:
            if n in virtual_node_to_edge:
                orig_idx = virtual_node_to_edge[n]
                selected_original_edges.append(edges[orig_idx])
                # Also ensure both endpoints are in selected nodes
                src, dst = edges[orig_idx]
                if src not in real_nodes:
                    real_nodes.append(src)
                if dst not in real_nodes:
                    real_nodes.append(dst)

        n_virtual = len(virtual_node_prizes)
        n_edge_prize = int(np.sum(e_prizes > 0))
        logger.debug(f"[EdgePrizePCST] threshold={adaptive_threshold:.4f}, "
                     f"node_prize={int(np.sum(node_prizes > 0))}, "
                     f"edge_prize={n_edge_prize}, virtual_nodes={n_virtual}, "
                     f"selected={len(real_nodes)} nodes")
        return real_nodes, selected_original_edges


@register("extractor", "SteinerBackbonePCSTExtractor")
class SteinerBackbonePCSTExtractor(AdaptivePCSTExtractor):
    """
    [아이디어 3] Two-Phase Extraction: Steiner Tree Backbone + PCST Expansion.

    Phase 1: Seed nodes 간 Steiner Tree 2-근사 (Kou et al., 1981)로
             connectivity를 확실히 확보한 backbone 구축.
    Phase 2: Backbone nodes에 bonus prize를 부여하고, backbone edges의
             cost를 0으로 설정하여 PCST로 추가 확장/가지치기.
    """
    def __init__(self,
                 backbone_bonus: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone_bonus = backbone_bonus
        logger.info(f"Initialized SteinerBackbonePCSTExtractor (backbone_bonus={backbone_bonus})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # Phase 1: Steiner Tree Backbone
        backbone_nodes = set()
        backbone_edges = set()
        if seed_nodes and len(seed_nodes) >= 2:
            import networkx as nx
            from .mst import steiner_tree_2approx

            G = nx.Graph()
            G.add_edges_from(edges)
            st_nodes, st_edges = steiner_tree_2approx(G, seed_nodes)
            backbone_nodes = set(st_nodes)
            backbone_edges = set((min(u, v), max(u, v)) for u, v in st_edges)

        # Phase 2: PCST Expansion with backbone bonuses
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Backbone nodes에 bonus prize 부여
        if backbone_nodes:
            max_prize = prizes.max() if prizes.max() > 0 else 0.1
            for bn in backbone_nodes:
                if bn < len(prizes):
                    prizes[bn] = max(prizes[bn], max_prize * self.backbone_bonus)

        # Backbone edges의 cost를 0으로 설정
        costs = np.empty(len(edges), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            edge_key = (min(u, v), max(u, v))
            if edge_key in backbone_edges:
                costs[i] = 0.0
            else:
                e_type = edge_types[i] if i < len(edge_types) else 'default'
                costs[i] = self._compute_cost(e_type)

        edges_arr = np.array(edges, dtype=np.int64)
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[SteinerBackbonePCST] backbone={len(backbone_nodes)} nodes, "
                     f"final={len(selected_nodes)} nodes")
        return selected_nodes.tolist(), selected_edges