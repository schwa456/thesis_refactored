import math
import time
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
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized PCST Extractor (Cost: {base_cost}, Threshold: {node_threshold})")

    def _compute_cost(self, edge_type: str) -> float:
        if edge_type == 'belongs_to':
            return self.belongs_to_cost
        elif edge_type in ['is_source_of', 'points_to']:
            return self.fk_cost
        elif edge_type == 'table_to_table':
            return self.macro_cost
        return self.base_cost

    def _build_ext_info(self,
                        *,
                        extract_type: str,
                        node_scores: List[float],
                        edges: List[Tuple[int, int]],
                        edge_types: List[str],
                        prizes: Optional[np.ndarray],
                        costs: Optional[np.ndarray],
                        selected_nodes: List[int],
                        selected_edges: List[Tuple[int, int]],
                        threshold: Optional[float],
                        t_start: float,
                        graph_data: Optional[Dict[str, Any]] = None,
                        **extras) -> Dict[str, Any]:
        """Standardized last_info builder for all PCST extractors.

        Captures: input/output sizes, prize/cost distribution, per-edge-type cost,
        selected-by-node-type counts (when builder metadata is available), elapsed
        time, and per-extractor extras.
        """
        elapsed = time.perf_counter() - t_start

        info: Dict[str, Any] = {
            "extractor_type": extract_type,
            "extractor_num_input_nodes": int(len(node_scores)),
            "extractor_num_edges": int(len(edges)),
            "extractor_num_selected_nodes": int(len(selected_nodes)),
            "extractor_num_selected_edges": int(len(selected_edges)),
            "extractor_threshold": (float(threshold) if threshold is not None else None),
            "extractor_time_s": float(elapsed),
        }

        # Selected-by-type split using builder metadata if provided
        if graph_data is not None:
            num_t = len(graph_data.get('table_to_id', {}) or {})
            num_c = len(graph_data.get('col_to_id', {}) or {})
            if num_t or num_c:
                sel_t = sum(1 for n in selected_nodes if n < num_t)
                sel_c = sum(1 for n in selected_nodes if num_t <= n < num_t + num_c)
                sel_fk = sum(1 for n in selected_nodes if n >= num_t + num_c)
                info["extractor_selected_by_type"] = {
                    "table": int(sel_t), "column": int(sel_c), "fk_node": int(sel_fk),
                }
                info["extractor_input_by_type"] = {
                    "table": int(num_t), "column": int(num_c),
                    "fk_node": int(max(len(node_scores) - num_t - num_c, 0)),
                }

        # Prize distribution
        if prizes is not None and len(prizes) > 0:
            prize_arr = np.asarray(prizes, dtype=np.float64)
            info["extractor_prize_nonzero"] = int((prize_arr > 0).sum())
            info["extractor_prize_mean"] = float(prize_arr.mean())
            info["extractor_prize_max"] = float(prize_arr.max())

        # Cost distribution + per-edge-type breakdown
        if costs is not None and len(costs) > 0:
            cost_arr = np.asarray(costs, dtype=np.float64)
            info["extractor_cost_mean"] = float(cost_arr.mean())
            info["extractor_cost_min"] = float(cost_arr.min())
            info["extractor_cost_max"] = float(cost_arr.max())
            if edge_types and len(edge_types) == len(cost_arr):
                bucket: Dict[str, List[float]] = defaultdict(list)
                for et, c in zip(edge_types, cost_arr.tolist()):
                    bucket[et].append(float(c))
                info["extractor_cost_by_type"] = {
                    et: {"count": len(vs), "mean": float(np.mean(vs))}
                    for et, vs in bucket.items()
                }

        # Edge-type input distribution (useful even when costs absent)
        if edge_types:
            et_counts: Dict[str, int] = defaultdict(int)
            for et in edge_types:
                et_counts[et] += 1
            info["extractor_edge_type_counts"] = dict(et_counts)

        if extras:
            info.update(extras)

        self.last_info = info
        return info

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed. Please install it to use PCSTExtractor.")

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])

        prizes = np.maximum(np.array(node_scores, dtype=np.float64) - self.node_threshold, 0.0)
        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[PCST] Extracted {len(selected_nodes)} nodes and {len(selected_edges)} edges.")
        self._build_ext_info(
            extract_type="PCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=self.node_threshold, t_start=t_start, graph_data=graph_data,
            node_threshold=float(self.node_threshold),
            base_cost=float(self.base_cost),
            belongs_to_cost=float(self.belongs_to_cost),
            fk_cost=float(self.fk_cost),
            macro_cost=float(self.macro_cost),
        )
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

        t_start = time.perf_counter()
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
        deg_values = list(degrees.values()) if degrees else []
        self._build_ext_info(
            extract_type="DynamicPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs_arr,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=self.node_threshold, t_start=t_start, graph_data=graph_data,
            hub_discount=float(self.hub_discount),
            degree_mean=float(np.mean(deg_values)) if deg_values else 0.0,
            degree_max=int(max(deg_values)) if deg_values else 0,
        )
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
        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        ppr_prizes = self._compute_ppr_prizes(len(node_scores), edges, node_scores)
        # 부모 클래스(동적 PCST)의 extract 메서드 호출
        result = super().extract(graph_data, ppr_prizes, seed_nodes, **kwargs)
        ppr_arr = np.asarray(ppr_prizes, dtype=np.float64)
        self.last_info = {
            **(self.last_info or {}),
            "extractor_type": "PPRPCSTExtractor",
            "ppr_alpha": float(self.ppr_alpha),
            "ppr_max_iter": int(self.ppr_max_iter),
            "ppr_prize_mean": float(ppr_arr.mean()) if ppr_arr.size else 0.0,
            "ppr_prize_max": float(ppr_arr.max()) if ppr_arr.size else 0.0,
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return result

@register("extractor", "UncertaintyPCSTExtractor")
class UncertaintyPCSTExtractor(PCSTExtractor):
    def __init__(self, alpha: float = 2.0, uncertainty_margin: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.uncertainty_margin = uncertainty_margin
    
    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed. Please install it to use PCSTExtractor.")

        t_start = time.perf_counter()
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
        self._build_ext_info(
            extract_type="UncertaintyPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=effective_threshold, t_start=t_start, graph_data=graph_data,
            alpha=float(self.alpha),
            uncertainty_margin=float(self.uncertainty_margin),
            raw_prize_nonzero=int((raw_prizes > 0).sum()),
        )
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
        t_start = time.perf_counter()
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
        deg_values = list(degrees.values()) if degrees else []
        self._build_ext_info(
            extract_type="DynamicUncertaintyPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs_arr,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=effective_threshold, t_start=t_start, graph_data=graph_data,
            alpha=float(self.alpha),
            uncertainty_margin=float(self.uncertainty_margin),
            hub_discount=float(self.hub_discount),
            degree_mean=float(np.mean(deg_values)) if deg_values else 0.0,
            degree_max=int(max(deg_values)) if deg_values else 0,
        )
        return selected_nodes.tolist(), selected_edges

@register("extractor", "GATAwarePCSTExtractor")
class GATAwarePCSTExtractor(PCSTExtractor):
    """
    본 논문의 제안 모델:
    PPR과 같은 정적 확산 대신, 학습된 GAT가 뱉어낸 '구조-문맥 융합 점수'를
    PCST의 Prize로 직접 사용하여 Implicit Bridge Table을 확정적으로 포착합니다.
    """
    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        result = super().extract(graph_data, node_scores, **kwargs)
        if isinstance(self.last_info, dict):
            self.last_info["extractor_type"] = "GATAwarePCSTExtractor"
        return result


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

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)

        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Edge-less 케이스 (isolated component): 점수 기반으로 prize 노드만 반환
        if len(edges) == 0:
            selected = np.where(prizes > 0)[0].tolist()
            self._build_ext_info(
                extract_type="AdaptivePCSTExtractor",
                node_scores=node_scores, edges=edges, edge_types=edge_types,
                prizes=prizes, costs=None,
                selected_nodes=selected, selected_edges=[],
                threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
                percentile=float(self.percentile),
                min_prize_nodes=int(self.min_prize_nodes),
                max_prize_nodes=int(self.max_prize_nodes),
                edgeless_case=True,
            )
            return selected, []

        costs = np.array([self._compute_cost(et) for et in edge_types], dtype=np.float64)
        edges_arr = np.array(edges, dtype=np.int64).reshape(-1, 2)

        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[AdaptivePCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, selected={len(selected_nodes)} nodes")
        self._build_ext_info(
            extract_type="AdaptivePCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            percentile=float(self.percentile),
            min_prize_nodes=int(self.min_prize_nodes),
            max_prize_nodes=int(self.max_prize_nodes),
        )
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

        t_start = time.perf_counter()
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
        self._build_ext_info(
            extract_type="ScoreDrivenPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            belongs_to_weight=float(self.belongs_to_weight),
            fk_weight=float(self.fk_weight),
            macro_weight=float(self.macro_weight),
            epsilon=float(self.epsilon),
        )
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

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # 1. Adaptive threshold & prizes (부모 클래스 로직 재사용)
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Edge-less 케이스 (isolated component): 점수 기반으로 prize 노드만 반환
        if len(edges) == 0:
            selected = np.where(prizes > 0)[0].tolist()
            self._build_ext_info(
                extract_type="ProductCostPCSTExtractor",
                node_scores=node_scores, edges=edges, edge_types=edge_types,
                prizes=prizes, costs=None,
                selected_nodes=selected, selected_edges=[],
                threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
                bt_weight=float(self.bt_weight),
                fk_weight=float(self.fk_weight),
                macro_weight=float(self.macro_weight),
                min_cost=float(self.min_cost),
                edgeless_case=True,
            )
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
        self._build_ext_info(
            extract_type="ProductCostPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            bt_weight=float(self.bt_weight),
            fk_weight=float(self.fk_weight),
            macro_weight=float(self.macro_weight),
            min_cost=float(self.min_cost),
            score_range_min=float(s_min),
            score_range_max=float(s_max),
        )
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
        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        num_nodes = len(node_scores)
        scores_arr = np.array(node_scores, dtype=np.float64)

        components = self._decompose(edges, num_nodes)

        # Single component → delegate directly (most common case in BIRD)
        if len(components) <= 1:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(getattr(self, 'last_info', None), dict):
                self.last_info["component_aware_wrapped"] = True
                self.last_info["component_aware_num_components"] = 1
                self.last_info["component_aware_skipped_components"] = 0
                self.last_info["extractor_type"] = f"ComponentAware+{self.last_info.get('extractor_type', 'Base')}"
            return result

        all_selected_nodes = []
        all_selected_edges = []
        components_run = 0
        components_skipped = 0

        for comp_node_ids, comp_edge_indices in components:
            # Skip components with no meaningful scores
            comp_scores = scores_arr[comp_node_ids]
            if comp_scores.max() < 0.01:
                components_skipped += 1
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
            components_run += 1

            # Remap back to global indices
            for ln in sel_nodes_local:
                all_selected_nodes.append(comp_node_ids[ln])
            for lu, lv in sel_edges_local:
                all_selected_edges.append((comp_node_ids[lu], comp_node_ids[lv]))

        logger.debug(f"[ComponentAware] {len(components)} components, "
                     f"selected {len(all_selected_nodes)} nodes total")
        # Overwrite last_info with a component-aware aggregate (preserve last inner info as nested)
        inner_info = dict(getattr(self, 'last_info', {}) or {})
        self.last_info = {
            "extractor_type": f"ComponentAware+{inner_info.get('extractor_type', 'Base')}",
            "extractor_num_input_nodes": int(num_nodes),
            "extractor_num_edges": int(len(edges)),
            "extractor_num_selected_nodes": int(len(all_selected_nodes)),
            "extractor_num_selected_edges": int(len(all_selected_edges)),
            "component_aware_wrapped": True,
            "component_aware_num_components": int(len(components)),
            "component_aware_components_run": int(components_run),
            "component_aware_skipped_components": int(components_skipped),
            "extractor_time_s": float(time.perf_counter() - t_start),
            "inner_last_info": inner_info,
        }
        return all_selected_nodes, all_selected_edges


@register("extractor", "ComponentAwareAdaptivePCSTExtractor")
class ComponentAwareAdaptivePCSTExtractor(ComponentAwareMixin, AdaptivePCSTExtractor):
    """ComponentAwareMixin + AdaptivePCSTExtractor 결합."""
    pass


@register("extractor", "ComponentAwareProductCostPCSTExtractor")
class ComponentAwareProductCostPCSTExtractor(ComponentAwareMixin, ProductCostPCSTExtractor):
    """ComponentAwareMixin + ProductCostPCSTExtractor 결합 (아이디어 2+4)."""
    pass


@register("extractor", "TopologyCostPCSTExtractor")
class TopologyCostPCSTExtractor(AdaptivePCSTExtractor):
    """
    [방향 2] Topology-derived cost PCST — edge-type parameter-free.

    Edge cost를 고정된 edge-type 비용(bt/fk/macro) 대신 endpoints의
    graph degree로부터 유도한다. Prize는 cost 공식 주도가 아니라
    tiebreaker로만 반영하여 구조적(bridge-friendly) 경로를 우선한다.

        c(u, v) = cost_scale
                × 1 / (1 + gamma * log(1 + combine(deg_u, deg_v)))
                × 1 / (1 + lambda_prize * (norm_p_u + norm_p_v))

    의도
    - 스키마 그래프의 degree 편중(Table hub ≫ Column leaf)이 자연스럽게
      "hub 경유 경로를 저렴하게" 만드는 효과 → implicit bridge table
      회복 의도와 정렬됨.
    - edge-type 파라미터(bt/fk/macro weight)가 구조적으로 제거되어
      튜닝 공간이 (gamma, lambda_prize, cost_scale) 3개로 축소.

    combine 옵션
    - 'max' (default): "한쪽이라도 hub이면 저렴" — belongs_to/macro 우호적
    - 'min': 가장 약한 endpoint가 지배 — belongs_to가 비싸짐 (주의)
    - 'mean': (deg_u + deg_v) / 2
    - 'product': deg_u * deg_v — hub 효과 증폭
    """
    def __init__(self,
                 gamma: float = 1.0,
                 lambda_prize: float = 0.3,
                 cost_scale: float = 0.1,
                 degree_combination: str = 'max',
                 **kwargs):
        super().__init__(**kwargs)
        if degree_combination not in ('max', 'min', 'mean', 'product'):
            raise ValueError(f"Unknown degree_combination: {degree_combination}")
        self.gamma = gamma
        self.lambda_prize = lambda_prize
        self.cost_scale = cost_scale
        self.degree_combination = degree_combination
        logger.info(f"Initialized TopologyCostPCSTExtractor "
                    f"(gamma={gamma}, lambda={lambda_prize}, "
                    f"cost_scale={cost_scale}, combination={degree_combination})")

    def _combine_degrees(self, deg_u: int, deg_v: int) -> float:
        if self.degree_combination == 'max':
            return float(max(deg_u, deg_v))
        if self.degree_combination == 'min':
            return float(min(deg_u, deg_v))
        if self.degree_combination == 'mean':
            return (deg_u + deg_v) / 2.0
        return float(deg_u * deg_v)

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        scores_arr = np.array(node_scores, dtype=np.float64)

        # Adaptive threshold + prize (부모 클래스 로직 재사용)
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # Edge-less 케이스 (isolated component)
        if len(edges) == 0:
            selected = np.where(prizes > 0)[0].tolist()
            self._build_ext_info(
                extract_type="TopologyCostPCSTExtractor",
                node_scores=node_scores, edges=edges, edge_types=edge_types,
                prizes=prizes, costs=None,
                selected_nodes=selected, selected_edges=[],
                threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
                gamma=float(self.gamma),
                lambda_prize=float(self.lambda_prize),
                cost_scale=float(self.cost_scale),
                degree_combination=self.degree_combination,
                edgeless_case=True,
            )
            return selected, []

        # Node degree (undirected)
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1

        # Prize 정규화 → [0, 1] (lambda 효과 스케일 안정화)
        p_max = prizes.max()
        if p_max > 1e-8:
            norm_prizes = prizes / p_max
        else:
            norm_prizes = np.zeros_like(prizes)

        # Topology + prize-tiebreaker cost
        costs = np.empty(len(edges), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            deg_combined = self._combine_degrees(degrees[u], degrees[v])
            topology_factor = 1.0 / (1.0 + self.gamma * math.log1p(deg_combined))
            prize_factor = 1.0 / (1.0 + self.lambda_prize * (norm_prizes[u] + norm_prizes[v]))
            costs[i] = self.cost_scale * topology_factor * prize_factor

        edges_arr = np.array(edges, dtype=np.int64).reshape(-1, 2)
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[TopologyCostPCST] threshold={adaptive_threshold:.4f}, "
                     f"prize_nodes={int(np.sum(prizes > 0))}, "
                     f"cost range=[{costs.min():.4f}, {costs.max():.4f}], "
                     f"selected={len(selected_nodes)} nodes")
        deg_values = list(degrees.values()) if degrees else []
        self._build_ext_info(
            extract_type="TopologyCostPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            gamma=float(self.gamma),
            lambda_prize=float(self.lambda_prize),
            cost_scale=float(self.cost_scale),
            degree_combination=self.degree_combination,
            degree_mean=float(np.mean(deg_values)) if deg_values else 0.0,
            degree_max=int(max(deg_values)) if deg_values else 0,
        )
        return selected_nodes.tolist(), selected_edges


@register("extractor", "ComponentAwareTopologyCostPCSTExtractor")
class ComponentAwareTopologyCostPCSTExtractor(ComponentAwareMixin, TopologyCostPCSTExtractor):
    """ComponentAwareMixin + TopologyCostPCSTExtractor 결합 (방향 2 + 아이디어 4)."""
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

        t_start = time.perf_counter()
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
        costs_arr_full = np.asarray(pcst_costs, dtype=np.float64) if pcst_costs else None
        self._build_ext_info(
            extract_type="EdgePrizePCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=node_prizes, costs=costs_arr_full,
            selected_nodes=real_nodes, selected_edges=selected_original_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            topk_e=int(self.topk_e),
            edge_cost=float(self.edge_cost),
            edge_prize_nonzero=int(n_edge_prize),
            edge_prize_max=float(e_prizes.max()) if e_prizes.size else 0.0,
            virtual_nodes_created=int(n_virtual),
            has_edge_embeddings=bool(edge_embeddings is not None),
            has_query_embedding=bool(query_embedding is not None),
        )
        return real_nodes, selected_original_edges


@register("extractor", "FKBackboneSteinerExtractor")
class FKBackboneSteinerExtractor(AdaptivePCSTExtractor):
    """
    [방향 2+] FK-Backbone Steiner Closure Extractor.

    기존 PCST 계열이 mixed-node graph(Table+Column) 위에서 prize-cost 최적화를
    하면서 bridge 테이블의 prize=0 문제로 recall을 잃는 구조적 한계를, 스키마
    그래프를 두 층으로 분리하여 해결한다.

    구조:
      1) Seed Table 추출
         - adaptive threshold 이상인 테이블 노드
         - adaptive threshold 이상인 컬럼 노드의 모교 테이블
         → Selector가 잘 선택한 노드들의 "소속 테이블 집합" S
      2) FK Backbone 구축
         - 노드: Table 만
         - 엣지: 'table_to_table' (이미 FK 기반 테이블 간 엣지)
      3) Steiner Tree 2-근사 (KMB)
         - Backbone 위에서 S 연결
         - 경로상의 bridge 테이블 무조건 포함 (prize 무관)
         - 비연결 component는 독립 수행 후 union
      4) Column + FK 복원
         - T_closed 각 테이블의 컬럼 중 score ≥ column_recovery_threshold 포함
         - T_closed 테이블들을 연결하는 FK 컬럼 쌍은 무조건 포함
         - 엣지: 선택된 노드 부분집합에 대한 induced subgraph

    핵심 차별점 vs SteinerBackbonePCSTExtractor:
      - SteinerBackbone: mixed-node 그래프 + backbone_bonus prize 트릭
      - FKBackboneSteiner: 테이블 전용 FK 백본에서 Steiner 후, 컬럼은 별도 복원
        (bridge 테이블이 prize/cost 계산과 완전 분리됨 → 구조적 회수 보장)
    """
    _PERCENTILE_SCOPES = ('global', 'all_cols', 'closed_cols', 'per_table')

    def __init__(self,
                 column_recovery_threshold: float = 0.0,
                 column_recovery_percentile: Optional[float] = None,
                 column_recovery_percentile_scope: str = 'closed_cols',
                 force_fk_columns: bool = True,
                 fallback_to_parent: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.column_recovery_threshold = column_recovery_threshold
        self.column_recovery_percentile = column_recovery_percentile
        if column_recovery_percentile_scope not in self._PERCENTILE_SCOPES:
            raise ValueError(
                f"column_recovery_percentile_scope must be one of "
                f"{self._PERCENTILE_SCOPES}, got {column_recovery_percentile_scope!r}")
        self.column_recovery_percentile_scope = column_recovery_percentile_scope
        self.force_fk_columns = force_fk_columns
        self.fallback_to_parent = fallback_to_parent
        mode = (f"percentile={column_recovery_percentile} scope={column_recovery_percentile_scope}"
                if column_recovery_percentile is not None
                else f"abs_threshold={column_recovery_threshold}")
        logger.info(f"Initialized FKBackboneSteinerExtractor "
                    f"({mode}, force_fk_columns={force_fk_columns})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        import networkx as nx
        from .mst import steiner_tree_2approx

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        table_to_id = graph_data.get('table_to_id', {})
        col_to_id = graph_data.get('col_to_id', {})
        num_t = len(table_to_id)
        num_c = len(col_to_id)

        scores_arr = np.array(node_scores, dtype=np.float64)

        if num_t == 0 or len(edges) == 0:
            if self.fallback_to_parent:
                result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
                if isinstance(self.last_info, dict):
                    self.last_info["extractor_type"] = "FKBackboneSteiner+fallback"
                    self.last_info["fallback_reason"] = (
                        "no_tables" if num_t == 0 else "no_edges")
                return result
            return [], []

        # --- Pre-index: col_flat_id -> table_flat_id, FK triples ---
        col_to_table: Dict[int, int] = {}
        fk_triples: Dict[int, Dict[str, Optional[int]]] = defaultdict(
            lambda: {'src': None, 'dst': None})

        for (u, v), et in zip(edges, edge_types):
            if et == 'belongs_to':
                # Builder 규약: (table_flat_id, column_flat_id)
                if u < num_t and num_t <= v < num_t + num_c:
                    col_to_table[v] = u
                elif v < num_t and num_t <= u < num_t + num_c:
                    col_to_table[u] = v
            elif et == 'is_source_of':
                # (column_flat_id, fk_flat_id)
                if u >= num_t + num_c:
                    fk_triples[u]['src'] = v
                else:
                    fk_triples[v]['src'] = u
            elif et == 'points_to':
                # (fk_flat_id, column_flat_id)
                if u >= num_t + num_c:
                    fk_triples[u]['dst'] = v
                else:
                    fk_triples[v]['dst'] = u

        # --- Step 1: Seed Table 추출 ---
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        seed_tables: set = set()
        # 직접 high-score 테이블
        for t_id in range(num_t):
            if scores_arr[t_id] > adaptive_threshold:
                seed_tables.add(t_id)
        # high-score 컬럼의 모교 테이블
        for c_flat in range(num_t, num_t + num_c):
            if scores_arr[c_flat] > adaptive_threshold:
                parent = col_to_table.get(c_flat)
                if parent is not None:
                    seed_tables.add(parent)

        if not seed_tables:
            if self.fallback_to_parent:
                logger.debug("[FKBackboneSteiner] No seed tables; falling back to AdaptivePCST")
                result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
                if isinstance(self.last_info, dict):
                    self.last_info["extractor_type"] = "FKBackboneSteiner+fallback"
                    self.last_info["fallback_reason"] = "no_seed_tables"
                return result
            return [], []

        # --- Step 2: FK Backbone 구축 (table only + table_to_table edges) ---
        G_fk = nx.Graph()
        G_fk.add_nodes_from(range(num_t))
        for (u, v), et in zip(edges, edge_types):
            if et == 'table_to_table' and u < num_t and v < num_t and u != v:
                G_fk.add_edge(u, v, weight=1.0)

        # --- Step 3: Steiner Tree 2-근사 (component별 독립 실행) ---
        closed_tables: set = set()
        if G_fk.number_of_edges() == 0:
            # FK 관계 없음 → seed tables만
            closed_tables = set(seed_tables)
        else:
            for comp in nx.connected_components(G_fk):
                comp_seeds = [s for s in seed_tables if s in comp]
                if not comp_seeds:
                    continue
                if len(comp_seeds) == 1:
                    closed_tables.add(comp_seeds[0])
                    continue
                sub_G = G_fk.subgraph(comp).copy()
                st_nodes, _ = steiner_tree_2approx(sub_G, comp_seeds, weight='weight')
                closed_tables.update(st_nodes)
            # 어떤 seed가 isolated component에 있어 Steiner에 포함 안 됐을 경우
            for s in seed_tables:
                if s not in closed_tables:
                    closed_tables.add(s)

        # --- Step 4a: Column 복원 (closed tables 각각의 high-score 컬럼) ---
        selected = set(closed_tables)
        candidate_cols = [
            c_flat for c_flat in range(num_t, num_t + num_c)
            if col_to_table.get(c_flat) in closed_tables
        ]

        if self.column_recovery_percentile is None:
            # 절댓값 θ_r 모드 (기존 동작)
            recovery_θ = self.column_recovery_threshold
            for c_flat in candidate_cols:
                if scores_arr[c_flat] >= recovery_θ:
                    selected.add(c_flat)
        else:
            # per-query percentile 모드: scope에 따라 threshold 기준 모집단이 달라짐
            p = float(self.column_recovery_percentile)
            scope = self.column_recovery_percentile_scope
            if scope == 'global':
                pool = scores_arr
                thr = float(np.percentile(pool, p)) if len(pool) > 0 else 0.0
                for c_flat in candidate_cols:
                    if scores_arr[c_flat] >= thr:
                        selected.add(c_flat)
            elif scope == 'all_cols':
                pool = scores_arr[num_t:num_t + num_c]
                thr = float(np.percentile(pool, p)) if len(pool) > 0 else 0.0
                for c_flat in candidate_cols:
                    if scores_arr[c_flat] >= thr:
                        selected.add(c_flat)
            elif scope == 'closed_cols':
                if candidate_cols:
                    pool = scores_arr[candidate_cols]
                    thr = float(np.percentile(pool, p))
                    for c_flat in candidate_cols:
                        if scores_arr[c_flat] >= thr:
                            selected.add(c_flat)
            elif scope == 'per_table':
                # 각 closed table 별로 독립 percentile
                per_table_cols: Dict[int, List[int]] = defaultdict(list)
                for c_flat in candidate_cols:
                    per_table_cols[col_to_table[c_flat]].append(c_flat)
                for t_id, cols_in_t in per_table_cols.items():
                    pool = scores_arr[cols_in_t]
                    thr = float(np.percentile(pool, p))
                    for c_flat in cols_in_t:
                        if scores_arr[c_flat] >= thr:
                            selected.add(c_flat)

        # --- Step 4b: FK 컬럼 강제 포함 (closed tables 간 FK 연결 컬럼) ---
        if self.force_fk_columns:
            for fk_flat, pair in fk_triples.items():
                src_col = pair.get('src')
                dst_col = pair.get('dst')
                if src_col is None or dst_col is None:
                    continue
                parent_s = col_to_table.get(src_col)
                parent_d = col_to_table.get(dst_col)
                if parent_s in closed_tables and parent_d in closed_tables:
                    selected.add(src_col)
                    selected.add(dst_col)
                    selected.add(fk_flat)

        # --- Step 4c: Induced edge set (선택 노드 간 엣지만) ---
        selected_edges = []
        for (u, v) in edges:
            if u in selected and v in selected:
                selected_edges.append((u, v))

        logger.debug(f"[FKBackboneSteiner] threshold={adaptive_threshold:.4f}, "
                     f"seed_tables={len(seed_tables)}, closed_tables={len(closed_tables)}, "
                     f"selected_nodes={len(selected)} ({len(selected)-len(closed_tables)} cols/fks), "
                     f"edges={len(selected_edges)}")
        selected_sorted = sorted(selected)
        self._build_ext_info(
            extract_type="FKBackboneSteinerExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=None, costs=None,
            selected_nodes=selected_sorted, selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            seed_tables_count=int(len(seed_tables)),
            closed_tables_count=int(len(closed_tables)),
            bridge_tables_added=int(len(closed_tables) - len(seed_tables)),
            fk_backbone_edges=int(G_fk.number_of_edges()),
            column_recovery_threshold=float(self.column_recovery_threshold),
            column_recovery_percentile=(
                None if self.column_recovery_percentile is None
                else float(self.column_recovery_percentile)),
            column_recovery_percentile_scope=self.column_recovery_percentile_scope,
            force_fk_columns=bool(self.force_fk_columns),
        )
        return selected_sorted, selected_edges


def _build_fk_table_graph(edges: List[Tuple[int, int]],
                          edge_types: List[str],
                          num_t: int):
    """Table-only FK subgraph 구축 (table_to_table edges). networkx Graph 반환."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_t))
    for (u, v), et in zip(edges, edge_types):
        if et == 'table_to_table' and u < num_t and v < num_t and u != v:
            G.add_edge(u, v)
    return G


def _select_top_k_table_anchors(scores_arr: np.ndarray,
                                 num_t: int,
                                 k: int) -> List[int]:
    """상위 k 테이블을 score 내림차순으로. 동점은 index 오름차순 tiebreaker."""
    if num_t == 0 or k <= 0:
        return []
    k = min(k, num_t)
    table_scores = scores_arr[:num_t]
    # argpartition 후 내림차순 재정렬 (deterministic)
    part_idx = np.argpartition(-table_scores, k - 1)[:k]
    order = np.lexsort((part_idx, -table_scores[part_idx]))
    return part_idx[order].tolist()


@register("extractor", "HybridFKPriorPCSTExtractor")
class HybridFKPriorPCSTExtractor(AdaptivePCSTExtractor):
    """
    [E-III, Neurosymbolic Layer 2] Hybrid PCST with FK-topology prior.

    Builder B-III 가 제공하는 FK reachability 를 PCST cost 조정에 직접 주입.
    - Anchor 식별: score 상위 k 테이블 (기본 k=5)
    - Anchor 간 FK shortest path 위 table_to_table edge: cost × fk_path_discount
    - Bridge (non-anchor) table node: prize boost (optional, bridge_bonus > 0 일 때)
    - Non-anchor FK edge: 기본 macro_cost 유지

    metadata (= graph_data) 의 `fk_shortest_paths` 키가 있으면 그대로 사용 (B-III 경로),
    없으면 extractor 내부에서 on-the-fly 계산 (fallback — B-III 완료 전에도 동작).

    기대 효과: 3-table JOIN 쿼리에서 bridge table 포함율 상승 (FKBackboneSteiner 와
    달리 structural closure 가 아니라 cost-side 완화이므로 precision 유지 가능).

    방안 A/B/Idea2/Idea4 와의 교차 실험은 가능 (부모 클래스 교체).
    """
    def __init__(self,
                 k_anchors: int = 5,
                 fk_path_discount: float = 0.3,
                 bridge_bonus: float = 0.0,
                 max_anchor_pairs: int = 20,
                 **kwargs):
        super().__init__(**kwargs)
        if not 0.0 < fk_path_discount <= 1.0:
            raise ValueError(f"fk_path_discount must be in (0, 1], got {fk_path_discount}")
        if bridge_bonus < 0.0:
            raise ValueError(f"bridge_bonus must be >= 0, got {bridge_bonus}")
        self.k_anchors = k_anchors
        self.fk_path_discount = fk_path_discount
        self.bridge_bonus = bridge_bonus
        self.max_anchor_pairs = max_anchor_pairs
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized HybridFKPriorPCSTExtractor "
                    f"(k_anchors={k_anchors}, fk_path_discount={fk_path_discount}, "
                    f"bridge_bonus={bridge_bonus})")

    def _consult_fk_paths(self,
                          graph_data: Dict[str, Any],
                          edges: List[Tuple[int, int]],
                          edge_types: List[str],
                          num_t: int,
                          anchors: List[int]
                          ) -> Tuple[set, set]:
        """
        Anchor 쌍 간 FK shortest path 위의 (table_edges_canonical, path_tables) 반환.

        우선순위:
          1) graph_data['fk_shortest_paths']: Dict[Tuple[int,int], List[int]] 형식
             (value = table-flat-id 경로. Builder B-III 규약.)
          2) 없으면 networkx 로 on-the-fly 계산.
        """
        import itertools
        precomputed = graph_data.get('fk_shortest_paths')
        pairs = list(itertools.combinations(sorted(anchors), 2))
        if self.max_anchor_pairs is not None and len(pairs) > self.max_anchor_pairs:
            pairs = pairs[:self.max_anchor_pairs]

        path_edges: set = set()  # canonical (min, max) table pair
        path_tables: set = set()

        if isinstance(precomputed, dict) and precomputed:
            for a, b in pairs:
                path = precomputed.get((a, b)) or precomputed.get((b, a))
                if not path:
                    continue
                for node in path:
                    if 0 <= node < num_t:
                        path_tables.add(node)
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    if 0 <= u < num_t and 0 <= v < num_t and u != v:
                        path_edges.add((min(u, v), max(u, v)))
            return path_edges, path_tables

        # Fallback: on-the-fly
        import networkx as nx
        G_fk = _build_fk_table_graph(edges, edge_types, num_t)
        if G_fk.number_of_edges() == 0:
            return path_edges, path_tables
        for a, b in pairs:
            if a not in G_fk or b not in G_fk:
                continue
            try:
                path = nx.shortest_path(G_fk, a, b)
            except nx.NetworkXNoPath:
                continue
            for node in path:
                path_tables.add(node)
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                path_edges.add((min(u, v), max(u, v)))
        return path_edges, path_tables

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if pcst_fast is None:
            raise ImportError("pcst_fast library is not installed.")

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        table_to_id = graph_data.get('table_to_id', {})
        num_t = len(table_to_id)

        if num_t == 0 or len(edges) == 0:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "HybridFKPrior+fallback"
                self.last_info["fallback_reason"] = (
                    "no_tables" if num_t == 0 else "no_edges")
            return result

        scores_arr = np.array(node_scores, dtype=np.float64)
        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        prizes = np.maximum(scores_arr - adaptive_threshold, 0.0)

        # 1. Top-k anchor tables
        anchors = _select_top_k_table_anchors(scores_arr, num_t, self.k_anchors)

        # 2. Anchor-pair FK shortest path edges (metadata or on-the-fly)
        path_edges_canonical, path_tables = (set(), set())
        if len(anchors) >= 2:
            path_edges_canonical, path_tables = self._consult_fk_paths(
                graph_data, edges, edge_types, num_t, anchors)

        # 3. Cost: macro edges on FK path 을 discount, 나머지 edge type 은 기본값
        costs = np.empty(len(edges), dtype=np.float64)
        n_discounted = 0
        for i, ((u, v), et) in enumerate(zip(edges, edge_types)):
            base = self._compute_cost(et)
            if et == 'table_to_table' and u < num_t and v < num_t:
                key = (min(u, v), max(u, v))
                if key in path_edges_canonical:
                    costs[i] = base * self.fk_path_discount
                    n_discounted += 1
                    continue
            costs[i] = base

        # 4. Prize: bridge tables 에 optional bonus
        n_boosted = 0
        if self.bridge_bonus > 0.0 and path_tables and prizes.max() > 1e-8:
            anchor_set = set(anchors)
            bridge_set = {t for t in path_tables if t not in anchor_set}
            if bridge_set:
                prize_scale = prizes.max()
                bonus = prize_scale * self.bridge_bonus
                for bt in bridge_set:
                    if 0 <= bt < len(prizes):
                        new_prize = max(prizes[bt], bonus)
                        if new_prize > prizes[bt]:
                            n_boosted += 1
                        prizes[bt] = new_prize

        edges_arr = np.array(edges, dtype=np.int64).reshape(-1, 2)
        selected_nodes, selected_edges_idx = pcst_fast.pcst_fast(
            edges_arr, prizes, costs, -1, 1, 'gw', 0)
        selected_edges = [edges[i] for i in selected_edges_idx]

        logger.debug(f"[HybridFKPriorPCST] threshold={adaptive_threshold:.4f}, "
                     f"anchors={len(anchors)}, path_edges={len(path_edges_canonical)}, "
                     f"path_tables={len(path_tables)}, "
                     f"discounted_edges={n_discounted}, boosted_bridges={n_boosted}, "
                     f"selected={len(selected_nodes)} nodes")
        self._build_ext_info(
            extract_type="HybridFKPriorPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            k_anchors=int(self.k_anchors),
            fk_path_discount=float(self.fk_path_discount),
            bridge_bonus=float(self.bridge_bonus),
            max_anchor_pairs=int(self.max_anchor_pairs),
            anchors_found=int(len(anchors)),
            path_edges_count=int(len(path_edges_canonical)),
            path_tables_count=int(len(path_tables)),
            discounted_edges=int(n_discounted),
            boosted_bridges=int(n_boosted),
            fk_paths_source=("precomputed" if graph_data.get('fk_shortest_paths') else "on_the_fly"),
        )
        return selected_nodes.tolist(), selected_edges


@register("extractor", "PathfindingEnsemblePCSTExtractor")
class PathfindingEnsemblePCSTExtractor(AdaptivePCSTExtractor):
    """
    [E-II] Base PCST + Steiner pathfinder ensemble.

    - Base PCST (AdaptivePCST default): score 기반 global subgraph 후보
    - Steiner 2-approx: top-k table anchor 간 결정론적 연결 경로 (전체 flat graph 위)
    - 결합 방식 (mode):
        * 'union'        : pcst ∪ steiner path — recall 우선, filter 로 precision 회복
        * '2pass'        : steiner path node 의 score 를 boost 한 뒤 PCST 재실행
        * 'intersection' : pcst ∩ steiner path — precision 우선, conservative

    Anchor 는 `node_scores[0:num_t]` 상위 k. Steiner 는 기존 `steiner_tree_2approx`
    재활용 (모듈 내부 함수), 전체 flat graph (table/column/fk_node) 에서 경로 탐색.

    의도: PCST 의 prize-cost 최적화가 놓치는 bridge 노드를 pathfinder 로 명시 보완.
    AdaptivePCST + XiYan 기준선 (F1 0.6987) 의 PCST✗ gold 을 경로로 복구.
    """
    _MODES = ('union', '2pass', 'intersection')

    def __init__(self,
                 mode: str = 'union',
                 k_anchors: int = 5,
                 prize_boost: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        if k_anchors < 2:
            raise ValueError(f"k_anchors must be >= 2, got {k_anchors}")
        if prize_boost < 0.0:
            raise ValueError(f"prize_boost must be >= 0, got {prize_boost}")
        self.mode = mode
        self.k_anchors = k_anchors
        self.prize_boost = prize_boost
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized PathfindingEnsemblePCSTExtractor "
                    f"(mode={mode}, k_anchors={k_anchors}, prize_boost={prize_boost})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        import networkx as nx
        from .mst import steiner_tree_2approx

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        table_to_id = graph_data.get('table_to_id', {})
        num_t = len(table_to_id)
        scores_arr = np.array(node_scores, dtype=np.float64)

        # Anchor 2개 미만 / edge 없음 → Base PCST 로 위임
        anchors = _select_top_k_table_anchors(scores_arr, num_t, self.k_anchors)
        if len(anchors) < 2 or not edges:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "PathfindingEnsemble+fallback"
                self.last_info["fallback_reason"] = (
                    "insufficient_anchors" if len(anchors) < 2 else "no_edges")
            return result

        # Steiner 2-approx on full flat graph
        G = nx.Graph()
        G.add_edges_from(edges)
        path_nodes_list, path_edges_list = steiner_tree_2approx(G, anchors)
        path_nodes = set(path_nodes_list)
        path_edges_canonical = {(min(u, v), max(u, v)) for u, v in path_edges_list}

        # 2pass: boost path nodes' score, rerun base PCST
        if self.mode == '2pass':
            boosted = scores_arr.copy()
            if scores_arr.size > 0:
                boost_val = self.prize_boost * float(scores_arr.max())
                for pn in path_nodes:
                    if 0 <= pn < len(boosted):
                        boosted[pn] = boosted[pn] + boost_val
            logger.debug(f"[PathEnsemble 2pass] path_nodes={len(path_nodes)}, "
                         f"anchors={len(anchors)}")
            result_nodes_2p, result_edges_2p = super().extract(
                graph_data, boosted.tolist(), seed_nodes, **kwargs)
            inner_info = dict(self.last_info or {})
            self._build_ext_info(
                extract_type="PathfindingEnsemblePCSTExtractor",
                node_scores=node_scores, edges=edges, edge_types=edge_types,
                prizes=None, costs=None,
                selected_nodes=result_nodes_2p, selected_edges=result_edges_2p,
                threshold=None, t_start=t_start, graph_data=graph_data,
                mode="2pass",
                k_anchors=int(self.k_anchors),
                prize_boost=float(self.prize_boost),
                anchors_found=int(len(anchors)),
                path_nodes_count=int(len(path_nodes)),
                path_edges_count=int(len(path_edges_canonical)),
                inner_last_info=inner_info,
            )
            return result_nodes_2p, result_edges_2p

        # union / intersection: 두 결과 결합
        pcst_nodes, pcst_edges = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
        inner_info = dict(self.last_info or {})
        pcst_nodes_set = set(pcst_nodes)
        pcst_edges_canonical = {(min(u, v), max(u, v)) for u, v in pcst_edges}

        if self.mode == 'union':
            result_nodes = pcst_nodes_set | path_nodes
            result_edges = pcst_edges_canonical | path_edges_canonical
        else:  # intersection
            result_nodes = pcst_nodes_set & path_nodes
            result_edges = pcst_edges_canonical & path_edges_canonical

        sorted_nodes = sorted(result_nodes)
        sorted_edges = [tuple(e) for e in sorted(result_edges)]
        logger.debug(f"[PathEnsemble {self.mode}] pcst={len(pcst_nodes_set)}, "
                     f"path={len(path_nodes)}, combined={len(sorted_nodes)} nodes, "
                     f"{len(sorted_edges)} edges")
        self._build_ext_info(
            extract_type="PathfindingEnsemblePCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=None, costs=None,
            selected_nodes=sorted_nodes, selected_edges=sorted_edges,
            threshold=None, t_start=t_start, graph_data=graph_data,
            mode=self.mode,
            k_anchors=int(self.k_anchors),
            prize_boost=float(self.prize_boost),
            anchors_found=int(len(anchors)),
            path_nodes_count=int(len(path_nodes)),
            path_edges_count=int(len(path_edges_canonical)),
            pcst_nodes_count=int(len(pcst_nodes_set)),
            pcst_edges_count=int(len(pcst_edges_canonical)),
            overlap_nodes=int(len(pcst_nodes_set & path_nodes)),
            inner_last_info=inner_info,
        )
        return sorted_nodes, sorted_edges


@register("extractor", "LouvainCommunityPCSTExtractor")
class LouvainCommunityPCSTExtractor(AdaptivePCSTExtractor):
    """
    [E-I] Louvain community-masked PCST.

    1) Table-only FK subgraph 위에서 Louvain community detection
    2) 각 community 를 member table 의 평균 score 로 rank
    3) Top-M community 의 tables + 해당 columns + 해당 FK nodes 만 유지한
       masked subgraph 생성
    4) Masked subgraph 에 Base PCST (AdaptivePCST) 실행 후 결과를 원 flat-ID 로 복원

    목적: large DB 에서 domain cluster 외부 노이즈 제거 (macro edge 비용이
    구조적으로 쿠션 되지 않는 케이스에서 정확한 전처리 프루닝 제공).

    fallback:
    - num_t == 0 / edges 없음 / FK edge 없음 / community 수 < min_communities_to_prune
      → Base PCST 그대로 실행.

    ComponentAwareProductCost 와 차이: CA 는 PCST 결과 **이후** 의 connected component,
    여기는 **입력 그래프** 의 community.
    """
    def __init__(self,
                 resolution: float = 1.0,
                 top_m_communities: int = 2,
                 min_communities_to_prune: int = 2,
                 adaptive_coverage: bool = False,
                 seed: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        if resolution <= 0.0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        if top_m_communities < 1:
            raise ValueError(f"top_m_communities must be >= 1, got {top_m_communities}")
        self.resolution = resolution
        self.top_m_communities = top_m_communities
        self.min_communities_to_prune = min_communities_to_prune
        self.adaptive_coverage = adaptive_coverage
        self.seed = seed
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized LouvainCommunityPCSTExtractor "
                    f"(resolution={resolution}, top_m={top_m_communities}, "
                    f"adaptive_coverage={adaptive_coverage})")

    def _select_communities(self,
                            communities: List[set],
                            scores_arr: np.ndarray,
                            threshold: float) -> set:
        """Top-M community 선택. adaptive_coverage 시 고score 테이블 전부 덮일 때까지 확장."""
        comm_scored = []
        for comm in communities:
            comm_list = list(comm)
            if not comm_list:
                continue
            mean_score = float(scores_arr[comm_list].mean())
            comm_scored.append((mean_score, comm_list))
        comm_scored.sort(reverse=True)

        selected = set()
        m_limit = min(self.top_m_communities, len(comm_scored))
        for _, cl in comm_scored[:m_limit]:
            selected.update(cl)

        if self.adaptive_coverage:
            # 쿼리의 고score 테이블을 모두 포함할 때까지 추가 community 편입
            num_t = len(scores_arr)  # caller ensures scores_arr[:num_t] here
            high_score_tables = {t for t in range(num_t) if scores_arr[t] > threshold}
            idx = m_limit
            while idx < len(comm_scored) and not high_score_tables.issubset(selected):
                _, cl = comm_scored[idx]
                selected.update(cl)
                idx += 1
        return selected

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        import networkx as nx
        try:
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            logger.warning("networkx louvain_communities not available; falling back")
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = "louvain_unavailable"
            return result

        t_start = time.perf_counter()
        edges = graph_data.get('edges', [])
        edge_types = graph_data.get('edge_types', [])
        table_to_id = graph_data.get('table_to_id', {})
        col_to_id = graph_data.get('col_to_id', {})
        num_t = len(table_to_id)
        num_c = len(col_to_id)

        if num_t == 0 or not edges:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = (
                    "no_tables" if num_t == 0 else "no_edges")
            return result

        scores_arr = np.array(node_scores, dtype=np.float64)

        # 1. Table FK subgraph
        G_fk = _build_fk_table_graph(edges, edge_types, num_t)
        if G_fk.number_of_edges() == 0:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = "no_fk_edges"
            return result

        # 2. Louvain (Table-level 군집; 연결성 없는 isolated table 은 solo community)
        try:
            communities = louvain_communities(
                G_fk, resolution=self.resolution, seed=self.seed)
        except Exception as e:
            logger.warning(f"Louvain failed ({e!r}); falling back to AdaptivePCST")
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = f"louvain_error:{type(e).__name__}"
            return result

        communities = [set(c) for c in communities if c]
        if len(communities) < self.min_communities_to_prune:
            # 단일 community → 프루닝 의미 없음, no-op
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = "too_few_communities"
                self.last_info["louvain_num_communities"] = int(len(communities))
            return result

        adaptive_threshold = self._compute_adaptive_threshold(scores_arr)
        selected_tables = self._select_communities(
            communities, scores_arr[:num_t], adaptive_threshold)

        # 3. Columns + FK nodes 중 selected_tables 소속만 허용
        col_to_table: Dict[int, int] = {}
        fk_pair: Dict[int, Dict[str, Optional[int]]] = defaultdict(
            lambda: {'src': None, 'dst': None})
        for (u, v), et in zip(edges, edge_types):
            if et == 'belongs_to':
                if u < num_t and num_t <= v < num_t + num_c:
                    col_to_table[v] = u
                elif v < num_t and num_t <= u < num_t + num_c:
                    col_to_table[u] = v
            elif et == 'is_source_of':
                if u >= num_t + num_c:
                    fk_pair[u]['src'] = v
                else:
                    fk_pair[v]['src'] = u
            elif et == 'points_to':
                if u >= num_t + num_c:
                    fk_pair[u]['dst'] = v
                else:
                    fk_pair[v]['dst'] = u
        selected_cols = {c for c, t in col_to_table.items() if t in selected_tables}
        selected_fks = set()
        for fk_flat, pair in fk_pair.items():
            src_col = pair.get('src')
            dst_col = pair.get('dst')
            if src_col in selected_cols and dst_col in selected_cols:
                selected_fks.add(fk_flat)
        allowed_nodes = selected_tables | selected_cols | selected_fks

        # 4. Masked local graph: flat-ID → local-ID remap, Base PCST 실행 후 복원
        allowed_list = sorted(allowed_nodes)
        if not allowed_list:
            result = super().extract(graph_data, node_scores, seed_nodes, **kwargs)
            if isinstance(self.last_info, dict):
                self.last_info["extractor_type"] = "LouvainCommunity+fallback"
                self.last_info["fallback_reason"] = "no_allowed_nodes"
            return result

        global_to_local = {g: l for l, g in enumerate(allowed_list)}
        local_edges: List[Tuple[int, int]] = []
        local_edge_types: List[str] = []
        for (u, v), et in zip(edges, edge_types):
            if u in allowed_nodes and v in allowed_nodes:
                local_edges.append((global_to_local[u], global_to_local[v]))
                local_edge_types.append(et)
        local_scores = scores_arr[allowed_list].tolist()
        local_graph = {
            'edges': local_edges,
            'edge_types': local_edge_types,
        }

        sel_local, sel_edges_local = super().extract(
            local_graph, local_scores, seed_nodes=None, **kwargs)
        selected_global = [allowed_list[l] for l in sel_local]
        selected_edges_global = [
            (allowed_list[lu], allowed_list[lv]) for lu, lv in sel_edges_local
        ]

        logger.debug(f"[LouvainPCST] communities={len(communities)}, "
                     f"selected_tables={len(selected_tables)}/{num_t}, "
                     f"allowed_nodes={len(allowed_nodes)}/{len(scores_arr)}, "
                     f"final={len(selected_global)} nodes")
        inner_info = dict(self.last_info or {})
        self._build_ext_info(
            extract_type="LouvainCommunityPCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=None, costs=None,
            selected_nodes=selected_global, selected_edges=selected_edges_global,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            resolution=float(self.resolution),
            top_m_communities=int(self.top_m_communities),
            adaptive_coverage=bool(self.adaptive_coverage),
            louvain_num_communities=int(len(communities)),
            louvain_selected_tables=int(len(selected_tables)),
            louvain_num_tables=int(num_t),
            louvain_allowed_nodes=int(len(allowed_nodes)),
            louvain_total_nodes=int(len(scores_arr)),
            community_sizes=sorted((len(c) for c in communities), reverse=True)[:10],
            inner_last_info=inner_info,
        )
        return selected_global, selected_edges_global


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

        t_start = time.perf_counter()
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
        self._build_ext_info(
            extract_type="SteinerBackbonePCSTExtractor",
            node_scores=node_scores, edges=edges, edge_types=edge_types,
            prizes=prizes, costs=costs,
            selected_nodes=selected_nodes.tolist(), selected_edges=selected_edges,
            threshold=adaptive_threshold, t_start=t_start, graph_data=graph_data,
            backbone_bonus=float(self.backbone_bonus),
            seed_nodes_count=int(len(seed_nodes) if seed_nodes else 0),
            backbone_nodes_count=int(len(backbone_nodes)),
            backbone_edges_count=int(len(backbone_edges)),
        )
        return selected_nodes.tolist(), selected_edges