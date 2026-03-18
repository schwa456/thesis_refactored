import numpy as np
import torch
try:
    import pcst_fast
except ImportError:
    pcst_fast = None
from torch_geometric.utils import degree
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

@register("extractor", "PCST")
class PCSTExtractor(BaseExtractor):
    def __init__(self, base_cost: float = 1.0, belongs_to_cost: float = 0.01, node_threshold: float = 0.5, **kwargs):
        self.base_cost = base_cost
        self.belongs_to_cost = belongs_to_cost
        self.node_threshold = node_threshold
        logger.info(f"Initialized PCST Extractor (Cost: {base_cost}, Threshold: {node_threshold})")

    def _compute_cost(self, edge_type: str) -> float:
        return self.belongs_to_cost if edge_type == 'belongs_to' else self.base_cost

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

@register("extractor", "AdvancedPCST")
class AdvancedPCSTExtractor(PCSTExtractor):
    def __init__(self, belongs_to_cost: float = 0.01, ppr_alpha: float = 0.15, ppr_max_iter: int = 50, node_threshold: float = 0.5, **kwargs):
        super().__init__(belongs_to_cost=belongs_to_cost, node_threshold=node_threshold)
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
        return super().extract(graph_data, ppr_prizes, seed_nodes, **kwargs)