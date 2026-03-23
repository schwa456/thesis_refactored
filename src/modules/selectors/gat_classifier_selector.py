import torch
from typing import List, Any, Dict
from modules.registry import register
from modules.base import BaseSelector
from models.node_classifier import SimpleNodeLinker
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "GATClassifierSelector")
class GATClassifierSelector(BaseSelector):
    def __init__(self, weight_path: str, hidden_dim: int = 256, threshold: float = 0.5, plm_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        self.threshold = threshold
        self.model = SimpleNodeLinker(hidden_dim=hidden_dim, plm_name=plm_name)
        
        # CPU/GPU Device 매핑 및 가중치 로드
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Loaded GATClassifierSelector (Threshold: {threshold}) from {weight_path}")

    def select(self, question: str, graph_data: Any, metadata: Dict[str, Any], **kwargs) -> List[str]:
        selected_nodes = []
        
        # 1. 모델 추론 (No Gradient)
        with torch.no_grad():
            graph_data = graph_data.to(next(self.model.parameters()).device)
            # out_probs = {'table': Tensor([N_t]), 'column': Tensor([N_c])}
            out_probs = self.model(graph_data, [question])
            
        # 2. 역방향 ID 매핑 딕셔너리 생성
        id_to_table = {v: k for k, v in metadata['table_to_id'].items()}
        id_to_col = {v: k for k, v in metadata['col_to_id'].items()}
        
        # 3. 임계값(Threshold) 기반 하드 필터링
        if 'table' in out_probs:
            t_probs = out_probs['table'].cpu().tolist()
            for i, prob in enumerate(t_probs):
                if prob >= self.threshold:
                    selected_nodes.append(id_to_table[i])
                    
        if 'column' in out_probs:
            c_probs = out_probs['column'].cpu().tolist()
            for i, prob in enumerate(c_probs):
                if prob >= self.threshold:
                    selected_nodes.append(id_to_col[i])
                    
        logger.debug(f"[GAT Selector] Selected {len(selected_nodes)} nodes for query.")
        return selected_nodes