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

        self.latest_scores = []
        
        logger.info(f"Loaded GATClassifierSelector (Threshold: {threshold}) from {weight_path}")

    def select(self, question: str, graph_data: Any, metadata: Dict[str, Any], **kwargs) -> List[int]:
        selected_node_ids = []
        
        with torch.no_grad():
            graph_data = graph_data.to(next(self.model.parameters()).device)
            out_logits = self.model(graph_data, [question])
        
        global_offset = len(metadata['table_to_id']) # 컬럼 ID가 테이블 ID 뒤에 이어지는 경우 오프셋
        total_nodes = len(metadata.get('node_metadata', {}))

        all_scores = [0.0] * total_nodes
        
        if 'table' in out_logits:
            t_probs = torch.sigmoid(out_logits['table']).cpu().tolist()
            for i, prob in enumerate(t_probs):
                if i < total_nodes:
                    all_scores[i] = prob
                if prob >= self.threshold:
                    selected_node_ids.append(i) # 테이블 글로벌 ID
                    
        if 'column' in out_logits:
            c_probs = torch.sigmoid(out_logits['column']).cpu().tolist()
            for i, prob in enumerate(c_probs):
                idx = i + global_offset
                if idx < total_nodes:
                    all_scores[idx] = prob
                if prob >= self.threshold:
                    selected_node_ids.append(idx)

        self.latest_scores = all_scores
                    
        return selected_node_ids