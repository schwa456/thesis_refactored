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

    def select(self, question: str, graph_data: Any, metadata: Dict[str, Any], **kwargs) -> List[int]:
        selected_node_ids = []
        
        with torch.no_grad():
            graph_data = graph_data.to(next(self.model.parameters()).device)
            out_probs = self.model(graph_data, [question])
            
        # [핵심] 딕셔너리를 순회하며 확률이 threshold를 넘는 글로벌 Node ID를 수집
        # metadata['table_to_id']와 metadata['col_to_id']는 로컬 ID를 글로벌 ID로 
        # 매핑하는 정보를 포함하고 있어야 합니다. (만약 분리되어 있다면 맞춰주어야 함)
        
        global_offset = len(metadata['table_to_id']) # 컬럼 ID가 테이블 ID 뒤에 이어지는 경우 오프셋
        
        if 'table' in out_probs:
            t_probs = out_probs['table'].cpu().tolist()
            for i, prob in enumerate(t_probs):
                if prob >= self.threshold:
                    selected_node_ids.append(i) # 테이블 글로벌 ID
                    
        if 'column' in out_probs:
            c_probs = out_probs['column'].cpu().tolist()
            for i, prob in enumerate(c_probs):
                if prob >= self.threshold:
                    selected_node_ids.append(i + global_offset) # 컬럼 글로벌 ID
                    
        return selected_node_ids