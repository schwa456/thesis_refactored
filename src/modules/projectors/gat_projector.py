import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from modules.registry import register
from modules.base import BaseProjector
from models.gat_network import SchemaHeteroGAT
from modules.projectors.dual_tower import DualTowerProjector
from utils.logger import get_logger

logger = get_logger(__name__)

@register("projector", "GATProjector")
class GATProjector(BaseProjector):
    """
    학습된 SchemaHeteroGAT와 DualTowerProjector를 통합하여,
    추론(Inference) 시에 Graph 구조를 반영한 최종 유사도 점수를 반환하는 브릿지 모듈입니다.
    """
    def __init__(self, hidden_channels: int = 256, num_layers: int = 2, heads: int = 4, checkpoint_path: str = None, **kwargs):
        super(GATProjector, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 모델 초기화
        self.gat = SchemaHeteroGAT(in_channels=384, hidden_channels=hidden_channels, out_channels=256, num_layers=num_layers, heads=heads).to(self.device)
        self.dual_tower = DualTowerProjector(text_dim=384, graph_dim=256, joint_dim=256).to(self.device)
        
        self.is_graph_aware = True # 파이프라인에서 Graph Data를 넘겨주도록 하는 플래그
        
        # 2. 체크포인트(가중치) 로드
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading trained weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.gat.load_state_dict(checkpoint['gat_state_dict'])
            self.dual_tower.load_state_dict(checkpoint['projector_state_dict'])
            
            logger.info(f"✅ Weights loaded successfully! (Best Val Recall@15: {checkpoint.get('best_val_recall', 'Unknown')})")
        else:
            logger.warning(f"🚨 Checkpoint not found at {checkpoint_path}! Using random initialization.")
            
        self.gat.eval()
        self.dual_tower.eval()

    def compute_scores(self, q_emb: torch.Tensor, graph_data: Any) -> torch.Tensor:
        """
        [추론 전용] NLQ 임베딩과 HeteroData를 받아 최종 노드별 유사도 점수(1D Tensor)를 반환합니다.
        """
        with torch.no_grad():
            # 데이터 디바이스 이동
            x_dict = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
            edge_index_dict = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}
            q_emb = q_emb.to(self.device)
            
            # 1. GAT Message Passing
            updated_embs_dict = self.gat(x_dict, edge_index_dict)
            
            # 2. Flattening (graph_builder.py의 Global ID 순서 보장: Table -> Column -> FK)
            embs_list = []
            for node_type in ['table', 'column', 'fk_node']:
                if node_type in updated_embs_dict:
                    embs_list.append(updated_embs_dict[node_type])
            
            flat_node_embs = torch.cat(embs_list, dim=0) # (Total_Nodes, 256)
            
            # 3. Dual Tower Projection 및 유사도 계산
            z_q, z_nodes = self.dual_tower(q_emb, flat_node_embs)
            scores = self.dual_tower.compute_similarity(z_q, z_nodes)
            
            return scores.cpu() # 후속 파이프라인(Selector 등) 처리를 위해 CPU로 반환