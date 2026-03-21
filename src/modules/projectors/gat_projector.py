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
        with torch.no_grad():
            x_dict = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
            edge_index_dict = {k: v.to(self.device) for k, v in graph_data.edge_index_dict.items()}
            q_emb = q_emb.to(self.device) # [1, 22, 384]
            
            # 1. GAT Message Passing & Flattening
            updated_embs_dict = self.gat(x_dict, edge_index_dict)
            embs_list = [updated_embs_dict[nt] for nt in ['table', 'column', 'fk_node'] if nt in updated_embs_dict]
            flat_node_embs = torch.cat(embs_list, dim=0) # [94, 256]
            
            # 2. Dual Tower Projection
            z_q, z_nodes = self.dual_tower(q_emb, flat_node_embs)
            z_q_sq = z_q.squeeze(0) # [22, 256]
            
            # 3. 22 x 94 Cross-Similarity 생성
            z_q_norm = torch.nn.functional.normalize(z_q_sq, p=2, dim=-1)
            z_n_norm = torch.nn.functional.normalize(z_nodes, p=2, dim=-1)
            sim_matrix = torch.matmul(z_q_norm, z_n_norm.transpose(0, 1)) # [22, 94]
            
            # 💡 [핵심 구현] "각 Token 별로 최대 점수를 가지는 Node 선택"
            # dim=1(노드 차원)을 기준으로 최댓값을 뽑으면, 각 토큰(22개)이 선택한 최고 점수와 해당 노드 인덱스가 나옵니다.
            best_scores_per_token, best_nodes_per_token = torch.max(sim_matrix, dim=1)
            
            # PCST에게 넘겨줄 94개짜리 빈 점수판 생성 (기본값 0.0)
            total_nodes = flat_node_embs.size(0)
            node_scores = torch.zeros(total_nodes, device=self.device)
            
            # 투표 결과 반영: 각 토큰이 선택한 노드에 점수를 부여합니다.
            for score, node_idx in zip(best_scores_per_token, best_nodes_per_token):
                # 한 노드가 여러 토큰의 선택을 받았을 수 있으므로, 더 큰 점수로 업데이트합니다.
                node_scores[node_idx] = torch.max(node_scores[node_idx], score)
            
            return node_scores.cpu() # 최종 [94] 형태 반환

    def forward(self, q_emb: torch.Tensor, node_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [BaseProjector 필수 구현] 
        단순 텐서 기반 투영이 필요할 경우, 내부의 DualTowerProjector로 연산을 위임합니다.
        """
        # 디바이스 동기화 후 위임
        return self.dual_tower(q_emb.to(self.device), node_embs.to(self.device))

    def compute_similarity(self, z_q: torch.Tensor, z_n: torch.Tensor) -> torch.Tensor:
        """
        [BaseProjector 필수 구현]
        투영된 벡터 간의 유사도 계산 역시 내부 DualTowerProjector로 위임합니다.
        """
        return self.dual_tower.compute_similarity(z_q, z_n)