import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.base import BaseSelector
from models.gat_network import SchemaHeteroGAT
from modules.projectors.dual_tower import DualTowerProjector
from modules.encoders.token_encoder import TokenEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


@register("selector", "EnsembleSelector")
class EnsembleSelector(BaseSelector):
    """
    [Phase B-2] Raw Cosine score와 GAT score를 가중 앙상블하는 Selector.

    final_score = alpha * raw_cosine + (1 - alpha) * gat_score

    Phase 2 분석에서 alpha=0.85가 최적으로 도출됨.
    Raw Cosine을 주 scorer로, GAT를 보조 신호로 활용한다.
    """
    def __init__(self,
                 weight_path: str,
                 alpha: float = 0.85,
                 top_k: int = 20,
                 in_channels: int = 384,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GAT + Projector 로드 (GATClassifierSelector와 동일)
        self.gat_model = SchemaHeteroGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        ).to(self.device)

        self.projector = DualTowerProjector(
            text_dim=in_channels,
            graph_dim=hidden_channels,
            joint_dim=hidden_channels
        ).to(self.device)

        self.encoder = TokenEncoder()

        logger.info(f"Loading GAT weights for ensemble from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.gat_model.load_state_dict(checkpoint['gat_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.gat_model.eval()
        self.projector.eval()

        self.latest_scores = []
        logger.info(f"Initialized EnsembleSelector (alpha={alpha}, top_k={top_k})")

    def _compute_gat_scores(self, question: str, graph_data: HeteroData, metadata: Dict[str, Any]) -> torch.Tensor:
        """GAT + DualTower로 node scores 계산"""
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            encoded_output = self.encoder.encode([question])
            if isinstance(encoded_output, tuple):
                q_emb = encoded_output[0].to(self.device)
            else:
                q_emb = encoded_output.to(self.device)

            if q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)
            elif q_emb.dim() == 2 and q_emb.size(0) > 1:
                q_emb = q_emb.mean(dim=0, keepdim=True)
            elif q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)

            node_embs_dict = self.gat_model(graph_data.x_dict, graph_data.edge_index_dict)

            num_nodes = len(metadata.get('node_metadata', {}))
            gat_scores = torch.zeros(num_nodes, device=self.device)

            num_t = graph_data['table'].num_nodes
            num_c = graph_data['column'].num_nodes

            if num_t > 0:
                z_q_t, z_n_t = self.projector(q_emb, node_embs_dict['table'])
                logits_t = self.projector.compute_similarity(z_q_t, z_n_t)
                gat_scores[0:num_t] = torch.sigmoid(logits_t).view(-1)

            if num_c > 0:
                z_q_c, z_n_c = self.projector(q_emb, node_embs_dict['column'])
                logits_c = self.projector.compute_similarity(z_q_c, z_n_c)
                gat_scores[num_t:num_t + num_c] = torch.sigmoid(logits_c).view(-1)

            if 'fk_node' in node_embs_dict and node_embs_dict['fk_node'].size(0) > 0:
                num_fk = node_embs_dict['fk_node'].size(0)
                z_q_fk, z_n_fk = self.projector(q_emb, node_embs_dict['fk_node'])
                logits_fk = self.projector.compute_similarity(z_q_fk, z_n_fk)
                gat_scores[num_t + num_c:num_t + num_c + num_fk] = torch.sigmoid(logits_fk).view(-1)

        return gat_scores.cpu()

    def select(self, scores: Optional[Union[torch.Tensor, List[float]]], candidates: List[int],
               question: str, graph_data: HeteroData, metadata: Dict[str, Any], **kwargs) -> List[int]:
        """
        Raw cosine scores (pipeline에서 전달)와 GAT scores를 앙상블하여 Top-K 선택.
        """
        # 1. Raw cosine scores 준비
        if scores is None:
            logger.warning("No raw cosine scores provided. Using GAT scores only.")
            raw_scores = torch.zeros(len(candidates))
        elif isinstance(scores, list):
            raw_scores = torch.tensor(scores, dtype=torch.float32)
        else:
            raw_scores = scores.squeeze().cpu()

        # 2. GAT scores 계산
        gat_scores = self._compute_gat_scores(question, graph_data, metadata)

        # 3. 앙상블: alpha * raw + (1 - alpha) * gat
        # 각각 [0, 1] 범위로 정규화 후 결합
        if raw_scores.max() > raw_scores.min():
            raw_norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        else:
            raw_norm = raw_scores

        if gat_scores.max() > gat_scores.min():
            gat_norm = (gat_scores - gat_scores.min()) / (gat_scores.max() - gat_scores.min())
        else:
            gat_norm = gat_scores

        ensemble_scores = self.alpha * raw_norm + (1.0 - self.alpha) * gat_norm

        # 4. Top-K 선택
        k_actual = min(self.top_k, len(candidates))
        top_scores, top_indices = torch.topk(ensemble_scores, k=k_actual)
        selected_seeds = [candidates[idx.item()] for idx in top_indices]

        # 5. PCST extractor에 넘길 scores 저장
        self.latest_scores = ensemble_scores.tolist()

        logger.debug(f"[Ensemble] alpha={self.alpha}, selected {k_actual} seeds")
        return selected_seeds
