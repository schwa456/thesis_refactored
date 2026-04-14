"""
DirectGATSelector — Query-Conditioned GAT + DirectClassifierHead 추론용 Selector.

기존 GATClassifierSelector(DualTowerProjector 기반)와 달리, query를
GAT forward에만 주입하고 projector 재입력을 제거한다. 이 selector가 반환하는
노드별 점수는 곧 각 노드의 '정답 여부' 확률이다.
"""
import torch
from typing import List, Dict, Any, Optional
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.base import BaseSelector
from models.gat_network import SchemaHeteroGAT
from models.direct_classifier import DirectClassifierHead
from modules.encoders.local_encoder import LocalPLMEncoder
from modules.encoders.token_encoder import TokenEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


@register("selector", "DirectGATSelector")
class DirectGATSelector(BaseSelector):
    def __init__(
        self,
        weight_path: str,
        in_channels: int = 384,
        hidden_channels: int = 256,
        out_channels: int = 256,
        classifier_hidden: int = 256,
        threshold: float = 0.5,
        query_conditioned: bool = False,
        query_supernode: bool = False,
        encoder_type: str = "plm",
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.apply_threshold = kwargs.get('apply_threshold', False)
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode
        self.encoder_type = encoder_type

        # GAT backbone
        self.gat_model = SchemaHeteroGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            query_conditioned=query_conditioned,
            query_supernode=query_supernode,
        ).to(self.device)

        # Direct classifier heads (node type 별)
        self.classifier_types = ['table', 'column', 'fk_node']
        self.classifier_heads = torch.nn.ModuleDict({
            nt: DirectClassifierHead(
                in_dim=out_channels,
                hidden_dim=classifier_hidden,
                dropout=0.0,  # 추론 시에는 dropout off
            ).to(self.device)
            for nt in self.classifier_types
        })

        # Encoder
        if self.encoder_type == "plm":
            self.encoder = LocalPLMEncoder()
            logger.info("DirectGATSelector: using LocalPLMEncoder (sentence-level)")
        else:
            self.encoder = TokenEncoder()
            logger.info("DirectGATSelector: using TokenEncoder (token-level)")

        # Checkpoint 로드
        logger.info(f"Loading Direct GAT weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device)
        if 'gat_state_dict' not in checkpoint or 'classifier_state_dict' not in checkpoint:
            raise RuntimeError(
                "Invalid checkpoint format. Expected 'gat_state_dict' and 'classifier_state_dict'. "
                "Train with src/train_gat_direct.py."
            )

        # Lazy GATv2Conv weights: dummy forward로 초기화 후 load
        self._lazy_init_gat()
        self.gat_model.load_state_dict(checkpoint['gat_state_dict'])
        self.classifier_heads.load_state_dict(checkpoint['classifier_state_dict'])

        recall_val = checkpoint.get('recall', None)
        recall_str = f"{recall_val:.4f}" if isinstance(recall_val, float) else str(recall_val)
        logger.info(
            f"Weights loaded. Trained Epoch: {checkpoint.get('epoch', 'Unknown')}, "
            f"Val Recall: {recall_str}"
        )

        self.gat_model.eval()
        for head in self.classifier_heads.values():
            head.eval()
        self.latest_scores: List[float] = []

    def _lazy_init_gat(self):
        """GATv2Conv의 lazy parameter를 dummy forward로 초기화."""
        dummy = HeteroData()
        dummy['table'].x = torch.zeros(1, self.gat_model.lin_dict['table'].in_channels // (2 if self.query_conditioned else 1), device=self.device)
        dummy['column'].x = torch.zeros(1, self.gat_model.lin_dict['column'].in_channels // (2 if self.query_conditioned else 1), device=self.device)
        dummy['fk_node'].x = torch.zeros(1, self.gat_model.lin_dict['fk_node'].in_channels // (2 if self.query_conditioned else 1), device=self.device)

        zero = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        dummy['table', 'has_column', 'column'].edge_index = zero
        dummy['column', 'belongs_to', 'table'].edge_index = zero
        dummy['column', 'is_source_of', 'fk_node'].edge_index = zero
        dummy['fk_node', 'points_to', 'column'].edge_index = zero
        dummy['table', 'table_to_table', 'table'].edge_index = zero

        if self.query_supernode:
            dummy['query_node'].x = torch.zeros(1, self.gat_model.lin_dict['query_node'].in_channels, device=self.device)
            for schema_nt in ['table', 'column', 'fk_node']:
                dummy['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = zero
                dummy[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = zero

        with torch.no_grad():
            if self.query_conditioned:
                dummy_q = torch.zeros(1, self.gat_model.lin_dict['table'].in_channels // 2, device=self.device)
                _ = self.gat_model(dummy.x_dict, dummy.edge_index_dict, query_emb=dummy_q)
            else:
                _ = self.gat_model(dummy.x_dict, dummy.edge_index_dict)

    def select(
        self,
        scores: Optional[List[float]],
        candidates: List[int],
        question: str,
        graph_data: HeteroData,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> List[int]:
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            # 1) Query embedding
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

            # 2) GAT forward (query 주입은 여기서 1회만)
            if self.query_supernode:
                graph_data['query_node'].x = q_emb
                for schema_nt in ['table', 'column', 'fk_node']:
                    num_nodes = graph_data[schema_nt].num_nodes
                    if num_nodes == 0:
                        graph_data['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = \
                            torch.zeros((2, 0), dtype=torch.long, device=self.device)
                        graph_data[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = \
                            torch.zeros((2, 0), dtype=torch.long, device=self.device)
                        continue
                    src = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                    dst = torch.arange(num_nodes, dtype=torch.long, device=self.device)
                    graph_data['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = \
                        torch.stack([src, dst], dim=0)
                    graph_data[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = \
                        torch.stack([dst, src], dim=0)
                node_embs_dict = self.gat_model(graph_data.x_dict, graph_data.edge_index_dict)
            elif self.query_conditioned:
                node_embs_dict = self.gat_model(
                    graph_data.x_dict, graph_data.edge_index_dict, query_emb=q_emb
                )
            else:
                node_embs_dict = self.gat_model(graph_data.x_dict, graph_data.edge_index_dict)

            # 3) Node type별 직접 classifier
            num_nodes = len(metadata.get('node_metadata', {}))
            final_scores = torch.zeros(num_nodes, device=self.device)

            num_t = graph_data['table'].num_nodes
            num_c = graph_data['column'].num_nodes

            if num_t > 0:
                logits_t = self.classifier_heads['table'](node_embs_dict['table'])
                final_scores[0:num_t] = torch.sigmoid(logits_t).view(-1)

            if num_c > 0:
                logits_c = self.classifier_heads['column'](node_embs_dict['column'])
                final_scores[num_t:num_t + num_c] = torch.sigmoid(logits_c).view(-1)

            if 'fk_node' in node_embs_dict and node_embs_dict['fk_node'].size(0) > 0:
                num_fk = node_embs_dict['fk_node'].size(0)
                logits_fk = self.classifier_heads['fk_node'](node_embs_dict['fk_node'])
                final_scores[num_t + num_c:num_t + num_c + num_fk] = torch.sigmoid(logits_fk).view(-1)

        self.latest_scores = final_scores.cpu().tolist()

        if self.apply_threshold:
            selected = [c for c, s in zip(candidates, self.latest_scores) if s >= self.threshold]
            logger.info(f"[DirectGAT] apply_threshold={self.threshold}: {len(selected)}/{len(candidates)} nodes selected")
            return selected if selected else candidates[:1]

        return candidates
