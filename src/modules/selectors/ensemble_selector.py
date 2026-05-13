import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.base import BaseSelector
from models.gat_network import SchemaHeteroGAT
from models.gat_network_v2 import SchemaHeteroGATv2
from modules.projectors.dual_tower import DualTowerProjector
from modules.encoders.token_encoder import TokenEncoder
from modules.encoders.local_encoder import LocalPLMEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


NUM_LAYERS_MODES = {"fixed", "per_db_dynamic", "D_max", "D_max_plus1"}
GAT_VERSIONS = {"v1", "v2"}


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
                 num_layers: int = 3,
                 query_conditioned: bool = False,
                 query_supernode: bool = False,
                 encoder_type: str = "token",
                 num_layers_mode: str = "fixed",
                 diameter_cache_path: Optional[str] = None,
                 num_layers_fallback: Optional[int] = None,
                 gat_version: str = "v1",
                 gat_v2_kwargs: Optional[Dict[str, Any]] = None,
                 score_normalization: str = "minmax",
                 supernode_edge_direction: str = "bidirectional",
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode
        self.encoder_type = encoder_type
        # score_normalization: "minmax" (default, backward compat) | "none" | "zscore"
        valid_norms = {"minmax", "none", "zscore"}
        if score_normalization not in valid_norms:
            raise ValueError(f"score_normalization must be in {valid_norms}, got '{score_normalization}'")
        self.score_normalization = score_normalization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Proposal C H2: per-DB dynamic num_layers (inference-time early-exit).
        if num_layers_mode not in NUM_LAYERS_MODES:
            raise ValueError(f"num_layers_mode must be in {NUM_LAYERS_MODES}, got {num_layers_mode}")
        if gat_version not in GAT_VERSIONS:
            raise ValueError(f"gat_version must be in {GAT_VERSIONS}, got {gat_version}")
        self.gat_version = gat_version
        self.num_layers_mode = num_layers_mode
        self.num_layers_fallback = num_layers_fallback if num_layers_fallback is not None else num_layers
        self.diameter_dict: Dict[str, int] = {}
        if num_layers_mode != "fixed":
            if not diameter_cache_path:
                raise ValueError(
                    f"num_layers_mode={num_layers_mode} requires diameter_cache_path")
            if not os.path.exists(diameter_cache_path):
                raise FileNotFoundError(
                    f"diameter_cache_path not found: {diameter_cache_path}")
            loaded = torch.load(diameter_cache_path, map_location="cpu")
            if isinstance(loaded, dict) and "diameters" in loaded:
                loaded = loaded["diameters"]
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"diameter cache at {diameter_cache_path} must be dict[db_name, int]")
            self.diameter_dict = {str(k): int(v) for k, v in loaded.items()}
            logger.info(
                f"[EnsembleSelector] num_layers_mode={num_layers_mode} | "
                f"loaded {len(self.diameter_dict)} DB diameters from {diameter_cache_path} | "
                f"fallback depth={self.num_layers_fallback}")

        # GAT backbone — v1 (default) or v2 (Proposal C H2 와 s06 계열 확장 기능 공유).
        # v1 과 v2 는 default options (pairnorm='none', JK='none', dual_stream=False, etc.)
        # 하에서 state_dict key 가 동일하므로 기존 checkpoint 재활용 가능. 재학습 불요.
        if gat_version == "v2":
            v2_kwargs = dict(gat_v2_kwargs) if gat_v2_kwargs else {}
            # Selector 가 depth resolution 을 단일 책임으로 가지므로 v2 모델 내부의 lookup 은 비활성.
            # num_layers_mode="fixed" + active_num_layers override 로 동작 (v1 과 동일 경로).
            v2_kwargs.setdefault("num_layers_mode", "fixed")
            # supernode_edge_direction 도 ckpt 에 맞춰 forward (kwargs override 우선).
            v2_kwargs.setdefault("supernode_edge_direction", supernode_edge_direction)
            self.gat_model = SchemaHeteroGATv2(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                query_conditioned=query_conditioned,
                query_supernode=query_supernode,
                **v2_kwargs,
            ).to(self.device)
        else:
            self.gat_model = SchemaHeteroGAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                query_conditioned=query_conditioned,
                query_supernode=query_supernode,
                supernode_edge_direction=supernode_edge_direction,
            ).to(self.device)

        self.projector = DualTowerProjector(
            text_dim=in_channels,
            graph_dim=hidden_channels,
            joint_dim=hidden_channels
        ).to(self.device)

        if encoder_type == "plm":
            self.encoder = LocalPLMEncoder()
            logger.info("EnsembleSelector: using LocalPLMEncoder (sentence-level)")
        else:
            self.encoder = TokenEncoder()
            logger.info("EnsembleSelector: using TokenEncoder (token-level)")

        logger.info(f"Loading GAT weights for ensemble from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.gat_model.load_state_dict(checkpoint['gat_state_dict'])
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.gat_model.eval()
        self.projector.eval()

        self.latest_scores = []
        # SGBE Phase 2 (DECISIONS 2026-05-12) — filter 단이 raw score 기반 gating 을 수행할 수 있도록
        # latest_raw_gat_scores (sigmoid 거친 GAT score, pre-normalize) + latest_raw_cos_scores 노출.
        # 기존 self.latest_scores (blended + normalize) 는 그대로 유지 — backward compat.
        self.latest_raw_gat_scores: Dict[int, float] = {}
        self.latest_raw_cos_scores: Dict[int, float] = {}
        self.last_resolved_depth: Optional[int] = None
        logger.info(
            f"Initialized EnsembleSelector (alpha={alpha}, top_k={top_k}, "
            f"num_layers={num_layers}, num_layers_mode={num_layers_mode}, gat_version={gat_version})")

    def _resolve_active_depth(self, metadata: Dict[str, Any]) -> Optional[int]:
        """Proposal C H2: resolve per-query forward depth from metadata['db_id'].

        Returns None when mode=='fixed' (caller then uses self.gat_model.num_layers).
        Otherwise returns an int in [1, self.gat_model.num_layers]:
          - D_max:          depth = min(D_max, num_layers)
          - D_max_plus1:    depth = min(D_max + 1, num_layers)
          - per_db_dynamic: alias for D_max (policy fixed; reserved for future mix).
        Missing db_id or unknown DB → self.num_layers_fallback (clamped).
        """
        if self.num_layers_mode == "fixed":
            return None
        max_depth = self.gat_model.num_layers
        db_id = metadata.get("db_id") if isinstance(metadata, dict) else None
        d_max = self.diameter_dict.get(str(db_id)) if db_id is not None else None

        if d_max is None:
            depth = self.num_layers_fallback
            logger.debug(
                f"[EnsembleSelector] db_id={db_id!r} missing from diameter dict; "
                f"fallback depth={depth}")
        elif self.num_layers_mode in ("D_max", "per_db_dynamic"):
            depth = d_max
        elif self.num_layers_mode == "D_max_plus1":
            depth = d_max + 1
        else:
            depth = self.num_layers_fallback
        return max(1, min(int(depth), max_depth))

    @staticmethod
    def get_raw_scores(query_emb: torch.Tensor, node_embs: torch.Tensor) -> torch.Tensor:
        """Pre-GAT raw cosine scores between a query and schema nodes.

        V-3 (SuperNode Top-k) 에서 재사용되는 utility. Encoder 출력 (pre-GAT) 에
        직접 cosine similarity 를 계산한다. Ensemble α-blend 를 거치지 않는 순수 raw 경로.

        Args:
            query_emb: [d] or [1, d] — single query embedding.
            node_embs: [N, d] — schema node embeddings (pre-GAT encoder output).

        Returns:
            [N] tensor of cosine similarities in [-1, 1].
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        elif query_emb.dim() == 2 and query_emb.size(0) > 1:
            query_emb = query_emb.mean(dim=0, keepdim=True)
        q_norm = F.normalize(query_emb, dim=-1)
        n_norm = F.normalize(node_embs, dim=-1)
        return (n_norm @ q_norm.t()).squeeze(-1)

    def _post_ensemble_hook(
        self,
        ensemble_scores: torch.Tensor,
        question: str,
        graph_data: HeteroData,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Subclass hook applied after (alpha*raw + (1-alpha)*gat), before top-k.
        Default: identity. Overridden by NeurosymbolicL1Selector for λ·reach boost.
        """
        return ensemble_scores

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

            active_depth = self._resolve_active_depth(metadata)
            # Proposal C H2 smoke / analysis hook — last resolved depth per forward pass.
            self.last_resolved_depth = active_depth

            if self.query_supernode:
                # Super Node 모드: query_node를 그래프에 동적 주입
                graph_data['query_node'].x = q_emb  # [1, 384]
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
                node_embs_dict = self.gat_model(
                    graph_data.x_dict, graph_data.edge_index_dict,
                    query_emb=q_emb if self.query_conditioned else None,
                    active_num_layers=active_depth)
            elif self.query_conditioned:
                node_embs_dict = self.gat_model(
                    graph_data.x_dict, graph_data.edge_index_dict,
                    query_emb=q_emb, active_num_layers=active_depth)
            else:
                node_embs_dict = self.gat_model(
                    graph_data.x_dict, graph_data.edge_index_dict,
                    active_num_layers=active_depth)

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
        # score_normalization: "minmax" (default, backward compat) | "none" | "zscore"
        def _normalize(scores: torch.Tensor) -> torch.Tensor:
            if self.score_normalization == "none":
                return scores
            elif self.score_normalization == "zscore":
                std = float(scores.std().item()) if scores.numel() > 0 else 0.0
                if std > 1e-8:
                    return (scores - scores.mean()) / scores.std()
                return scores
            else:  # "minmax"
                if scores.max() > scores.min():
                    return (scores - scores.min()) / (scores.max() - scores.min())
                return scores

        raw_norm = _normalize(raw_scores)
        gat_norm = _normalize(gat_scores)
        ensemble_scores = self.alpha * raw_norm + (1.0 - self.alpha) * gat_norm

        # 3-a. Subclass hook (Neurosymbolic Layer 1 등) — default no-op
        ensemble_scores = self._post_ensemble_hook(
            ensemble_scores, question=question, graph_data=graph_data, metadata=metadata
        )

        # 4. Top-K 선택
        k_actual = min(self.top_k, len(candidates))
        top_scores, top_indices = torch.topk(ensemble_scores, k=k_actual)
        selected_seeds = [candidates[idx.item()] for idx in top_indices]

        # 5. PCST extractor에 넘길 scores 저장 (기존 — blended + normalize)
        self.latest_scores = ensemble_scores.tolist()

        # 5-a. SGBE Phase 2 — filter 단에 전달할 raw score (sigmoid 거친 GAT score, normalize 전) + raw cosine
        # node index = candidates[i] (보통 단순 0..N-1, schema_linking.py:160-163 참조)
        gat_list = gat_scores.tolist()
        self.latest_raw_gat_scores = {int(candidates[i]): float(gat_list[i])
                                       for i in range(min(len(candidates), len(gat_list)))}
        raw_list = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        self.latest_raw_cos_scores = {int(candidates[i]): float(raw_list[i])
                                       for i in range(min(len(candidates), len(raw_list)))}

        logger.debug(f"[Ensemble] alpha={self.alpha}, selected {k_actual} seeds, "
                     f"latest_raw_gat_scores={len(self.latest_raw_gat_scores)} "
                     f"latest_raw_cos_scores={len(self.latest_raw_cos_scores)}")
        return selected_seeds
