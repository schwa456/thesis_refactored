"""DirectedTopKSuperNodeSelector — 학위 논문 Part III (advisor 제안 reflect).

Inference-time SuperNode 변형:
  - **Out-only directed edge**: query_node → schema (`attends_to_*`) 만 등록.
    schema → query_node (`attended_by_*`) 역방향 edge 제거.
  - **Threshold filter**: per-query min-max normalized cosine 기준으로 schema 노드를 가려서
    query_node 와 directed edge 로 연결한다. 통과 못 한 노드는 query_node 와 비연결.
  - **Three threshold modes**:
      * 'percentile' (default, primary): per-query Pn — `value=80` → P80 (선택률 ~20%).
      * 'top_k': per-query top-K (기존 SuperNode 와 직접 비교 baseline).
      * 'abs_tau': per-query min-max norm 후 절대 cutoff (예: 0.7).

Selector 단에서 graph 의 query_node 주입 + edge filter 를 직접 수행하므로 v1/v2 GAT 모델 모두에서
동작한다. 학습 시점에는 `gat_network.py` / `gat_network_v2.py` 의 동일한 mask compute 를 통해
`supernode_threshold_mode` 가 forward 에서 발동한다 (양 path 의 동치).

근거:
  - planning/DECISIONS.md 2026-05-05 (Directed Top-K SuperNode 학습 변형 3 종 + threshold P80 primary 채택)
  - notebooks/analysis_results/raw_score_distribution_for_directed_topk.md §5 / §7
  - 기존 SuperNode 양방향 edge 분기 (ensemble_selector.py:230-246) 의 directed-only 변형

주의: 학습 시 동일 threshold + edge 구조로 학습된 ckpt 가 필요. 기존 SuperNode 양방향 ckpt
재활용은 score quality 측면 부정 가능성 — DECISIONS 시나리오 A/B/C 분기.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.selectors.ensemble_selector import EnsembleSelector
from utils.logger import get_logger

logger = get_logger(__name__)


SUPERNODE_THRESHOLD_MODES = {"percentile", "top_k", "abs_tau"}
SUPERNODE_SCORE_NORMS = {"minmax", "none"}


@register("selector", "DirectedTopKSuperNodeSelector")
class DirectedTopKSuperNodeSelector(EnsembleSelector):
    """EnsembleSelector 의 query_supernode 분기를 directed + threshold 변형으로 override.

    Args (EnsembleSelector 외 추가):
        threshold_mode: 'percentile' | 'top_k' | 'abs_tau'.
        threshold_value: percentile=80 → P80 cutoff. top_k=20 → top-20. abs_tau=0.7 → score>=0.7.
        score_normalization: 'minmax' (default, analyzer 정의와 일치) | 'none'.

    추가 검증:
        - query_supernode=True 강제 (super class 옵션 그대로 활용).
        - GAT model 측 supernode_topk / supernode_threshold_mode 와 별개로 selector 가 직접 mask 산출.
    """

    def __init__(
        self,
        weight_path: str,
        threshold_mode: str = "percentile",
        threshold_value: float = 80.0,
        score_normalization: str = "minmax",
        **kwargs,
    ):
        kwargs.setdefault("query_supernode", True)
        if not kwargs["query_supernode"]:
            raise ValueError("DirectedTopKSuperNodeSelector requires query_supernode=True")
        # 학습 ckpt 가 directed_from_sn 으로 학습되면 GAT 모델도 그 edge type 으로 instantiate
        # 해야 state_dict key 매치. inference 시에도 selector 가 attended_by_* edge 를 비워두므로
        # bidirectional ckpt 와도 호환되지만 default 는 directed_from_sn (학위 논문 설계 의도).
        kwargs.setdefault("supernode_edge_direction", "directed_from_sn")
        super().__init__(weight_path=weight_path, **kwargs)

        if threshold_mode not in SUPERNODE_THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be in {SUPERNODE_THRESHOLD_MODES}, got {threshold_mode}"
            )
        if score_normalization not in SUPERNODE_SCORE_NORMS:
            raise ValueError(
                f"score_normalization must be in {SUPERNODE_SCORE_NORMS}, got {score_normalization}"
            )
        self.threshold_mode = threshold_mode
        self.threshold_value = float(threshold_value)
        self.supernode_score_norm = score_normalization
        self.last_selected_count: Optional[int] = None
        logger.info(
            f"Initialized DirectedTopKSuperNodeSelector "
            f"(threshold_mode={threshold_mode}, threshold_value={threshold_value}, "
            f"score_normalization={score_normalization})"
        )

    # ──────────────────────────────────────────────────────────────
    # mask helper
    # ──────────────────────────────────────────────────────────────
    def _compute_threshold_mask(
        self,
        query_emb: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-query cosine score 산출 후 threshold_mode 별 mask 반환.

        gat_network[v2]._compute_supernode_mask 와 같은 정의 — 일관 유지.
        """
        q = query_emb if query_emb.dim() == 2 else query_emb.unsqueeze(0)
        if q.size(0) > 1:
            q = q.mean(dim=0, keepdim=True)
        q_norm = F.normalize(q, dim=-1)

        schema_types = ["table", "column", "fk_node"]
        scores_list = []
        type_tags: List[str] = []
        local_idx: List[int] = []
        for nt in schema_types:
            if nt not in x_dict or x_dict[nt].size(0) == 0:
                continue
            n_norm = F.normalize(x_dict[nt], dim=-1)
            sc = (n_norm @ q_norm.t()).squeeze(-1)
            scores_list.append(sc)
            type_tags.extend([nt] * sc.size(0))
            local_idx.extend(range(sc.size(0)))

        masks: Dict[str, torch.Tensor] = {}
        for nt in schema_types:
            if nt in x_dict:
                masks[nt] = torch.zeros(
                    x_dict[nt].size(0), dtype=torch.bool, device=x_dict[nt].device
                )
        if not scores_list:
            return masks

        combined = torch.cat(scores_list, dim=0)
        if self.supernode_score_norm == "minmax" and combined.numel() > 0:
            cmin, cmax = combined.min(), combined.max()
            combined_norm = (combined - cmin) / (cmax - cmin) if cmax > cmin else combined
        else:
            combined_norm = combined

        if self.threshold_mode == "top_k":
            k = min(int(self.threshold_value), combined_norm.size(0))
            keep_idx = torch.topk(combined_norm, k=k).indices.tolist()
        elif self.threshold_mode == "percentile":
            p = self.threshold_value / 100.0
            cutoff = torch.quantile(combined_norm.to(torch.float32), p)
            keep_idx = (combined_norm >= cutoff).nonzero(as_tuple=False).flatten().tolist()
        else:  # abs_tau
            cutoff = self.threshold_value
            keep_idx = (combined_norm >= cutoff).nonzero(as_tuple=False).flatten().tolist()

        for i in keep_idx:
            masks[type_tags[i]][local_idx[i]] = True
        return masks

    # ──────────────────────────────────────────────────────────────
    # _compute_gat_scores override — directed edge + threshold filter
    # ──────────────────────────────────────────────────────────────
    def _compute_gat_scores(
        self,
        question: str,
        graph_data: HeteroData,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """SuperNode 분기 변형: directed edge only + threshold-filtered schema.

        부모 EnsembleSelector._compute_gat_scores 와 비교:
          - bidirectional 에서 attended_by_* 역방향 edge 등록 X
          - 모든 schema 노드 → threshold 통과 노드만 attends_to_* edge 등록
          - 나머지 node embedding 추출 / projector 호출은 동일
        """
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
            self.last_resolved_depth = active_depth

            # --- SuperNode 주입 (directed) ---
            graph_data["query_node"].x = q_emb  # [1, in_dim]

            # threshold mask 산출 (raw schema feature 기준 — pre-GAT cosine)
            schema_x_dict = {
                nt: graph_data[nt].x
                for nt in ("table", "column", "fk_node")
                if nt in graph_data.node_types
            }
            mask = self._compute_threshold_mask(q_emb, schema_x_dict)
            total_selected = int(sum(int(m.sum().item()) for m in mask.values()))
            self.last_selected_count = total_selected

            for schema_nt in ("table", "column", "fk_node"):
                num_nodes = graph_data[schema_nt].num_nodes
                if num_nodes == 0:
                    graph_data[
                        "query_node", f"attends_to_{schema_nt}", schema_nt
                    ].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    continue

                m = mask.get(schema_nt)
                if m is None or not bool(m.any()):
                    # threshold 가 너무 강해서 이 type 에서 통과한 노드 0 개. 빈 edge_index 등록.
                    graph_data[
                        "query_node", f"attends_to_{schema_nt}", schema_nt
                    ].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    continue

                kept = m.nonzero(as_tuple=False).flatten().to(self.device)
                src = torch.zeros(kept.size(0), dtype=torch.long, device=self.device)
                dst = kept.to(torch.long)
                graph_data[
                    "query_node", f"attends_to_{schema_nt}", schema_nt
                ].edge_index = torch.stack([src, dst], dim=0)

                # 역방향 edge 가 v1/v2 model 측에서 등록된 edge type (bidirectional 기본값)일 경우,
                # 반드시 0-len edge_index 로 명시 등록해야 PyG 가 missing-key 로 에러 내지 않는다.
                rev_key = (schema_nt, f"attended_by_{schema_nt}", "query_node")
                if rev_key in graph_data.edge_types or self._model_has_edge(rev_key):
                    graph_data[rev_key].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long, device=self.device
                    )

            # --- v2 GAT 가 supernode_edge_direction='directed_from_sn' 일 경우 self_loop 가 필요.
            # 학습 ckpt 가 directed_from_sn 으로 학습됐으면 v2 forward 가 _inject_sn_self_loop 로
            # 자동 주입. v1 GAT 도 동일.
            node_embs_dict = self.gat_model(
                graph_data.x_dict,
                graph_data.edge_index_dict,
                query_emb=q_emb if self.query_conditioned else None,
                active_num_layers=active_depth,
            )

            num_nodes_total = len(metadata.get("node_metadata", {}))
            gat_scores = torch.zeros(num_nodes_total, device=self.device)

            num_t = graph_data["table"].num_nodes
            num_c = graph_data["column"].num_nodes

            if num_t > 0 and "table" in node_embs_dict:
                z_q_t, z_n_t = self.projector(q_emb, node_embs_dict["table"])
                logits_t = self.projector.compute_similarity(z_q_t, z_n_t)
                gat_scores[0:num_t] = torch.sigmoid(logits_t).view(-1)

            if num_c > 0 and "column" in node_embs_dict:
                z_q_c, z_n_c = self.projector(q_emb, node_embs_dict["column"])
                logits_c = self.projector.compute_similarity(z_q_c, z_n_c)
                gat_scores[num_t:num_t + num_c] = torch.sigmoid(logits_c).view(-1)

            if "fk_node" in node_embs_dict and node_embs_dict["fk_node"].size(0) > 0:
                num_fk = node_embs_dict["fk_node"].size(0)
                z_q_fk, z_n_fk = self.projector(q_emb, node_embs_dict["fk_node"])
                logits_fk = self.projector.compute_similarity(z_q_fk, z_n_fk)
                gat_scores[num_t + num_c:num_t + num_c + num_fk] = torch.sigmoid(logits_fk).view(-1)

        return gat_scores.cpu()

    def _model_has_edge(self, edge_type) -> bool:
        """GAT 모델이 해당 edge type 의 conv 를 등록해 두었는지 확인."""
        first_conv = self.gat_model.convs[0] if len(self.gat_model.convs) > 0 else None
        if first_conv is None:
            return False
        # HeteroConv stores convs in `convs` dict keyed by edge_type tuple
        try:
            return edge_type in first_conv.convs
        except (AttributeError, TypeError):
            return False
