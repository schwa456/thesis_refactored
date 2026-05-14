"""GRASTFDFilter with Step 2 Graph Transformer reranking (Direction C-GT, Hoang 2025 Option β).

학술 Agent Phase 5 Q2 (planning/filter_proposal_by_scholar_agent_phase5_2026-05-14.md §3)
의 Option β 권고 구현:

  Step 1  (Skipped — h^0 재활용)
          현 anchor 의 LLM column scorer 출력 (XiYan ranking output 등) 을 h^0 로 사용.
          BIRD-Train fine-tune (Qwen3-Reranker-0.6B LoRA) 은 post-paper.

  Step 2  Graph Transformer Reranking (신규)
          Relation-aware GT (3 layers, hidden=1024, heads=8) — Hoang 2025 §3.3.
          edge types: R={fk, col→fk, col→pk} directed + reverse = 6 channels.
          h^0 → GT → per-column refined score.

  Step 3  Steiner tree restore (기존 GRASTFDFilter)
          terminal_source="graph_transformer" 신규 mode — GT score top-K + threshold filter.

  Step 4  FK/PK hardcode (기존)

학술 frame (학술 Agent Phase 5 §0 + Q4(c) 종합 권고): "Filter-Invariant 경계 확정 실험".
positive (R-P trade-off mitigation) / null (Filter-Invariance 경계 추가 evidence)
모두 학술적 가치. EX 개선 기대 낮음.
"""
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from modules.registry import register
from modules.filters.grast_fd_filter import (
    GRASTFDFilter,
    _VALID_STEINER_METHODS,
)
from modules.filters.grast_fd_transformer import (
    GraphTransformerEncoder,
    EDGE_TYPE_FK_FORWARD,
    EDGE_TYPE_FK_REVERSE,
    EDGE_TYPE_COL_TO_FK_FORWARD,
    EDGE_TYPE_COL_TO_FK_REVERSE,
    EDGE_TYPE_COL_TO_PK_FORWARD,
    EDGE_TYPE_COL_TO_PK_REVERSE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_TERMINAL_SOURCES_GT: Tuple[str, ...] = (
    "forward", "gat_topk", "prelim_sql", "graph_transformer",
)


@register("filter", "GRASTFDFilterWithTransformer")
class GRASTFDFilterWithTransformer(GRASTFDFilter):
    """GRAST-FD + Graph Transformer Step 2 add-on (Option β)."""

    def __init__(
        self,
        # GT-specific args
        transformer_checkpoint_path: Optional[str] = None,
        transformer_in_dim: int = 16,
        transformer_hidden_dim: int = 1024,
        transformer_num_layers: int = 3,
        transformer_num_heads: int = 8,
        transformer_dropout: float = 0.1,
        transformer_score_top_k: int = 10,
        transformer_score_threshold: Optional[float] = None,
        transformer_device: str = "cpu",
        # h^0 fallback policy
        h0_fallback: str = "anchor_scorer",  # "anchor_scorer" | "zeros"
        # GRAST-FD args 그대로 전달
        terminal_source: str = "graph_transformer",
        **kwargs,
    ):
        if terminal_source not in _VALID_TERMINAL_SOURCES_GT:
            raise ValueError(
                f"terminal_source='{terminal_source}' invalid. "
                f"Expected one of {_VALID_TERMINAL_SOURCES_GT}."
            )
        # GRASTFDFilter 의 terminal_source 검증을 우회하기 위해 임시로 "forward" 전달 후
        # 본 클래스에서 다시 set.
        kwargs.setdefault("steiner_method", "default")
        super().__init__(terminal_source="forward", **kwargs)
        self.terminal_source = terminal_source  # override after super().__init__()

        self.transformer_in_dim = int(transformer_in_dim)
        self.transformer_hidden_dim = int(transformer_hidden_dim)
        self.transformer_num_layers = int(transformer_num_layers)
        self.transformer_num_heads = int(transformer_num_heads)
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_score_top_k = int(transformer_score_top_k)
        self.transformer_score_threshold = (
            None if transformer_score_threshold is None
            else float(transformer_score_threshold)
        )
        self.transformer_device = transformer_device
        self.h0_fallback = h0_fallback
        self.transformer_checkpoint_path = transformer_checkpoint_path

        self.transformer: Optional[GraphTransformerEncoder] = None
        self._transformer_load_error: Optional[str] = None
        self._maybe_load_transformer()

        logger.info(
            "Initialized GRASTFDFilterWithTransformer "
            f"(terminal_source={self.terminal_source}, "
            f"hidden_dim={self.transformer_hidden_dim}, "
            f"num_layers={self.transformer_num_layers}, "
            f"num_heads={self.transformer_num_heads}, "
            f"score_top_k={self.transformer_score_top_k}, "
            f"score_threshold={self.transformer_score_threshold}, "
            f"device={self.transformer_device}, "
            f"checkpoint={'loaded' if self.transformer is not None else 'fallback'})"
        )

    # ------------------------------------------------------------------
    # Transformer load / init
    # ------------------------------------------------------------------
    def _maybe_load_transformer(self) -> None:
        """Load checkpoint if path exists, else leave self.transformer = None (fallback active)."""
        path = self.transformer_checkpoint_path
        if not path or not os.path.exists(path):
            if path:
                self._transformer_load_error = f"checkpoint not found: {path}"
                logger.warning(
                    f"[GRAST-FD-GT] checkpoint path '{path}' does not exist — "
                    "fallback active (terminal_source='forward')."
                )
            else:
                self._transformer_load_error = "no checkpoint_path provided"
            return
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                cfg = state.get("config", {}) or {}
                self.transformer_in_dim = int(cfg.get("in_dim", self.transformer_in_dim))
                self.transformer_hidden_dim = int(
                    cfg.get("hidden_dim", self.transformer_hidden_dim)
                )
                self.transformer_num_layers = int(
                    cfg.get("num_layers", self.transformer_num_layers)
                )
                self.transformer_num_heads = int(
                    cfg.get("num_heads", self.transformer_num_heads)
                )
                model = self._build_transformer()
                model.load_state_dict(state["state_dict"])
            else:
                model = self._build_transformer()
                model.load_state_dict(state)
            model.eval()
            self.transformer = model.to(self.transformer_device)
        except Exception as e:
            self._transformer_load_error = f"{type(e).__name__}: {e}"
            logger.warning(
                f"[GRAST-FD-GT] checkpoint load failed: {e} — fallback active."
            )
            self.transformer = None

    def _build_transformer(self) -> GraphTransformerEncoder:
        return GraphTransformerEncoder(
            in_dim=self.transformer_in_dim,
            hidden_dim=self.transformer_hidden_dim,
            num_layers=self.transformer_num_layers,
            num_heads=self.transformer_num_heads,
            dropout=self.transformer_dropout,
        )

    # ------------------------------------------------------------------
    # Graph encoding (col-only graph for GT — table nodes excluded so 모든 노드가 column)
    # ------------------------------------------------------------------
    def _encode_graph_for_transformer(
        self,
        full_schema: Dict[str, List[str]],
        metadata: Optional[Dict[str, Any]],
        fk_pk_columns: Set[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Build (column_nodes, edge_index[2,E], edge_type[E]) for GT.

        Node 들은 column 만. Edge types:
          - fk_to_id 의 src.col -> dst.col 양방향 (FK_FORWARD / FK_REVERSE)
          - col → FK column (source 가 FK column 인 경우): src 가 FK 면 forward,
            dst 가 FK 면 reverse — 단순화 위해 모든 edge endpoint 중 FK column 으로
            가는 path 만 별도 표시.
          - col → PK column: 마찬가지.
        belongs_to 는 학술 Agent §1.1 spec 대로 별도 채널 없음 — table membership 은
        node feature (input_proj 의 in_dim 에 인코딩) 에서 처리.
        """
        col_nodes = [
            f"{t}.{c}" for t, cols in full_schema.items() for c in (cols or [])
        ]
        node_idx = {n: i for i, n in enumerate(col_nodes)}

        fk_to_id = (metadata or {}).get("fk_to_id", {}) or {}
        inferred_fk = self.inferred_fk

        src_list: List[int] = []
        dst_list: List[int] = []
        et_list: List[int] = []

        def _add(s: str, d: str, et_fwd: int, et_rev: int):
            if s not in node_idx or d not in node_idx:
                return
            i, j = node_idx[s], node_idx[d]
            src_list.append(i); dst_list.append(j); et_list.append(et_fwd)
            src_list.append(j); dst_list.append(i); et_list.append(et_rev)

        # FK edges (declared + inferred)
        for fk_key in list(fk_to_id.keys()) + list(inferred_fk):
            parsed = self._parse_fk_key(fk_key)
            if parsed is None:
                continue
            s, d = parsed
            _add(s, d, EDGE_TYPE_FK_FORWARD, EDGE_TYPE_FK_REVERSE)

        # column → FK / column → PK: same-table column pairs where target endpoint is FK/PK
        fk_endpoints: Set[str] = set()
        for fk_key in list(fk_to_id.keys()) + list(inferred_fk):
            parsed = self._parse_fk_key(fk_key)
            if parsed is None:
                continue
            fk_endpoints.update(parsed)

        for tbl, cols in full_schema.items():
            full_cols = [f"{tbl}.{c}" for c in (cols or [])]
            for src in full_cols:
                for dst in full_cols:
                    if src == dst:
                        continue
                    if dst in fk_endpoints:
                        _add(src, dst,
                             EDGE_TYPE_COL_TO_FK_FORWARD, EDGE_TYPE_COL_TO_FK_REVERSE)
                    if dst in fk_pk_columns and dst not in fk_endpoints:
                        # PK 이지만 FK endpoint 아닌 경우만 col→PK channel 로 분리.
                        _add(src, dst,
                             EDGE_TYPE_COL_TO_PK_FORWARD, EDGE_TYPE_COL_TO_PK_REVERSE)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_type = torch.tensor(et_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)
        return col_nodes, edge_index, edge_type

    def _build_h0(
        self,
        col_nodes: List[str],
        s_fwd_set: Set[str],
        gat_scores: Optional[Dict[str, float]],
        fk_pk_columns: Set[str],
    ) -> torch.Tensor:
        """h^0 재활용: anchor LLM column scorer + GAT score + struct flag concatenation.

        학술 Agent Q2 권장: "현재 anchor LLM column scorer 출력 (XiYan 의 ranking output)
        을 h^0 로 사용 — Step 1 추가 학습 불필요". XiYan 은 binary selection 만 노출
        하므로 실용적으로 다음 feature 를 concat:

            [0] xiyan_selected      : float (1.0 if in S_fwd else 0.0)
            [1] gat_score           : float (gat_scores 의 column 값 or 0.0)
            [2] is_fk_pk            : float
            [3..N-1]                : zero-padding to in_dim
        """
        N = len(col_nodes)
        in_dim = self.transformer_in_dim
        if N == 0:
            return torch.zeros((0, in_dim))
        feat = torch.zeros((N, in_dim), dtype=torch.float32)
        for i, node in enumerate(col_nodes):
            feat[i, 0] = 1.0 if node in s_fwd_set else 0.0
            if gat_scores is not None:
                # 'table.col' 우선, fallback 'col' single
                v = gat_scores.get(node)
                if v is None and "." in node:
                    _, c = node.split(".", 1)
                    v = gat_scores.get(c)
                if v is not None:
                    try:
                        feat[i, 1] = float(v)
                    except Exception:
                        pass
            if node in fk_pk_columns:
                feat[i, 2] = 1.0
        return feat

    # ------------------------------------------------------------------
    # Override terminal resolution
    # ------------------------------------------------------------------
    def _resolve_terminals(
        self,
        query: str,
        s_fwd: List[str],
        full_schema: Dict[str, List[str]],
        db_id: Optional[str],
        gat_scores: Optional[Dict[str, float]],
        evidence: Optional[str],
        diag: Dict[str, Any],
    ) -> Set[str]:
        if self.terminal_source != "graph_transformer":
            return super()._resolve_terminals(
                query, s_fwd, full_schema, db_id, gat_scores, evidence, diag,
            )

        # transformer 미준비 시 자동 fallback to 'forward' (학술 Agent Q5 fallback plan)
        if self.transformer is None:
            logger.info(
                f"[GRAST-FD-GT] transformer unavailable "
                f"({self._transformer_load_error}) — fallback to 'forward'."
            )
            diag["terminal_fallback"] = "forward (transformer unavailable)"
            full_keys = {
                f"{t}.{c}" for t, cols in full_schema.items() for c in (cols or [])
            }
            return {n for n in s_fwd if isinstance(n, str) and "." in n and n in full_keys}

        # FK/PK columns 미리 추출 (h^0 + edge type 분리에 필요)
        fk_pk_columns = self._extract_fk_pk_columns(
            {t: cols for t, cols in full_schema.items()}, db_id,
            {"fk_to_id": (self.transformer is not None and
                          {} or {}), **(self._last_metadata_ref or {})},
        ) if False else self._extract_fk_pk_columns_for_full_schema(full_schema, db_id)

        col_nodes, edge_index, edge_type = self._encode_graph_for_transformer(
            full_schema, self._last_metadata_ref, fk_pk_columns,
        )
        h0 = self._build_h0(col_nodes, set(s_fwd), gat_scores, fk_pk_columns)

        device = self.transformer_device
        try:
            h0 = h0.to(device)
            edge_index = edge_index.to(device)
            edge_type = edge_type.to(device)
            with torch.no_grad():
                _, scores = self.transformer(h0, edge_index, edge_type)
            scores = scores.detach().cpu()
        except Exception as e:
            logger.warning(
                f"[GRAST-FD-GT] transformer forward failed: {e} — fallback to 'forward'."
            )
            diag["terminal_fallback"] = f"forward (gt forward error: {type(e).__name__})"
            full_keys = {n for n in col_nodes}
            return {n for n in s_fwd if isinstance(n, str) and "." in n and n in full_keys}

        # Terminal selection: top-K + optional threshold filter
        n = scores.numel()
        if n == 0:
            return set()
        # 점수 안정성 위해 sigmoid 통과
        relevance = torch.sigmoid(scores)
        diag["gt_score_min"] = float(relevance.min().item())
        diag["gt_score_max"] = float(relevance.max().item())
        diag["gt_score_mean"] = float(relevance.mean().item())

        if self.transformer_score_threshold is not None:
            mask = relevance >= self.transformer_score_threshold
            picked_idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        else:
            k = max(1, min(self.transformer_score_top_k, n))
            picked_idx = torch.topk(relevance, k=k).indices.tolist()
        terms = {col_nodes[i] for i in picked_idx}
        # forward 도 함께 terminal 로 — Steiner tree 가 connectivity 확보에 유리
        terms.update(n for n in s_fwd if isinstance(n, str) and "." in n)
        return terms

    def _extract_fk_pk_columns_for_full_schema(
        self,
        full_schema: Dict[str, List[str]],
        db_id: Optional[str],
    ) -> Set[str]:
        """full_schema 기준 FK/PK 추출 — _resolve_terminals 내부 edge type 분리용."""
        dummy_subgraph = {t: list(cols or []) for t, cols in full_schema.items()}
        return self._extract_fk_pk_columns(
            dummy_subgraph, db_id, self._last_metadata_ref,
        )

    # ------------------------------------------------------------------
    # Capture metadata on entry so _resolve_terminals can reach it
    # ------------------------------------------------------------------
    def refine(
        self, query: str, subgraph: Dict[str, List[str]], db_id: Optional[str] = None,
        tier2_pool: Optional[Any] = None,
        gat_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._last_metadata_ref = metadata or {}
        result = super().refine(
            query=query, subgraph=subgraph, db_id=db_id,
            tier2_pool=tier2_pool, gat_scores=gat_scores,
            metadata=metadata, evidence=evidence, **kwargs,
        )
        # transformer-specific diag 추가
        info = result.get("filter_info") or {}
        info["filter_transformer_available"] = self.transformer is not None
        info["filter_transformer_load_error"] = self._transformer_load_error
        info["filter_transformer_hidden_dim"] = self.transformer_hidden_dim
        info["filter_transformer_num_layers"] = self.transformer_num_layers
        info["filter_transformer_num_heads"] = self.transformer_num_heads
        info["filter_transformer_top_k"] = self.transformer_score_top_k
        info["filter_transformer_threshold"] = self.transformer_score_threshold
        result["filter_info"] = info
        return result
