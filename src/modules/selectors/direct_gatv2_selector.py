"""
DirectGATv2Selector — SchemaHeteroGATv2 + DirectClassifierHead 추론용 Selector.

기존 DirectGATSelector(models.gat_network.SchemaHeteroGAT v1 기반)와 동일한
Direct(BCE) 추론 경로를 유지하되, backbone 을 SchemaHeteroGATv2 (v2) 로 교체한다.
v2 는 PairNorm / Initial-Residual / Jumping-Knowledge / Edge-Type 분리(V6-W2) 등
oversmoothing mitigation axis 를 model-level flag 로 노출하므로, 이 selector 는
해당 flag 전체를 생성자에서 받아 그대로 전달한다.

핵심: v6w2 (그리고 v6w1) 학습 ckpt 는 `train_gat_s06.py` 가 `config['model']`
블록을 통째로 함께 저장한다. `auto_config_from_ckpt=True` (기본) 면 그 블록을 읽어
모델 구조 파라미터를 자동 복원하므로, 4 cells (p2_standalone / p2_phase1 /
p2_standalone_no_selfloop / p2_sum) 가 동일 selector class + 다른 ckpt 만으로
정확히 호환된다. config 에 없는 항목은 생성자 인자(또는 v2 기본값)로 fallback.

ckpt 형식 (train_gat_s06.py 사양):
    ckpt['gat_state_dict']         168 keys → SchemaHeteroGATv2.state_dict()
    ckpt['classifier_state_dict']   18 keys → DirectClassifierHead × {table,column,fk_node}
    ckpt['config']['model']                 → 모델 구조 파라미터 전체 (auto-config 원천)
"""
import torch
from typing import List, Dict, Any, Optional
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.base import BaseSelector
from models.gat_network_v2 import SchemaHeteroGATv2
from models.direct_classifier import DirectClassifierHead
from modules.encoders.local_encoder import LocalPLMEncoder
from modules.encoders.token_encoder import TokenEncoder
from utils.logger import get_logger

logger = get_logger(__name__)

# SchemaHeteroGATv2 생성자가 받는 구조 파라미터 화이트리스트.
# ckpt config['model'] 또는 selector kwargs 에서 이 키들만 추려 모델로 전달한다.
# (capture_layerwise_outputs / diameter_path 는 추론 경로에서 강제 제어 — 아래 참고)
_V2_MODEL_KEYS = (
    "in_channels", "hidden_channels", "out_channels", "num_layers", "heads",
    "query_conditioned", "query_supernode",
    "pairnorm_mode", "pairnorm_scale", "initial_residual_alpha", "jumping_knowledge",
    "dual_stream",
    "num_layers_mode", "num_layers_fallback", "diameter_dict",
    "supernode_edge_direction", "supernode_topk", "supernode_topk_criterion",
    "supernode_threshold_mode", "supernode_threshold_value", "supernode_score_normalization",
    "drop_message_p", "use_layernorm_pre_softmax", "aggregation_type",
    "gat_layer_type", "softplus_symmetric_norm",
    "gcnii_beta_lambda", "aero_hop_attention", "aero_cumulative_attention", "aero_cumulative_decay",
    "v6w1_pairnorm_scale", "v6w1_jk_mode",
    "edge_type_split", "edge_type_split_self_loops", "edge_type_split_aggr",
    "edge_type_split_key_relations",
    "v6w3_variant", "v6w5_variant", "v6w6_variant",
)


@register("selector", "DirectGATv2Selector")
class DirectGATv2Selector(BaseSelector):
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
        auto_config_from_ckpt: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.apply_threshold = kwargs.get('apply_threshold', False)
        self.encoder_type = encoder_type

        # ── 1) ckpt 선로딩 (config 자동 복원을 위해 모델 생성 전에 읽는다) ──
        logger.info(f"Loading Direct GATv2 weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        if 'gat_state_dict' not in checkpoint or 'classifier_state_dict' not in checkpoint:
            raise RuntimeError(
                "Invalid checkpoint format. Expected 'gat_state_dict' and 'classifier_state_dict'. "
                "Train with src/train_gat_s06.py (SchemaHeteroGATv2 + DirectClassifierHead)."
            )
        ckpt_mcfg: Dict[str, Any] = {}
        if auto_config_from_ckpt:
            ckpt_mcfg = dict(checkpoint.get('config', {}).get('model', {}))
            if ckpt_mcfg:
                logger.info(
                    f"[DirectGATv2] auto-config from ckpt: "
                    f"QC={ckpt_mcfg.get('query_conditioned')} SN={ckpt_mcfg.get('query_supernode')} "
                    f"split={ckpt_mcfg.get('edge_type_split')} "
                    f"self_loops={ckpt_mcfg.get('edge_type_split_self_loops')} "
                    f"aggr={ckpt_mcfg.get('edge_type_split_aggr')} "
                    f"PN={ckpt_mcfg.get('pairnorm_mode')} IR={ckpt_mcfg.get('initial_residual_alpha')} "
                    f"JK={ckpt_mcfg.get('jumping_knowledge')} "
                    f"v6w3={ckpt_mcfg.get('v6w3_variant')} "
                    f"v6w5={ckpt_mcfg.get('v6w5_variant')} "
                    f"v6w6={ckpt_mcfg.get('v6w6_variant')} "
                    f"SN={ckpt_mcfg.get('query_supernode')}/{ckpt_mcfg.get('supernode_edge_direction')}"
                )
            else:
                logger.warning(
                    "[DirectGATv2] auto_config_from_ckpt=True 이나 ckpt 에 config['model'] 없음 — "
                    "생성자 인자/기본값으로 모델 구성."
                )

        # ── 2) 모델 구조 kwargs 병합: ckpt config < 명시 kwargs (호출자가 강제 override 가능) ──
        explicit = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "query_conditioned": query_conditioned,
            "query_supernode": query_supernode,
        }
        # 명시 kwargs 중 v2 화이트리스트에 속하는 것만 추가 수집
        for k in _V2_MODEL_KEYS:
            if k in kwargs:
                explicit[k] = kwargs[k]

        model_kwargs: Dict[str, Any] = {}
        for k in _V2_MODEL_KEYS:
            if k in ckpt_mcfg:
                model_kwargs[k] = ckpt_mcfg[k]
        # 명시 인자가 ckpt 값을 덮어쓴다. 단, query_conditioned/query_supernode 의 기본 False 가
        # ckpt 의 True 를 잘못 덮지 않도록: 명시 인자는 'kwargs 로 직접 전달됐거나 ckpt 에 키가 없을 때'만 적용.
        for k, v in explicit.items():
            if k in kwargs or k not in model_kwargs:
                model_kwargs[k] = v

        # core dim 은 항상 존재하도록 보강
        model_kwargs.setdefault("in_channels", in_channels)
        model_kwargs.setdefault("hidden_channels", hidden_channels)
        model_kwargs.setdefault("out_channels", out_channels)

        # 추론 경로 강제 제어: analyzer 용 layer capture 는 추론 중 불필요(버퍼 누적 방지) → 항상 OFF.
        # diameter_path 는 머신 의존 경로일 수 있어 전달하지 않는다(diameter_dict 만 허용).
        model_kwargs["capture_layerwise_outputs"] = False

        self.query_conditioned = bool(model_kwargs.get("query_conditioned", False))
        self.query_supernode = bool(model_kwargs.get("query_supernode", False))

        # ── 3) 모델 생성 ──
        self.gat_model = SchemaHeteroGATv2(**model_kwargs).to(self.device)

        # Direct classifier heads (node type 별) — out_channels 기준
        resolved_out = int(model_kwargs.get("out_channels", out_channels))
        self.classifier_types = ['table', 'column', 'fk_node']
        self.classifier_heads = torch.nn.ModuleDict({
            nt: DirectClassifierHead(
                in_dim=resolved_out,
                hidden_dim=classifier_hidden,
                dropout=0.0,  # 추론 시 dropout off
            ).to(self.device)
            for nt in self.classifier_types
        })

        # ── 4) Encoder ──
        if self.encoder_type == "plm":
            self.encoder = LocalPLMEncoder()
            logger.info("DirectGATv2Selector: using LocalPLMEncoder (sentence-level)")
        else:
            self.encoder = TokenEncoder()
            logger.info("DirectGATv2Selector: using TokenEncoder (token-level)")

        # ── 5) Lazy GATv2Conv weights: dummy forward 로 초기화 후 load ──
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
        """GATv2Conv lazy parameter 를 dummy forward 로 초기화.

        모델의 `node_types` / `all_edge_types` 를 직접 introspect 하여 dummy graph 를 구성 —
        V6-W2 (edge_type_split: per-relation conv + self-loop) / V6-W3 (table_summary / local_vn
        신규 node·edge type) 변형을 하드코딩 없이 자동 포괄. 모든 node type 에 1개 노드, 모든
        edge type 에 1개 edge (src=dst=0) 를 두어 각 관계별 lazy conv 가 파라미터를 생성하도록 함
        (이 초기화가 누락되면 신규 relation conv 의 state_dict 키가 비어 load_state_dict 실패)."""
        model = self.gat_model
        lin = model.lin_dict
        # query_conditioned 면 lin_dict in_channels 가 2x (query concat). 원래 노드 feature dim 복원.
        # effective_in 은 모든 node type 공통이므로 base_in 단일값으로 통일.
        base_in = lin['table'].in_channels // (2 if self.query_conditioned else 1)

        dummy = HeteroData()
        for nt in model.node_types:
            dummy[nt].x = torch.zeros(1, base_in, device=self.device)

        zero = torch.zeros((2, 1), dtype=torch.long, device=self.device)
        for et in model.all_edge_types:
            dummy[et].edge_index = zero

        with torch.no_grad():
            if self.query_conditioned:
                dummy_q = torch.zeros(1, base_in, device=self.device)
                _ = model(dummy.x_dict, dummy.edge_index_dict, query_emb=dummy_q)
            else:
                _ = model(dummy.x_dict, dummy.edge_index_dict)

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
        # SGBE Phase 2 (DECISIONS 2026-05-12) — filter 단이 raw GAT score 기반 gating 가능하도록 노출.
        # Direct variant 는 classifier head sigmoid output 자체가 raw GAT score. cosine 분기 없음.
        self.latest_raw_gat_scores: Dict[int, float] = {
            int(candidates[i]): float(self.latest_scores[i])
            for i in range(min(len(candidates), len(self.latest_scores)))
        }
        self.latest_raw_cos_scores: Dict[int, float] = {}

        if self.apply_threshold:
            selected = [c for c, s in zip(candidates, self.latest_scores) if s >= self.threshold]
            logger.info(
                f"[DirectGATv2] apply_threshold={self.threshold}: "
                f"{len(selected)}/{len(candidates)} nodes selected"
            )
            return selected if selected else candidates[:1]

        return candidates
