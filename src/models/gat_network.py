import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData

class SchemaHeteroGAT(nn.Module):
    """
    이종 그래프(Heterogeneous Graph) 구조의 DB 스키마를 학습하는 모델.
    Table, Column, FK_Node가 서로 Attention 기반의 메시지 패싱을 수행하여,
    텍스트 의미(Semantic)와 구조적 위치(Structure)가 융합된 임베딩을 출력합니다.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, heads: int = 4, query_conditioned: bool = False,
                 query_supernode: bool = False):
        super(SchemaHeteroGAT, self).__init__()
        self.num_layers = num_layers
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode

        # query_conditioned=True 시 node feature에 query를 concat → 입력 차원 2배
        effective_in = in_channels * 2 if query_conditioned else in_channels

        # 1. 초기 차원 축소 및 통일 (PLM의 384 차원 -> GAT의 256 차원 등)
        node_types = ['table', 'column', 'fk_node']
        if query_supernode:
            node_types.append('query_node')

        self.lin_dict = nn.ModuleDict({
            nt: Linear(effective_in, hidden_channels) for nt in node_types
        })

        # 2. Heterogeneous GAT 레이어 정의
        base_edge_types = {
            # A. Table <-> Column 상호작용
            ('table', 'has_column', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
            ('column', 'belongs_to', 'table'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),

            # B. Column <-> FK_Node 상호작용 (논문의 핵심 기여 포인트)
            ('column', 'is_source_of', 'fk_node'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
            ('fk_node', 'points_to', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),

            # C. Table <-> Table 거시적 상호작용
            ('table', 'table_to_table', 'table'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
        }

        # Query Super Node: 모든 schema 노드와 양방향 edge
        supernode_edge_types = {}
        if query_supernode:
            for schema_nt in ['table', 'column', 'fk_node']:
                supernode_edge_types[('query_node', f'attends_to_{schema_nt}', schema_nt)] = \
                    GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
                supernode_edge_types[(schema_nt, f'attended_by_{schema_nt}', 'query_node')] = \
                    GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            all_edge_types = {**base_edge_types, **supernode_edge_types}
            # 각 layer마다 새로운 GATv2Conv 인스턴스 생성
            conv_dict = {}
            for edge_type, _ in all_edge_types.items():
                conv_dict[edge_type] = GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)

        # 3. 최종 출력 차원 변환 (Alignment Layer와 연결될 차원)
        self.out_lin_dict = nn.ModuleDict({
            nt: Linear(hidden_channels * heads, out_channels) for nt in node_types
        })

        self.skip_dict = nn.ModuleDict({
            nt: Linear(effective_in, out_channels) for nt in node_types
        })

    def forward(self, x_dict: dict, edge_index_dict: dict,
                query_emb: torch.Tensor = None) -> dict:
        """
        x_dict: {'table': Tensor, 'column': Tensor, 'fk_node': Tensor}
        edge_index_dict: {('table', 'has_column', 'column'): Tensor, ...}
        query_emb: Optional[Tensor] — [1, dim] or [dim]. query_conditioned=True 시 사용.
        """
        # Query Conditioning: query_emb를 모든 노드 feature에 concat
        if self.query_conditioned and query_emb is not None:
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
            augmented = {}
            for nt, x in x_dict.items():
                q_exp = query_emb.expand(x.size(0), -1)
                augmented[nt] = torch.cat([x, q_exp], dim=-1)
            x_dict = augmented

        # Step 1: Input Projection (Linear 통과 및 Activation)
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = F.leaky_relu(self.lin_dict[node_type](x))

        # Step 2: Message Passing (GAT Layers)
        for i in range(self.num_layers):
            out_dict = self.convs[i](out_dict, edge_index_dict)
            
            # 레이어 사이에 비선형 활성화 함수 적용
            out_dict = {node_type: F.elu(x) for node_type, x in out_dict.items()}

        # Step 3: Output Projection
        final_dict = {}
        for node_type, x in out_dict.items():
            final_dict[node_type] = self.out_lin_dict[node_type](x) + self.skip_dict[node_type](x_dict[node_type])

        # 최종 반환 형태: 각 노드 타입별 업데이트된 임베딩 텐서
        return final_dict