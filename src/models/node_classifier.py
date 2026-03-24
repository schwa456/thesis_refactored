import torch
import torch.nn as nn
from .gat_network import SchemaHeteroGAT
from .plm_encoder import PLMEncoder

class SimpleNodeLinker(nn.Module):
    def __init__(self, hidden_dim, plm_name: str="", in_channels: int=384):
        super().__init__()

        #1. Encoders
        self.plm_encoder = PLMEncoder(plm_name)
        self.gat = SchemaHeteroGAT(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=hidden_dim)

        self.table_classifier = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.column_classifier = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_data, queries):
        # 1. NLQ Encoding
        if isinstance(queries, list) and isinstance(queries[0], str):
            q_emb = self.plm_encoder(queries)
        else:
            q_emb = queries # [Batch, 384]
            
        # 2. GAT Node Encoding
        # batch 객체를 직접 넘기는 경우와 dict를 넘기는 경우 호환성 유지
        x_dict = graph_data.x_dict if hasattr(graph_data, 'x_dict') else graph_data
        edge_index_dict = graph_data.edge_index_dict if hasattr(graph_data, 'edge_index_dict') else None
        node_feats = self.gat(x_dict, edge_index_dict) # [Total_Nodes, 256]

        # 3. Feature Fusion & Classification
        out_probs = {}
        
        # Table 처리
        if 'table' in node_feats:
            t_feats = node_feats['table']
            q_emb_t = queries[graph_data['table'].batch]
            out_probs['table'] = self.table_classifier(torch.cat([t_feats, q_emb_t], -1)).squeeze(-1)
            
        # Column 처리
        if 'column' in node_feats:
            c_feats = node_feats['column']
            q_emb_c = queries[graph_data['column'].batch]
            out_probs['column'] = self.column_classifier(torch.cat([c_feats, q_emb_c], -1)).squeeze(-1)
            
        return out_probs