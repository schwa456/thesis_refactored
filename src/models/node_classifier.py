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
            encoded = self.plm_encoder(queries)
            q_emb = encoded[0] if isinstance(encoded, tuple) else encoded

            # 3차원(Batch, Seq_Len, Dim)인 경우 Mean Pooling으로 2차원(Batch, Dim)으로 축소
            if q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)
        else:
            q_emb = queries # [Batch, 384]
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            elif q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)
            
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
            if hasattr(graph_data['table'], 'batch'):
                t_batch = graph_data['table'].batch
            else:
                t_batch = torch.zeros(t_feats.size(0), dtype=torch.long, device=t_feats.device)

            q_emb_t = q_emb[t_batch]
            out_probs['table'] = self.table_classifier(torch.cat([t_feats, q_emb_t], -1)).squeeze(-1)
            
        # Column 처리
        if 'column' in node_feats:
            c_feats = node_feats['column']
            if hasattr(graph_data['column'], 'batch'):
                c_batch = graph_data['column'].batch
            else:
                c_batch = torch.zeros(c_feats.size(0), dtype=torch.long, device=c_feats.device)
            
            q_emb_c = q_emb[c_batch]
            out_probs['column'] = self.column_classifier(torch.cat([c_feats, q_emb_c], -1)).squeeze(-1)
            
        return out_probs