import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
from collections import defaultdict
import torch.nn.functional as F

class DMGI(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)])
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.Tensor(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)

        return loss

class DMGI_FULL_EMBED(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata,hetero_info):
        super().__init__()
        node_types=metadata[0]
        num_mp=len(metadata[1])
        self.convs=[GCNConv(in_channels, out_channels) for _ in range(num_mp)] 
        
        self.hetero_info=hetero_info
        
        
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        
        self.Z = {n:torch.nn.Parameter(torch.Tensor(s, out_channels)) for n,s in hetero_info.items()}
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)

        self.M.bias.data.zero_()
        for k in self.Z:
            torch.nn.init.xavier_uniform_(self.Z[k])

    def forward(self, hetero_data:HeteroData):

        hetero_hs={key:defaultdict(list) for key in self.hetero_info.keys()}
        for conv,(mp,edge_index_dict) in zip(self.convs,hetero_data.edge_items()):
            
            node_type=mp[0]
            
            x=hetero_data.x_dict[node_type]
            edge_index=edge_index_dict['edge_index']

            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            hetero_hs[node_type]['pos_h'].append(pos_h)
            

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            hetero_hs[node_type]['neg_h'].append(neg_h)

            hetero_hs[node_type]['summary'].append(pos_h.mean(dim=0, keepdim=True))

        return hetero_hs

    def loss(self, hetero_hs):
        loss = 0.
        for node_type in hetero_hs:
            for pos_h, neg_h, s,in zip(hetero_hs[node_type]['pos_h'], hetero_hs[node_type]['neg_h'], hetero_hs[node_type]['summray']):
                s = s.expand_as(pos_h)
                loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
                loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

            pos_hs=hetero_hs[node_type]['pos_h']
            neg_hs=hetero_hs[node_type]['neg_h']
            pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
            neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

            pos_reg_loss = (self.Z[node_type] - pos_mean).pow(2).sum()
            neg_reg_loss = (self.Z[node_type] - neg_mean).pow(2).sum()
            loss += 0.001 * (pos_reg_loss - neg_reg_loss)

        return loss
    
    def embed(self):
        return torch.vstack(list(self.Z.values()))
