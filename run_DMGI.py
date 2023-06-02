# An implementation of "Unsupervised Attributed Multiplex Network
# Embedding" (DMGI) for unsupervised learning on  heterogeneous graphs:
# * Paper: <https://arxiv.org/abs/1911.06750> (AAAI 2020)

import os.path as osp
from models import DMGI
import torch
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB

path = osp.join(osp.dirname(osp.realpath(__file__)), './data/IMDB')
dataset = IMDB(path)

metapaths = [
    [('movie', 'actor'), ('actor', 'movie')],  # MAM
    [('movie', 'director'), ('director', 'movie')],  # MDM
]
data = T.AddMetaPaths(metapaths, drop_orig_edge_types=True)(dataset[0])



model = DMGI(data['movie'].num_nodes, data['movie'].x.size(-1),
             out_channels=64, num_relations=len(data.edge_types))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)


def train():
    model.train()
    optimizer.zero_grad()
    x = data['movie'].x
    edge_indices = data.edge_index_dict.values()
    pos_hs, neg_hs, summaries = model(x, edge_indices)
    loss = model.loss(pos_hs, neg_hs, summaries)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    train_emb = model.Z[data['movie'].train_mask].cpu()
    val_emb = model.Z[data['movie'].val_mask].cpu()
    test_emb = model.Z[data['movie'].test_mask].cpu()

    train_y = data['movie'].y[data['movie'].train_mask].cpu()
    val_y = data['movie'].y[data['movie'].val_mask].cpu()
    test_y = data['movie'].y[data['movie'].test_mask].cpu()

    clf = LogisticRegression().fit(train_emb, train_y)
    return clf.score(val_emb, val_y), clf.score(test_emb, test_y)


for epoch in range(1, 1001):
    loss = train()
    if epoch % 50 == 0:
        val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')