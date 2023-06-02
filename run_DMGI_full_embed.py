import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB
import os.path as osp
from models import DMGI_FULL_EMBED
import torch
from sklearn.linear_model import LogisticRegression

path = osp.join(osp.dirname(osp.realpath(__file__)), './data/IMDB')
dataset = IMDB(path)
FEAT_SIZE,EMBED_SIZE=2

metapaths = [
    [('movie', 'actor'), ('actor', 'movie')],  # MAM
    [('movie', 'director'), ('director', 'movie')],  # MDM
    [('actor', 'movie'), ('movie', 'actor')],  # AMA
    [('director' ,'movie'), ('movie', 'director')],  # DMD
    ]
num_rel=len(metapaths)
data = T.AddMetaPaths(metapaths, drop_orig_edge_types=True)(dataset[0])

hetero_info={}
for node_type,x_dict in data.node_items():
    hetero_info[node_type]=len(x_dict['x'])
    
model=DMGI_FULL_EMBED(data['movie'].x.size(-1),64,data.metadata(),hetero_info)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.005)



def train():
    model.train()
    optimizer.zero_grad()
    hetero_h = model(data)
    loss = model.loss(hetero_h)
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