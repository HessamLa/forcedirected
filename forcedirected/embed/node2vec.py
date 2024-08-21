import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

import numpy as np
import pandas as pd
import networkx as nx
from ..utilities import load_graph

def embed_node2vec(edgelist:str, n_dim:int, p=1.0, q=0.5, walk_length=100, context_size=20, walks_per_node=20, num_negative_samples=1, 
                   epochs:int=1000, device:str='auto', **kwargs):
    """
    Returns a Pandas dataframe with node embeddings generated using the Node2Vec algorithm.
    Columns: ['id', 'dim_1', 'dim_2', ..., 'dim_n']
    Example usage might include specifying the path to the graph and dimensions for embeddings
    embeddings_df = embed(n_dim=128, file_path='path_to_your_graph_file.graphml')
    print(embeddings_df.head())
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the grpah into a Networkx object
    Gx = load_graph(edgelist, **kwargs)

    # Convert NetworkX graph to PyTorch Geometric data object
    data = from_networkx(Gx)

    model = Node2Vec(
        data.edge_index,
        embedding_dim=n_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=True,
    ).to(device)

    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, epochs + 1):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    embeddings = model().detach().cpu().numpy()
    # make the pandas dataframe from the node id and embeddings
    

    # Convert embeddings to a DataFrame
    node_labels = list(Gx.nodes())
    data = np.column_stack([node_labels, embeddings])
    embeddings_df = pd.DataFrame(data)
    return embeddings_df

    # @torch.no_grad()
    # def plot_points(colors):
    # model.eval()
    # z = model().cpu().numpy()
    # z = TSNE(n_components=2).fit_transform(z)
    # y = data.y.cpu().numpy()
    # @torch.no_grad()
    # def plot_points(colors):
    # model.eval()
    # z = model().cpu().numpy()
    # z = TSNE(n_components=2).fit_transform(z)
    # y = data.y.cpu().numpy()