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

def embed_line(edgelist:str, n_dim:int, # place the required parameters here
                   epochs:int=1000, device:str='auto', **kwargs):
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the grpah into a Networkx object
    Gx = load_graph(**kwargs)
    # Convert NetworkX graph to PyTorch Geometric data object
    data = from_networkx(Gx)

    # REST OF THE CODE HERE
    print("The Code is not implemented yet.")
    exit(1)
    
    # Get the embeddings
    embeddings = None # model().detach().cpu().numpy()
    # make the pandas dataframe from the node id and embeddings
    # Convert embeddings to a DataFrame
    node_labels = list(Gx.nodes())
    data = np.column_stack([node_labels, embeddings])
    embeddings_df = pd.DataFrame(data)
    return embeddings_df
