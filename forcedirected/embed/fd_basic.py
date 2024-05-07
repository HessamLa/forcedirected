import os
import importlib
import networkx as nx
import torch

from ..utilities import load_graph

# print("TESTING IMPORT METHOD 1")
# from ..models.ForceDirected import ForceDirected
# print("  success 1")

# print("TESTING IMPORT METHOD 2")
# ForceDirectedModule = importlib.import_module('..models.ForceDirected', package='forcedirected.embed')
# ForceDirected = getattr(ForceDirectedModule, 'ForceDirected')
# print("  success 2")


__all__ = [
    "embed_basic",
]

def embed_basic(n_dim:int, epochs:int=1000, device:str='auto', **kwargs):
    
    # load the graph
    Gx = load_graph(edgelist=kwargs['edgelist'], adjlist=kwargs['adjlist'], nodelist=kwargs['nodelist'])
    if(Gx.number_of_nodes()==0):
        print("Graph is empty")
        exit(1)
    print("Number of nodes:", Gx.number_of_nodes())
    print("Number of edges:", Gx.number_of_edges())
    print("Number of connected components:", nx.number_connected_components(Gx))    

    # create the embedding function
    print("Creating the forcedirected object")
    from ..models.model_201_basic import FDModel
    fdobj = FDModel(Gx, n_dim, **kwargs)
    print(fdobj)
    
    # set the device
    if(device=='auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make the embeddings
    fdobj.embed(epochs=epochs, device=device)
    embeddings_df = fdobj.get_embeddings_df()
    return embeddings_df # End of embed(.)
