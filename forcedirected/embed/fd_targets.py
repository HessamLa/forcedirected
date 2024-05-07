import os
import importlib
import networkx as nx
import torch

from ..utilities import load_graph

__all__ = [
    "embed",
]

def embed_targets(n_dim:int, model:str, epochs:int=1000, device:str='auto', **kwargs):
    # load the forcedirected model
    if(isinstance(model, str)):
        importpath = model
        # print(f"Importing model from '{importpath}'")
        try:
            model_module = importlib.import_module(importpath)
        except ImportError:
            # print(f"Model not found at {importpath}")
            importpath_2='forcedirected.models.'+importpath
            # print(f"Trying '{importpath_2}'")
            try:
                model_module = importlib.import_module(importpath_2)
            except ImportError:
                raise ImportError(f"Model not found at {importpath} or {importpath_2}")

    fdmodel = getattr(model_module, 'FDModel')

    # load the graph
    Gx = load_graph(**kwargs)
    if(Gx.number_of_nodes()==0):
        print("Graph is empty")
        exit(1)
    print("Number of nodes:", Gx.number_of_nodes())
    print("Number of edges:", Gx.number_of_edges())
    print("Number of connected components:", nx.number_connected_components(Gx))    

    # create the embedding function
    print("Creating the forcedirected object")
    fdobj = fdmodel(Gx, n_dim, **kwargs)
    print(fdobj)
    
    # set the device
    if(device=='auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make the embeddings
    fdobj.embed(epochs=epochs, device=device)
    embeddings_df = fdobj.get_embeddings_df()
    return embeddings_df # End of embed(.)
