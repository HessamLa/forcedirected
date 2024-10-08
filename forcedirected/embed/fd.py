import os
from types import ModuleType
from typing import Any, Union

import importlib
import networkx as nx
import torch 
from forcedirected.utilities import load_graph

# print("TESTING IMPORT METHOD 1")
# from ..models.ForceDirected import ForceDirected
# print("  success 1")

# print("TESTING IMPORT METHOD 2")
# ForceDirectedModule = importlib.import_module('..models.ForceDirected', package='forcedirected.embed')
# ForceDirected = getattr(ForceDirectedModule, 'ForceDirected')
# print("  success 2")


__all__ = [
    "embed",
    "embed_basic",
    "embed_shell",
    "embed_targets",
]

def embed(edgelist:str, n_dim:int, model_module:Union[str, ModuleType], epochs:int=1000, device:str='auto', **kwargs):
    """Function for graph embedding using a forcedirected method. The model module is found in 'models' directory and is specified by 'model' parameter."""
    # load the forcedirected model
    if(isinstance(model_module, str)):
        importpath = model_module
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
    print("==+==  Module file:", model_module.__file__)
    print("==+==  Model class:", fdmodel)
    print("")

    # load the graph
    Gx = load_graph(edgelist, **kwargs)
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

def embed_basic(edgelist:str, n_dim:int, epochs:int=1000, device:str='auto', **kwargs):
    """Function for graph embedding using forcedirected basic method."""
    from forcedirected.models import model_201_basic as model_module
    return embed(edgelist, n_dim, model_module, epochs, device, **kwargs)

def embed_shell(edgelist:str, n_dim:int, epochs:int=1000, device:str='auto', **kwargs):
    """Function for graph embedding using forcedirected with shell averaging method."""
    from forcedirected.models import model_204_shell as model_module
    return embed(edgelist, n_dim, model_module, epochs, device, **kwargs)


def embed_targets(edgelist:str, n_dim:int, epochs:int=1000, device:str='auto', **kwargs):
    """Function for graph embedding using forcedirected with shell averaging and selected targets method."""
    from forcedirected.models import model_214_targets as model_module
    return embed(edgelist, n_dim, model_module, epochs, device, **kwargs)
