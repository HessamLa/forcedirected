# %%
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric as pyg
from torch_geometric.utils import to_networkx, to_networkit, from_networkit
from torch_geometric.datasets import SNAPDataset

import os
import numpy as np
import networkx as nx
import networkit as nk

# %%
def convert_networkit_to_pyg(graph):
    # Extract graph properties
    num_nodes = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    edge_list = [(edge[0], edge[1]) for edge in graph.iterEdges()]

    # Create PyTorch Geometric Data object
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.zeros(num_nodes, 1)  # Assign feature vectors if available, otherwise use zeros
    y = None  # Assign class labels if available, otherwise use None
    data = Data(x=x, edge_index=edge_index, y=y)

    return data

def load_graph_networkx(datasetname="cora", rootpath='./datasets', node_attrs=["x"]):
    rootpath = os.path.abspath(rootpath)
    print(rootpath)
    undirected=True
    if(datasetname == "cora"):
        undirected=True
        dataset = Planetoid(root=rootpath, name='Cora')
        data = dataset[0]
    elif(datasetname == "ego-Facebook" or datasetname == "ego-facebook"):
        undirected=True
        # dataset = SNAPDataset(root=rootpath, name='ego-Facebook')
        filepath=f"{rootpath}/snap/ego-facebook/facebook_combined.txt"
        G = nk.readGraph(filepath, nk.graphio.Format.EdgeListSpaceZero)
        # dataset = pyg.utils.from_networkit(G)
        data = convert_networkit_to_pyg(G)
    else:
        print(f"Dataset {datasetname} is not recognized")
        raise
        undirected=True
        print("loading dataset:", datasetname)
        dataset = SNAPDataset(root=rootpath, name=datasetname)
    
    # print(type(dataset), len(dataset))
    # print(len(dataset[0][0]))

    # data = dataset[0]
    # print(f'Dataset: {dataset}:')
    # print('======================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    Gx = to_networkx(data, to_undirected=undirected, node_attrs=["x"])
    print(Gx.number_of_nodes())
    
    return (Gx, data)

# Gx, data = load_graph_networkx('ego-Facebook', rootpath='../../datasets')
# Gx, data = load_graph_networkx('cora', rootpath='../../datasets')

"""receives a networkx graph and returns (G, A, degrees, hops) as the following:
G: a graph in networkit format
A: adjacency matrix in numpy format
degrees: nodes degrees in numpy format
hops: hops distance matrix in numpy format


"""
def process_graph_networkx(Gx):
    print("\nProcess Graph")
    print('======================')
    # ccgenerator = nx.connected_components(Gx)
    # cclist = [c for c in ccgenerator]
    # print("\nnumber of components ", len(cclist))

    # get the adjacency matrix
    A = nx.to_numpy_array(Gx)

    # Gt = to_networkit(data, directed=False) # this doesn't work
    # networkit is better that networkx for finding hops count
    G = nk.nxadapter.nx2nk(Gx)
    print("number of nodes and edges:", G.numberOfNodes(), G.numberOfEdges())
    cc = nk.components.StronglyConnectedComponents(G)
    cc.run()
    print("number of components ", cc.numberOfComponents())

    # edge_index = data.edge_index.numpy()
    # # print(edge_index.shape)
    # edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
    
    # G = nk.Graph(data.num_nodes, directed=False)
    # # print(edge_index.shape[1])
    # for i in range(edge_index.shape[1]):
    #     e = edge_index[:,i].T
    #     G.addEdge(e[0], e[1], checkMultiEdge=True)
    
    # print("number of nodes and edges:", G.numberOfNodes(), G.numberOfEdges())
    # cc = nk.components.ConnectedComponents(G)
    # cc.run()
    # print("number of components ", cc.numberOfComponents())
    
    n = G.numberOfNodes()
    # node degrees
    degrees = np.array([G.degree(u) for u in G.iterNodes()])

    # get distance between all pairs. How about this one
    asps = nk.distance.APSP(G)
    asps.run()
    # all pair hops distance
    hops = asps.getDistances(asarray=True)
    # print(hops.shape) # nxn matrix
    return (G, A, degrees, hops)


import random
def remove_random_edges(G, ratio):
    G = nk.nxadapter.nx2nk(G)
    num_edges = G.numberOfEdges()
    num_edges_to_remove = int(num_edges * ratio)

    edges = list(G.edges())
    random.shuffle(edges)
    edges_to_remove = []

    for edge in edges:
        G_removed = G.copy()
        G_removed.removeEdge(edge[0], edge[1])
        if nk.components.ConnectedComponents(G_removed).numberOfComponents() == nk.components.ConnectedComponents(G).numberOfComponents():
            G = G_removed
            edges_to_remove.append(edge)
        if len(edges_to_remove) >= num_edges_to_remove:
            break

    return G, edges_to_remove

# Gx,  excluded_edges = remove_random_edges(Gx, 0.1)


# if(__name__=='__main__'):
#     Gx, data = load_graph_networkx('Facebook', rootpath='../../datasets')
#     G, A, degrees, hops = process_graph_networkx(Gx)
#     n = A.shape[0]
#     # find max hops
#     maxhops = max(hops[hops<n+1]) # hops<n+1 to exclude infinity values
#     hops[hops>maxhops]=maxhops+1
#     print("max hops:", maxhops)

