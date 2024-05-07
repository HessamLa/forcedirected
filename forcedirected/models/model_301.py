
# %%         
from typing import Any

import numpy as np
import pandas as pd

from forcedirected.utilities.graphtools import get_hops

from forcedirected.algorithms import get_alpha_hops, pairwise_difference
from forcedirected.Functions import attractive_force_ahops, repulsive_force_hops_exp

from forcedirected.Functions import DropSteadyRate, DropLinearChange, DropExponentialDimish
from forcedirected.Functions import generate_random_points
from recursivenamespace import recursivenamespace as rn

import torch

from forcedirected.utilityclasses import ForceClass_base

# from forcedirected.models.ForceDirected import ForceDirected
from forcedirected.structs.ForceDirected_3 import ForceDirected

# def force_function(ForceClass_base):
#     def __init__(self, name, func, shape=None, dimension_coef=None, **kwargs) -> None:
#         self.name = name
#         pass

#     def forward(D, H, params, **kwargs):
#         n = D.shape[1] # total number of nodes that exert node (to be included in the calculation)
#         N = torch.norm(D, dim=-1)     # pairwise distance between points
        
#         k1,k2,k3,k4 = params.k1, params.k2, params.k3, params.k4

#         # attractive component
#         Fa = torch.where(H>0, k1*N*torch.exp(-(H-1)/k2), torch.zeros_like(H))/n
#         dFa = torch.where(H>0, k1*torch.exp(-(H-1)/k2), torch.zeros_like(H))/n # derivative of Fa w.r.t. N
#         # repulsive component
#         Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n
#         dFr = torch.where(H>0, k3/k4*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n # derivative of Fr w.r.t. N
        
#         F = Fa + Fr
#         dF = dFa + dFr

#         # for report
#         NFa, NFr, NF = (torch.norm(t, dim=-1) for t in [Fa, Fr, F])
#         batch_count, batch_no = kwargs.get('batch_count', 0), kwargs.get('batch_no', 0)
#         print(f"batch {batch_no}/{batch_count} |  Fa: {NFa.sum():.3f}, Fr: {NFr.sum():.3f}, F: {NF.sum():.3f}({NF.mean():.3f})")
        
#         # Use the Newtonian method to regress to the root of the equation
#         dZ = torch.where(N == 0 & dF == 0, torch.zeros_like(F), -(F/dF)/N).unsqueeze(-1)*D # multiply by the unit vector D/N
#         dZ = dZ.sum(axis=1)
#         return dZ

class FDModel(ForceDirected):
    """Force Directed Model"""
    VERSION = ForceDirected.VER_MAJ+"01"
    DESCRIPTION = f"Same as 201, with the ForceDirected-v{ForceDirected.VER_MAJ} base implementation."
    def __init__(self, k1=0.999, k2=1.0, k3=None, k4=0.01, lr=1.0, drop_rate=0.5, **kwargs):        
        super().__init__(**kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.lr = lr
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=drop_rate)
        
    def setup(self, Gx, n_dim=None, Z=None, **kwargs):
        """Initialize the embedding variables with the graph and the dimension of the embedding space."""
        if(n_dim is None and Z is None): raise ValueError("Either n_dim or Z must be provided")

        self.n_nodes = Gx.number_of_nodes()
        # Z is provided
        if(n_dim is None): n_dim = Z.shape[1] 
        # n_dim is provided, and Z should be randomly generated
        elif(Z is None):
            mean, standard_dev = 0, self.n_nodes/n_dim**2
            Z = np.random.normal(mean, standard_dev, size=(self.n_nodes, n_dim))
            Z = torch.tensor(Z)
        self.Z = torch.nn.Parameter(Z, requires_grad=False)

        # get the APSP matrix
        hops = get_hops(Gx)
    
        # find maxhops, which is in range [0, n_nodes-1]. 0 is for totally disconnected graph.
        self.maxhops = max(hops[hops<self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>=self.n_nodes]=self.n_nodes  # disconncted nodes are 'n_nodes' hops away, to avoid infinity
        self.register_buffer('hops', torch.tensor(hops))

        # get the degrees of the nodes, as self.degrees
        self.register_buffer('degrees', torch.tensor([d for n, d in Gx.degree()]))


        if(self.k3 is None):
            self.k3 = self.n_nodes
        # End of initialize


    def netforce(self, bmask, **kwargs):
        # k1,k2,k3,k4 = (1, 0.83058, 10, 0.9)
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        D = pairwise_difference(self.Z[bmask], self.Z)
        N = torch.norm(D, dim=-1)     # pairwise distance between points
        H = self.hops[bmask]

        n = D.shape[1] # total number of nodes
        

        # calculate force magnitudes
        # Fa = k1 * N * torch.exp(-(H-1)/k2) / n        
        Fa = torch.where(H>0, k1*N*torch.exp(-(H-1)/k2), torch.zeros_like(H))/n

        # Fr = -k3 * (H-1) * torch.exp(-N/k4) / n
        # Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n
        Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))

        F = Fa + Fr

        # # Get force magnitudes for report
        NFa = Fa.abs().sum()
        # # get the mean of force magnitudes over Fa: NFa_mean := Fa.sum().sum() / (Fa.shape[0]*Fa.shape[1])
        NFa_mean = Fa.abs().mean()
        NFr = Fr.abs().sum()
        NFr_mean = Fr.abs().mean()
        NF = F.abs().sum()
        NF_mean = F.abs().mean()
        print(f"{n}  Fa: {NFa:.3f}, Fr: {NFr:.3f}, F: {NF:.3f}({NF_mean:.8f})")

        # apply unit directions
        F = torch.where(N == 0, torch.zeros_like(F), F/N).unsqueeze(-1)*D # multiply by unit vector D/N
        # for each row, sum up the force vectors
        F = F.sum(axis=1)

        # F = F.sum(axis=1)
        return F

if __name__ == "__main__":
    import networkx as nx

    from forcedirected.utilityclasses import Callback_Base
    
    class SaveEmbedding (Callback_Base):
        def __init__(self, save_every=1, **kwargs) -> None:
            super().__init__(**kwargs)
            self.save_every = save_every

        def on_epoch_end(self, fd_model, epoch, **kwargs):
            if(epoch%self.save_every!=0): return
            Z = fd_model.get_embeddings()
            # Save embedding
            pd.DataFrame(Z, index=Gx.nodes()).to_pickle("Z.pkl")

    # generate a random graph
    # Gx = nx.powerlaw_cluster_graph(100, 4, 0.1)
    Gx = nx.scale_free_graph(300)
    # make it undirected
    Gx = Gx.to_undirected()
    # remove self loops
    Gx.remove_edges_from(nx.selfloop_edges(Gx))
    from forcedirected.utilities import load_graph
    Gx = load_graph("/N/u/hessamla/BigRed200/gnn/forcedirected-graph-embedding/data/graphs/cora/cora_edgelist.txt")
    # plot the graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    nx.draw(Gx, node_size=0.5, node_color='red', edge_color='gray', width=0.2)
    # save image to file
    plt.savefig("graph_raw.pdf")

    model = FDModel(params={'k1':0.999, 'k2':1.0, 'k3':10, 'k4':0.9}, lr=1.0, drop_rate=0.5)
    Z = model.embed(Gx, epochs=1000, n_dim=10)

    # plot the embeddings in 2d
    from sklearn.manifold import TSNE
    Z2d = TSNE(n_components=2).fit_transform(Z)

    # plot the embeddings
    plt.figure(figsize=(4,4))
    nx.draw(Gx, pos=Z2d, node_size=0.5, node_color='red', edge_color='gray', width=0.2)
    plt.savefig("graph_embedded.pdf")

    # save the embeddings
    pd.DataFrame(Z, index=Gx.nodes()).to_pickle("Z.pkl")
    pd.DataFrame(Z2d, index=Gx.nodes()).to_pickle("Z2d.pkl")

# %%
