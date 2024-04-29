
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

from forcedirected.models.ForceDirected import ForceDirected

class FDModel(ForceDirected):
    """Force Directed Model"""
    VERSION="0201"
    DESCRIPTION=f"A new iteration. Derives from the class ForceDirected"
    def __init__(self, Gx, n_dim,
                random_points_generator:callable = generate_random_points, 
                lr=1.0, random_drop_rate=0.5,
                **kwargs):
        super().__init__(Gx, n_dim, **kwargs)
        self.n_dim = n_dim
        self.Gx = Gx
        self.n_nodes = Gx.number_of_nodes()
        self.register_buffer('degrees', torch.tensor([d for n, d in Gx.degree()]))

        # get hops matrix
        hops = get_hops(Gx)
    
        # find maxhops, which is in range [0, n_nodes-1]. 0 is for totally disconnected graph.
        self.maxhops = max(hops[hops<self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>=self.n_nodes]=self.n_nodes  # disconncted nodes are 'n_nodes' hops away, to avoid infinity
        self.register_buffer('hops', torch.tensor(hops))

        print("max hops:", self.maxhops)
        
        self.random_points_generator = random_points_generator
        Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dim), )
        # set Z as a torch Parameter
        self.Z = torch.nn.Parameter(Z, requires_grad=False)
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=random_drop_rate)
        
        # define force vectors
        self.forcev = rn()
        # To be used like the following
        # result_F = self.fmodel_attr(self) # pass the current model (with its contained embeddings) to calculate the force
        pass
        

    def forward(self, bmask, **kwargs):
        k1,k2,k3,k4 = (1, 0.83058, 10, 0.9)
        
        D = pairwise_difference(self.Z[bmask], self.Z)
        N = torch.norm(D, dim=-1)     # pairwise distance between points
        H = self.hops[bmask]

        n = D.shape[1] # total number of nodes
        

        # attractive component
        def func_attr():
            v = torch.where(H>0, k1*N*torch.exp(-(H-1)/k2), torch.zeros_like(H))/n
            v = v.sum(axis=1)
            Nv = torch.norm(v, dim=-1)
            return v.sum(axis=1), Nv.sum(), Nv.mean()
        
        def func_repl():
            v = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n
            v = v.sum(axis=1)
            Nv = torch.norm(v, dim=-1)
            return v.sum(axis=1), Nv.sum(), Nv.mean()


        # calculate force magnitudes
        # Fa = k1 * N * torch.exp(-(H-1)/k2) / n        
        Fa = torch.where(H>0, k1*N*torch.exp(-(H-1)/k2), torch.zeros_like(H))/n
        # Get force magnitudes for report
        NFa = Fa.abs().sum()
        # get the mean of force magnitudes over Fa: NFa_mean := Fa.sum().sum() / (Fa.shape[0]*Fa.shape[1])
        NFa_mean = Fa.abs().mean()

        # Fr = -k3 * (H-1) * torch.exp(-N/k4) / n
        # Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n
        Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))
        NFr = Fr.abs().sum()
        NFr_mean = Fr.abs().mean()

        F = Fa + Fr
        NF = F.abs().sum()
        NF_mean = F.abs().mean()

        # print(f"{n}  Fa: {NFa:.3f}, Fr: {NFr:.3f}, F: {NF:.3f}({NF_mean:.8f})")

        # apply unit directions
        F = torch.where(N == 0, torch.zeros_like(F), F/N).unsqueeze(-1)*D # multiply by unit vector D/N
        # for each row, sum up the force vectors
        F = F.sum(axis=1)

        d = self.degrees[bmask].unsqueeze(-1)
        dZ = torch.where(d!=0, F/d, torch.zeros_like(F))

        # random drop
        dZ = self.random_drop(dZ)
        return dZ
            

# %%
