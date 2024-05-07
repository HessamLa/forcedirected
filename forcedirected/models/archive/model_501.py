
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
    VERSION="0301"
    DESCRIPTION=f"Use Newtonian method to regress to the root of the equation. Version {VERSION}"
    def __init__(self, Gx, n_dim, alpha:float,
                random_points_generator:callable = generate_random_points, 
                lr=1.0,
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
        
        # self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)

        self.random_points_generator = random_points_generator
        Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dim), )
        # set Z as a torch Parameter
        self.Z = torch.nn.Parameter(Z, requires_grad=False)
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=0.5)
        
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

        # attractive component
        Fa  = torch.where(H==0, torch.zeros_like(H), k1*N*torch.exp(-(H-1)/k2))/n
        dFa = torch.where(H==0, torch.zeros_like(H), k1*torch.exp(-(H-1)/k2))/n # derivative of Fa w.r.t. N
        # repulsive component
        Fr  = torch.where(H==0, torch.zeros_like(H), -k3*(H-1)*torch.exp(-N/k4))/n
        dFr = torch.where(H==0, torch.zeros_like(H), k3/k4*(H-1)*torch.exp(-N/k4))/n # derivative of Fr w.r.t. N
        
        F = Fa + Fr
        dF = dFa + dFr
        # jacobian
        
        NFa, NFr, NF = (torch.norm(t.sum(axis=1), dim=-1) for t in [Fa, Fr, F])
        NdFa, NdFr, NdF = (torch.norm(t.sum(axis=1), dim=-1) for t in [dFa, dFr, dF])

        #print the number of Force vectors where NF is zero

        print(f"{n}  Fa: {NFa.sum():8.3f},  Fr: {NFr.sum():8.3f},  F: {NF.sum():8.3f}({NF.mean():8.3f})")
        print(f"{n} dFa: {NdFa.sum():8.3f}, dFr: {NdFr.sum():8.3f}, dF: {NdF.sum():8.3f}({NdF.mean():8.3f})")

        # F = torch.where(dF==0, torch.zeros_like(F), -F/dF)
        F = -dF.inverse()@F # Use the Newtonian method to regress to the root of the equation
        F = torch.where(N == 0, torch.zeros_like(F), F/N).unsqueeze(-1)*D # multiply by the unit vector D/N
        F = F.sum(axis=1)

        d = self.degrees[bmask].unsqueeze(-1)
        dZ = torch.where(d!=0, F/d, torch.zeros_like(F))

        # random drop
        dZ = self.random_drop(dZ)
        return dZ
            

# %%
