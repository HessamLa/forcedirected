
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
    VER_MIN="01"
    DESCRIPTION=f"The basic FD model: f(x) = xe^(-h+1) - he^(-x) . Derives from the class ForceDirected"
    def __init__(self, Gx, n_dim,
                k1=0.999, k2=0.83058, k3=10.0, k4=0.9,
                lr=1.0, random_drop_rate=0.5,
                **kwargs):
        super().__init__(Gx, n_dim, **kwargs)
        self.VERSION=f"{self.VER_MAJ}{self.VER_MIN}"
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4

        self.lr = lr
        self.n_dim = n_dim
        self.Gx = Gx
        self.random_drop_rate = random_drop_rate

        # setup the model    
        self.n_nodes = self.Gx.number_of_nodes()
        if(self.k3 is None): 
            self.k3 = self.n_nodes

        self.register_buffer('degrees', torch.tensor([d for n, d in self.Gx.degree()]))
        
        # get ASPS matrix (hops distance)
        hops = get_hops(self.Gx)

        # find maxhops, which is in range [0, n_nodes-1]. 0 is for totally disconnected graph.
        self.maxhops = max(hops[hops<self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>=self.n_nodes]=self.n_nodes  # disconncted nodes are 'n_nodes' hops away, to avoid infinity
        self.register_buffer('hops', torch.tensor(hops))
        if(self.verbosity>=1):
            print("max hops:", self.maxhops)
        
        # generate the random embeddings
        self.random_points_generator = generate_random_points
        Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dim), )
        self.Z = torch.nn.Parameter(Z, requires_grad=False)
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=self.random_drop_rate)

        # # get neigborhood hops counts
        # self.register_buffer('hop_occurence', torch.tensor(get_occurence(hops)))
        
        # self.register_buffer('hop_normalized', 1/self.hop_occurence) # 1/|N_h|, average hops to set of nodes at h hops away

        
        # print('hops.shape', self.hops.shape)
        # print('hop_occurence.shape', self.hop_occurence.shape)
        
    def forward(self, bmask, **kwargs):
        k1,k2,k3,k4 = self.k1, self.k2, self.k3, self.k4

        
        D = pairwise_difference(self.Z[bmask], self.Z)
        N = torch.norm(D, dim=-1)   # pairwise Euclidean distance between points
        H = self.hops[bmask]        # hops distance between points

        n = D.shape[1] # total number of nodes
        
        # calculate force magnitudes
        # Fa = k1 * N * torch.exp(-(H-1)/k2) / n        
        Fa = 1/n*torch.where(H>0, k1*N*torch.exp(-(H-1)/k2), torch.zeros_like(H))

        # Fr = -k3 * (H-1) * torch.exp(-N/k4) / n
        # Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))/n
        Fr = torch.where(H>0, -k3*(H-1)*torch.exp(-N/k4), torch.zeros_like(H))

        F = Fa + Fr

        if(self.verbosity>=3):
            # Get force magnitudes for report
            NFa = Fa.abs().sum()
            # get the mean of force magnitudes over Fa: NFa_mean := Fa.sum().sum() / (Fa.shape[0]*Fa.shape[1])
            NFa_mean = Fa.abs().mean()
            NFr = Fr.abs().sum()
            NFr_mean = Fr.abs().mean()
            NF = F.abs().sum()
            NF_mean = F.abs().mean()
            print(f" Fa: {NFa:.3f}, Fr: {NFr:.3f}, F: {NF:.3f}({NF_mean:.8f})")

        # apply unit directions
        F = torch.where(N == 0, torch.zeros_like(F), F/N).unsqueeze(-1)*D # multiply by unit vector D/N
        # for each row, sum up the force vectors
        F = F.sum(axis=1)

        mass = self.degrees[bmask].unsqueeze(-1)
        dZ = torch.where(mass!=0, F/mass, torch.zeros_like(F))

        # random drop
        dZ = self.random_drop(dZ)
        return dZ
            

# %%
