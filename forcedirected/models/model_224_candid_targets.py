
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

def get_occurence (M): # along rows
    def get_row_count(row):
        unique_values, counts = np.unique(row, return_counts=True) # number of nodes at i-hop distance
        c = counts[np.searchsorted(unique_values, row)] 
        return c
    return np.apply_along_axis(get_row_count, axis=1, arr=M)

class FDModel(ForceDirected):
    """Force Directed Model"""
    VER_MIN="24"
    DESCRIPTION=f"Same as the shell model (v204). "\
         "The set of target nodes is node in the 1st and 2nd proximity, and a fixed number of nodes chosed randomly."
    def __init__(self, Gx, n_dim,
                k1:float=0.999, k2:float=1.0, k3:float=10.0, k4:float=0.01,
                lr:float=1.0, random_drop_rate:float=0.5,
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
        ###########################
        # # HERE IS THE MAIN CHANGE
        # For each row and its equivalent column (its symmetric), only keep the values 1 and 2 and maximum 
        # of t values not in {0, 1, 2}. Zero out the rest. 
        # This is to keep the target nodes in the 1st and 2nd proximity, and a fixed number of nodes chosed randomly.
        print("Settings the candid nodes.")
        def update_hops_matrix(H, h, k):
            import random
            n = H.shape[0]  # number of nodes
            Hupdate = np.zeros_like(H, dtype=float)
            
            for i in range(n):
                if(i%1000==0):
                    print(i)
                # Nodes within h-hops
                within_h = set(np.where((H[i] <= h) & (H[i] > 0))[0])
                
                # Nodes more than h-hops away
                beyond_h = set(np.where(H[i] > h)[0])
                
                # Randomly select up to k nodes from beyond_h
                random_beyond_h = set(random.sample(beyond_h, min(k, len(beyond_h))))
                
                # Combine the sets
                reachable = within_h.union(random_beyond_h)
                
                # Update Hupdate matrix
                for j in reachable:
                    Hupdate[i, j] = H[i, j]
            
            return Hupdate
        
        keep_max = 2 # keep the maximum of t values not in {0, 1, 2}
        t = int(np.log(self.n_nodes))
        print(f"keep_max={keep_max},  random nodes={t}")
        hops = update_hops_matrix(hops, keep_max, t)
        # count non-zero elements in the matrix
        non_zero_count = (hops>0).sum().item()
        print(f"non-zero count:{non_zero_count} ({np.sqrt(non_zero_count)})")
        print(f"ratio saved: {100-non_zero_count*100/(self.n_nodes**2):.3f}%")


        # t = int(np.log(self.n_nodes))
        # keep_max = 2 # keep the maximum of t values not in {0, 1, 2}
        # idx = self.hops[i,:]>keep_max
        # for i in range(self.hops.shape[0]):    
        #     true_count = idx.sum().item()
        #     print(f"Node {i}: count of entries > {keep_max} = {true_count:3d} ", end="")
        #     if true_count > t:
        #         # Get the indices of True values (values > keep_max)
        #         true_indices = torch.where(idx[i, :])[0]
                
        #         # Randomly select indices to change to False
        #         # change_count = true_count - t
        #         change_indices = torch.randperm(true_count)[:change_count]
                
        #         # cnt_to_keep = t
        #         # keep_indices = torch.randperm(true_count)[:cnt_to_keep]
        #     # self.hops[i, true_indices[keep_indices]] = 0
        #     # self.hops[true_indices[keep_indices], i] = 0
        #     self.hops[i, true_indices[change_indices]] = 0
        #     self.hops[true_indices[change_indices], i] = 0
        #     print(f"Node {i}: count of entries <= {keep_max} = {self.hops}, count of entries > {keep_max} = {change_count}")
        #     assert (self.hops[i, idx]>keep_max).sum().item() == t
        #     assert (self.hops[idx, i]>keep_max).sum().item() == t
        #     print()
        ###########################


        self.register_buffer('hops', torch.tensor(hops))
        if(self.verbosity>=1):
            print("max hops:", self.maxhops)


        # generate the random embeddings
        Z = torch.tensor(generate_random_points(self.n_nodes, self.n_dim), )
        self.Z = torch.nn.Parameter(Z, requires_grad=False)
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=self.random_drop_rate)

        # Get the shell averaging coefficient
        self.register_buffer('hop_occurence', torch.tensor(get_occurence(hops)))
        # 1/|S_h(u)|, averaging factor based on shell size. S_h(u) is the set of nodes at h-hops away from node u.
        self.register_buffer('shell_avg_coeff', 1/self.hop_occurence) 

        
        if(self.verbosity>=3):
            print('hops.shape', self.hops.shape)
            print('hop_occurence.shape', self.hop_occurence.shape)
        pass
        

    def forward(self, bmask, **kwargs):
        k1,k2,k3,k4 = self.k1, self.k2, self.k3, self.k4

        H = self.hops[bmask]
        
        D = pairwise_difference(self.Z[bmask], self.Z)
        N = torch.norm(D, dim=-1)     # pairwise Euclidean distance between points
        
        sh_avg_coeff = self.shell_avg_coeff[bmask] # averaging factor
        
        n = D.shape[1] # total number of nodes

        # calculate force magnitudes
        # Attractive force Fa = 1/|S_h(u)| * k1 * x * exp(-k2 * (h - 1))
        Fa = torch.where(H>0, k1 * sh_avg_coeff * N * torch.exp(-k2 * (H - 1)), torch.zeros_like(H))
        # Repulsive force Fr = -k3 * x * exp(-k4 * h)
        Fr = torch.where(H>0, -k3 * H * torch.exp(-N*k4), torch.zeros_like(H))
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

        d = self.degrees[bmask].unsqueeze(-1)
        dZ = torch.where(d!=0, F/d, torch.zeros_like(F))

        # random drop
        dZ = self.random_drop(dZ)
        return dZ
            

# %%
