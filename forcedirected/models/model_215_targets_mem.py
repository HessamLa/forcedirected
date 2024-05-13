
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

"""
Model To reduce complexity, only the forces from neighbor+landmark nodes are considered.
Currently, the criteria for landmark node is based on the node degree.
Other possible criteria worth experimenting:
1) Degree centrality
2) Closeness centrality
3) Betweenness centrality
4) Eigencentrality
5) Communicability centrality
6) Information centrality 
7) Katz centrality
8) Load centrality
.. see this paper also https://www.mdpi.com/1099-4300/25/8/1196

S_h(u) is the set of nodes exactly h-hops away from node u.
N_h(u) is the set of nodes at most h-hops away from node u.

N_h(u) = Union S_i(u) for i=1 to h

"""
def get_landmarks_idx_by_degree(Gx, ratio=0.05):
    """Sort nodes by degree and select the top-degree nodes as landmark nodes."""
    assert 0 <= ratio and ratio < 1
    k = int(ratio * Gx.number_of_nodes())

    # get nodes
    # If nodes are not inherently indexed, create an index mapping (optional, depends on your needs)
    node_to_index = {node: idx for idx, node in enumerate(Gx.nodes())}

    degrees = Gx.degree()

    # Sort the nodes by degree in descending order
    sorted_nodes_by_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    landmarks = [node for node, degree in sorted_nodes_by_degree[:k]]

    landmarks_idx = [node_to_index[node] for node in landmarks]
    return landmarks_idx
    # for l in landmarks:
    #     print(l, degrees[l])

    # n = len(Gx.nodes())  # Number of nodes in the graph
    # N_h_lists = {}  # Dictionary to store the N_h list for each node

    # # Iterate through each node in the graph
    # for i in range(n):
    #     # Find nodes that are at most h hops away from node i
    #     N_h = [j for j in range(n) if H[i, j] <= h and H[i, j] != 0]  # Exclude self (H[i, j] != 0 if you don't want to include the node itself)
    #     # Store the N_h list in the dictionary
    #     N_h_lists[i] = N_h
    # return landmarks_idx, N_h_lists
    

class FDModel(ForceDirected):
    """Force Directed Model"""
    VER_MIN="25"
    DESCRIPTION=f"The shell FD model with target nodes: f(x) =1/|S_h(u)| * ( xe^(-h+1) - he^(-x) ) . Fa from S_a(u)+L, F_r from S_r(u)+L. Better memory utilization."
    def __init__(self, Gx, n_dim,
                k1:float=0.999, k2:float=1.0, k3:float=10.0, k4:float=0.01,
                reach_a:int=1, reach_r:int=4,
                landmarks_ratio:float = 0.01,  # top 1% of the nodes (based on degree) are landmark nodes
                lr:float=1.0, random_drop_rate:float=0.5,
                **kwargs):
        super().__init__(Gx, n_dim, **kwargs)
        self.VERSION=f"{self.VER_MAJ}{self.VER_MIN}"
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4
        self.reach_a = reach_a
        self.reach_r = reach_r
        self.reach_max = max(reach_a, reach_r)
        self.landmarks_ratio = landmarks_ratio

        self.lr = lr
        self.n_dim = n_dim
        self.Gx = Gx
        self.random_drop_rate = random_drop_rate

        # setup the model    
        self.n_nodes = self.Gx.number_of_nodes()
        if(self.k3 is None): 
            self.k3 = self.n_nodes

        self.register_buffer('degrees', torch.tensor([d for n, d in self.Gx.degree()]))
        
        # generate the random embeddings
        Z = torch.tensor(generate_random_points(self.n_nodes, self.n_dim), )
        self.Z = torch.nn.Parameter(Z, requires_grad=False)
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=self.random_drop_rate)

        # get ASPS matrix (hops distance)
        hops = get_hops(self.Gx)

        # find maxhops, which is in range [0, n_nodes-1]. 0 is for totally disconnected graph.
        self.maxhops = max(hops[hops<self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>=self.n_nodes]=self.n_nodes  # disconncted nodes are 'n_nodes' hops away, to avoid infinity
        self.register_buffer('hops', torch.tensor(hops))
        if(self.verbosity>=1):
            print("max hops:", self.maxhops)
        
        # max-out indices where hops > reach_max since they are not used for force calculation
        self.hops[self.hops>self.reach_max] = self.maxhops

        # Get the shell averaging coefficient
        self.register_buffer('hop_occurence', torch.tensor(get_occurence(self.hops)))
        
        self.register_buffer('shell_avg_coeff', 1/self.hop_occurence) # 1/|S_h(u)|, averaging factor based on shell size. S_h(u) is the set of nodes at h-hops away from node u.

        if(self.verbosity>=3):
            print('hops.shape', self.hops.shape)
            print('hop_occurence.shape', self.hop_occurence.shape)

        landmarks_idx = get_landmarks_idx_by_degree(Gx, ratio=landmarks_ratio)
        self.register_buffer('landmarks_idx', torch.tensor(landmarks_idx))
        if(self.verbosity>=1):
            print(f'total landmark nodes:{len(self.landmarks_idx)} ({landmarks_ratio*100:.3f}%)')
        # zero-out the shell_avg_coeff for targets at hops > reach_max
        self.shell_avg_coeff[self.hops>self.reach_max] = 0
        # update shell_avg_coeff for landmark nodes
        self.shell_avg_coeff[:, self.landmarks_idx] += 1/len(self.landmarks_idx)

        # let B = shell_avg_coeff, C = hops
        # Downsize B(n,n) to Br(n,r), and C(n,n) to Cr(n,r) 
        # where r is the maximum number of non-zero elements in each row of B
        # idx_map(n,r) is the index mapping tensor, Br[i] = B[i, idx_map[i]] and Cr[i] = C[i, idx_map[i]]
        # B is the input tensor with shape (n, n)
        B = self.shell_avg_coeff 
        C = self.hops

        n = B.size(0)
        # Find the maximum number of larger-than-zero elements in each row
        r = torch.max(torch.sum(B > 0, dim=1)).item()

        # Create an empty tensor Br with shape (n, r)
        
        Br = torch.zeros(n, r, dtype=B.dtype)
        Cr = torch.zeros(n, r, dtype=C.dtype)

        # Create an empty index mapping tensor idx_map with shape (n, r)
        self.idx_map = torch.zeros(n, r, dtype=torch.long)

        # Iterate over each row of B
        for i in range(n):
            # Find the indices of larger-than-zero elements in the current row
            indices = torch.where(B[i] > 0)[0]
            # Copy the larger-than-zero elements to Br
            Br[i, :indices.size(0)] = B[i, indices]
            Cr[i, :indices.size(0)] = C[i, indices]
            # Initialize all elements of the i-th row of idx_map to i
            self.idx_map[i] = i
            # Update the corresponding indices in idx_map
            self.idx_map[i, :indices.size(0)] = indices

        self.r = r
        self.shell_avg_coeff = Br
        self.hops = Cr
        pass
        

    def forward(self, bmask, **kwargs):
        k1,k2,k3,k4 = self.k1, self.k2, self.k3, self.k4

        H = self.hops[bmask]
        
        # Create a new dimension for Z to match the shape of idx_map
        n,r,d = self.Z[bmask].shape[0], self.r, self.Z[bmask].shape[1]
        Z = self.Z[bmask].unsqueeze(1).expand(n, r, d)
        # Use idx_map to index Z along the first dimension
        idx = self.idx_map[bmask]
        Z_target = self.Z[[idx]]
        
        # print("\n", self.idx_map.shape, idx.shape, Z_target.shape)
        D = Z_target - Z

        # D = pairwise_difference(self.Z[bmask], self.Z)
        N = torch.norm(D, dim=-1)     # pairwise Euclidean distance between points
        
        avg_coeff = self.shell_avg_coeff[bmask] # averaging factor
        
        # # count of the landmark nodes
        # if(len(self.landmarks_idx) > 0):
        #     shell_avg[:, self.landmarks_idx] += 1/len(self.landmarks_idx) # add averaging factor for the landmarks nodes
        
        
        n = D.shape[1] # total number of nodes

        # calculate force magnitudes
        # Fa = 1/|S_h(u)| * k1 * x * exp(-k2 * (h - 1)), for pairs at reach_a hops away or with a landmark node
        # Fr = -k3 * x * exp(-k4 * h), for pairs at reach_r hops away or with a landmark node
        Fa = avg_coeff * k1 * N * torch.exp(-k2 * (H - 1))

        # mask = (0 < H) & (H <= self.reach_a) # include nodes reach_a hops away
        # if(len(self.landmarks_idx)>0):
        #     mask[:, self.landmarks_idx] = True
        # Fa = torch.where(mask==True, avg_coeff * k1 * N * torch.exp(-k2 * (H - 1)), torch.zeros_like(H))
        
        Fr = -k3 * H *torch.exp(-N*k4)
        # mask = (0 < H) & (H <= self.reach_r) # include nodes reach_r hops away
        # if(len(self.landmarks_idx)>0):
        #     mask[:, self.landmarks_idx] = True # include landmarks
        # Fr = torch.where(mask==True, -k3 * H *torch.exp(-N*k4), torch.zeros_like(H))

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
