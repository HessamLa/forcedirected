
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
        "Force on each node is calculated with respect to some landmark nodes. "\
        "The set of landmark nodes is the set node in the hmax-th neighborhood, and a fixed number of landmark nodes chosen randomly."
    def __init__(self, Gx, n_dim,
                hmax:int=2, tlog_coeff:int=100, random_nodes_ratio:list=[0.5, 0.25],
                k1:float=0.999, k2:float=1.0, k3:float=1.0, k4:float=0.01,
                lr:float=1.0, random_drop_rate:float=0.5,
                **kwargs):
        super().__init__(Gx, n_dim, **kwargs)
        # enforce the types
        k1, k2, k3, k4 = float(k1), float(k2), float(k3), float(k4)
        lr, random_drop_rate = float(lr), float(random_drop_rate)
        hmax, tlog_coeff = int(hmax), int(tlog_coeff)

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
        non_zero_count_before = (hops>0).sum().item()
        ###########################
        # # HERE IS THE MAIN CHANGE
        # For each row and its equivalent column (its symmetric), only keep the values {0,1, 2,..., hmax} as well as a
        # maximum number of t values not in {0, 1, ... hmax}. Zero out the rest. 
        # This is to keep the landmark nodes in hmax-th neighborhood, and a fixed number of nodes chosen randomly.
        print("Settings the landmark nodes.")
        import random
        def update_hops_matrix(H, h, k):
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
        def update_hops_matrix_2(H, h, k):
            n = H.shape[0]  # number of nodes
            Hupdate = np.zeros_like(H, dtype=float)
            h_counts = np.zeros(n, dtype=int)
            
            for i in range(n):
                # Nodes within h-hops
                within_h = set(np.where((H[i] <= h) & (H[i] > 0))[0])
                
                # Nodes more than h-hops away
                h_plus_1 = set(np.where(H[i] == h + 1)[0])
                h_plus_2 = set(np.where(H[i] == h + 2)[0])
                beyond_h_plus_2 = set(np.where(H[i] > h + 2)[0])
                
                # Calculate the number of nodes to select from each category
                k_h_plus_1 = min(int(k * 0.5), len(h_plus_1))
                k_h_plus_2 = min(int(k * 0.25), len(h_plus_2))
                k_beyond_h_plus_2 = k - k_h_plus_1 - k_h_plus_2
                
                # Randomly select nodes from each category
                selected_h_plus_1 = set(random.sample(h_plus_1, k_h_plus_1))
                selected_h_plus_2 = set(random.sample(h_plus_2, k_h_plus_2))
                selected_beyond_h_plus_2 = set(random.sample(beyond_h_plus_2, min(k_beyond_h_plus_2, len(beyond_h_plus_2))))
                
                # Combine the sets
                random_beyond_h = selected_h_plus_1.union(selected_h_plus_2, selected_beyond_h_plus_2)
                reachable = within_h.union(random_beyond_h)
                
                # Update Hupdate matrix
                for j in reachable:
                    Hupdate[i, j] = H[i, j]
                
                # Count entries in range (0, h]
                h_counts[i] = np.sum((Hupdate[i, :] > 0) & (Hupdate[i, :] <= h))
            
            return Hupdate, h_counts
        
        def update_hops_matrix_3(H, hmax, max_random_count, p: list = [0.50, 0.25]):
            n = H.shape[0]  # number of nodes
            Hupdate = np.zeros_like(H, dtype=float)
            h_counts = np.zeros(n, dtype=int)
            
            # Validate p
            if(sum(p) > 1.0):
                raise ValueError("Sum of elements in p must be less than or equal to 1.0")
            # Adjust p
            if(sum(p) < 1.0):
                p.append(1.0 - sum(p))
            
            print("p:", p)
            
            # Calculate k_count
            def generate_k_count(max_random_count, p):
                """
                Generate a list of integers k_count such that k_count[i] is approximately max_random_count * p[i]
                and the sum of k_count equals max_random_count.
                
                Args:
                max_random_count (int): The total number of nodes to be selected
                p (list): A list of floating point values that sum to 1.0
                
                Returns:
                list: A list of integers that sum to max_random_count
                """
                # Initial calculation
                k_count = [int(max_random_count * pi) for pi in p]
                
                # Adjust for rounding errors
                diff = max_random_count - sum(k_count)
                
                # Distribute the difference
                for i in range(diff):
                    # Find the index where adding 1 would lead to the closest match to the desired proportion
                    errors = [(k_count[j] + 1) / max_random_count - p[j] for j in range(len(p))]
                    k_count[errors.index(min(errors))] += 1
                
                return k_count
            k_count = generate_k_count(max_random_count, p)
            
            # Calculate the number of categories
            num_categories = len(p)
            
            for i in range(n):
                # Nodes within hmax-hops
                within_hmax = set(np.where((H[i] <= hmax) & (H[i] > 0))[0])
                
                # Categorize nodes beyond hmax-hops
                beyond_hmax_categories = [set(np.where(H[i] == hmax + j + 1)[0]) for j in range(num_categories - 1)]
                beyond_hmax_categories.append(set(np.where(H[i] > hmax + num_categories - 1)[0]))
                
                # Randomly select nodes from each category
                selected_beyond_hmax = set()
                for category, k in zip(beyond_hmax_categories, k_count):
                    selected_beyond_hmax.update(random.sample(category, min(k, len(category))))
                
                # Combine the sets
                reachable = within_hmax.union(selected_beyond_hmax)
                
                # Update Hupdate matrix
                for j in reachable:
                    Hupdate[i, j] = H[i, j]
                
                # Count entries in range (0, hmax]
                h_counts[i] = np.sum((Hupdate[i, :] > 0) & (Hupdate[i, :] <= hmax))
            
            return Hupdate, h_counts        
        
        t = tlog_coeff*int(np.log(self.n_nodes))# keep the maximum of t values not in the intrange [0, hmax] 
        print(f"hmax={hmax}, number of random nodes beyond hmax={t}")
        hops, _ = update_hops_matrix_3(hops, hmax, t, p=random_nodes_ratio)
        # find maxhops, which is in range [0, n_nodes-1]. 0 is for totally disconnected graph.
        self.max_distance = max(hops[hops<self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>=self.n_nodes]=self.n_nodes  # disconncted nodes are 'n_nodes' hops away, to avoid infinity

        self.register_buffer('hops', torch.tensor(hops))

        # count non-zero elements in the matrix
        non_zero_count_after = (self.hops>0).sum().item()
        print(f"non-zero count before updating:{non_zero_count_before}/{self.n_nodes**2}={non_zero_count_before*100/self.n_nodes**2:.2f}% ({np.sqrt(non_zero_count_before):.2f})")
        print(f"non-zero count after updating:{non_zero_count_after}/{self.n_nodes**2}={non_zero_count_after*100/self.n_nodes**2:.2f}% ({np.sqrt(non_zero_count_after):.2f})")
        print(f"ratio saved: {(non_zero_count_before-non_zero_count_after)*100/non_zero_count_before:.3f}%")
        print(f"Non-zero percentage: {100-non_zero_count_after*100/(self.n_nodes**2):.3f}%")
        ###########################


        self.register_buffer('hops', torch.tensor(hops))
        if(self.verbosity>=1):
            print("max distance:", self.max_distance)

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
        # Fa = torch.where(H==1, k1 * sh_avg_coeff * N * torch.exp(-k2 * (H - 1)), torch.zeros_like(H))
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
