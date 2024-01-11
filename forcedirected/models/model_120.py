
# %%
from typing import Any

import numpy as np
import pandas as pd

from forcedirected.utilities.graphtools import process_graph_networkx
from forcedirected.utilities import batchify

from forcedirected.algorithms import get_alpha_hops, pairwise_difference
from forcedirected.Functions import attractive_force_ahops, repulsive_force_hops_exp

from forcedirected.Functions import DropSteadyRate, DropLinearChange, DropExponentialDimish
from forcedirected.Functions import generate_random_points
from forcedirected.utilityclasses import ForceClass
from forcedirected.utilityclasses import Model_Base
from recursivenamespace import recursivenamespace as rn
import torch

def force_standard(D, H, k1=1, k2=1, k3=1, k4=1, return_sum=True):
    """
    f(x) = f_attr + f_repl
    f(x) = k1*x*exp(-h/k2) - k3*h*exp(-x/k4)
    D (m,n,d) is the pairwaise difference (force). D[i,j]=Z[j]-Z[i], from i to j
    H (m,n) is the hops distance between nodes, H[i,j] = h_ij hops from i to j
    """
    n = D.shape[1] # total number of nodes
    N = torch.norm(D, dim=-1)     # pairwise distance between points
    F = k1*N*torch.exp(-H/k2) - k3*H*torch.exp(-N/k4)
    F = torch.where(N == 0, torch.zeros_like(F), F/N).unsqueeze(-1)*D # multiply by unit vector D/N
    F = F/n
    if(return_sum):
        F = F.sum(axis=1)
    # print(D)
    # print(N)
    # print(F)
    return F

class FDModel(Model_Base):
    """Force Directed Model"""
    VERSION="0120"
    DESCRIPTION="Standard force model f_ij(x) = k1*x*exp(-h/k2) - k3*h*exp(-x/k4), k1=0.8"
    def __str__(self):
        return f"FDModel v{FDModel.VERSION},{FDModel.DESCRIPTION}"
    
    def __init__(self, Gx, n_dim, alpha:float,
                random_points_generator:callable = generate_random_points, 
                **kwargs):
        
        super().__init__(**kwargs)
        self.Gx = Gx
        self.n_nodes = Gx.number_of_nodes()
        self.n_dim = n_dim
        self.alpha = alpha
        Gnk, A, degrees, hops = process_graph_networkx(Gx)
        self.lr = kwargs.get('lr', 1.0)
        
        self.degrees = degrees

        # find max hops
        self.maxhops = max(hops[hops<=self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>self.maxhops]=self.maxhops+1  # disconncted nodes are 'maxhops+1' hops away, to avoid infinity
        self.hops = hops
        print("max hops:", self.maxhops)

        self.dim_bias = np.array([1+i*4//self.n_dim for i in range(self.n_dim)])

        self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)
        
        self.random_points_generator = random_points_generator
        self.Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dim), )
        
        self.force_std = ForceClass(name='force_standard',
                            func=lambda *args, **kwargs: force_standard(*args, **kwargs)
        )

        # define force vectors
        self.forcev = rn(
            F1 = rn(tag='standard', description='Standard Force', 
                    v=torch.zeros_like(self.Z)
                    ),
            )
        # To be used like the following
        # result_F = self.fmodel_attr(self) # pass the current model (with its contained embeddings) to calculate the force
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=0.5)
        
        self.statsdf = pd.DataFrame()
        # initialize embeddings
        
        FDModel.DESCRIPTION = FDModel.DESCRIPTION.format(alpha=self.alpha)
        pass
    
    def get_embeddings(self):   
        return self.Z.detach().cpu().numpy()

    @torch.no_grad()
    def train(self, epochs=100, device='cpu', row_batch_size='auto', **kwargs):
        # train begin
        kwargs['epochs'] = epochs
        self.notify_train_begin_callbacks(**kwargs)

        self.hops = torch.tensor(self.hops).to(device)
        self.dim_bias = torch.tensor(self.dim_bias).to(device)
        
        self.alpha_hops = torch.tensor(self.alpha_hops).to(device)
        self.degrees = torch.tensor(self.degrees).to(device) # mass of a node is equivalent to its degree
        self.Z = self.Z.to(device)
        for F in self.forcev.values():
            F.v = F.v.to(device)
        
        def do_pairwise(Z_source, Z_landmark):
            D = pairwise_difference(Z_source, Z_landmark) # D[i,j] = Z_lmrk[j] - Z_src[i]
            N = torch.norm(D, dim=-1)     # pairwise distance between points
            unitD = torch.where(N.unsqueeze(-1)!=0, D / N.unsqueeze(-1), torch.zeros_like(D))
            return D, N, unitD

        from forcedirected.utilities import optimize_batch_count
        @optimize_batch_count(max_batch_count=self.n_nodes)
        def run_batches(batch_count=1, **kwargs):
            kwargs['batch_count'] = batch_count
            print(f"run_batches: batch count: {kwargs['batch_count']}")
            batch_size=int(self.Z.shape[0]//batch_count +0.5)
            for i, bmask in enumerate (batchify(list(range(self.Z.shape[0])), batch_size=batch_size)):
                # batch begin
                kwargs['batch'] = i+1
                kwargs['batch_size'] = batch_size
                self.notify_batch_begin_callbacks(**kwargs)
                
                ###################################
                # this is the forward pass
                hops=self.hops[bmask,:]
                Z_source, Z_landmark = self.Z[bmask], self.Z
                D = pairwise_difference(Z_source, Z_landmark) # D[i,j] = Z_lmrk[j] - Z_src[i]
                
                self.forcev.F1.v[bmask] = self.force_std(D, hops, k1=0.8)
                del D

                # batch ends
                self.notify_batch_end_callbacks(**kwargs)
            
            return batch_count

        for epoch in range(epochs):
            if(self.stop_training): break

            # epoch begin
            kwargs['epoch'] = epoch
            self.notify_epoch_begin_callbacks(**kwargs)

            batch_count = run_batches(**kwargs)

            kwargs['batch_count'] = batch_count
            kwargs['batch_size'] = int(self.Z.shape[0]//batch_count +0.5)
            
            # aggregate forces
            F = torch.zeros_like(self.Z).to(device)
            F = sum([F.v for F in self.forcev.values()])

            # apply random drop
            F = self.random_drop(F, **kwargs)
            ###################################
            # finally calculate the gradient and udpate the embeddings
            # find acceleration on each point a = F/m. And X-X0 = a*t^2, Assume X0 = 0 and t = 1
            self.dZ = torch.where(self.degrees[..., None] != 0, 
                                    F / self.degrees[..., None], 
                                    torch.zeros_like(F))
            self.Z += self.dZ*self.lr
        
            self.notify_epoch_end_callbacks(**kwargs)
        # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass

# %%
