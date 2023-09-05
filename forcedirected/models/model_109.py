
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
import forcedirected.utilities.RecursiveNamespace as rn
import torch

def force_exp_x(D, N, unitD, A, k1=1, k2=1, return_sum=True):
    """
    f(x) = k1*A*exp(-x/k2)
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N

    A (n,n) or (n,n,d) is the coefficient for each 
    k1 is the amplifying factor, scalar: k1*f(x)
    k2 is decaying factor factor, scalar: f(x/k2)
    """
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1*torch.exp(-N/k2)

    # apply negative direction
    F = A * unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.mean(axis=1) # sum the i-th row to get all forces to i

    return F

class FDModel(Model_Base):
    """Force Directed Model"""
    VERSION="0109"
    DESCRIPTION="Same as 108. Decaying biased dim starting from 100. Exponential Decay repulsive(k1=10, k2=0.9), alpha={alpha}"
    def __str__(self):
        return f"FDModel v{FDModel.VERSION},{FDModel.DESCRIPTION}"
    
    def __init__(self, Gx, n_dims, alpha:float,
                random_points_generator:callable = generate_random_points, 
                **kwargs):
        
        super().__init__(**kwargs)
        self.Gx = Gx
        self.n_nodes = Gx.number_of_nodes()
        self.n_dims = n_dims
        self.alpha = alpha
        Gnk, A, degrees, hops = process_graph_networkx(Gx)
        
        self.degrees = degrees

        # find max hops
        self.maxhops = max(hops[hops<=self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>self.maxhops]=self.maxhops+1  # disconncted nodes are 'maxhops+1' hops away, to avoid infinity
        self.hops = hops
        print("max hops:", self.maxhops)

        self.dim_bias = np.array([1+i*4//self.n_dims for i in range(self.n_dims)])

        self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)
        
        self.random_points_generator = random_points_generator
        self.Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dims), )
        
        self.fmodel_attr = ForceClass(name='attractive_force_ahops', 
                            func=lambda *args, **kwargs: attractive_force_ahops(*args, k1=1, k2=1, **kwargs)
                            )
        self.fmodel_repl = ForceClass(name='repulsive_force_hops_exp',
                            #  func=lambda fd_model: repulsive_force_hops_exp(fd_model.D, fd_model.N, fd_model.unitD, torch.tensor(fd_model.hops).to(fd_model.device), k1=10, k2=0.9)
                            func=lambda *args, **kwargs: repulsive_force_hops_exp(*args, k1=10, k2=0.9, **kwargs)
                            )
        self.fmodel_repl_biased = ForceClass(name='force_exp_x',
                            func=lambda *args, **kwargs: force_exp_x(*args, k1=10, k2=0.9, **kwargs)
                            )
        # define force vectors
        self.forcev = rn(
            F1 = rn(tag='attr', description='attractive force', 
                    v=torch.zeros_like(self.Z)
                    ),
            F2 = rn(tag='repl1', description='repulsive force',
                    v=torch.zeros_like(self.Z)
                    ),
            F3 = rn(tag='repl2', description='biased dimension repl force',
                    v=torch.zeros_like(self.Z)
            )
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
        
        def do_pairwise(Z_source, Z_landmark):
            D = pairwise_difference(Z_source, Z_landmark) # D[i,j] = Z_lmrk[j] - Z_src[i]
            N = torch.norm(D, dim=-1)     # pairwise distance between points
            unitD = torch.where(N.unsqueeze(-1)!=0, D / N.unsqueeze(-1), torch.zeros_like(D))
            return D, N, unitD

        # Fa = torch.zeros_like(self.Z).to(device)
        # Fr = torch.zeros_like(self.Z).to(device)

        for F in self.forcev.values():
            F.v = F.v.to(device)
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
                D, N, unitD = do_pairwise(self.Z[bmask], self.Z)
                # Fa[bmask] = self.fmodel_attr(D, N, unitD, alpha_hops=self.alpha_hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                # Fr[bmask] = self.fmodel_repl(D, N, unitD, hops=self.hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force

                
                # create the biased dimension matrix
                dim_bias = torch.tensor([2+i*10//self.n_dims for i in range(self.n_dims)])
                dim_bias[dim_bias>=4] = 4 # 10% is 2, 10% is 3 and the rest is 4
                
                mat_dim_bias = self.dim_bias.reshape(1, 1, -1).repeat(D.shape[0], D.shape[1], 1)
                mat_dim_bias = mat_dim_bias<=hops[...,None] # for hops 0 and 1 is False. For the rest it is True
                mat_dim_bias = mat_dim_bias.type(torch.int)

                mat_dim_bias = torch.where(hops[...,None]>=2, mat_dim_bias/(hops[...,None]-1), torch.zeros_like(mat_dim_bias))
                mat_dim_bias *= 100
                
                self.forcev.F1.v[bmask] = self.fmodel_attr(D, N, unitD, alpha_hops=self.alpha_hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                self.forcev.F2.v[bmask] = self.fmodel_repl(D, N, unitD, hops=hops) # pass the current model (with its contained embeddings) to calculate the force
                self.forcev.F3.v[bmask] = self.fmodel_repl_biased(D, N, unitD, A=-mat_dim_bias) # pass the current model (with its contained embeddings) to calculate the force
                del D, N, unitD

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
            self.Z += self.dZ
        
            self.notify_epoch_end_callbacks(**kwargs)
        # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass

# %%
