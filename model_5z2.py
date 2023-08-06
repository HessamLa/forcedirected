
# %%
from typing import Any

import numpy as np
import pandas as pd
import functools
import math
import sys

from forcedirected.utilities.graphtools import load_graph_networkx, process_graph_networkx
from forcedirected.utilities.reportlog import ReportLog
from forcedirected.utilities import batchify

from forcedirected.algorithms import get_alpha_hops, pairwise_difference
from forcedirected.Functions import attractive_force_ahops, repulsive_force_hops_exp

from forcedirected.Functions import DropSteadyRate, DropLinearChange, DropExponentialDimish
from forcedirected.Functions import generate_random_points
from forcedirected.utilityclasses import ForceClass, NodeEmbeddingClass
from forcedirected.utilityclasses import Model_Base, Callback_Base
import torch

# Function to check available GPU memory
def check_gpu_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

class FDModel(Model_Base):
    """Force Directed Model"""
    VERSION="0005z2"
    DESCRIPTION="Same as Model_5. dZ+=dZ^2 to induce skewness in relocations. Exponential Decay repulsive(k1=10, k2=0.9), alpha={alpha}"
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
        self.hops = hops

        # find max hops
        self.maxhops = max(hops[hops<=self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>self.maxhops]=self.maxhops+1  # disconncted nodes are 'maxhops+1' hops away, to avoid infinity
        print("max hops:", self.maxhops)
        
        # test 1
        self.alpha_hops = 1/self.n_nodes*np.power(self.alpha, self.hops-1)
        np.fill_diagonal(self.alpha_hops, 0)
        # test 2
        # self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)
        
        self.random_points_generator = random_points_generator
        self.Z = torch.tensor(self.random_points_generator(self.n_nodes, self.n_dims), )
        
        self.fmodel_attr = ForceClass(name='attractive_force_ahops', 
                            func=lambda *args, **kwargs: attractive_force_ahops(*args, k1=1, k2=1, **kwargs)
                            )
        self.fmodel_repl = ForceClass(name='repulsive_force_hops_exp',
                            #  func=lambda fd_model: repulsive_force_hops_exp(fd_model.D, fd_model.N, fd_model.unitD, torch.tensor(fd_model.hops).to(fd_model.device), k1=10, k2=0.9)
                            func=lambda *args, **kwargs: repulsive_force_hops_exp(*args, k1=10, k2=0.9, **kwargs)
                            )
        # To be used like the following
        # result_F = self.fmodel_attr(self) # pass the current model (with its contained embeddings) to calculate the force
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=0.5)
        
        self.statsdf = pd.DataFrame()
        # initialize embeddings
        
        FDModel.DESCRIPTION = FDModel.DESCRIPTION.format(alpha=self.alpha)
        pass
    
    def get_embeddings(self):   return self.Z.detach().cpu().numpy()

    @torch.no_grad()
    def train(self, epochs=100, device='cpu', row_batch_size='auto', **kwargs):
        # train begin
        kwargs['epochs'] = epochs
        self.notify_train_begin_callbacks(**kwargs)

        self.hops = torch.tensor(self.hops).to(device)
        self.alpha_hops = torch.tensor(self.alpha_hops).to(device)
        self.degrees = torch.tensor(self.degrees).to(device) # mass of a node is equivalent to its degree

        self.Z = self.Z.to(device)
        
        def do_pairwise(Z_source, Z_landmark):
            D = pairwise_difference(Z_source, Z_landmark) # D[i,j] = Z_lmrk[j] - Z_src[i]
            N = torch.norm(D, dim=-1)     # pairwise distance between points
            unitD = torch.where(N.unsqueeze(-1)!=0, D / N.unsqueeze(-1), torch.zeros_like(D))
            return D, N, unitD

        self.dZ = torch.zeros_like(self.Z).to(device)
        self.Fa = torch.zeros_like(self.Z).to(device)
        self.Fr = torch.zeros_like(self.Z).to(device)
        F = torch.zeros_like(self.Z).to(device)

        
        max_batch_size = self.n_nodes
        if(row_batch_size  == 'auto'): 
            row_batch_size = max_batch_size
        else: 
            row_batch_size = min(row_batch_size, max_batch_size)
        
        for epoch in range(epochs):
            if(self.stop_training): break

            # epoch begin
            kwargs['epoch'] = epoch
            self.notify_epoch_begin_callbacks(**kwargs)

            def optimize_batch_size(func):
                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    
                    row_batch_size = kwargs.get('row_batch_size')
                    max_batch_size = kwargs.get('max_batch_size')
                    if row_batch_size<max_batch_size: # row_batch_size was the last successful batch size
                        min_batch_size = row_batch_size
                        # upgrade row_batch_size
                        row_batch_size = (row_batch_size + max_batch_size + 1)//2
                        # print(f"Test new batch size ({min_batch_size},{row_batch_size},{max_batch_size})", file=sys.stderr)
                    else:                        # either row_batch_size was the last unsuccessful batch size or it is to be determined
                        min_batch_size = None
                    

                    while True:
                        try:
                            kwargs['row_batch_size'] = row_batch_size
                            kwargs['max_batch_size'] = max_batch_size
                            return func(self, *args, **kwargs)
                            
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                if(row_batch_size == 1):
                                    print("One row is too big for GPU memory. Please reduce the number of dimensions or the number of nodes.", file=sys.stderr)
                                    raise e
                                if(min_batch_size is None): # a successful batch size has not been found yet
                                    max_batch_size = row_batch_size-1
                                    row_batch_size = max_batch_size//2
                                else:           # a successful batch size is in range [[min_batch_size, row_batch_size))
                                    max_batch_size = row_batch_size-1
                                    row_batch_size = (min_batch_size + max_batch_size)//2
                                # print(f"REDUCE row_batch_size: {row_batch_size}, max_batch_size: {max_batch_size}")
                            else:
                                print(f"Exception: {e}")
                                raise e
                return wrapper
            
            @optimize_batch_size
            def run_batches(obj, row_batch_size=row_batch_size, max_batch_size=max_batch_size, **kwargs):
                kwargs['batches'] = int(obj.Z.shape[0]//row_batch_size +0.5)
                for i, bmask in enumerate (batchify(list(range(obj.Z.shape[0])), batch_size=row_batch_size)):
                    # batch begin
                    kwargs['batch'] = i+1
                    kwargs['row_batch_size'] = row_batch_size
                    kwargs['max_batch_size'] = max_batch_size
                    self.notify_batch_begin_callbacks(**kwargs)
                    
                    ###################################
                    # this is the forward pass
                    
                    D, N, unitD = do_pairwise(obj.Z[bmask], obj.Z)
                    # print(D.shape, N.shape, unitD.shape)
                    obj.Fa[bmask] = obj.fmodel_attr(D, N, unitD, alpha_hops=obj.alpha_hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                    obj.Fr[bmask] = obj.fmodel_repl(D, N, unitD, hops=obj.hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                    del D, N, unitD

                    # batch ends
                    self.notify_batch_end_callbacks(**kwargs)
                return row_batch_size, max_batch_size
            
            row_batch_size, max_batch_size = run_batches(self, row_batch_size=row_batch_size, max_batch_size=max_batch_size, **kwargs)
            kwargs['batch_size'] = row_batch_size
            kwargs['batches'] = int(self.Z.shape[0]//row_batch_size +0.5)
            
            ### JUST A HACK, We need this hack to avoid indexing problem with the statslog callback
            self.fmodel_attr.F = self.Fa
            self.fmodel_repl.F = self.Fr

            F = self.random_drop(self.Fa+self.Fr, **kwargs)
            ###################################
            # finally calculate the gradient and udpate the embeddings
            # find acceleration on each point a = F/m. And X-X0 = a*t^2, Assume X0 = 0 and t = 1
            self.dZ = torch.where(self.degrees[..., None] != 0, 
                                    F / self.degrees[..., None], 
                                    torch.zeros_like(F))
            
            # skew each row vector towards the direction of the row's max


            # to induce skewness in relocations, does it help?
            # calculate skewed direction
            dZ_skewed = self.dZ*torch.abs(self.dZ)/torch.abs(self.dZ).max(dim=-1, keepdim=True)[0] 
            dZ_snorm = dZ_skewed.norm(dim=1, keepdim=True)
            # get the unit direction for skewed direction
            dZ_sunit = torch.where(dZ_snorm!=0, dZ_skewed / dZ_snorm, torch.zeros_like(dZ_snorm))

            # Align each row of dZ with corresponding row of dZ_skewed
            self.dZ = dZ_sunit * torch.norm(self.dZ, dim=-1, keepdim=True)
            
            self.Z += self.dZ
        
            #### FIX ME. ForceClass should be able to work with batch methods
            ###################################

            self.notify_epoch_end_callbacks(**kwargs)
        # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass

# %%
