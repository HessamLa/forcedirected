
# %%
from typing import Any

import numpy as np
import pandas as pd

from forcedirected.utilities.graphtools import load_graph_networkx, process_graph_networkx
from forcedirected.utilities.reportlog import ReportLog

from forcedirected.algorithms import get_alpha_hops, pairwise_difference
from forcedirected.Functions import attractive_force_ahops, repulsive_force_hops_exp

from forcedirected.Functions import DropSteadyRate, DropLinearChange, DropExponentialDimish
from forcedirected.Functions import generate_random_points
from forcedirected.utilityclasses import ForceClass, NodeEmbeddingClass
from forcedirected.utilityclasses import Model_Base, Callback_Base
import torch

class FDModel(Model_Base):
    """Force Directed Model"""
    VERSION="0004"
    DESCRIPTION="Exponential Decay repulsive(k1=10, k2=0.9), alpha={alpha}"
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
        
        self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)
        
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
        self.random_points_generator = random_points_generator
        self.embeddings = NodeEmbeddingClass(self.n_nodes, self.n_dims)
        self.embeddings.set(self.random_points_generator(self.n_nodes, self.n_dims))
        
        FDModel.DESCRIPTION = FDModel.DESCRIPTION.format(alpha=self.alpha)
        pass
    
    @property
    def Z(self):        return self.embeddings.Z
    @property
    def D(self):        return self.embeddings.D
    @property
    def N(self):        return self.embeddings.N
    @property
    def unitD(self):    return self.embeddings.unitD

    def get_embeddings(self):   return self.embeddings.Z.detach().cpu().numpy()

    def train(self, epochs=100, device=None, **kwargs):
        # train begin
        kwargs['epochs'] = epochs
        self.notify_train_begin_callbacks(**kwargs)

        # initialize train
        self.device = device
        if(device is None):
            self.device = torch.device('cpu')
        
        
        self.hops = torch.tensor(self.hops).to(device)
        self.alpha_hops = torch.tensor(self.alpha_hops).to(device)
        self.degrees = torch.tensor(self.degrees).to(device) # mass of a node is equivalent to its degree
        self.embeddings.to(device)

        self.dZ = torch.zeros_like(self.Z).to(device)
        
        def make_index_batches(idxlist:list, batch_size):
            # index = torch.randperm(self.n_nodes)
            import random
            random.shuffle(idxlist)
            for i in range(0, len(idxlist), batch_size):
                yield idxlist[i:i+batch_size]
        def do_pairwise(Z_source, Z_landmark):

            D = pairwise_difference(Z_source, Z_landmark) # D[i,j] = Z_lmrk[j] - Z_src[i]
            N = torch.norm(D, dim=-1)     # pairwise distance between points
            # Element-wise division with mask
            unitD = torch.zeros_like(D)   # unit direction
            mask = N!=0 
            unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

            print(Z_source.shape, Z_landmark.shape,'\n', D.shape, N.shape, unitD.shape)
            return D, N, unitD

        for epoch in range(epochs):
            if(self.stop_training): break
            # epoch begin
            kwargs['epoch'] = epoch
            self.notify_epoch_begin_callbacks(**kwargs)
            self.notify_batch_begin_callbacks(**kwargs)
            
            print(f'Epoch {epoch+1}/{epochs}')
            Z = self.embeddings.Z.to(self.device)
            dZ = torch.zeros_like(self.Z).to(self.device)
            Fa = torch.zeros_like(self.Z).to(self.device)
            Fr = torch.zeros_like(self.Z).to(self.device)
            F = torch.zeros_like(self.Z).to(self.device)
            for i, bmask in enumerate (make_index_batches(list(range(Z.shape[0])), batch_size=1000)):
                print(bmask[:10])
                print(bmask[-10:])
                # batch begin
                kwargs['batch'] = i
                print(f'   Batch {i+1}/{self.n_nodes//1000}', end=' ')
                self.notify_batch_begin_callbacks(**kwargs)
                
                ###################################
                # this is the forward pass
                D, N, unitD = do_pairwise(Z[bmask], Z)
                print(D.shape, N.shape, unitD.shape)
                Fa[bmask] = self.fmodel_attr(D, N, unitD, alpha_hops=self.alpha_hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                Fr[bmask] = self.fmodel_repl(D, N, unitD, hops=self.hops[bmask,:]) # pass the current model (with its contained embeddings) to calculate the force
                F[bmask] = self.random_drop(Fa[bmask]+Fr[bmask], **kwargs)
                # force = torch.norm(Fa, dim=1)
                # fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
                # print(f'attr: {fsum:,.3f}({fmean:.3f})  ', end='')

                # force = torch.norm(Fr, dim=1)
                # fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
                # print(f'repl: {fsum:,.3f}({fmean:.3f})  ' , end='')
                # Fsum = Fa+Fr
                # force = torch.norm(Fsum, dim=1)
                # fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
                # print(f'sum: {fsum:,.3f}({fmean:.3f})  ' , end='')
                
                ###################################
                # finally calculate the gradient and udpate the embeddings
                # find acceleration on each point a = F/m. And X-X0 = a*t^2, Assume X0 = 0 and t = 1
                dZ[bmask] = torch.where(self.degrees[bmask, None] != 0, 
                                        F[bmask] / self.degrees[bmask, None], 
                                        torch.zeros_like(F[bmask]))
            print(Z.shape, dZ.shape)
            self.dZ = dZ
            relocs = torch.norm(self.dZ, dim=1)
            # print(f'relocs: {relocs.sum():,.3f}({relocs.mean():.5f})')
            # self.embeddings.update(self.dZ)
            self.embeddings.Z += dZ
        
            ###################################

            # batch ends
            self.notify_batch_end_callbacks(**kwargs)
            self.notify_epoch_end_callbacks(**kwargs)
        # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass
