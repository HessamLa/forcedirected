# %%
import numpy as np
import matplotlib.pyplot as plt

import networkx, networkit # this line is for typing hints used in this code
import networkx as nx
import networkit as nk
from types import *

from graphtools import load_graph_networkx, process_graph_networkx
from algorithms import get_alpha_hops, pairwise_difference
from forces import attractive_force_ahops
from forces import repulsive_force_ahops_recip_x


# %%
import torch

# %%
def generate_random_points_on_sphere(n, d, sphere_radius=1.0):
    '''
    Generates n random points in on surface of a d-dimensional sphere with radius sphere_size.
    '''
    P = np.random.normal(0, 1, size=(n, d))
    # Normalize the points to have unit norm
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms * sphere_radius
    return P
    
def generate_random_points_1(n, d, s=1):
    '''
    Generates n random points in a d-dimensional space with normal distribution.
    '''
    # Generate random points from a standard normal distribution
    P = np.random.normal(0, 1, size=(n, d))
    # Normalize the points to have unit norm
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    # Scale the points to have the desired standard deviation
    distances = np.random.normal(0, s, size=(n, 1))
    P *= distances
    return P

# random drops
def steady_drop(X, drop_rate=0.5, **kwargs):
    idx = torch.rand(X.shape) < drop_rate
    X[idx] = 0
    return X

def force_attractive(fd_model):
    return attractive_force_ahops(fd_model.D, fd_model.N, fd_model.unitD, fd_model.alpha_hops, k1=1, k2=1)

def force_repulsive(fd_model):
    return repulsive_force_ahops_recip_x(fd_model.D, fd_model.N, fd_model.unitD, torch.tensor(fd_model.hops).to(fd_model.device), k1=10, k2=0.9)

# %%
def generate_random_points(n, d, mean=0, standard_dev=1):
    """
    Generates n random points in a d-dimensional space with normal distribution.
    """
    P = np.random.normal(mean, standard_dev, size=(n, d))
    return P

class Callbacks_Base:
    def __init__(self, model, *args, **kwargs):
        # Initialize any necessary variables or objects
        pass

    def on_train_begin(self, model, *args, **kwargs):
        # Perform any necessary actions at the beginning of training
        pass

    def on_train_end(self, model, *args, **kwargs):
        # Perform any necessary actions at the end of training
        pass

    def on_epoch_begin(self, model, *args, **kwargs):
        # Perform any necessary actions at the beginning of each epoch
        pass

    def on_epoch_end(self, model, *args, **kwargs):
        # Perform any necessary actions at the end of each epoch
        pass

    def on_batch_begin(self, model, *args, **kwargs):
        # Perform any necessary actions at the beginning of each batch
        pass

    def on_batch_end(self, model, *args, **kwargs):
        # Perform any necessary actions at the end of each batch
        pass

class ForceDirected():
    def __init__(self, 
                 G:networkx, ndim:int, alpha:float, 
                 random_points_generator:callable = generate_random_points, 
                 random_drop:callable = steady_drop,
                 callbacks:list = [], 
                 **kwargs):
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.n_dim = ndim
        self.alpha = alpha

        self.random_points_generator = random_points_generator
        self.random_drop = random_drop
        
        self.callbacks = callbacks
        
        self.forces = [force_attractive, force_repulsive]

        self.initialize()
        pass
    def initialize(self, **kwargs):
        self.Z = self.random_points_generator(self.n_nodes, self.n_dim)
        G, A, self.degrees, self.hops = process_graph_networkx(self.G)
        # alpha_hops = get_alpha_hops(hops, alpha)
        self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)

        
        # if(max(hops)<2**8):
        #     self.hops = torch.tensor(hops, dtype=torch.uint8)
        # elif(max(hops)<2**16):
        #     self.hops = torch.tensor(hops, dtype=torch.uint16)
        # else:
        #     self.hops = torch.tensor(hops, dtype=torch.uint32)
        
    def calculate_distances(self):
        ''''
        Calculates the pairwise differences (D) and distances (N) and unit direction vectors (unitD).
        '''
        self.D = pairwise_difference(self.Z) # D[i,j] = Z[i] - Z[j]
        self.N = torch.norm(self.D, dim=-1)     # pairwise distance between points
        # Element-wise division with mask
        mask = self.N!=0 
        self.unitD = torch.zeros_like(self.D)   # unit direction
        self.unitD[mask] = self.D[mask] / self.N[mask].unsqueeze(-1)

    def train(self, epochs=100, device_str='cpu', **kwargs):
        kwargs['epochs'] = epochs
        
        for callback in self.callbacks:
            callback.on_train_begin(self, **kwargs)


        device = torch.device(device_str)
        kwargs['device'] = device
        self.device = device

        self.alpha_hops = torch.tensor(self.alpha_hops).to(device)
        self.degrees = torch.tensor(self.degrees).to(device) # mass of a node is equivalent to its degree
        self.Z = torch.tensor(self.Z).to(device)
        self.Z_0 = self.Z.detach().clone().to(device)
        # Fa = torch.zeros_like(Z).to(device)
        # Fr = torch.zeros_like(Z).to(device)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            kwargs['epoch'] = epoch
            # epoch begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, **kwargs)
            # batch begin
            for callback in self.callbacks:
                callback.on_batch_begin(self, epoch, **kwargs)

            self.calculate_distances()

            F = self.forces[0](self)
            for force_func in self.forces[1:]:
                F += force_func(self)

            F = self.random_drop(F, **kwargs)
            
            # find acceleration on each point a = F/m
            a = torch.where(self.degrees[:, None] != 0, F / self.degrees[:, None], torch.zeros_like(F))
            
            # finally apply relocations
            self.Z_0 = self.Z.detach().clone() # save current points
            self.Z += a

            # batch ends
            for callback in self.callbacks:
                callback.on_batch_end(self, epoch, **kwargs)
            # epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, **kwargs)

        
        for callback in self.callbacks:
            callback.on_train_end(self, **kwargs)
        pass
    
    def get_embeddings(self):
        return self.Z.cpu().numpy()

# %%
class dummyclass:
    def __init__(self) -> None:
        pass
args = dummyclass()
args.dataset = 'cora'

#reload tools
%load_ext autoreload
%autoreload 2

Gx, data = load_graph_networkx(datasetname=args.dataset, rootpath='../datasets')
G, A, degrees, hops = process_graph_networkx(Gx)

model = ForceDirected(Gx, ndim=12, alpha=0.3)
model.train(epochs=100)
# %%
