import torch
from forcedirected.algorithms import pairwise_difference


class NodeEmbeddingClass:
    def __init__(self, n_nodes, n_dims) -> None:
        self.shape = (n_nodes, n_dims)
        self.Z = None
        
        pass

    def update(self, dZ=0):
        self.Z += dZ
        ''''
        Calculates the pairwise differences (D) and distances (N) and unit direction vectors (unitD).
        '''
        self.D = pairwise_difference(self.Z) # D[i,j] = Z[i] - Z[j]
        self.N = torch.norm(self.D, dim=-1)     # pairwise distance between points
        # Element-wise division with mask
        self.unitD = torch.zeros_like(self.D)   # unit direction
        mask = self.N!=0 
        self.unitD[mask] = self.D[mask] / self.N[mask].unsqueeze(-1)
    
    def set(self, Z):
        if(type(Z) is not torch.Tensor):
            Z = torch.tensor(Z)
        self.Z = Z
        self.update()
        pass

    def to(self, device):
        self.Z = self.Z.to(device)
        self.D = self.D.to(device)
        self.N = self.N.to(device)
        self.unitD = self.unitD.to(device)

