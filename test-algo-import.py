# %%
from nodeforce import *
import numpy as np
import torch

hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])

alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)
alpha_hops = torch.tensor(alpha_hops)

a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
a = torch.tensor(a).float()
D = pairwise_difference(a)
N = torch.norm(D, dim=-1)

# Element-wise division with mask
unitD = torch.zeros_like(D)
mask = N!=0 # Create a mask for non-zero division
unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

print("\nTest: repulsive_force_exp(.)")
t = repulsive_force_exp(D, N, unitD)
print(t)

print("\nTest: repulsive_force_recip_x(.)")
t = repulsive_force_recip_x(D, N, unitD)
print(t)

print("\nTest: repulsive_force_ahops_recip_x(.)")
t = repulsive_force_ahops_recip_x(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True)
print(t)

print("\nTest: attractive_force_ahops(.)")
t = attractive_force_ahops(D, N, unitD, alpha_hops)
print(t)
