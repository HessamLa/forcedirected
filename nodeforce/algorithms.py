import torch
import numpy as np
############################################################################
# EMBEDDING 
############################################################################
"""
Returns a matrix such that M[i,j] = P[j]-P[i].
Summing row i corresponds with force from i to others
Summing col j corresponds with forces from other to j
"""
def pairwise_difference(P: torch.tensor) -> torch.tensor:
    n, d = P.size()
    # Expand dimensions of P to create row-wise repetitions
    P_row = P.unsqueeze(1)    
    # Expand dimensions of P to create column-wise repetitions
    P_col = P.unsqueeze(0)
    # Compute the matrix M
    # print(P_row.size(), P_col.size())
    D = P_col - P_row
    return D

"""
hops is numpy.narray
"""
def get_alpha_to_hops (hops, alpha: float):
    hops=hops.astype(float)
    # alpha^(h-1)
    alpha_to_hops = np.power(alpha, hops-1, out=np.zeros_like(hops), where=hops!=0)
    return alpha_to_hops
def test_get_alpha_to_hops ():
    print("Test get_alpha_to_hops(.)")
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])    
    a = get_alpha_to_hops(hops, alpha=0.3)
    print(a)
# test_get_alpha_to_hops()

"""
Given an array of hops, it calculates the coefficient of each corresponding position
and returns a corresponding array.

For element if hops[i] is h, result[i] is as follows:
result[i] = 1/|N_h| * alpha^(h-1)

|N_h| is the total number of elements with value h in the array.
Essentially it means the total number of neighbors at minimum h-hop distance.
example:
hops   = [1  , 3  , 1  , 2   , 2   , 2   , 4   ]
result = [0.5, 1.0, 0.5, 0.33, 0.33, 0.33, 1.0 ]

Actually:
|N_1| = 2, |N_2| = 3, |N_3| = 1, |N_4| = 1
"""
def get_alpha_hops (hops, alpha: float):
    hops=hops.astype(float)
    
    # alpha^(h-1)
    alpha_to_hops = get_alpha_to_hops(hops, alpha)
    
    unique_values, counts = np.unique(hops, return_counts=True) # number of nodes at i-hop distance
    # |N_h| per entry. For any (i,j) with h-hops distance, abs_Nh[i,j]=s|N_h|
    abs_Nh = counts[np.searchsorted(unique_values, hops)] 
    
    result = alpha_to_hops/abs_Nh
    return result

def test_get_alpha_hops ():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)
    print(alpha_hops)
# test_get_alpha_hops()
# print(hops)
# print(alpha_hops)

"""
f(x) = k1*exp(-x/k2)
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar: k1*f(x) 
k2 is decaying factor factor, scalar: f(x/k2)
"""
def repulsive_force_exp(D, N, unitD, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1/n * torch.exp(-N/k2)

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F

"""
f(x) = k1*h*exp(-x/k2)
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N

hops (n,n) is the coefficient for each hop-distant neighbor
    alpha_hops[i,j] = alpha^(h-1) where h is hops-distance between i,j
k1 is amplitude factor, scalar: k1*f(x) 
k2 is decaying factor factor, scalar: f(x/k2)
"""
def repulsive_force_hops_exp(D, N, unitD, hops, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1/n * hops * torch.exp(-N/k2)

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F

"""
Calculates repulsive force by inverse of the distance
f(x) = k1/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
def repulsive_force_recip_x(D, N, unitD, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    x_to_k2 = torch.pow(N, k2) # raise x to power of k2
    F = k1/n * torch.where(x_to_k2!= 0, k1 / x_to_k2, torch.zeros_like(x_to_k2))

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)

    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F

"""
Calculates repulsive force by inverse of the distance
f(x) = k1*alpha^(h-1)/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
alpha, scalar is the coefficient for each hop-distant neighbor

alpha_hops (n,n) is the coefficient for each hop-distant neighbor
    alpha_hops[i,j] = alpha^(h-1) where h is hops-distance between i,j
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
def repulsive_force_ahops_recip_x(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    x_to_k2 = torch.pow(N, k2) # raise x to power of k2. x is distance between two points
    F = k1/n * alpha_hops * torch.where(x_to_k2!= 0, k1 / x_to_k2, torch.zeros_like(x_to_k2))
    
    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
def test_repulsive_force_ahops_recip_x():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

    a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
    a = torch.tensor(a).float()
    D = pairwise_difference(a)
    N = torch.norm(D, dim=-1)

    # Element-wise division with mask
    unitD = torch.zeros_like(D)
    mask = N!=0 # Create a mask for non-zero division
    unitD[mask] = D[mask] / N[mask].unsqueeze(-1)
    
    t = repulsive_force_ahops_recip_x(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True)
    print(t)

# test_repulsive_force_ahops_recip_x()

"""
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N

alpha_hops (n,n) is the coefficient for each hop-distant neighbor
    alpha_hops[i,j] = alpha^(h-1) where h is hops-distance between i,j
k1 is amplitude factor: k1*f(x)
k2 is intensity factor factor over distance: f(x^k2)
"""
def attractive_force_ahops(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True):
    # calculate the amplitude
    F = k1 * alpha_hops * N
    
    # finally apply the direction
    F = unitD * F.unsqueeze(-1)

    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
    
if __name__ == "__main__":
    import networkx as nx

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


