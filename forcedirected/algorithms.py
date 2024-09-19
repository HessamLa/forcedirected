import torch
import numpy as np
############################################################################
# EMBEDDING 
############################################################################
def pairwise_difference(P0: torch.tensor, P1: torch.tensor = None) -> torch.tensor:
    """
    Calculates pairwise difference between points from P0 to P1.
    If P1 is None, do pairwise difference between all pairs from P0.

    Returns a matrix such that M[i,j] = P_1[j]-P_0[i].
    Summing row i corresponds with force from i to others
    Summing col j corresponds with forces from other to j
    """
    if(P1 is None):
        P1 = P0
    # Expand dimensions of P to create row-wise repetitions
    P_row = P0.unsqueeze(1)    
    # Expand dimensions of P to create column-wise repetitions
    P_col = P1.unsqueeze(0)
    # Compute the matrix M
    # print(P_row.size(), P_col.size())
    D = P_col - P_row
    return D

def pairwise_difference_masked(P0: torch.tensor, P1: torch.tensor = None, indices: torch.tensor = None) -> torch.tensor:
    """
    Calculates pairwise difference between points from P0 to P1 based on the indices.
    If P1 is None, do pairwise difference between all pairs from P0.

    

    Returns a matrix such that M[i,j] = P_1[j]-P_0[i].
    Summing row i corresponds with force from i to others
    Summing col j corresponds with forces from other to j
    """
    if(P1 is None):
        P1 = P0
    raise NotImplementedError("pairwise_difference_masked(.) is not implemented yet.")
    return D
"""
hops is numpy.narray
"""
def get_alpha_to_hops (hops, alpha: float):
    hops=hops.astype(float)
    # alpha^(h-1)
    alpha_to_hops = np.power(alpha, hops-1, out=np.zeros_like(hops), where=hops!=0)
    return alpha_to_hops
# def test_get_alpha_to_hops ():
#     print("Test get_alpha_to_hops(.)")
#     hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])    
#     a = get_alpha_to_hops(hops, alpha=0.3)
#     print(a)
# test_get_alpha_to_hops()

def get_alpha_hops (hops, alpha: float):
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
    hops=hops.astype(float)
    
    # alpha^(h-1)
    alpha_to_hops = get_alpha_to_hops(hops, alpha)
    
    unique_values, counts = np.unique(hops, return_counts=True) # number of nodes at i-hop distance
    # |N_h| per entry. For any (i,j) with h-hops distance, abs_Nh[i,j]=s|N_h|
    abs_Nh = counts[np.searchsorted(unique_values, hops)] 
    
    result = alpha_to_hops/abs_Nh
    return result

# def test_get_alpha_hops ():
#     hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
#     alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)
#     print(alpha_hops)

# test_get_alpha_hops()
# print(hops)
# print(alpha_hops)


