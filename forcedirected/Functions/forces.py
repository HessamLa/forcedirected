import numpy as np
import torch

from ..algorithms import get_alpha_hops, pairwise_difference


def repulsive_force_exp(D, N, unitD, k1=1, k2=1, return_sum=True):
    """
    f(x) = k1*exp(-x/k2)
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N
    k1 is amplitude factor, scalar: k1*f(x) 
    k2 is decaying factor factor, scalar: f(x/k2)
    """
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1 * torch.exp(-N/k2)

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F
def repulsive_force_hops(D, N, unitD, hops, k1=1, k2=1, return_sum=True):
    """
    linear repulsive force proportional to hops distance
    f(x) = k1*h(1-x/k2) if x<k2, else 0
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N

    hops (n,n) is the coefficient for each hop-distant neighbor
    k1 is amplitude factor, scalar: k1*f(x) 
    k2 is decaying factor factor, scalar: f(x/k2)
    """
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = torch.where(N<k2,   k1/n * hops * (1-N/k2),
                            torch.zeros_like(N))

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F

def repulsive_force_exp_hops_x(D, N, unitD, hops, k1=1, k2=1, k3=1, return_sum=True):
    """
    f(x) = k1*exp((h-1)/k2)*exp(-x/k3) = k1*exp( (h-1)/k2 - x/k3 )
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N

    hops (n,n) is the coefficient for each hop-distant neighbor
    k1 is amplitude factor, scalar: k1*f(x) 
    k2 and k3 are decaying factors, scalar: f(x/k2)
    """
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1/n * torch.exp((hops-1)/k2-N/k3)

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F

def repulsive_force_exp_hops_x_2(D, N, unitD, hops, k1=1, k2=1, k3=1, return_sum=True):
    """
    f(x) = k1*(1-exp(-(h-1)/k2))*exp(-x/k3) = k1*exp( (h-1)/k2 - x/k3 )
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N
    effectively 0 repulsion between neighbors

    hops (n,n) is the coefficient for each hop-distant neighbor
    k1 is amplitude factor, scalar: k1*f(x) 
    k2 and k3 are decaying factors, scalar: f(x/k2)
    """
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1/n * (1-torch.exp((1-hops)/k2))*torch.exp(-N/k3)

    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

    return F

def repulsive_force_hops_exp(D, N, unitD, hops, k1=1, k2=1, return_sum=True):
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
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    F = k1 * hops * torch.exp(-N/k2)

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

def repulsive_force_ahops_recip_x(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True):
    """
    Calculates repulsive force by inverse of the distance
    f(x) = k1*alpha^(h-1)/(x^k2)

    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N
    alpha, scalar is the coefficient for each hop-distant neighbor

    alpha_hops (n,n) is the coefficient for each hop-distant neighbor
        alpha_hops[i,j] = alpha^(h-1)/n where h is hops-distance between i,j
    k1 is amplitude factor, scalar.
    k2 is power factor, scalar.
    """
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate forces amplitude
    x_to_k2 = torch.pow(N, k2) # raise x to power of k2. x is distance between two points
    F = k1 * alpha_hops * torch.where(x_to_k2!= 0, k1 / x_to_k2, torch.zeros_like(x_to_k2))
    
    # apply negative direction
    F = -unitD * F.unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i
        # print("After sum")
        # print("F[2]\n", F[2])
    return F
    
# def test_repulsive_force_ahops_recip_x():
#     hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
#     alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

#     a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
#     a = torch.tensor(a).float()
#     D = pairwise_difference(a)
#     N = torch.norm(D, dim=-1)

#     # Element-wise division with mask
#     unitD = torch.zeros_like(D)
#     mask = N!=0 # Create a mask for non-zero division
#     unitD[mask] = D[mask] / N[mask].unsqueeze(-1)
    
#     t = repulsive_force_ahops_recip_x(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True)
#     print(t)

# test_repulsive_force_ahops_recip_x()
def attractive_force_ahops(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True):
    """
    D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N

    alpha_hops (n,n) is the coefficient for each hop-distant neighbor
        alpha_hops[i,j] = alpha^(h-1)/n where h is hops-distance between i,j
    k1 is amplitude factor: k1*f(x)
    k2 is intensity factor factor over distance: f(x^k2)
    """
    # calculate the amplitude
    F = k1 * alpha_hops * N
    
    # finally apply the direction
    F = unitD * F.unsqueeze(-1)

    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
    
def attractive_force_base(D, N, unitD, e_hops, k1=1, k2=1, return_sum=True):
    """
    D (n,n,d) is the pairwaise difference (force). D[i,j]=Z[j]-Z[i], from i to j
    N (n,n) is norm of each pairwise diff element, i.e x.
    unitD (n,n,d) is the unit direction, D/N

    e_hops (n,n) is the coefficient for each hop-distant neighbor
        e_hops[i,j] = e^-(h-1) where h is hops-distance between i,j
    k1 is amplitude factor: k1*f(x)
    k2 is intensity factor factor over distance: f(x^k2)
    """
    # calculate the amplitude
    F = k1 * e_hops * N
    
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

