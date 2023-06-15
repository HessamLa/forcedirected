############################################################################
# EMBEDDING 
############################################################################
# from .embedder import embed_forcedirected

"""
Returns a matrix such that M[i,j] = P[j]-P[i].
Summing row i corresponds with force from i to others
Summing col j corresponds with forces from other to j
"""
from .algorithms import pairwise_difference
from .algorithms import get_alpha_to_hops
######################################################################
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
from .algorithms import get_alpha_hops
######################################################################
"""
f(x) = k1*exp(-x/k2)
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar: k1*f(x) 
k2 is decaying factor factor, scalar: f(x/k2)
"""
from .algorithms import repulsive_force_exp
######################################################################
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
from .algorithms import repulsive_force_hops_exp
######################################################################
"""
Calculates repulsive force by inverse of the distance
f(x) = k1/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
from .algorithms import repulsive_force_recip_x
######################################################################
"""
Calculates repulsive force by inverse of the distance
f(x) = k1*alpha^(h-1)/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
alpha, scalar is the coefficient for each hop-distant neighbor
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
from .algorithms import repulsive_force_ahops_recip_x
######################################################################
"""
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
alpha_hops (n,n) is the coefficient for each hop-distant neighbor
k1 is amplitude factor: k1*f(x)
k2 is decaying factor factor over distance: f(x/k2)
"""
from .algorithms import attractive_force_ahops