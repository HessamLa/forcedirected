from .utilities.graphtools import load_graph_networkx
from .utilities.graphtools import process_graph_networkx
from .utilities.graphtools import remove_random_edges
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
