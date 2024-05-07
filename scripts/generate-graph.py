#%%
import networkx as nx
import networkit as nk

import os, sys
import argparse
from typing import Any, Union

# %%
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a graph")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="The type of graph to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for generating the graph",
    )
    parser.add_argument(
        "--transitive_closure",
        action="store_true",
        help="Whether to compute the transitive closure of the graph",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="The directory to save the graph to",
    )
    return parser.parse_args()



# %%
import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

print("Imported all libraries successfully.", flush=True)
# %%
# tau1, gamma : Power law exponent for the degree distribution
# tau2, beta  : Power law exponent for the community size distribution

def genrate_lfr_graph(n, tau1=2.01, tau2=1.051, mu=0.1, 
                      min_degree=None, max_degree=80, average_degree=None,
                      min_community=10, max_community=100):
    G = LFR_benchmark_graph(
                        n, tau1, tau2, mu, 
                        min_degree=min_degree, max_degree=max_degree, average_degree=average_degree,
                        min_community=min_community, max_community=max_community,
                        max_iters=5000, seed = 0,
                        )
    return G  

# Step 2: Compute the APSP matrix using Floyd-Warshall algorithm
def compute_apsp_matrix(G):
    """
    Returns a matrix where the element at (i, j) is the shortest path length from node i to node j.
    If no path exists, the entry will be np.inf.
    """
    length = dict(nx.floyd_warshall(G))
    n = G.number_of_nodes()
    apsp_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if j in length[i]:
                apsp_matrix[i][j] = length[i][j]
    return apsp_matrix

def convert_communities_to_labels(n, communities):
    """ Convert a set of communities (where each community is a set of nodes) to a label list. """
    labels = [0] * n
    for idx, community in enumerate(communities):
        for node in community:
            labels[node] = idx
    return labels

# Conductance, Cut Ratio are not directly available in NetworkX, but we can compute them manually:
def compute_conductance_cut_ratio(G, communities):
    conductance_list = []
    cut_ratio_list = []
    for community in communities:
        # Nodes in the community
        comm = set(community)
        # Nodes outside the community
        non_comm = set(G.nodes) - comm
        
        # Edges within the community
        internal_edges = G.subgraph(comm).number_of_edges()
        # Edges from community to outside
        boundary_edges = len([e for e in G.edges(comm) if e[1] in non_comm])

        # Total possible edges between community and outside
        possible_boundary_edges = len(comm) * len(non_comm)

        # Conductance calculation
        if boundary_edges + internal_edges > 0:
            conductance = boundary_edges / (boundary_edges + internal_edges)
        else:
            conductance = 0
        conductance_list.append(conductance)
        
        # Cut ratio calculation
        if possible_boundary_edges > 0:
            cut_ratio = boundary_edges / possible_boundary_edges
        else:
            cut_ratio = 0
        cut_ratio_list.append(cut_ratio)

    return conductance_list, cut_ratio_list

# Function to calculate normalized cut
def normalized_cut(G, communities):
    ncut_values = []
    for community in communities:
        A = set(community)
        B = set(G.nodes) - A
        cut = sum([G[u][v]['weight'] if 'weight' in G[u][v] else 1 for u, v in G.edges() if (u in A and v in B) or (u in B and v in A)])
        assoc_A = sum([G[u][v]['weight'] if 'weight' in G[u][v] else 1 for u, v in G.edges() if u in A or v in A])
        assoc_B = sum([G[u][v]['weight'] if 'weight' in G[u][v] else 1 for u, v in G.edges() if u in B or v in B])
        ncut = (cut / float(assoc_A)) + (cut / float(assoc_B))
        ncut_values.append(ncut)
    return np.mean(ncut_values)  # Returning the average normalized cut across all communities

# Function to calculate silhouette score
def calculate_silhouette(G, communities):
    # Create a list of node labels based on their community
    community_labels = {node: i for i, community in enumerate(communities) for node in community}
    labels = [community_labels[node] for node in G.nodes()]
    
    # Create the adjacency matrix as the feature set
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0)
    
    # Calculate silhouette score
    return silhouette_score(A, labels, metric='precomputed')

n=1000
mus = [round(0.1*i, 1) for i in range(1, 11)]

graphs = {}
for mu in mus:
    min_degree=1
    average_degree = 20
    while min_degree<=20:
        try:
            # G = genrate_lfr_graph(n=n, mu=mu, min_degree=min_degree)
            G = genrate_lfr_graph(n=n, mu=mu, average_degree=average_degree)
            graphs[mu] = G

            # apsp_matrix = compute_apsp_matrix(G)

            # print(f"\nGraph with n={n} mu={mu}, min_degree={min_degree} generated successfully.")
            original_communities = {frozenset(G.nodes[v]["community"]) for v in G}
            # print("Communities original:", len(original_communities))
            lv_communities = nx.community.louvain_communities(G)
            # print("Communities with nx.community.louvain_communities:", len(lv_communities))

            # calculate NMI
            # CODE HERE ...
            
            # Convert communities to labels for NMI calculation
            original_labels = convert_communities_to_labels(n, original_communities)
            lv_labels = convert_communities_to_labels(n, lv_communities)

            # Calculate NMI
            nmi = normalized_mutual_info_score(original_labels, lv_labels)
            
            # calculate modularity
            modularity = nx.community.modularity(G, original_communities)
            avg_clustering = nx.average_clustering(G)
            conductance, cut_ratio = compute_conductance_cut_ratio(G, original_communities)
            ncut = normalized_cut(G, original_communities)
            silhouette = calculate_silhouette(G, original_communities)

            coverage, performance = nx.algorithms.community.quality.partition_quality(G, original_communities)

            print(f"mu={mu} |NMI={nmi:.4f} |Modularity={modularity:.4f} "
                  f"|Average Clustering={avg_clustering:.4f} "
                  f"|Conductance={sum(conductance) / len(conductance):.4f} "
                  f"|Avg Cut Ratio={sum(cut_ratio) / len(cut_ratio):.4f} "
                  f"|Normalized Cut={ncut:.4f} "
                  f"|Silhouette Score={silhouette:.4f}"
                  f"|Coverage={coverage:.4f} "
                  f"|Performance={performance:.4f}")
            
            break
        except Exception as e:
            print(e)
            min_degree += 1
            continue

# # save the graphs
# paths={}
# for mu, G in graphs.items():
# 	filename=...
#     filepath=os.path.join(out_dir, filename)

# 	# save edgelist to filename
# 	...
# 	paths[mu] = filepath

# # verify by drawing
# for mu, path in paths.items():
#     G = nx.read_edgelist(path)
#     plt.figure()
#     nx.draw(G, node_size=0.5, node_color="red", edge_color="grey", width=0.5)

# get the community ids
for mu, G in graphs.items():
    print(f"\nmu={mu}")
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    print("Communities original:", len(communities))
    communities = nx.community.louvain_communities(G)
    print("Communities with nx.community.louvain_communities:", len(communities))
    # visualize
    plt.figure()
    nx.draw(G, node_size=0.5, node_color="red", edge_color="grey", width=0.5)
# %%
