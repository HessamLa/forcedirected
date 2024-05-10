# generate_graph.py
import networkx as nx
import numpy as np



__all__ = [
    "generate_sbm",
]

def generate_sbm(n_communities, p_intra, p_inter, seed=None, **kwargs):
    # print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, gamma={gamma}, beta={beta}, mu={mu}, min_degree={min_degree}, max_degree={max_degree}, average_degree={average_degree}, min_community={min_community}, max_community={max_community}.")
    n_nodes = kwargs.get("n_nodes", 500)
    print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, ....")
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes % n_communities  # Adjust the last community size if needed
    prob_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(prob_matrix, p_intra)
    G = nx.stochastic_block_model(sizes, prob_matrix)
    return G
    # # change 'block' to 'community'
    # for node, comm in nx.get_node_attributes(G, 'block').items():
    #     G.nodes[node]['community'] = G.nodes[node].pop('block')
    # print(G.nodes[0])
    # for v in G:
    #     print(G.nodes[v])
    #     G.nodes[v]["community"]
    #     break
    # return G

    pass

if __name__ == '__main__':
    print("Example usage")
    n = 1000
    outpath = "./sbm.edgelist"
    G = generate_sbm(n_nodes=500, n_communities=5, p_intra=0.25, p_inter=0.01,seed=None)
    
    # print(f"SBM graph generated at {outpath} with node count {n} and parameters .....")