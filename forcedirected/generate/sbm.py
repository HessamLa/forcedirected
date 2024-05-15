# generate_graph.py
import networkx as nx
import numpy as np


__all__ = [
    "sbm",
]

def generate_sizes(n_nodes, n_communities, equal_size=True, seed=None):
    if(equal_size):
        sizes = [n_nodes//n_communities]*n_communities
        for i in range(n_nodes%n_communities):
            sizes[i] += 1
    else:
        np.random.seed(seed)
        # Generate n-1 random numbers between 0 and C
        sizes = np.random.uniform(0, n_communities, size=n_nodes-1)    
        # Calculate the remaining value to make the sum equal to C
        remaining = n_communities - np.sum(sizes)
        sizes = np.append(sizes, remaining)
        np.random.shuffle(sizes)
    assert len(sizes) == n_communities
    assert sum(sizes) == n_nodes
    return sizes

def sbm(n_nodes, n_communities, p_intra, p_inter, seed=None, **kwargs):
    # print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, gamma={gamma}, beta={beta}, mu={mu}, min_degree={min_degree}, max_degree={max_degree}, average_degree={average_degree}, min_community={min_community}, max_community={max_community}.")
    print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, ....")
    # set seed
    # generate size of each community

    sizes = generate_sizes(n_nodes, n_communities, seed=seed)
    
    np.random.seed(seed)
    prob_matrix = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(prob_matrix, p_intra)
    G = nx.stochastic_block_model(sizes=sizes, p=prob_matrix, seed=seed)
    # for node, comm in nx.get_node_attributes(G, 'block').items():
    #     G.nodes[node]['community'] = G.nodes[node].pop('block')
    return G
    
if __name__ == '__main__':
    print("Example usage")
    n = 1000
    outpath = "./sbm.edgelist"
    G = sbm(n_nodes=500, n_communities=5, p_intra=0.25, p_inter=0.01,seed=None)
    
    # print(f"SBM graph generated at {outpath} with node count {n} and parameters .....")