import networkx as nx

__all__ = [
    "generate_sbm",
]

def generate_sbm(n_nodes, # other parameters
            seed=None, **kwargs):
    # print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, gamma={gamma}, beta={beta}, mu={mu}, min_degree={min_degree}, max_degree={max_degree}, average_degree={average_degree}, min_community={min_community}, max_community={max_community}.")
    print(f"Generating graph using Stochastic Block Model with parameters: n={n_nodes}, ....")
    # CODE HERE
    # G = sbm_graph(
    #     ...
    # )
    # return G
    pass

if __name__ == '__main__':
    print("Example usage")
    n = 1000
    outpath = "./sbm.edgelist"
    generate_sbm(n, gamma=2.01, beta=1.051, mu=0.1, outpath=outpath)
    print(f"SBM graph generated at {outpath} with node count {n} and parameters .....")
