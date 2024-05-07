import networkx as nx
from networkx.generators.community import LFR_benchmark_graph

try:
    from .utilities import write_graph
except ImportError:
    from utilities import write_graph

__all__ = [
    "generate",
]

def generate(n_nodes, gamma=2.01, beta=1.051, mu=0.1,
             min_degree=None, max_degree=80, average_degree=20,
             min_community=10, max_community=100, seed=None, **kwargs):
    print(f"Generating LFR graph with parameters: n={n_nodes}, gamma={gamma}, beta={beta}, mu={mu}, min_degree={min_degree}, max_degree={max_degree}, average_degree={average_degree}, min_community={min_community}, max_community={max_community}.")
    G = LFR_benchmark_graph(
        n_nodes, gamma, beta, mu,
        min_degree=min_degree, max_degree=max_degree, average_degree=average_degree,
        min_community=min_community, max_community=max_community,
        max_iters=5000, seed=seed,
    )
    return G

if __name__ == '__main__':
    print("Example usage")
    n = 1000
    outpath = "./lfr.edgelist"
    generate(n, gamma=2.01, beta=1.051, mu=0.1, outpath=outpath)
    print(f"LFR graph generated at {outpath} with node count {n} and parameters tau1=2.01, tau2=1.051, mu=0.1, min_community=10, max_community=100.")
