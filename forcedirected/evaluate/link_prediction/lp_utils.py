# %%
import networkx as nx
import networkit as nk
import numpy as np
import random
from itertools import combinations
from sklearn.model_selection import train_test_split  # Import train_test_split function
from typing import Union

npconcat = lambda *x, axis=0: np.concatenate((x), axis=axis)

def get_asps_matrix(G):
    # Convert to Networkit graph if it's a NetworkX graph
    if isinstance(G, nx.Graph):
        G = nk.nxadapter.nx2nk(G, weightAttr=None)
    asps = nk.distance.APSP(G)
    asps.run()
    # all pair hops distance
    asps_matrix = asps.getDistances(asarray=True)
    return asps_matrix

def generate_negative_samples(G:nx, k:int=None, sampling_mode:str='random', seed:int=None):
    """
    Generate negative edge samples from a graph.

    Parameters:
    - G (nx.Graph): The graph from which to generate negative samples.
    - sampling_mode (str): The mode for negative sampling ('random' or '{n}hop{+,-, or empty}').
      For example, '3hop+' means the negative samples are from pairs that are 3 or more hops away. '4hop-' is 4 or less hops away.
    - k (int): The number of negative samples to generate.
    Returns:
    - List[Tuple]: A list of tuples representing the negative edge samples.
    """
    # Get all possible pairs of nodes that are not connected by an edge
    all_nodes = list(G.nodes)
    all_possible_negative_edges = list(set(combinations(all_nodes, 2)) - set(G.edges))
    if(seed is not None):
        all_possible_negative_edges.sort()
        random.seed(seed)
    random.shuffle(all_possible_negative_edges)

    # Function to filter negative samples based on hop conditions
    def filter_by_hop_condition(hop_condition, n_hops):
        # convert node id to node index
        node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
        edges_idx = [(node_to_index[u], node_to_index[v]) for u, v in all_possible_negative_edges]
        # Get the hop distance matrix for the entire graph
        hop_matrix = get_asps_matrix(G)
        filtered_edges = []
        for u, v in edges_idx:
            distance = hop_matrix[u, v]
            if hop_condition(distance, n_hops):
                filtered_edges.append((u, v))
        return filtered_edges

    # Random mode
    if sampling_mode == 'random':
        negative_samples = all_possible_negative_edges
    # Parse the {n}hop{+,-, or empty} format
    elif 'hop' in sampling_mode:
        n_hops = int(sampling_mode.split('hop')[0])
        if sampling_mode.endswith('hop'):
            # Exactly n hops away
            hop_condition = lambda distance, n_hops: distance == n_hops
        elif sampling_mode.endswith('hop+'):
            # At least n hops away
            hop_condition = lambda distance, n_hops: distance >= n_hops
        elif sampling_mode.endswith('hop-'):
            # At most n hops away
            hop_condition = lambda distance, n_hops: distance <= n_hops
        else:
            raise ValueError("Invalid sampling mode format.")
        
        filtered_edges = filter_by_hop_condition(hop_condition, n_hops)
        negative_samples = filtered_edges
        # return random.sample(filtered_edges, min(k, len(filtered_edges)))
    else:
        raise ValueError("Invalid sampling mode provided.")
    if(k is None):
        return negative_samples
    else:
        return random.sample(negative_samples, min(k, len(negative_samples)))

def _test_generate_negative_samples():
    # Example usage:
    # Create a simple graph for demonstration
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0,3), (3,4), (4,5), (4,6), (6,7)])

    # Generate negative samples with random mode
    random_negative_samples = generate_negative_samples(G, k=4, sampling_mode='random')
    print("Random negative samples:", random_negative_samples)

    # Generate negative samples with 2hop mode
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='2hop+')
    print("2hop+ negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='3hop+')
    print("3hop+ negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='4hop+')
    print("4hop+ negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='5hop+')
    print("5hop+ negative samples:")
    print(hop2_negative_samples)

    # Generate negative samples with 2hop mode
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='2hop-')
    print("2hop- negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='3hop-')
    print("3hop- negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='4hop-')
    print("4hop- negative samples:")
    print(hop2_negative_samples)
    hop2_negative_samples = generate_negative_samples(G, sampling_mode='5hop-')
    print("5hop- negative samples:")
    print(hop2_negative_samples)

#%%
""" generates train and test sets of edges from a graph G (Networkx), Z is the embeddings
negative_samlpes: 'random' or '<n>hop<+/->'. n is the hop distance and + or - is more or less.
For example, '3hop+' means the negative samples are 2 or more hops away from the positive samples. 4hop- 4 or less hops away.
"""
def prepare_edge_dataset(G:nx, train_size:Union[float,int]=0.5, test_size:Union[float,int]=None, negative_sampling_mode:str='random', seed:int=None):
    """
    Prepare the data for link prediction.
    train_size: the size or ratio of training data. If int [0, len(edges)], it's the number of training samples. If float [0,1], it's the ratio of training samples.
    """
    if(test_size is not None):
        if(isinstance(test_size, int)):
            train_size = len(G.edges()) - test_size
        else:
            train_size = 1.0-test_size

    if(train_size<0 or train_size>len(G.edges())):
        raise ValueError("train_size or test_size must be in range [0.0, 1.0] (float) or [0, len(edges)] (int).")
    
    # generate positive samples
    pos_edges = list(G.edges())
    pos_edges.sort()
    
    # generate negative samples
    neg_edges = generate_negative_samples(G, k=len(pos_edges), sampling_mode=negative_sampling_mode, seed=seed)
    neg_edges.sort()

    if(seed is not None):
        random.seed(seed)
    random.shuffle(pos_edges)
    random.shuffle(neg_edges)

    Epos = np.array([[u,v, 1] for u,v in pos_edges])
    Eneg = np.array([[u,v, 0] for u,v in neg_edges])

    pos_train, pos_test = train_test_split(Epos, train_size=train_size, test_size=test_size, shuffle=True, random_state=seed)
    neg_train, neg_test = train_test_split(Eneg, train_size=train_size, test_size=test_size, shuffle=True, random_state=seed)

    # concatenate positive and negative samples
    E_train = npconcat(pos_train, neg_train, axis=0)
    E_test = npconcat(pos_test, neg_test, axis=0)

    if(seed is not None):
        np.random.seed(seed)
    np.random.shuffle(E_train)
    np.random.shuffle(E_test)

    # assert (E_train[:,2]==1).sum() == (E_train[:,2]==0).sum()
    # assert (E_test[:,2]==1).sum() == (E_test[:,2]==0).sum()    
    return (E_train, E_test)

def _test_prepare_edge_dataset():
    # Example usage:
    # Create a simple graph for demonstration
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0,3), (3,4), (4,5), (4,6), (6,7), (7, 8) ])
    
    # Generate negative samples with random mode
    data=prepare_edge_dataset(G, train_size=0.5, sampling_mode='random')
    prepare_edge_dataset(G, train_size=0.5, sampling_mode='2hop+')
    prepare_edge_dataset(G, train_size=0.5, sampling_mode='3hop')
    prepare_edge_dataset(G, train_size=0.5, sampling_mode='4hop+')
    data=prepare_edge_dataset(G, train_size=0.5, sampling_mode='5hop+')
    
if __name__ == "__main__":
    print("Testing generate_negative_samples")
    _test_generate_negative_samples()
    print("Testing prepare_edge_dataset")
    _test_prepare_edge_dataset()