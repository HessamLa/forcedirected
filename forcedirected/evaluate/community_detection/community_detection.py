# %%
import networkx as nx
import numpy as np
import pandas as pd
# import nmi from scikit learn
from sklearn.metrics.cluster import normalized_mutual_info_score

class CommunityDetection:
    def __init__(self, graph:nx, embeddings:pd.DataFrame, communites:dict|list=None, k:int=None, clustering_model:str='knn', seed=None, **kwargs):
        self.G = None
        self.Z = None
        self.communities = None
        self.k = None
        self.clustering_model = None
        self.seed = None

        self.setup(graph=graph, embeddings=embeddings, communites=communites, k=k, clustering_model=clustering_model, seed=seed, **kwargs)
    
    def setup(self, graph=None, embeddings=None, clustering_model:str='knn', seed=None, **kwargs):

        if(graph is not None):
            self.G = graph

        if(embeddings is not None):
            self.embeddings = embeddings
            self.Z = embeddings.iloc[:,1:]
            nodes = embeddings['id']
            nodes2 = list(self.G.nodes())
            if(not all(nodes == nodes2)):
                print("Graph nodes:", nodes2)
                print("Embedding nodes:", nodes)
                raise Exception("ERROR: The nodes in the graph and the embeddings do not match.")

        if(clustering_model is not None):
            self.clustering_model = clustering_model

        if(seed is not None):
            self.seed = seed
def eval_cd(graph:nx, embeddings:pd.DataFrame, communites:dict|list=None, k:int=None, clustering_model:str='knn', seed=None, **kwargs):
    """
    Evaluate the community detection performance of the embeddings.
    graph is a networkx object, with n nodes.
    embeddings is a pandas dataframe with shape (n+1,d). The first column is the node ID. Each row is the embedding of a node in a d-dimensional space.
    communities is the ground truth communities of the nodes. It can be a dictionary with node ID as key and community ID as value, or a list of lists, where each list contains the node IDs of a community.
    k is the number of communities. If communities is provided, k is ignored.
    clustering_model is the clustering model to use. It can be 'knn' for K-Nearest Neighbors, 'gmm' for Gaussian Mixture Model, 'dbscan' for DBSCAN, 'spectral' for Spectral Clustering, 'hierarchical' for Agglomerative Clustering, 'kmeans' for K-Means, 'birch' for Birch, 'affinity' for Affinity Propagation, 'mean_shift' for Mean Shift, 'optics' for OPTICS, 'hdbscan' for HDBSCAN, 'agglomerative' for Agglomerative Clustering, 'minibatch' for Mini Batch K-Means
    seed is the random seed for reproducibility.
    """


    raise NotImplementedError
