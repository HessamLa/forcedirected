# %%
import networkx as nx
# import nmi from scikit learn
from sklearn.metrics.cluster import normalized_mutual_info_score

def eval_cd(Gx, embeddings, clustering_model:str='knn', seed=None, **kwargs):
    raise NotImplementedError
