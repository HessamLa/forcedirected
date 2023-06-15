# from:
# https://github.com/vatsal220/medium_articles/blob/main/node_classification/classify_node.ipynb
# %%
# !pip install arxiv
# %%
import networkx as nx
import pandas as pd
import numpy as np
import arxiv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
# from node2vec import Node2Vec as n2v
# %% Fetch Data
def search_arxiv(queries, max_results = 100):
    '''
    This function will search arxiv associated to a set of queries and store
    the latest 10000 (max_results) associated to that search.
    
    params:
        queries (List -> Str) : A list of strings containing keywords you want
                                to search on Arxiv
        max_results (Int) : The maximum number of results you want to see associated
                            to your search. Default value is 1000, capped at 300000
                            
    returns:
        This function will return a DataFrame holding the following columns associated
        to the queries the user has passed. 
            `title`, `date`, `article_id`, `url`, `main_topic`, `all_topics`
    
    example:
        research_df = search_arxiv(
            queries = ['automl', 'recommender system', 'nlp', 'data science'],
            max_results = 10000
        )
    '''
    d = []
    searches = []
    # hitting the API
    for query in queries:
        search = arxiv.Search(
          query = query,
          max_results = max_results,
          sort_by = arxiv.SortCriterion.SubmittedDate,
          sort_order = arxiv.SortOrder.Descending
        )
        searches.append(search)
    
    # Converting search result into df
    for search in searches:
        for res in search.results():
            data = {
                'title' : res.title,
                'date' : res.published,
                'article_id' : res.entry_id,
                'url' : res.pdf_url,
                'main_topic' : res.primary_category,
                'all_topics' : res.categories,
                'authors' : res.authors
            }
            d.append(data)
        
    d = pd.DataFrame(d)
    d['year'] = pd.DatetimeIndex(d['date']).year
    
    # change article id from url to integer
    unique_article_ids = d.article_id.unique()
    article_mapping = {art:idx for idx,art in enumerate(unique_article_ids)}
    d['article_id'] = d['article_id'].map(article_mapping)
    return d

# constants
queries = [
    'automl', 'machinelearning', 'data', 'phyiscs','mathematics', 'recommendation system', 'nlp', 'neural networks'
]
%%time
research_df = search_arxiv(
    queries = queries,
    max_results = 250
)
research_df.shape
# %%
research_df.head()
# %%
def generate_network(df, node_col, edge_col):
    '''
    This function will generate a article to article network given an input DataFrame.
    It will do so by creating an edge_dictionary where each key is going to be a node
    referenced by unique values in node_col and the values will be a list of other nodes
    connected to the key through the edge_col.
    
    params:
        df (DataFrame) : The dataset which holds the node and edge columns
        node_col (String) : The column name associated to the nodes of the network
        edge_col (String) : The column name associated to the edges of the network
        
    returns:
        A networkx graph corresponding to the input dataset
        
    example:
        generate_network(
            research_df,
            node_col = 'article_id',
            edge_col = 'main_topic'
        )
    '''
    edge_dct = {}
    for i,g in df.groupby(node_col):
        topics = g[edge_col].unique()
        edge_df = df[(df[node_col] != i) & (df[edge_col].isin(topics))]
        edges = list(edge_df[node_col].unique())
        edge_dct[i] = edges
    
    # create nx network
    g = nx.Graph(edge_dct, create_using = nx.MultiGraph)
    return g
Gnx = generate_network(
    research_df, 
    node_col = 'article_id', 
    edge_col = 'main_topic'
)
print(nx.info(Gnx))
# %% Get the adjacency matrix
A = nx.to_numpy_array(Gnx)
print(A.shape)

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
print("is symmetric?", check_symmetric(A))
# %%
distances = nx.all_pairs_shortest_path_length(Gnx)

hops = np.zeros_like(A)
for u,dist_dict in distances:
    print(u, dist_dict)
    for v,d in dist_dict.items():
        hops[u,v]=d
# %%

# find connected components
V = [v in nx.connected_components(Gnx)]

print(len(V))


# %%
h = hops[10, :]
print(set(h))
total=0
for i in range(int(h.max())+1):
    print(i, len(h[h==i]))
    total+=len(h[h==i])
print("total", total)

h.shape

# %%
emb_df = (
    pd.DataFrame(
        [mdl.wv.get_vector(str(n)) for n in tp_nx.nodes()],
        index = tp_nx.nodes
    )
)
# %% Generate random embeddings
# first set random positions
d = 12 # a 6 dimensional space
n = emb_df.shape[0]
Z = np.random.rand(n,d)
print(f"A matrix of shape {Z.shape} where each row corresponds to embedding of a node in a {d}-dimensional space.")

