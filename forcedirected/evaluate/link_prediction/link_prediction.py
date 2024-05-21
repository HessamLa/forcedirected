import networkx as nx
# import networkit as nk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_score

from recursivenamespace import rns

from forcedirected.evaluate import classifier_models
from .lp_utils import prepare_edge_dataset, npconcat

class LinkPrediction:
    def __init__(self, graph:nx, embeddings:pd, classification_model:str='rf', train_size:float=0.5,  test_size:float=None, 
              negative_sampling_mode:str='random', prepare_data:bool=True, fit_transform=False, seed=None, **kwargs):
        self.G = None
        self.Z = None
        self.clf = None

        self.setup(graph=graph, embeddings=embeddings, classification_model=classification_model, test_size=test_size, 
                   negative_sampling_mode=negative_sampling_mode, prepare_data=prepare_data, fit_transform=fit_transform, seed=seed, **kwargs)
        
    def setup(self, graph=None, embeddings=None, classification_model:str='rf', train_size:float=0.5, test_size:float=None, 
              negative_sampling_mode:str='random', prepare_data=True, fit_transform=False, seed=None, **kwargs):

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

        if(classification_model is not None):
            self.clf = classifier_models[classification_model].instantiate()

        if(test_size is not None):
            self.test_size = test_size
            if(isinstance(test_size, int)):
                self.train_size = len(self.G.edges()) - test_size
            else:
                self.train_size = 1.0-test_size
        else:
            self.train_size = train_size
            if(isinstance(train_size, int)):
                self.test_size = len(self.G.edges()) - train_size
            else:
                self.test_size = 1.0-train_size
        
        if(prepare_data):
            if(self.embeddings is None or self.G is None):
                raise ValueError("ERROR: LinkPrediction, Embeddings or Graph not provided.")
            
            # get the node ids from embeddings dataframe and compare it with the graph node ids
            ids_emb = set(self.embeddings['id'].tolist())
            ids_g = set(self.G.nodes())
            if(ids_emb != ids_g):
                raise Exception("ERROR: The nodes IDs in the graph and the embeddings dataframe do not match.")
                
            
            # make the node_id to node_index mapping according to embeddings 'id' column
            # self.node_to_index = {node: idx for idx, node in enumerate(self.embeddings['id'])}
            node_to_index = {self.embeddings['id'][i]:i for i in range(len(self.embeddings['id']))}
            self.Z = self.embeddings.iloc[:,1:].values
            

            if(fit_transform):
                scaler = StandardScaler()
                self.Z = scaler.fit_transform(self.Z)

            E_train, E_test = prepare_edge_dataset(self.G, train_size=self.train_size, test_size=self.test_size,
                                negative_sampling_mode=negative_sampling_mode, seed=seed)
            # convert node id to node index (only apply to the first 2 columns)
            # node_to_index = {node: idx for idx, node in enumerate(self.G.nodes())}
            E_train[:, :2] = np.vectorize(node_to_index.get)(E_train[:, :2])
            E_test[:, :2] = np.vectorize(node_to_index.get)(E_test[:, :2])
            # convert all to int
            E_train = E_train.astype(int)
            E_test = E_test.astype(int)
            self.E_train, self.E_test = E_train, E_test

    def evaluate(self, product='hadamard', metrics='all', top_k=100, **kwargs):
        # get features
        X = self.Z
        if(product == 'hadamard'):
            X_train = X[self.E_train[:,0]] * X[self.E_train[:,1]]
            X_test = X[self.E_test[:,0]] * X[self.E_test[:,1]]
        elif(product == 'average'):
            X_train = (X[self.E_train[:,0]] + X[self.E_train[:,1]]) / 2
            X_test = (X[self.E_test[:,0]] + X[self.E_test[:,1]]) / 2
        elif(product == 'concat'):
            X_train = npconcat(X[self.E_train[:,0]], X[self.E_train[:,1]], axis=1)
            X_test = npconcat(X[self.E_test[:,0]], X[self.E_test[:,1]], axis=1)
        else:
            raise ValueError(f"ERROR: product \"{product}\" not supported.")

        # get labels
        y_train = self.E_train[:,2]
        y_test = self.E_test[:,2]

        # fit
        self.clf.fit(X_train, y_train)

        # obtain the results
        y_pred = self.clf.predict(X_test)

        if(metrics == 'all'):
            metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc',
                            'at_top_k']
        result = rns()
        if('accuracy' in metrics):
            result.accuracy = accuracy_score(y_test, y_pred)
        if('f1' in metrics):
            result.f1 = f1_score(y_test, y_pred)
        if('precision' in metrics):
            result.precision = precision_score(y_test, y_pred)
        if('recall' in metrics):
            result.recall = recall_score(y_test, y_pred)
        if('roc_auc' in metrics):
            result.roc_auc = roc_auc_score(y_test, y_pred)
        # precision at top k
        if('at_top_k' in metrics and X_test.shape[0] >= top_k):
            try:
                # Make predictions on the test set
                y_prob = self.clf.predict_proba(X_test)[:, 1]
                # Sort the predicted probabilities in descending order
                sorted_indices = np.argsort(y_prob)[::-1]
                # Calculate P@k
                result[f'p_at_{top_k}'] = precision_score(y_test[sorted_indices[:top_k]], np.ones(top_k))
                # Calculate hit@100
                result[f'hit_at_{top_k}'] = np.mean(y_test[sorted_indices[:top_k]] == 1)
            except Exception as e:
                raise e
                print(e)
                pass
        return result    

def eval_lp(graph, embeddings, train_size:float=0.5, test_size:float=None, classification_model:str='rf', metrics='all', negative_sampling_mode='random', top_k=100, seed=None, **kwargs):
    lp = LinkPrediction(graph, embeddings, classification_model=classification_model, 
                        train_size=train_size, test_size=test_size, negative_sampling_mode=negative_sampling_mode, seed=seed, **kwargs)
    results =  lp.evaluate(metrics=metrics, top_k=top_k, **kwargs)
    for k, v in results.items():
        print(f"{str(k):<16s}: {v}")
    return results

__all__ = ['LinkPrediction', 'eval_lp']

