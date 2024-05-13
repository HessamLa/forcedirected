import networkx as nx
import pandas as pd
import numpy as np
from forcedirected.utilities import load_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import all classifiers
from sklearn.ensemble import RandomForestClassifier # rf
from sklearn.linear_model import LogisticRegression # lr
from sklearn.neural_network import MLPClassifier    # mlp
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.ensemble import AdaBoostClassifier     # ada
from sklearn.tree import DecisionTreeClassifier     # dt
from sklearn.svm import LinearSVC                   # svc
# import all metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from recursivenamespace import rns

__all__ = [
    'NodeClassification', 'eval_nc',
]

# from ..evaluate import classifier_models
from forcedirected.evaluate import classifier_models
# classifier_models=rns(
#     lr  =rns(fullname='Logistic Reg.', 
#             instantiate=lambda *args, max_iter=500, n_jobs=-1, **kwargs: LogisticRegression(*args, max_iter=max_iter, n_jobs=n_jobs, **kwargs)),
#     knn =rns(fullname='K Nearest Neighbors',  
#             instantiate=lambda *args, n_jobs=-1, **kwargs: KNeighborsClassifier(*args, n_jobs=n_jobs, **kwargs)),
#     rf  =rns(fullname='Random Forest',  
#             instantiate=lambda *args, min_samples_split=0.02, n_jobs=-1, **kwargs: RandomForestClassifier(*args, min_samples_split=min_samples_split, n_jobs=n_jobs, **kwargs)),
#     dt  =rns(fullname='Decision Tree',  
#             instantiate=lambda *args, min_samples_split=0.02, **kwargs: DecisionTreeClassifier(*args, min_samples_split=min_samples_split, **kwargs)),
#     ada =rns(fullname='AdaBoost',  
#             instantiate=lambda *args, **kwargs: AdaBoostClassifier(*args, **kwargs)),
#     svc =rns(fullname='Linear SVC',  
#             instantiate=lambda *args, **kwargs: LinearSVC(*args, **kwargs)),
#     mlp =rns(fullname='MLP',
#             instantiate=lambda *args, hidden_layer_sizes=(256,), solver='adam', **kwargs: MLPClassifier(*args, hidden_layer_sizes=hidden_layer_sizes, solver=solver, **kwargs)),
# )

class NodeClassification:
    def __init__(self, embeddings:np.ndarray|pd.DataFrame=None, labels:np.ndarray|pd.DataFrame=None, test_size:float=0.5, classification_model:str='rf', seed=None, **kwargs) -> None:
        """
        Initializes the NodeClassification object.
        - embeddings: np.ndarray or pd.DataFrame of node embeddings shape (n_nodes, n_features)
        - labels: np.ndarray or pd.DataFrame of node labels shape (n_nodes, 1)
        - test_size: float, ratio of test size
        - classification_model: str, name of the classification model to use
        - seed: int, random seed
        """
        self.Z = None # the embeddings
        self.y = None # the labels
        self.clf = None
        self.setup(embeddings, labels, test_size, classification_model, seed, **kwargs)

    def setup(self, embeddings:np.ndarray|pd.DataFrame=None, labels:np.ndarray|pd.DataFrame=None, test_size:float=None, classification_model:str=None, seed=None, fit_transform=False, prepare_data=True, **kwargs):
        """Setup the data and classifier.
        - prepare_data: bool, whether to prepare the data for evaluation. If True, the data is split into train test sets.
        """
        if(embeddings is not None):
            if(isinstance(embeddings, pd.DataFrame)):
                embeddings = embeddings.values
            self.Z = embeddings
        
        if(labels is not None):
            if(isinstance(labels, pd.DataFrame)):
                labels = labels.values
            self.y = labels.ravel()

        if(classification_model is not None):
            self.clf = classifier_models[classification_model].instantiate(random_state=seed)

        if(test_size is not None):
            self.test_size = test_size

        if(self.clf is None or self.Z is None or self.y is None or self.test_size is None):
            print("WARNING NodeClassification: Node embeddings and/or labels are not setup.")
            return

        if(fit_transform):
            scaler = StandardScaler()
            self.embeddings = scaler.fit_transform(self.embeddings)

        if(prepare_data):    
            print("Preparing data for node classification...",seed)
            Xidx = np.arange(self.Z.shape[0])
            yidx = np.arange(self.y.shape[0])
            self.X_train_idx, self.X_test_idx, self.y_train_idx, self.y_test_idx = \
                train_test_split(Xidx, yidx, test_size=test_size, shuffle=True, random_state=seed)
        pass

    def evaluate(self, embeddings:np.ndarray|pd.DataFrame=None, **kwargs):
        """Evaluate the node classification performance of the embeddings.
        - embeddings: np.ndarray or pd.DataFrame of node embeddings shape (n_nodes, n_features). If provided, overwrite the existing embeddings.
        """
        self.setup(embeddings=embeddings, prepare_data=False)

        X_train = self.Z[self.X_train_idx]
        X_test = self.Z[self.X_test_idx]
        y_train = self.y[self.y_train_idx].ravel()
        y_test = self.y[self.y_test_idx].ravel()

        # fit the model
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        
        # produce metrics
        accuracy  = accuracy_score(y_test, y_pred)
        f1_micro  = f1_score(y_test, y_pred, average='micro')
        f1_macro  = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
        except ValueError:
            roc_auc = None

        results = rns(
            accuracy=accuracy,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            confusion=confusion,
        )
        return results


def eval_nc(embeddings, labels, test_size:float=0.5, classification_model:str='rf', seed=None, **kwargs):
    kwargs = rns(kwargs)
    kwargs.test_size = test_size
    kwargs.classification_model = classification_model
    kwargs.seed = seed
    nc = NodeClassification(embeddings, labels, **kwargs)
    return nc.evaluate(embeddings=embeddings, **kwargs)
