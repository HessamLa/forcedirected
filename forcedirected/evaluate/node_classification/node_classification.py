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
# import all metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from recursivenamespace import rns

__all__ = [
    'NodeClassification', 'eval_nc',
]

class NodeClassification:
    def __init__(self, embeddings=None, labels=None, test_size:float=0.5, classification_model:str='rf', seed=None, **kwargs) -> None:
        self.embeddings = None
        self.labels = None
        self.setup(embeddings, labels, test_size, classification_model, seed, **kwargs)

    def setup(self, embeddings=None, labels=None, test_size:float=None, classification_model:str=None, seed=None, fit_transform=False, **kwargs):
        # create the classification model
        if(classification_model is None):
            pass
        elif(classification_model=='rf'):
            self.clf = RandomForestClassifier(random_state=seed, **kwargs)
        elif(classification_model=='lr'):
            self.clf = LogisticRegression(random_state=seed, **kwargs)
        elif(classification_model=='mlp'):
            self.clf = MLPClassifier(random_state=seed, **kwargs)
        elif(classification_model=='knn'):
            self.clf = KNeighborsClassifier(**kwargs)
        elif(classification_model=='ada'):
            self.clf = AdaBoostClassifier(random_state=seed, **kwargs)
        else:
            raise ValueError(f"Invalid classification model: {classification_model}")
        
        if(embeddings is not None):
            if(isinstance(embeddings, pd.DataFrame)):
                embeddings = embeddings.values
            self.embeddings = embeddings
        
        if(labels is not None):
            if(isinstance(labels, pd.DataFrame)):
                labels = labels.values
            self.labels = labels

        if(test_size is not None):
            self.test_size = test_size

        if(self.embeddings is None or self.labels is None or self.test_size is None):
            print("WARNING NodeClassification: Node embeddings and/or labels are not setup.")
            return

        if(fit_transform):
            scaler = StandardScaler()
            self.embeddings = scaler.fit_transform(self.embeddings)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.embeddings, self.labels, test_size=test_size, shuffle=True, random_state=seed)
        # Reshape y_train and y_test to ensure they are 1D arrays
        self.y_train = self.y_train.ravel()
        self.y_test = self.y_test.ravel()
        pass

    def evaluate(self):
        # fit the model
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        
        # produce metrics
        accuracy  = accuracy_score(self.y_test, self.y_pred)
        f1_micro  = f1_score(self.y_test, self.y_pred, average='micro')
        f1_macro  = f1_score(self.y_test, self.y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, self.y_pred, average='weighted')

        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall    = recall_score(self.y_test, self.y_pred, average='weighted')
        confusion = confusion_matrix(self.y_test, self.y_pred)
        try:
            roc_auc = roc_auc_score(self.y_test, self.y_pred, average='weighted')
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
    nc = NodeClassification(embeddings, labels, test_size, classification_model, seed, **kwargs)
    return nc.evaluate()
