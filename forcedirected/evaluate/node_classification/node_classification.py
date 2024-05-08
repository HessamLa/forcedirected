import networkx as nx
from ..utilities import load_graph
from sklearn.model_selection import train_test_split
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
    'eval_nc',
]


def eval_nc(embeddings, labels, test_size:float=0.2, classification_model:str='rf', seed=None, **kwargs):
    """Function to evaluate node classification using embeddings."""
    # load the graph
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, shuffle=True, random_state=seed)

    # create the classification model
    if(classification_model=='rf'):
        clf = RandomForestClassifier(random_state=seed, **kwargs)
    elif(classification_model=='lr'):
        clf = LogisticRegression(random_state=seed, **kwargs)
    elif(classification_model=='mlp'):
        clf = MLPClassifier(random_state=seed, **kwargs)
    elif(classification_model=='knn'):
        clf = KNeighborsClassifier(**kwargs)
    elif(classification_model=='ada'):
        clf = AdaBoostClassifier(random_state=seed, **kwargs)
    else:
        raise ValueError(f"Invalid classification model: {classification_model}")
    
    # fit the model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # produce metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
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



