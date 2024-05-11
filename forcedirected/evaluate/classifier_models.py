from sklearn.ensemble import RandomForestClassifier # rf
from sklearn.linear_model import LogisticRegression # lr
from sklearn.neural_network import MLPClassifier    # mlp
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.ensemble import AdaBoostClassifier     # ada
from sklearn.tree import DecisionTreeClassifier     # dt
from sklearn.svm import LinearSVC                   # svc

from recursivenamespace import rns

classifier_models=rns(
    lr  =rns(fullname='Logistic Reg.', 
            instantiate=lambda *args, max_iter=500, n_jobs=-1, **kwargs: LogisticRegression(*args, max_iter=max_iter, n_jobs=n_jobs, **kwargs)),
    knn =rns(fullname='K Nearest Neighbors',  
            instantiate=lambda *args, n_jobs=-1, **kwargs: KNeighborsClassifier(*args, n_jobs=n_jobs, **kwargs)),
    rf  =rns(fullname='Random Forest',  
        #     instantiate=lambda *args, min_samples_split=0.02, n_jobs=-1, **kwargs: RandomForestClassifier(*args, min_samples_split=min_samples_split, n_jobs=n_jobs, **kwargs)),
            instantiate=lambda *args, n_jobs=-1, **kwargs: RandomForestClassifier(*args, n_jobs=n_jobs, **kwargs)),
    dt  =rns(fullname='Decision Tree',  
        #     instantiate=lambda *args, min_samples_split=0.02, **kwargs: DecisionTreeClassifier(*args, min_samples_split=min_samples_split, **kwargs)),
            instantiate=lambda *args, **kwargs: DecisionTreeClassifier(*args, **kwargs)),
    ada =rns(fullname='AdaBoost',  
            instantiate=lambda *args, **kwargs: AdaBoostClassifier(*args, **kwargs)),
    svc =rns(fullname='Linear SVC',  
            instantiate=lambda *args, **kwargs: LinearSVC(*args, **kwargs)),
    mlp =rns(fullname='MLP',
            instantiate=lambda *args, hidden_layer_sizes=(256,), solver='adam', **kwargs: MLPClassifier(*args, hidden_layer_sizes=hidden_layer_sizes, solver=solver, **kwargs)),
)

__all__ = ['classifier_models']