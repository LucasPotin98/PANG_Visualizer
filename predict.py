import numpy as np
from joblib import load
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from statsData import get_dataset_id, get_graphs_for_dataset,get_filtered_patterns,gspan_to_networkx

def load_model(dataset_name):
    model_path = f"{dataset_name}.pkl"
    with open(model_path, "rb") as f:
        return load(f)


def computePerfs(model,dataset_name):
    if dataset_name == "MUTAG":
        mean = 0.81
        std = 0.16
    if dataset_name == "PTC":
        mean = 0.49
        std = 0.11
    if dataset_name == "AIDS":
        mean = 0.92
        std = 0.03
    return mean,std

def constructRepresentation(presentPatterns,nbPatterns):
    return np.array([1 if pid in presentPatterns else 0 for pid in range(0,nbPatterns)]).reshape(1, -1)
    # Faire la prediction


def predict(model,vector):
    prediction = model.predict(vector)
    return prediction


def get_decision_path(model, X_instance, feature_names=None):
    print(X_instance)
    features = []
    node_indicator = model.decision_path(X_instance)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    for node_id in node_index:
        if model.tree_.feature[node_id] != -2:
            features.append(((model.tree_.feature[node_id]),X_instance[0][model.tree_.feature[node_id]]))
    return features
