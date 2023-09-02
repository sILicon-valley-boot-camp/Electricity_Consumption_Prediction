import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def create_weighted_knn(feature, k, metric, graph_type=nx.Graph):
    if metric=='cosine':
        adj = _cal_cosine(feature)
    elif metric=='minkowski':
        adj = _cal_L2(feature)
    else:
        raise 'not implemented metric'
    mask_index = np.argpartition(adj, -k, axis=1)[:, :-k]
    mask_index = np.stack([np.arange(adj.shape[0])[:, None].repeat(adj.shape[0]-k, axis=1), mask_index], axis=-1).reshape(-1, 2).T
    adj[mask_index[0], mask_index[1]] = 0
    
    return nx.from_numpy_array(adj, create_using=graph_type)

def create_weighted_graph(feature, metric, graph_type=nx.Graph):
    if metric=='cosine':
        adj = _cal_cosine(feature)
    elif metric=='minkowski':
        adj = _cal_L2(feature)
    else:
        raise 'not implemented metric'
    
    return nx.from_numpy_array(adj, create_using=graph_type)

def _cal_L2(feature):
    adj = 0 - euclidean_distances(feature, feature)
    mask = ~np.eye(adj.shape[0],dtype=bool)
    adj[mask] = MinMaxScaler().fit_transform(adj[mask].reshape(-1, 1)).reshape(-1)
    np.fill_diagonal(adj, 1)
    return adj

def _cal_cosine(feature):
    adj = cosine_similarity(feature, feature)
    np.fill_diagonal(adj, 1)
    return adj