import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def create_weighted_graph(feature, metric, graph_type=nx.Graph):
    if metric=='cosine':
        adj = cosine_similarity(feature, feature)
    elif metric=='minkowski':
        adj = euclidean_distances(feature, feature)
    else:
        raise 'not implemented metric'
    
    return nx.from_numpy_array(adj, create_using=graph_type)