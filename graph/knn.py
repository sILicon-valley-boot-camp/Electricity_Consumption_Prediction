import networkx as nx
from sklearn.neighbors import kneighbors_graph as knn

def create_knn(feature, k, metric, graph_type=nx.Graph):
    return nx.from_numpy_array(knn(feature, k, metric=metric), create_using=graph_type)