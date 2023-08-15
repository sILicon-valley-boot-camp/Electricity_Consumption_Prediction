from networkx import from_numpy_array
from sklearn.neighbors import kneighbors_graph as knn

def create_knn(feature, k, metric):
    return from_numpy_array(knn(feature, k, metric=metric))