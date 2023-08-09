import os
import networkx as nx
import matplotlib.pyplot as plt

from .knn import create_knn

def get_graph(args, features, result_path):
    if args.graph == 'knn':
        graph = create_knn(features, args.k)

    nx.draw(graph, with_labels=True)
    plt.savefig(os.path.join(result_path, 'graph.png'))
    return graph