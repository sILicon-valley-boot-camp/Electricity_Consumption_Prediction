import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .knn import create_knn
from .weighted_graph import create_weighted_graph 

def get_graph(args, time_series, flat, result_path):
    graph_type = nx.Graph if args.graph_type=='graph' else nx.DiGraph

    if 'EU_mean' in args.graph:
        feature = np.stack(time_series.groupby('건물번호').apply(lambda x: x['전력소비량(kWh)'].mean())).reshape(-1, 1)
    elif 'EU_ts' in args.graph:
        feature = np.stack(time_series.groupby('건물번호').apply(lambda x: x['전력소비량(kWh)'].values.reshape(-1)))
    elif args.graph == 'ts_knn':
        col = ['기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
        ts_data = np.stack(time_series.groupby('건물번호').apply(lambda x: x[col].values.reshape(-1)))
        feature = np.stack(ts_data)
    elif 'EU_weather_ts' in args.graph:
        col = ['기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '전력소비량(kWh)']
        ts_data = np.stack(time_series.groupby('건물번호').apply(lambda x: x[col].values.reshape(-1)))
        feature = np.stack(ts_data)
    elif 'ts_all' in args.graph:
        col = list(set(time_series.columns) - {'num_date_time', '건물번호', '일시'})
        ts_data = np.stack(time_series.groupby('건물번호').apply(lambda x: x[col].values.reshape(-1)))
        feature = np.stack(ts_data)
    else:
        print('using default graph')
        feature = flat.values

    if 'knn' in args.graph:
        graph = create_knn(feature, args.k, metric=args.sim, graph_type=graph_type)
    elif 'weighted' in args.graph:
        graph = create_weighted_graph(feature, metric=args.sim, graph_type=graph_type)
    else:
        raise 'graph not defined'
        
    nx.draw(graph, with_labels=True)
    plt.savefig(os.path.join(result_path, 'graph.png'))
    return graph