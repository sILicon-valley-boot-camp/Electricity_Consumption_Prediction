import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .knn import create_knn

def get_graph(args, time_series, flat, result_path):
    if args.graph == 'knn':
        graph = create_knn(flat.values, args.k, metric=args.sim)

    if args.graph == 'EU_mean_knn':
        graph = create_knn(np.stack(time_series.groupby('건물번호').apply(lambda x: x['전력소비량(kWh)'].mean())).reshape(-1, 1), args.k, metric=args.sim)

    if args.graph == 'EU_ts_knn':
        graph = create_knn(np.stack(time_series.groupby('건물번호').apply(lambda x: x['전력소비량(kWh)'].values.reshape(-1))), args.k, metric=args.sim)

    if args.graph == 'ts_knn':
        col = ['기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
        ts_data = np.stack(time_series.groupby('건물번호').apply(lambda x: x[col].values.reshape(-1)))
        graph = create_knn(np.stack(ts_data), args.k, metric=args.sim)

    if args.graph == 'ts_all_knn':
        col = list(set(time_series.columns) - {'num_date_time', '건물번호', '일시'})
        ts_data = np.stack(time_series.groupby('건물번호').apply(lambda x: x[col].values.reshape(-1)))
        graph = create_knn(np.stack(ts_data), args.k, metric=args.sim)

    nx.draw(graph, with_labels=True)
    plt.savefig(os.path.join(result_path, 'graph.png'))
    return graph