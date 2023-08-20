import numpy as np
import pandas as pd

import torch
from torch.utils.data import  Dataset
from torch_geometric.data import Data   
from torch_geometric.utils import from_networkx

class GraphTimeDataset(Dataset): #get graph data at t time step
    def __init__(self, ts_df, flat_df, graph, window_size, time_index, label):
        super().__init__()
        self.ts = ts_df.sort_values(by=['건물번호', '일시'], ignore_index=True)
        self.flat = torch.tensor(flat_df.values, dtype=torch.float)        
        self.window_size = window_size
        self.time_index = time_index
        self.drop = ['num_date_time', '건물번호', '일시']
        self.graph = from_networkx(graph) if graph is not None else None
        self.label = label

    def __len__(self):
        return len(self.time_index)
    
    def __getitem__(self, index):
        time = pd.date_range(self.time_index[index] - pd.Timedelta(hours=(self.window_size-1)) , self.time_index[index], freq='H')
        ts_data = self.ts[self.ts['일시'].isin(time)]

        data = np.array([group.drop(columns = [self.label] + self.drop).values for _, group in ts_data.groupby('건물번호')])

        dict_data = {'x': torch.tensor(data, dtype=torch.float), #(window_size, feat_dim),
                     'y': torch.tensor(ts_data[ts_data['일시']==self.time_index[index]][self.label].values, dtype=torch.float)}
                
        if self.graph is not None:
            dict_data['edge_index'] = self.graph.edge_index

        if self.graph is not None and self.graph.weight is not None:
            dict_data['edge_attr'] = self.graph.weight

        return Data(**dict_data)