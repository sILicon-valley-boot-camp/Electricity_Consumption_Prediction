import pandas as pd

import torch
from torch.utils.data import  Dataset
from torch_geometric.data import Data   
from torch_geometric.utils import from_networkx

class GraphTimeDataset(Dataset): #get graph data at t time step
    def __init__(self, ts_df, flat_df, graph, window_size, time_index, label=None):
        super().__init__()
        self.ts = ts_df.sort_values(by=['건물번호', '일시'], ignore_index=True)
        self.flat = flat_df
        self.window_size = window_size
        self.time_index = time_index
        self.label = label
        self.drop = ['num_date_time', '건물번호', '일시']
        self.graph = from_networkx(graph)

    def __len__(self):
        return len(self.time_index)
    
    def __getitem__(self, index):
        time = pd.date_range(self.time_index[index] - pd.Timedelta(hours=(self.window_size-1)) , self.time_index[index], freq='H')
        ts_data = self.ts[self.ts['일시'].isin(time)]

        data = ts_data.drop(columns = [self.label] + self.drop)

        dict_data = {'node_feat': torch.tensor(data.values, dtype=torch.float), #(window_size, feat_dim),
                     'flat': torch.tensor(self.flat.values, dtype=torch.float),
                     'edge_index': self.graph.edge_index}
        
        if self.label is not None:
            dict_data['y'] = torch.tensor(ts_data[ts_data['일시']==self.time_index[index]][self.label].values, dtype=torch.float)
        
        if self.graph.edge_weight is not None:
            dict_data['edge_weight'] = self.graph.edge_weight

        return dict_data