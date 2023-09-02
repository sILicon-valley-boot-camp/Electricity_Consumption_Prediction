import numpy as np
import pandas as pd

import torch
from torch.utils.data import  Dataset
from torch_geometric.data import Data   
from torch_geometric.utils import from_networkx

class GraphTimeDataset(Dataset): #get graph data at t time step
    def __init__(self, ts_df, flat_df, graph, window_size, time_index, label, device):
        super().__init__()
        self.ts = ts_df.sort_values(by=['건물번호', '일시'], ignore_index=True)
        self.flat = torch.tensor(flat_df.values, dtype=torch.float, device=device) 

        self.window_size = window_size
        self.time_index = time_index
        self.times = self.ts[self.ts['건물번호'] == 1]['일시'].reset_index(drop=True)
        self.drop = list(set(self.ts.columns) & {'num_date_time', '건물번호', '일시'})

        self.data = torch.stack([torch.tensor(group.drop(columns = [label] + self.drop).values, dtype=torch.float, device=device) for _, group in self.ts.groupby('건물번호')])
        self.y = torch.stack([torch.tensor(group[label].values, dtype=torch.float, device=device) for _, group in self.ts.groupby('건물번호')])
        self.graph = from_networkx(graph).to(device) if graph is not None else None

    def __len__(self):
        return len(self.time_index)
    
    def __getitem__(self, index):
        time = pd.date_range(self.time_index[index] - pd.Timedelta(hours=(self.window_size-1)) , self.time_index[index], freq='H')
        data_index = self.times[self.times.isin(time)].index
        target_index = self.times[self.times == self.time_index[index]].index

        data = self.data[:, data_index, :]
        y = self.y[:, target_index].squeeze(-1)
        dict_data = {'x': data,
                     'y': y,}
                
        if self.graph is not None:
            dict_data['edge_index'] = self.graph.edge_index

        if self.graph is not None and self.graph.weight is not None:
            dict_data['edge_attr'] = self.graph.weight

        return Data(**dict_data)