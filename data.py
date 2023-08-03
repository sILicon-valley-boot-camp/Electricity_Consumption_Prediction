import pandas as pd

import torch
from torch.utils.data import  Dataset

class DataSet(Dataset):
    def __init__(self, data, label, window_size, target_index):
        self.label = label
        self.data = data.sort_values(by=['건물번호', '일시'], ignore_index=True)
        self.drop = ['num_date_time', '건물번호', '일시']
        self.window_size = window_size
        self.target_index = target_index

    def __len__(self):
        return len(self.target_index)
    
    def __getitem__(self, index):
        data = self.data[self.target_index[index]-(self.window_size): self.target_index[index]+1]
        y = data[self.label].values[-1] # current time step - 11 ~ current time step, (when window_size=10)
        x = data.drop(columns = [self.label] + self.drop)  # current time step - 10 ~ current time step, (when window_size=10)

        return {'x': torch.tensor(x.values, dtype=torch.float), #(window_size, feat_dim)
                'y': torch.tensor(y, dtype=torch.float)} #(window_size+1, )
    
class TestDataSet(Dataset):
    def __init__(self, data, window_size, test_start, test_end):
        self.data = data
        self.window_size = window_size-1
        target_time = pd.date_range(test_start , test_end, freq='H')
        self.target_index = self.data[self.data['일시'].isin(target_time)].index
        self.drop = ['num_date_time', '건물번호', '일시']

    def __len__(self):
        return len(self.target_index)
    
    def __getitem__(self, index):
        data = self.data[self.target_index[index]-(self.window_size): self.target_index[index]+1]
        x = data.drop(columns = self.drop)  # current time step - 10 ~ current time step, (when window_size=10)

        return {'x': torch.tensor(x.values, dtype=torch.float)}
    
class TestDataSetByBuilding(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.drop = ['num_date_time', '건물번호', '일시']

    def __len__(self):
        return 100
    
    def __getitem__(self, index):
        data = self.data[self.data['건물번호'] == index+1]
        x = data.drop(columns = self.drop + ['전력소비량(kWh)'])  # current time step - 10 ~ current time step, (when window_size=10)

        return {'x': torch.tensor(x.values, dtype=torch.float)}