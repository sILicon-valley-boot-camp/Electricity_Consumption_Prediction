import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def load_data(data_path, info_path):
    data = pd.read_csv(data_path)
    info = pd.read_csv(info_path)
    data = pd.merge(data, info, on='건물번호')

    data = data.sort_values(by=['건물번호', '일시'])
    data['PCS용량(kW)'] = data['PCS용량(kW)'].astype('float64')
    data.reset_index(drop=True, inplace=True)

    start_time = 2022060100
    end_time = 2022082423

    for building_num in data['건물번호'].unique():
        data.loc[(data['건물번호'] == building_num) & (data['일시'] == start_time), ['일조(hr)', '일사(MJ/m2)']] = 0
        data.loc[(data['건물번호'] == building_num) & (data['일시'] == end_time), ['일조(hr)', '일사(MJ/m2)']] = 0

    for building_num in data['건물번호'].unique():
        building_data = data[data['건물번호'] == building_num]
        data.loc[data['건물번호'] == building_num, '일조(hr)'] = building_data['일조(hr)'].interpolate(method='linear')
        data.loc[data['건물번호'] == building_num, '일사(MJ/m2)'] = building_data['일사(MJ/m2)'].interpolate(method='linear')
    
    return data


class BuildingDataset(Dataset):
    def __init__(self, dataframe, window_size=10, mode='train'):
        self.data = dataframe
        self.window_size = window_size
        self.mode = mode
        self.scaler = MinMaxScaler()

        self.building_data = []
        self.building_nums = pd.factorize(self.data['건물번호'])[0]
        self.building_types = pd.factorize(self.data['건물유형'])[0]

        continuous_columns = [
            "연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)",
            "PE1", "PE2", "PE3", "PE4", "PE5", 
            "기온(C)", "습도(%)", "풍속(m/s)", "강수량(mm)", "일조(hr)", "일사(MJ/m2)"
        ]

        for building_num in self.data['건물번호'].unique():
            building_data = self.data[self.data['건물번호'] == building_num].copy()
            building_data[continuous_columns] = self.scaler.fit_transform(building_data[continuous_columns])

            building_num_index = np.where(self.data['건물번호'].unique() == building_num)[0][0]
            building_type = self.data[self.data['건물번호'] == building_num]['건물유형'].iloc[0]
            building_type_index = np.where(self.data['건물유형'].unique() == building_type)[0][0]

            building_data['건물번호'] = building_num_index
            building_data['건물유형'] = building_type_index
            self.building_data.append(building_data)

    def __len__(self):
        total_length = sum(max(0, len(data) - self.window_size + 1) for data in self.building_data)
        return total_length

    def __getitem__(self, idx):
        for building_data in self.building_data:
            if idx < len(building_data) - self.window_size + 1:
                window_data = building_data.iloc[idx:idx+self.window_size]
                break
            idx -= len(building_data) - self.window_size + 1

        if window_data is None:
            raise IndexError('The given index is out of bounds.')

        features = [
            '건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
            '일시', 'PE1', 'PE2', 'PE3', 'PE4', 'PE5', '기온(C)', '습도(%)', '풍속(m/s)', '강수량(mm)', '일조(hr)', '일사(MJ/m2)'
        ]
        input_data = torch.tensor(window_data[features].values, dtype=torch.float32)
        power_consumption = torch.tensor(window_data['전력소비량(kWh)'].values, dtype=torch.float32)

        if self.mode == 'train':
            return {'input': input_data, 'label': power_consumption}
        else:  # self.mode == 'test'
            return {'input': input_data}
