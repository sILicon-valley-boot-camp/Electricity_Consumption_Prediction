import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, info_path):
    data = pd.read_csv(data_path)
    info = pd.read_csv(info_path)
    data = pd. merge(data, info, on='건물번호')
    data = data.sort_values(by=['건물번호', '일시'])
    data.reset_index(drop=True)
    return data

def handle_nan(data):
    data['강수량(mm)'].fillna(0, inplace=True)  # 강수량의 누락된 값을 0으로 채움
    data['풍속(m/s)'].interpolate(inplace=True)  # 풍속의 누락된 값을 선형보간으로 채움
    data['습도(%)'].interpolate(inplace=True)  # 습도의 누락된 값을 선형보간으로 채움
    data['일조(hr)'].fillna(data['일조(hr)'].mean(), inplace=True)  # 일조의 누락된 값을 평균값으로 채움
    data['일사(MJ/m2)'].fillna(data['일사(MJ/m2)'].mean(), inplace=True)  # 일사의 누락된 값을 평균값으로 채움

    data['태양광용량(kW)'].replace('-', 0, inplace=True)  # 태양광용량의 누락된 값을 0으로 채움
    data['ESS저장용량(kWh)'].replace('-', 0, inplace=True)  # ESS저장용량의 누락된 값을 0으로 채움
    data['PCS용량(kW)'].replace('-', 0, inplace=True)  # PCS용량의 누락된 값을 0으로 채움

    data['태양광용량(kW)'] = pd.to_numeric(data['태양광용량(kW)'], errors='coerce').fillna(0)
    data['ESS저장용량(kWh)'] = pd.to_numeric(data['ESS저장용량(kWh)'], errors='coerce').fillna(0)
    data['PCS용량(kW)'] = pd.to_numeric(data['PCS용량(kW)'], errors='coerce').fillna(0)
    return data

class BuildingDataset(Dataset):
    def __init__(self, dataframe, window_size=10, mode='train'):
        self.data = dataframe
        self.window_size = window_size
        self.mode = mode
        self.scaler = MinMaxScaler()  # Min-Max Scaler to normalize features

        self.building_data = []  # 각 건물의 데이터를 저장할 리스트
        self.building_nums = pd.factorize(self.data['건물번호'])[0]  # 각 건물번호를 고유한 정수 인덱스로 변환
        self.building_types = pd.factorize(self.data['건물유형'])[0]  # 각 건물유형을 고유한 정수 인덱스로 변환

        continuous_columns = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 
                                  'PCS용량(kW)', '기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 
                                  '일조(hr)', '일사(MJ/m2)']

        # 각 건물의 데이터를 구분하여 저장
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
        total_length = 0
        for building_data in self.building_data:
            total_length += max(0, len(building_data) - self.window_size + 1)
        return total_length


    def __getitem__(self, idx):
        window_data = None  # Set initial value
        for building_data in self.building_data:
            if idx < len(building_data) - self.window_size + 1:
                window_data = building_data.iloc[idx:idx+self.window_size]
                break
            else:
                idx -= len(building_data) - self.window_size + 1

        if window_data is None:
            raise IndexError('The given index is out of bounds.')

        building_num = torch.tensor(window_data['건물번호'].values)
        building_type = torch.tensor(window_data['건물유형'].astype('category').cat.codes.values)  # 건물유형을 카테고리 코드로 변환
        total_area = torch.tensor(window_data['연면적(m2)'].values)
        cooling_area = torch.tensor(window_data['냉방면적(m2)'].values)
        solar_capacity = torch.tensor(window_data['태양광용량(kW)'].values)
        ess_capacity = torch.tensor(window_data['ESS저장용량(kWh)'].values)
        pcs_capacity = torch.tensor(window_data['PCS용량(kW)'].values)
        date_time = torch.tensor([int(dt.replace(' ', '')) for dt in window_data['일시']])
        temperature = torch.tensor(window_data['기온(C)'].values)
        rainfall = torch.tensor(window_data['강수량(mm)'].values)
        wind_speed = torch.tensor(window_data['풍속(m/s)'].values)
        humidity = torch.tensor(window_data['습도(%)'].values)
        sunshine = torch.tensor(window_data['일조(hr)'].values)
        solar_radiation = torch.tensor(window_data['일사(MJ/m2)'].values)

        power_consumption = torch.tensor(window_data['전력소비량(kWh)'].values)
        input_data = torch.stack([
            building_num,
            building_type,
            total_area,
            cooling_area,
            solar_capacity,
            ess_capacity,
            pcs_capacity,
            date_time,
            temperature,
            rainfall,
            wind_speed,
            humidity,
            sunshine,
            solar_radiation
        ], dim=-1)  # Stacks the tensors along a new last dimension
        
        if self.mode == 'train':
            return {'input': input_data, 'label': power_consumption}
        else: # self.mode == 'test'
            return {'input': input_data}
    
