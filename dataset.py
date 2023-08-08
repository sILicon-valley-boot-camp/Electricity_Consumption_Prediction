import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, info_path):
    data = pd.read_csv(data_path)
    info = pd.read_csv(info_path)
    data = pd.merge(data, info, on='건물번호')

    data = data.sort_values(by=['건물번호', '일시'])
    data['PCS용량(kW)'] = data['PCS용량(kW)'].astype('float64')

    data.reset_index(drop=True)
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

        continuous_columns = [
            "연면적(m2)", "냉방면적(m2)", "태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)",
            "PE1", "PE2", "PE3", "PE4",	"PE5", 
            "기온(C)", "습도(%)", "풍속(m/s)", "강수량(mm)", "일조(hr)", "일사(MJ/m2)",
            ]

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
        date_time = torch.tensor(window_data['일시'].values)
        pe1 = torch.tensor(window_data['PE1'].values)
        pe2 = torch.tensor(window_data['PE2'].values)
        pe3 = torch.tensor(window_data['PE3'].values)
        pe4 = torch.tensor(window_data['PE4'].values)
        pe5 = torch.tensor(window_data['PE5'].values)
        temperature = torch.tensor(window_data['기온(C)'].values)
        humidity = torch.tensor(window_data['습도(%)'].values)
        wind_speed = torch.tensor(window_data['풍속(m/s)'].values)
        rainfall = torch.tensor(window_data['강수량(mm)'].values)
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
            pe1,
            pe2,
            pe3,
            pe4,
            pe5,
            temperature,
            humidity,
            wind_speed,
            rainfall,
            sunshine,
            solar_radiation
        ], dim=-1)  # Stacks the tensors along a new last dimension
        
        if self.mode == 'train':
            return {'input': input_data, 'label': power_consumption}
        else: # self.mode == 'test'
            return {'input': input_data}
    
