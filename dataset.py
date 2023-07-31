import pandas as pd
import torch
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, csv_file, window_size=5, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.data['강수량(mm)'].fillna(0, inplace=True)  # 강수량의 누락된 값을 0으로 채움
        self.window_size = window_size
        self.mode = mode
        self.building_data = []  # 각 건물의 데이터를 저장할 리스트

        # 각 건물의 데이터를 구분하여 저장
        for building_num in self.data['건물번호'].unique():
            building_data = self.data[self.data['건물번호'] == building_num]
            self.building_data.append(building_data)

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        # window를 사용하여 데이터를 추출
        for building_data in self.building_data:
            if idx < len(building_data) - self.window_size + 1:
                window_data = building_data.iloc[idx:idx+self.window_size]
                break
            else:
                idx -= len(building_data) - self.window_size + 1

        # 각 열을 PyTorch 텐서로 변환
        building_num = torch.tensor(window_data['건물번호'].values)
        date_time = torch.tensor([int(dt.replace(' ', '')) for dt in window_data['일시']])
        temperature = torch.tensor(window_data['기온(C)'].values)
        rainfall = torch.tensor(window_data['강수량(mm)'].values)
        wind_speed = torch.tensor(window_data['풍속(m/s)'].values)
        humidity = torch.tensor(window_data['습도(%)'].values)
        sunshine = torch.tensor(window_data['일조(hr)'].values)
        solar_radiation = torch.tensor(window_data['일사(MJ/m2)'].values)

        if self.mode == 'train':
            power_consumption = torch.tensor(window_data['전력소비량(kWh)'].values)
            return {
                'input': {
                    'building_num': building_num,
                    'date_time': date_time,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    'sunshine': sunshine,
                    'solar_radiation': solar_radiation
                },
                'label': power_consumption
            }
        else:  # 'test' mode
            return {
                'input': {
                    'building_num': building_num,
                    'date_time': date_time,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    'sunshine': sunshine,
                    'solar_radiation': solar_radiation
                }
            }
