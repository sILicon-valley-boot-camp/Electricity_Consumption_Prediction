import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet, TestDataSet

train_data = pd.read_csv("data/train.csv")
train_data['일시'] = pd.to_datetime(train_data['일시'])
train_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
train_start = min(train_data['일시'])
train_end = max(train_data['일시'])

test_data = pd.read_csv("data/test.csv")
test_data['일시'] = pd.to_datetime(test_data['일시'])
test_start = min(test_data['일시'])
test_end = max(test_data['일시'])

window_size = 10

total_data = pd.concat([train_data, test_data], join='inner').reset_index(drop=True)
total_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
test_time = pd.date_range(test_start - pd.Timedelta(hours=(window_size)) , test_end, freq='H') #for compatibility
test_data = total_data[total_data['일시'].isin(test_time)].reset_index(drop=True)

train_time = pd.date_range(train_start + pd.Timedelta(hours=(window_size)) , train_end, freq='H') #args.window_size의 이후의 시간부터 loss function에 제공
train_target_data = train_data[train_data['일시'].isin(train_time)] #train_start time_stamp's data will not be used(except electricity consumption) 

output_index = '전력소비량(kWh)'
train_dataset = DataSet(data=train_data, label=output_index, window_size=window_size, target_index=kfold_train_index)
