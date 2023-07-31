# Standard library imports
import os
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

# Local application imports
from config import get_args
from dataset import CustomDataset
import models
from loss_functions import MSE, MAE, MAPE, SMAPE
import train
import test
from utils import seed_everything

def prepare_data(data_path):
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:].values
    return data

def prepare_model(input_dim, hidden_dim, output_dim, device):
    model = models.RNNModel(input_dim, hidden_dim, output_dim).to(device)
    return model

def run_train(train_data, train_label, valid_data, valid_label, model, batch_size, device):
    train_data_set = CustomDataset(train_data, train_label)
    valid_data_set = CustomDataset(valid_data, valid_label)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data_set, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Choose loss function
    loss_function = MSE()

    trainer = train.Trainer(train_loader, valid_loader, model, loss_function, optimizer, epochs)
    trainer.train()

def run_test(test_data, model, batch_size, device):
    test_data_set = CustomDataset(test_data, None)

    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    # Choose loss function
    loss_function = MSE()

    tester = test.Tester(test_loader, model, loss_function)
    tester.test()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    device = torch.device('cuda:0')

    model = prepare_model(args.input_dim, args.hidden_dim, args.output_dim, device)

    if args.mode == 'train':
        train_data = prepare_data(args.train_data_path)
        train_label = prepare_data(args.train_label_path)
        valid_data = prepare_data(args.valid_data_path)
        valid_label = prepare_data(args.valid_label_path)
        
        run_train(train_data, train_label, valid_data, valid_label, model, args.batch_size, device)

    else: # args.mode == 'test'
        test_data = prepare_data(args.test_data_path)

        run_test(test_data, model, args.batch_size, device)
