import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim

from config import get_args
from utils import seed_everything
from dataset import CustomDataset

import models
from loss_functions import MSE, MAE, MAPE, SMAPE

if __name__ == "__main__":
    args = get_args()
    # get args

    seed_everything(args.seed)
    # fix seed

    device = torch.device('cuda:0')
    # use cuda:0

    if args.mode == 'train':
        train_data_path = pd.read_csv(args.data_path)
        train_label_path = pd.read_csv(args.data_path)
        valid_data_path = pd.read_csv(args.data_path)
        valid_label_path = pd.read_csv(args.data_path)

        train_data = train_data_path.iloc[:, 1:].values
        train_label = train_label_path.iloc[:, 1:].values
        valid_data = valid_data_path.iloc[:, 1:].values
        valid_label = valid_label_path.iloc[:, 1:].values

        train_data_set = CustomDataset(train_data, train_label)
        valid_data_set = CustomDataset(valid_data, valid_label)

        train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data_set, batch_size=args.batch_size, shuffle=False)
    else:
        test_data_path = pd.read_csv(args.data_path)

        test_data = test_data_path.iloc[:, 1:].values

        test_data_set = CustomDataset(test_data, None)

        test_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False)
    # load data

    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    model = models.RNNModel(input_dim, hidden_dim, output_dim).to(device)
    # build model

    epoch = args.epochs
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # set training hyperparameters

    loss_function = MSE()
    # loss_function = MAE()
    # loss_function = MAPE()
    # loss_function = SMAPE()
    # set loss function

    for epoch in range(epoch):
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()