# Third party imports
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim

# Local application imports
from config import get_args
from dataset import BuildingDataset
from models import RNNModel, LSTMModel, GRUModel
from loss_functions import MSE, MAE, MAPE, SMAPE
import train
import test
from utils import seed_everything

def prepare_model(input_dim, hidden_dim, output_dim, device):
    model = RNNModel(input_dim, hidden_dim, output_dim).to(device)
    # model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    # model = GRUModel(input_dim, hidden_dim, output_dim).to(device)
    return model

def run_train(dataset, model, lr, epochs, batch_size):

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Choose loss function
    loss_function = MSE()
    # loss_function = MAE()
    # loss_function = MAPE()
    # loss_function = SMAPE()

    trainer = train.Trainer(train_loader, valid_loader, model, loss_function, optimizer, epochs)
    trainer.train()

def run_test(dataset, model, batch_size):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        lr = args.lr
        epochs = args.epochs
        dataset = BuildingDataset(args.data_path, args.window_size, args.mode)

        run_train(dataset, model, lr, epochs, args.batch_size)

    else: # args.mode == 'test'
        dataset = BuildingDataset(args.data_path, args.window_size, args.mode)

        run_test(dataset, model, args.batch_size)
