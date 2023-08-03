# Third party imports
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
import logging

# Local application imports
from config import get_args
from dataset import BuildingDataset
from models import RNNModel, LSTMModel, GRUModel
from loss_functions import MSE, MAE, MAPE, SMAPE
import train
import test
from utils import seed_everything


def prepare_model(input_dim, hidden_dim, output_dim, num_layers, device):
    model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    # model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    # model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    return model

def run_train(dataset, model, lr, epochs, batch_size, logger, device):

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_function = MSE
    # loss_function = MAE
    # loss_function = MAPE
    # loss_function = SMAPE

    trainer = train.Trainer(train_loader, valid_loader, model, loss_function, optimizer, epochs, logger, device)
    trainer.train(logger)

def run_test(dataset, model, batch_size, logger, device):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_function = MSE
    # loss_function = MAE
    # loss_function = MAPE
    # loss_function = SMAPE

    tester = test.Tester(test_loader, model, loss_function, logger, device)
    tester.test(logger)

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda:0')
    model = prepare_model(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, device)

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('training_log.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.mode == 'train':
        lr = args.lr
        epochs = args.epochs
        dataset = BuildingDataset(args.data_path, args.info_path, args.window_size, args.mode)

        run_train(dataset, model, lr, epochs, args.batch_size, logger, device)

    else: # args.mode == 'test'
        dataset = BuildingDataset(args.data_path, args.info_path, args.window_size, args.mode)

        run_test(dataset, model, args.batch_size,logger, device)
