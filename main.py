import gc
import time
import logging
from sklearn.model_selection import KFold
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import get_args
from dataset import BuildingDataset, load_data
from models import RNNModel, LSTMModel, GRUModel
from loss_functions import MSE, MAE, MAPE, SMAPE
import train
import test
from utils import seed_everything


def setup_logging():
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"log_{current_time}.txt"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("This is a new log file.")
    return logger


def prepare_model(input_dim, hidden_dim, output_dim, num_layers, device):
    model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    # model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    # model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    return model


def run_train(dataset, lr, epochs, batch_size, logger, device, n_splits=5):
    kf = KFold(n_splits=n_splits)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)

        model = prepare_model(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # loss_function = MSE
        # loss_function = MAE
        # loss_function = MAPE
        loss_function = SMAPE

        trainer = train.Trainer(train_loader, valid_loader, model, loss_function, optimizer, epochs, device)
        fold_result = trainer.train(fold, logger)

        del model, optimizer, trainer
        gc.collect()


def run_test(dataset, model, batch_size, logger, device):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # loss_function = MSE
    # loss_function = MAE
    # loss_function = MAPE
    loss_function = SMAPE

    tester = test.Tester(test_loader, model, loss_function, device)
    tester.test(logger)

    del model
    gc.collect()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda:0')
    logger = setup_logging()

    if args.mode == 'train':
        data = load_data(args.data_path, args.info_path)
        dataset = BuildingDataset(data, args.window_size, args.mode)
        del data
        gc.collect()
        run_train(dataset, args.lr, args.epochs, args.batch_size, logger, device)

    else: # args.mode == 'test'
        data = load_data(args.data_path, args.info_path)
        dataset = BuildingDataset(data, args.window_size, args.mode)
        model = prepare_model(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, device)
        weights = torch.load(args.model_path)
        model.load_state_dict(weights)
        run_test(dataset, model, args.batch_size, logger, device)

