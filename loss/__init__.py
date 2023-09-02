import torch
from .smape import smape

def args_for_loss(parser):
    parser.add_argument('--loss_name', type=str, default="mse", choices=['mse', 'mae', 'smape'])

def get_loss(loss_name):
    if loss_name == 'mse':
        return torch.nn.functional.mse_loss
    elif loss_name == 'mae':
        return torch.nn.functional.l1_loss
    if loss_name == 'smape':
        return smape