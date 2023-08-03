import numpy as np
import torch
from torch.nn import MSELoss, L1Loss

def MSE(pred, true):
    return MSELoss()(pred, true)

def MAE(pred, true):
    return L1Loss()(pred, true)

def MAPE(pred, true):
    return torch.mean(torch.abs((true - pred) / true)) * 100

def SMAPE(pred, true):
    v = 2 * abs(true - pred) / (abs(true) + abs(pred))
    loss = v.mean() * 100
    return loss
