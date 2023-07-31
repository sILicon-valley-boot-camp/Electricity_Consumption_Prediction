import numpy as np
import torch
from torch.nn import MSELoss, L1Loss

def MSE(true, pred):
    return MSELoss(true, pred)

def MAE(predicted, target):
    return L1Loss(predicted, target)

def MAPE(predicted, target):
    return torch.mean(torch.abs((target - predicted) / target)) * 100

def SMAPE(true, pred):
    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    loss = np.mean(v) * 100
    return loss
