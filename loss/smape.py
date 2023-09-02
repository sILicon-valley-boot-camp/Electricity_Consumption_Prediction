import torch

def smape(true, pred):
    v = 2 * torch.abs(pred - true) / ((torch.abs(pred) + torch.abs(true)) + 1e-9)
    output = torch.mean(v) * 100
    return output