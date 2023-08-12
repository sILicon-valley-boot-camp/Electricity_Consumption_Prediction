import torch

def smape(true, pred):
    v = 2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true))
    output = torch.mean(v) * 100
    return output