import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer import TimeSeriesTransformerEncoder
from .RNN import RNN

def args_for_model(parser, model):
    parser.add_argument('--pooling', type=str, default="last", choices=['mean', 'max', 'last', 'first', 'all'])
    parser.add_argument('--dropout', type=float, default=0.5)
    
    if model == 'transformer':
        parser.add_argument('--n_head', type=int, default=5)
        parser.add_argument('--num_layers', type=int, default=1)

    if model == 'LSTM' or model == 'GRU':
        parser.add_argument('--hidden', type=int, default=50)
        parser.add_argument('--num_layers', type=int, default=1)

class TimeSeriesModel(nn.Module):
    def __init__(self, args, feature_size):
        super().__init__()
        if args.model == 'transformer':
            self.encoder = TimeSeriesTransformerEncoder(args, feature_size)
            self.linear = nn.Linear(feature_size, 1)

        if args.model == 'LSTM' or args.model == 'GRU':
            self.encoder = RNN(args, feature_size)
            self.linear = nn.Linear(args.hidden, 1)

    def forward(self, src):
        src = torch.transpose(src, 0, 1).contiguous() # change to (seq, bs, feat) shape
        out = self.encoder(src) # return (bs, feat)
        return self.linear(out).squeeze(1)