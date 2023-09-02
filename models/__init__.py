import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import TimeSeriesTransformerEncoder
from .RNN import RNN
from .GNN import RnnGnn

def args_for_model(parser, model, GNN_model=None):
    parser.add_argument('--pooling', type=str, default="last", choices=['mean', 'max', 'last', 'first', 'all'])
    parser.add_argument('--dropout', type=float, default=0.5)
    
    if model == 'transformer':
        parser.add_argument('--n_head', type=int, default=5)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--transformer_dropout', type=float, default=0.5)

    if model == 'LSTM' or model == 'GRU':
        parser.add_argument('--hidden', type=int, default=50)
        parser.add_argument('--num_layers', type=int, default=1)

    if GNN_model is not None:
        parser.add_argument('--gnn_hidden', type=int, default=50)
        parser.add_argument('--gnn_n_layers', type=int, default=2)
        parser.add_argument('--gnn_output_size', type=int, default=10)
        parser.add_argument('--gnn_drop_p', type=float, default=0.5)
        parser.add_argument('--norm', type=str, default=None)
        parser.add_argument('--jk', type=str, default=None)
        parser.add_argument('--flat_out', type=int, default=10)
        parser.add_argument('--emb_dim', type=int, default=10)    

        #experiments
        parser.add_argument('--gnn_input', nargs='+', default=['enc_out', 'node_emb'])
        parser.add_argument('--outputs_after_gnn', nargs='+', default=['enc_out', 'flat'])

        if GNN_model == 'GAT':
            parser.add_argument('--v2', type=bool, default=True)
        elif GNN_model == 'GraphSAGE':
            parser.add_argument('--aggr', type=str, default='mean')

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