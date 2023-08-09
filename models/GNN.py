import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as g_nn

from . import TimeSeriesTransformerEncoder, RNN

class RnnGnn(nn.Module):
    def __init__(self, args, feature_size):
        super().__init__()

        if args.model == 'transformer':
            self.encoder = TimeSeriesTransformerEncoder(args, feature_size)
            gnn_in = feature_size

        if args.model == 'LSTM' or args.model == 'GRU':
            self.encoder = RNN(args, feature_size)
            gnn_in = args.hidden

        args_gnn = {}
        if args.gnn == 'GAT':
            args_gnn = {'v2':args.v2}
        elif args.gnn == 'GraphSAGE':
            args_gnn = {'aggr':args.aggr}

        self.gnn = getattr(g_nn , args.gnn)(
            in_channels=gnn_in, hidden_channels=args.gnn_hidden, num_layers=args.gnn_n_layers, 
            out_channels=args.gnn_output_size, dropout=args.gnn_drop_p, norm=args.norm, jk=args.jk, #jk jumping knowdledge -> layer wise aggregation
            **args_gnn
        )

        self.flat_encoder = nn.Linear(args.flat_dim, args.flat_out)
        self.out_layer = nn.Linear(args.gnn_output_size+args.flat_out+gnn_in, 1)

    def forward(self, src, flat, edge_index, edge_weight):
        src = torch.transpose(src, 0, 1).contiguous() # change to (seq, bs, feat) shape
        out = self.encoder(src) # return (bs, feat)

        out = out.view(out.shape[0], -1) # all_nodes, rnn_outdim
        x = self.gnn(out, edge_index, edge_weight)
        
        x_flat = self.flat_encoder(flat)
        x = torch.concat([x, x_flat, out])
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.out_layer(x)