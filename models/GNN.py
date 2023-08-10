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
        if args.GNN == 'GAT':
            args_gnn = {'v2':args.v2}
        elif args.GNN == 'GraphSAGE':
            args_gnn = {'aggr':args.aggr}

        self.node_embedding = nn.Embedding(100, args.emb_dim)

        self.gnn = getattr(g_nn , args.GNN)(
            in_channels=gnn_in+args.emb_dim, hidden_channels=args.gnn_hidden, num_layers=args.gnn_n_layers, 
            out_channels=args.gnn_output_size, dropout=args.gnn_drop_p, norm=args.norm, jk=args.jk, #jk jumping knowdledge -> layer wise aggregation
            **args_gnn
        )

        self.dropout = nn.Dropout(p=args.dropout)
        self.flat_encoder = nn.Linear(args.flat_dim, args.flat_out)
        self.out_layer = nn.Linear(args.gnn_output_size+args.flat_out+gnn_in, 1)

    def forward(self, node_feat, flat, edge_index, edge_weight=None):
        node_feat = torch.transpose(node_feat, 0, 1).contiguous() # change to (seq, bs, feat) shape
        out = self.encoder(node_feat) # return (bs, feat)

        out = out.view(out.shape[0], -1) # all_nodes, rnn_outdim
        gnn_input = torch.concat([out, self.node_embedding.weight], dim=-1)
        x = self.gnn(gnn_input, edge_index, edge_weight=edge_weight)
        
        x_flat = self.flat_encoder(flat)
        x = torch.concat([x, x_flat, out], dim=-1)
        x = self.dropout(x)

        return self.out_layer(x).squeeze(-1)