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

        last_output_size = args.gnn_output_size
        self.node_embedding = nn.Embedding(100, args.emb_dim)

        self.concat_rnn = args.concat_rnn
        if self.concat_rnn:
            last_output_size += gnn_in

        self.concat_before = args.concat_before
        if self.concat_before:
            gnn_in += args.emb_dim+args.flat_out
        else:
            last_output_size += args.emb_dim+args.flat_out
        
        self.gnn = getattr(g_nn , args.GNN)(
            in_channels=gnn_in, hidden_channels=args.gnn_hidden, num_layers=args.gnn_n_layers, 
            out_channels=args.gnn_output_size, dropout=args.gnn_drop_p, norm=args.norm, jk=args.jk, #jk jumping knowdledge -> layer wise aggregation
            **args_gnn
        )

        self.dropout = nn.Dropout(p=args.dropout)
        self.flat_encoder = nn.Linear(args.flat_dim, args.flat_out)
        self.out_layer = nn.Linear(last_output_size, 1)

    def forward(self, node_feat, flat, edge_index, edge_weight=None):
        node_feat = torch.transpose(node_feat, 0, 1).contiguous() # change to (seq, bs, feat) shape
        enc_out = self.encoder(node_feat) # return (bs, feat)

        enc_out = enc_out.view(enc_out.shape[0], -1) # all_nodes, rnn_outdim

        if self.concat_before:
            x_flat = self.flat_encoder(flat)
            gnn_input = torch.concat([enc_out, self.node_embedding.weight, x_flat], dim=-1)
        else:
            gnn_input = enc_out

        gnn_output = self.gnn(gnn_input, edge_index, edge_weight=edge_weight)
        
        if not self.concat_before:
            x_flat = self.flat_encoder(flat)
            gnn_output = torch.concat([gnn_output, self.node_embedding.weight, x_flat], dim=-1)

        if self.concat_rnn:
            gnn_output = torch.concat([gnn_output, enc_out], dim=-1)   
        
        gnn_output = self.dropout(gnn_output)

        return self.out_layer(gnn_output).squeeze(-1)