import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as g_nn

from . import TimeSeriesTransformerEncoder, RNN

class RnnGnn(nn.Module):
    def __init__(self, args, feature_size):
        super().__init__()
        self.outputs_after_gnn = args.outputs_after_gnn
        self.gnn_input = args.gnn_input

        if args.model == 'transformer':
            self.encoder = TimeSeriesTransformerEncoder(args, feature_size)
            gnn_in = feature_size

        if args.model == 'LSTM' or args.model == 'GRU':
            self.encoder = RNN(args, feature_size)
            gnn_in = args.hidden

        gnn_in = self.get_size_for_gnn(gnn_in, args.flat_out, args.emb_dim)
        gnn_out = self.get_size_for_gnn(args.gnn_output_size, args.flat_out, args.emb_dim)

        args_gnn = {}
        if args.GNN == 'GAT':
            args_gnn = {'v2':args.v2}
        elif args.GNN == 'GraphSAGE':
            args_gnn = {'aggr':args.aggr}

        self.gnn = getattr(g_nn , args.GNN)(
            in_channels=gnn_in, hidden_channels=args.gnn_hidden, num_layers=args.gnn_n_layers, 
            out_channels=args.gnn_output_size, dropout=args.gnn_drop_p, norm=args.norm, jk=args.jk, #jk jumping knowdledge -> layer wise aggregation
            **args_gnn
        )

        self.node_embedding = nn.Embedding(100, args.emb_dim)
        self.flat_encoder = nn.Linear(args.flat_dim, args.flat_out)
        self.dropout = nn.Dropout(p=args.dropout)
        self.out_layer = nn.Linear(gnn_out, 1)

    def get_size_for_gnn(self, enc_size, flat_size, emb_size):
        size = 0
        if 'enc_out' in self.gnn_input:
            size += enc_size
        if 'flat' in self.gnn_input:
            size += flat_size
        if 'node_emb' in self.gnn_input:
            size += emb_size

        return size
    
    def get_size_for_out(self, enc_size, flat_size, emb_size):
        size = 0
        if 'enc_out' in self.outputs_after_gnn:
            size += enc_size
        if 'flat' in self.outputs_after_gnn:
            size += flat_size
        if 'node_emb' in self.outputs_after_gnn:
            size += emb_size

        return size

    def prepare_input_for_gnn(self, enc_out, flat, node_emb):
        array = []
        if 'enc_out' in self.gnn_input:
            array.append(enc_out)
        if 'flat' in self.gnn_input:
            array.append(flat)
        if 'node_emb' in self.gnn_input:
            array.append(node_emb)

        return torch.concat(array, dim=-1)

    def prepare_output_with_gnn(self, gnn_output, enc_out, flat, node_emb):
        array = [gnn_output]
        if 'enc_out' in self.outputs_after_gnn:
            array.append(enc_out)
        if 'flat' in self.outputs_after_gnn:
            array.append(flat)
        if 'node_emb' in self.outputs_after_gnn:
            array.append(node_emb)

        return torch.concat(array, dim=-1)

    def forward(self, node_feat, flat, edge_index, edge_weight=None):
        node_feat = torch.transpose(node_feat, 0, 1).contiguous() # change to (seq, bs, feat) shape
        enc_out = self.encoder(node_feat) # return (bs, feat)
        enc_out = enc_out.view(enc_out.shape[0], -1) # all_nodes, rnn_outdim

        flat = self.flat_encoder(flat)
        node_emb = self.node_embedding.weight

        gnn_input = self.prepare_input_for_gnn(enc_out, flat, node_emb)
        gnn_output = self.gnn(gnn_input, edge_index, edge_weight=edge_weight)

        output = self.prepare_output_with_gnn(gnn_output, enc_out, flat, node_emb)
        output = self.dropout(output)

        return self.out_layer(output).squeeze(-1)