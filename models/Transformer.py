import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from src.models.pyg_ns import define_ns_gnn_encoder
from src.models.utils import init_weights, get_act_fn
from src.models.utils import init_weights, get_act_fn, trunc_normal_

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :] #(length, 1, feat)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer_pooling = args['transformer_pooling'] 
        self.pos_encoder = PositionalEncoding(args['feature_size'])
        self.use_cls = False
        if args['transformer_pooling']== 'first':
            self.use_cls = True
            self.cls_token = nn.Parameter(torch.zeros(1, 1, args['feature_size']))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args['feature_size'], nhead=args['n_head'], dropout=args['transformer_dropout'], register_hook=args['fp_attn_transformer']) #(seq, bs, feat)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
                
    def forward(self, src):
        if self.use_cls:
            cls_tokens = self.cls_token.expand(-1, src.shape[1], -1)
            src = torch.cat((cls_tokens, src), dim=0)
        mask = self._generate_square_subsequent_mask(len(src)).cuda() #delete
        self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = torch.transpose(output, 0, 1).contiguous()
        
        if self.transformer_pooling == 'mean':
            output = torch.mean(output, 1).squeeze()
        elif self.transformer_pooling == 'max':
            output = torch.max(output, 1)[0].squeeze()
        elif self.transformer_pooling == 'last':
            output = output[:, -1, :]
        elif self.transformer_pooling == 'first':
            output = output[:, 0, :]
        elif self.transformer_pooling == 'all':
            pass
        else:
            raise NotImplementedError('only transformer_pooling mean / all for now.')
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerGNN(torch.nn.Module):
    """
    model class for Transformer-GNN with node-sampling scheme.
    """
    def __init__(self, args):
        super().__init__()
        self.input_layer = nn.Linear(args['input_size'], args['feature_size'])
        self.transformer_encoder = TimeSeriesTransformer(args)        
        self.gnn_name = args['gnn_name']
        self.gnn_encoder = define_ns_gnn_encoder(args['gnn_name'])(args)
        self.last_act = get_act_fn(args['final_act_fn'])
        self.transformer_out = nn.Linear(args['feature_size'], args['out_dim'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        x = self.input_layer(x) #change dimension of input to match pos encoder
        seq = x.permute(1, 0, 2)
        out = self.transformer_encoder.forward(seq)
        last = out[:, -1, :] if len(out.shape)==3 else out
        last = last[:batch_size]
        out = out.view(out.size(0), -1) # all_nodes, transformer_outdim
        x = out
        x = self.gnn_encoder(x, flat, adjs, edge_weight, last)
        y = self.last_act(x)        
        transformer_y = self.last_act(self.transformer_out(last))
        return y, transformer_y
    
    def infer_transformer_by_batch(self, ts_loader, device):
        transformer_outs = []
        lasts = []
        transformer_ys = []
        for inputs, labels, ids in ts_loader:
            seq, flat = inputs
            seq = seq.to(device)
            seq = self.input_layer(seq)
            seq = seq.permute(1, 0, 2)
            out = self.transformer_encoder.forward(seq)
            last = out[:, -1, :] if len(out.shape)==3 else out
            out = out.view(out.size(0), -1)
            transformer_y = self.last_act(self.transformer_out(last))
            transformer_outs.append(out)
            lasts.append(last)
            transformer_ys.append(transformer_y)
        transformer_outs = torch.cat(transformer_outs, dim=0) # [entire_g, dim]
        lasts = torch.cat(lasts, dim=0) # [entire_g, dim]
        transformer_ys = torch.cat(transformer_ys, dim=0)
        print('Got all transformer output.')
        return transformer_outs, lasts, transformer_ys
    
    def infer_transformer_by_batch_attn(self, ts_loader, device):
        transformer_outs = []
        lasts = []
        transformer_ys = []
        transformer_input = []
        attentions = []

        def hook(m, i, o):
            attentions.append(o[1].detach().cpu().numpy())

        for encoder_layer in self.transformer_encoder.transformer_encoder.layers:
            encoder_layer.self_attn.register_forward_hook(hook)

        for inputs, labels, ids in ts_loader:
            seq, flat = inputs
            transformer_input.append(seq.detach().cpu().numpy())
            seq = seq.to(device)
            seq = self.input_layer(seq)
            seq = seq.permute(1, 0, 2)
            out = self.transformer_encoder.forward(seq)
            last = out[:, -1, :] if len(out.shape)==3 else out
            out = out.view(out.size(0), -1)
            transformer_y = self.last_act(self.transformer_out(last))
            transformer_outs.append(out)
            lasts.append(last)
            transformer_ys.append(transformer_y)
        transformer_outs = torch.cat(transformer_outs, dim=0) # [entire_g, dim]
        lasts = torch.cat(lasts, dim=0) # [entire_g, dim]
        transformer_ys = torch.cat(transformer_ys, dim=0)
        print('Got all transformer output.')
        return transformer_outs, lasts, transformer_ys, (np.stack(transformer_input), np.stack(attentions))

    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False, is_gat=False, get_attention = False):
        # first collect transformer outputs by minibatching:
        if get_attention:
            transformer_outs, last_all, transformer_ys, input_with_attention = self.infer_transformer_by_batch_attn(ts_loader, device)
        else:
            transformer_outs, last_all, transformer_ys = self.infer_transformer_by_batch(ts_loader, device)

        # then pass transformer outputs to gnn
        x_all = transformer_outs
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight, last_all, get_emb=get_emb)

        if is_gat:
            out = out[0]
        out = self.last_act(out)

        if get_attention:
            return input_with_attention

        return out, transformer_ys