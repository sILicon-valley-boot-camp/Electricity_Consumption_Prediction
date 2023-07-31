from Transformer import TimeSeriesTransformer

def args_for_model(parser, model):
    if model == 'TimeSeriesTransformer':
        parser.add_argument('--transformer_pooling', type=str, default="last", choices=['mean', 'max', 'last', 'first', 'all'])
        parser.add_argument('--feature_size', type=int, default=50)
        parser.add_argument('--n_head', type=int, default=5)
        parser.add_argument('--num_layers', type=int, default=1)