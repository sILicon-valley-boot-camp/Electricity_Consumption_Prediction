import sklearn.preprocessing as sp


def args_for_scaler(parser):
    parser.add_argument('--scaler_name', type=str, default="MinMax", choices=['MinMax', 'Standard', 'Quantile_u', 'Quantile_g'])

def get_scaler(scaler_name):
    if scaler_name == 'MinMaxScaler':
        return getattr(sp , 'MinMax')()
    elif scaler_name == 'Standard':
        return getattr(sp , 'Standard')()
    elif scaler_name == 'Quantile_u':
        return getattr(sp , 'QuantileTransformer')()
    elif scaler_name == 'Quantile_g':
        return getattr(sp , 'QuantileTransformer')(output_distribution='normal')
    else:
        return None