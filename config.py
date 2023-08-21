import json
import argparse

from loss import args_for_loss
from models import args_for_model
from scaler import args_for_scaler

def args_for_data(parser):
    parser.add_argument('--train', type=str, default='../data/train.csv')
    parser.add_argument('--test', type=str, default='../data/test.csv')
    parser.add_argument('--flat', type=str, default='../data/building_info.csv')
    parser.add_argument('--submission', type=str, default='../data/sample_submission.csv')
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--window_size', type=int, default=10)
    
def args_for_train(parser):
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--cv_k', type=int, default=10, help='k-fold stratified cross validation')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epoch of lr scheduler(Not Implemented)')
    parser.add_argument('--no_sigmoid', default=False, help='whether to use sigmoid at the last', action='store_true')

    parser.add_argument('--continue_train', type=int, default=-1, help='continue training from fold x') 
    parser.add_argument('--continue_from_folder', type=str, help='continue training from args.continue_from')

def args_for_graph(parser):
    parser.add_argument('--graph', type=str, default='knn')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sim', type=str, default='minkowski')
    parser.add_argument('--graph_type', type=str, default='graph', choices=['graph', 'directed'])

def args_for_tuning(parser):
    parser.add_argument('--test_ratio', type=float, default=0.3, help='train test split ratio(only used in hyper-parmeter tuning)')
    parser.add_argument('--n_trials', type=int, default=None, help='n_trials')
    parser.add_argument('--timeout', type=int, default=None, help='optuna training timeout(sec)')
    parser.add_argument('--n_job_parallel', type=int, default=1, help='n_job_parallel')

def args_for_config_file(parser):
    parser.add_argument('--config', default=None, type=str, help='read from config file')
    parser.add_argument('--target_args', nargs='*', help='target args to change from config file')

def modify_args(config_args, args):
    for key in args.target_args:
        try:
            if config_args[key] != args[key]:
                print(f'using {args[key]} for {key}')
        except KeyError:
            print(f'added {key}')
            config_args[key] = args[key]

    return argparse.Namespace(**config_args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='LSTM', type=str)
    parser.add_argument('--GNN', default='GAT', type=str)

    args_for_data(parser)
    args_for_train(parser)
    args_for_loss(parser)
    args_for_scaler(parser)
    args_for_graph(parser)
    args_for_tuning(parser)
    args_for_config_file(parser)

    _args, _ = parser.parse_known_args()
    args_for_model(parser, _args.model, _args.GNN)
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config) as json_file:
            config_args = argparse.Namespace(**json.load(json_file))
        args = modify_args(config_args.__dict__, args.__dict__)

    return args
