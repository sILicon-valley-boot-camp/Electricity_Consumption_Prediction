import argparse

from loss import args_for_loss
from models import args_for_model

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
    #parser.add_argument('--batch_size', type=int, default=None, help='batch_size') batch_size fixed to 1
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epoch of lr scheduler(Not Implemented)')

    parser.add_argument('--continue_train', type=int, default=-1, help='continue training from fold x') 
    parser.add_argument('--continue_from_folder', type=str, help='continue training from args.continue_from')

def args_for_graph(parser):
    parser.add_argument('--graph', type=str, default='knn')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sim', type=str, default='minkowski')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='LSTM', type=str)
    parser.add_argument('--GNN', default='GAT', type=str)

    args_for_data(parser)
    args_for_train(parser)
    args_for_loss(parser)
    args_for_graph(parser)
    _args, _ = parser.parse_known_args()
    args_for_model(parser, _args.model, _args.GNN)

    args = parser.parse_args()
    return args
