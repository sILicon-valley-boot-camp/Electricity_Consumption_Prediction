import os
import sys
import json
import optuna
import logging
import pandas as pd
from functools import partial
from argparse import Namespace
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import optim
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

import models
from loss import get_loss
from config import get_args
from graph import get_graph
from trainer import Trainer
from scaler import get_scaler
from lr_scheduler import get_sch
from data import GraphTimeDataset
from utils import seed_everything, handle_unhandled_exception, save_to_json

from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

def main(args):
    args = Namespace(**args)
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    result_path = os.path.join(args.result_path, args.model + args.GNN + '_' + str(trial.number))
    os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('file_logger'+str(trial.number))
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    flat_data = pd.read_csv(args.flat)
    args.flat_dim = len(flat_data.columns)

    train_data = pd.read_csv(args.train)
    train_data['일시'] = pd.to_datetime(train_data['일시'])
    train_data_length = len(set(train_data['일시'])) #todo check train_data length
    train_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
    train_start = min(train_data['일시'])
    train_end = max(train_data['일시'])

    train_time = pd.date_range(train_start + pd.Timedelta(hours=(args.window_size-1)) , train_end, freq='H') #for compatibility
    
    output_index = '전력소비량(kWh)'
    scaling_col = list(set(train_data.columns) - {'num_date_time', '건물번호', '일시', '전력소비량(kWh)'})
    input_size = len(scaling_col)
    data_scaler = MinMaxScaler()
    target_scaler = get_scaler(args.scaler_name)
    
    train_data[scaling_col] = data_scaler.fit_transform(train_data[scaling_col])

    if target_scaler is not None:
        train_data[output_index] = target_scaler.fit_transform(train_data[output_index].values.reshape(-1, 1))
        scaling_fn = target_scaler.inverse_transform
    else:
        scaling_fn = lambda x:x
        
    test_data = pd.read_csv(args.test)
    test_data['일시'] = pd.to_datetime(test_data['일시'])
    test_start = min(test_data['일시'])
    test_end = max(test_data['일시'])

    test_data[scaling_col] = data_scaler.transform(test_data[scaling_col])

    total_data = pd.concat([train_data, test_data], join='outer').reset_index(drop=True)
    total_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
    test_time = pd.date_range(test_start - pd.Timedelta(hours=(args.window_size-1)) , test_end, freq='H') #for compatibility
    test_data = total_data[total_data['일시'].isin(test_time)].reset_index(drop=True)
    test_time = pd.date_range(test_start, test_end, freq='H') #for compatibility

    graph = get_graph(args, train_data, flat_data, result_path) if args.graph != 'node_emb' else None

    kfold_train_time, kfold_valid_time = train_test_split(train_time, test_size=int(train_data_length * args.test_ratio), random_state=args.seed, shuffle=True)

    logger.info(f'start training')

    train_dataset = GraphTimeDataset(ts_df=train_data, flat_df=flat_data, graph=graph, label=output_index, window_size=args.window_size, time_index=kfold_train_time)
    valid_dataset = GraphTimeDataset(ts_df=train_data, flat_df=flat_data, graph=graph, label=output_index, window_size=args.window_size, time_index=kfold_valid_time)

    model = getattr(models , 'RnnGnn')(args, input_size).to(device)
    loss_fn = get_loss(args.loss_name)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler)(optimizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers #pin_memory=True
    )
    
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, scaling_fn, device, args.patience, args.epochs, args.base_epochs, result_path, logger, len(train_dataset), len(valid_dataset), use_ray=True)
    return trainer.train()

def tune_args(args):
    # data
    args.window_size = tune.randint(10, 128)

    # training
    args.lr = tune.uniform(1e-5, 1e-2)

    # graph
    args.graph = tune.choice(["EU_ts_weighted_knn", "ts_all_weighted_knn"])
    args.k = tune.randint(2, 90)
    args.sim = tune.choice(['minkowski', 'cosine'])
    #args.graph_type = tune.choice(['graph', 'directed'])
    args.graph_type = 'directed'

    # model
    args.model = tune.choice(['LSTM', 'GRU'])
    args.GNN = tune.choice(["GCN", "GraphSAGE", "GIN", "GAT", "EdgeCNN"])
    
    # LSTM & GRU
    args.pooling = tune.choice(['mean', 'max', 'last', 'first'])
    args.dropout = tune.uniform(0.2, 0.8)
    args.hidden = tune.randint(10, 512)
    args.num_layers = tune.randint(1, 4)

    # GNN
    args.gnn_hidden = tune.randint(10, 512)
    args.gnn_n_layers = tune.randint(1, 3)
    args.gnn_output_size = tune.randint(10, 512)
    args.gnn_drop_p = tune.uniform(0.2, 0.8)
    args.norm = tune.choice(["BatchNorm", "InstanceNorm", "LayerNorm", None])
    args.jk = tune.choice(["cat", "max", "lstm", None])
    args.emb_dim = tune.randint(10, 512)
    # args.flat_out = tune.randint(10, 512)
    # args.gnn_input = tune.choice([])
    # args.outputs_after_gnn = tune.choice([])
    # args.aggr = tune.choice([])

    return args

def align_args(config_args, args):
    for key in args.keys():
        if key not in config_args:
            config_args[key] = args[key]
    return config_args

if __name__ == '__main__':
    args = get_args()
    args.patience = -1
    args.num_workers = 0
    path = os.path.join(args.result_path, 'tuning_'+args.comment+'_'+str(len(os.listdir(args.result_path))))
    args.result_path = path
    os.makedirs(path)

    search_space = tune_args(args).__dict__

    with open('base_model.json') as json_file:
        config_args = json.load(json_file)

    optuna_search = OptunaSearch(
        points_to_evaluate=[align_args(config_args, args.__dict__)], 
        sampler=optuna.samplers.TPESampler()
    )

    scheduler = ASHAScheduler(
        max_t=args.epochs,
    )

    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=-1 if args.n_trials==None else args.n_trials, 
        time_budget_s=args.timeout,
        search_alg=optuna_search,
        scheduler=scheduler 
    )

    run_config = air.RunConfig(
        storage_path=path, 
        name=path.split('/')[-1],
        log_to_file=os.path.join(path, "log.log")
    )
    
    trainable = tune.with_resources(main, {"gpu": args.gpu_ratio})
    tuner = tune.Tuner(main, param_space=search_space, tune_config=tune_config, run_config=run_config)
    results = tuner.fit()

    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.log_dir  # Get best trial's logdir
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    df_results = results.get_dataframe()
    # Get a dataframe of results for a specific score or mode
    results.get_dataframe(filter_metric="loss", filter_mode="min").to_csv(os.path.join(path, 'result.csv'), index=False)