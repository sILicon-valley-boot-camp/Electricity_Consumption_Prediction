import os
import sys
import joblib
import optuna
import logging
import pandas as pd
from functools import partial
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

from optuna.visualization import plot_edf
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_timeline
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history

class SaveVisCallback:
    def __init__(self, path):
        self.path = path

    def visualize_optuna(self, study):
        for method, vis in zip(
            ['plot_edf', 'plot_rank', 'plot_slice', 'plot_contour', 'plot_timeline', 'plot_param_importances', 'plot_intermediate_values', 'plot_parallel_coordinate', 'plot_optimization_history'],
            [plot_edf, plot_rank, plot_slice, plot_contour, plot_timeline, plot_param_importances, plot_intermediate_values, plot_parallel_coordinate, plot_optimization_history]
         ):
            try:
                fig = vis(study)
                fig.write_image(file=os.path.join(self.path, f'{method}.png'), format='png') 
            except Exception as e:
                print(f'Error using {method}\n{e}')

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        joblib.dump(study, os.path.join(self.path, f"study.pkl"))

        if (trial.number+1) % 10 == 0:
            self.visualize_optuna(study)

def main(trial, args=None):
    args = tune_args(args, trial)
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

    train_dataset = GraphTimeDataset(ts_df=train_data, flat_df=flat_data, graph=graph, label=output_index, window_size=args.window_size, time_index=kfold_train_time, device=device)
    valid_dataset = GraphTimeDataset(ts_df=train_data, flat_df=flat_data, graph=graph, label=output_index, window_size=args.window_size, time_index=kfold_valid_time, device=device)

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
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, scaling_fn, device, args.patience, args.epochs, args.base_epochs, result_path, logger, len(train_dataset), len(valid_dataset), trial)
    
    return trainer.train()

def tune_args(args, trial):
    # data
    #args.window_size = trial.suggest_int("window_size", 10, 128)
    args.window_size = 24

    # training
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2)

    # graph
    args.graph = trial.suggest_categorical("graph", ["EU_ts_weighted_knn", "ts_all_weighted_knn"])
    args.k = trial.suggest_int("k", 2, 90)
    args.sim = trial.suggest_categorical("sim", ['minkowski', 'cosine'])
    #args.graph_type = trial.suggest_categorical("graph_type", ['graph', 'directed'])
    args.graph_type = 'directed'

    # model
    args.model = trial.suggest_categorical("model", ['LSTM', 'GRU', 'transformer'])
    args.GNN = trial.suggest_categorical("GNN", ["GCN", "GraphSAGE", "GIN", "GAT", "EdgeCNN"])
    args.dropout = trial.suggest_float("dropout", 0.2, 0.8)
    args.pooling = trial.suggest_categorical("pooling", ['mean', 'max', 'last', 'first'])
    
    # LSTM & GRU
    if args.model=='LSTM' or args.model=='GRU':
        args.hidden = trial.suggest_int("hidden", 10, 512)
        args.num_layers = trial.suggest_int("num_layers", 1, 4)
    else:
        args.n_head = 2 #trial.suggest_int("n_head", 2, 8, 2)
        args.num_layers = trial.suggest_int("Transformer_num_layers", 1, 4)
        args.transformer_dropout = trial.suggest_float("transformer_dropout", 0.2, 0.8)

    # GNN
    args.gnn_hidden = trial.suggest_int("gnn_hidden", 10, 512)
    args.gnn_n_layers = trial.suggest_int("gnn_n_layers", 1, 3)
    args.gnn_output_size = trial.suggest_int("gnn_output_size", 10, 512)
    args.gnn_drop_p = trial.suggest_float("gnn_drop_p", 0.2, 0.8)
    args.norm = trial.suggest_categorical("norm", ["BatchNorm", "InstanceNorm", "LayerNorm", None])
    args.jk = trial.suggest_categorical("jk", ["cat", "max", "lstm", None])
    args.emb_dim = trial.suggest_int("emb_dim", 10, 512)
    # args.flat_out = trial.suggest_int("flat_out", 10, 512)
    # args.gnn_input = trial.suggest_categorical("gnn_input", [])
    # args.outputs_after_gnn = trial.suggest_categorical("outputs_after_gnn", [])
    # args.aggr = trial.suggest_categorical("aggr", [])

    return args

if __name__ == '__main__':
    args = get_args()
    args.patience = -1
    args.num_workers = 0

    if args.continue_tune_from is None:
        path = os.path.join(args.result_path, 'tuning_'+args.comment+'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(path)
        args.result_path = path
        objective =  partial(main,args=args)
        callback = SaveVisCallback(path)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed), pruner=optuna.pruners.HyperbandPruner())
    
    else:
        path = args.continue_tune_from
        args.result_path = path
        objective =  partial(main,args=args)
        callback = SaveVisCallback(path)    
        study = joblib.load(os.path.join(path, "study.pkl"))
    
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_job_parallel, callbacks=[callback], show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    callback.visualize_optuna(study)