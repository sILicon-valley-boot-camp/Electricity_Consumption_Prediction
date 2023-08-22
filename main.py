import os
import sys
import logging
import pandas as pd
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold

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

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.comment + '_' + args.model + args.GNN + '_' + str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    flat_data = pd.read_csv(args.flat)
    args.flat_dim = len(flat_data.columns)

    train_data = pd.read_csv(args.train)
    train_data['일시'] = pd.to_datetime(train_data['일시'])
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

    prediction = pd.read_csv(args.submission)
    stackking_input = pd.DataFrame(columns = [output_index], index=range(len(train_data))) #dataframe for saving OOF predictions

    if args.continue_train > 0:
        prediction = pd.read_csv(os.path.join(result_path, 'sum.csv'))
        stackking_input = pd.read_csv(os.path.join(result_path, f'for_stacking_input.csv'))

    graph = get_graph(args, train_data, flat_data, result_path) if args.graph != 'node_emb' else None

    skf = KFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True) #Using StratifiedKFold for cross-validation    
    for fold, (train_index, valid_index) in enumerate(skf.split(train_time)): #using the target_data's index for kfold cross validation split
        kfold_train_time = train_time[train_index]
        kfold_valid_time = train_time[valid_index]
        
        if args.continue_train > fold+1:
            logger.info(f'skipping {fold+1}-fold')
            continue
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')
        #logger to log current n-fold output

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
            train_loader, valid_loader, model, loss_fn, optimizer, scheduler, scaling_fn, device, args.patience, args.epochs, fold_result_path, fold_logger, len(train_dataset), len(valid_dataset))
        trainer.train() #start training

        test_dataset = GraphTimeDataset(ts_df=test_data, flat_df=flat_data, graph=graph, label=output_index, window_size=args.window_size, time_index=test_time, device=device)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ) #make test data loader

        prediction['answer'] += trainer.test(test_loader).squeeze(-1)
        prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 
        
        stackking_input.loc[train_data['일시'].isin(train_time[valid_index]), output_index] = trainer.test(valid_loader).squeeze(-1) #use the validation data(hold out dataset) to make input for Stacking Ensemble model(out of fold prediction)
        #may need testing with trainer.inference()
        stackking_input.to_csv(os.path.join(result_path, f'for_stacking_input.csv'), index=False)

prediction['answer'] = prediction['answer'] / 10 #use the most likely results as my final prediction
prediction.to_csv(os.path.join(result_path, 'prediction.csv'), index=False)