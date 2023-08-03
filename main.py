import os
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import models
from config import get_args
from trainer import Trainer
from lr_scheduler import get_sch
from utils import seed_everything
from data import DataSet, TestDataSet

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    #logger to log result of every output

    train_data = pd.read_csv(args.train)
    train_data['일시'] = pd.to_datetime(train_data['일시'])
    train_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
    train_start = min(train_data['일시'])
    train_end = max(train_data['일시'])

    output_index = '전력소비량(kWh)'
    scaling_col = set(train_data.columns) ^ {'num_date_time', '건물번호', '일시', '전력소비량(kWh)'}
    input_size = len(scaling_col)+1
    data_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    train_data[scaling_col] = data_scaler.fit_transform(train_data[scaling_col])
    train_data[output_index] = target_scaler.fit_transform(train_data[output_index])

    test_data = pd.read_csv(args.test)
    test_data['일시'] = pd.to_datetime(test_data['일시'])
    test_start = min(test_data['일시'])
    test_end = max(test_data['일시'])

    test_data[scaling_col] = data_scaler.transform(test_data[scaling_col])

    total_data = pd.concat([train_data, test_data], join='inner').reset_index(drop=True)
    total_data.sort_values(by=['건물번호', '일시'], inplace=True, ignore_index=True)
    test_time = pd.date_range(test_start - pd.Timedelta(hours=(args.window_size-1)) , test_end, freq='H') #for compatibility
    test_data = total_data[total_data['일시'].isin(test_time)].reset_index(drop=True)

    train_time = pd.date_range(train_start + pd.Timedelta(hours=(args.window_size)) , train_end, freq='H') #args.window_size의 이후의 시간부터 loss function에 제공
    train_target_data = train_data[train_data['일시'].isin(train_time)] #train_start time_stamp's data will not be used(except electricity consumption) 

    prediction = pd.read_csv(args.submission)
    stackking_input = pd.DataFrame(columns = [output_index], index=range(len(train_data))) #dataframe for saving OOF predictions

    if args.continue_train > 0:
        prediction = pd.read_csv(os.path.join(result_path, 'sum.csv'))
        stackking_input = pd.read_csv(os.path.join(result_path, f'for_stacking_input.csv'))

    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True) #Using StratifiedKFold for cross-validation    
    for fold, (train_index, valid_index) in enumerate(skf.split(train_target_data.index, train_target_data['건물번호'])): #using the target_data's index for kfold cross validation split
        kfold_train_index = train_target_data.index[train_index]
        kfold_valid_index = train_target_data.index[valid_index]
        
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

        train_dataset = DataSet(data=train_data, label=output_index, window_size=args.window_size, target_index=kfold_train_index)
        valid_dataset = DataSet(data=train_data, label=output_index, window_size=args.window_size, target_index=kfold_valid_index)

        model = getattr(models , args.model)(args, input_size).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = get_sch(args.scheduler)(optimizer)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
        )
        
        trainer = Trainer(
            train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, fold_result_path, fold_logger, len(train_dataset), len(valid_dataset))
        trainer.train() #start training

        test_dataset = TestDataSet(data=test_data, window_size=args.window_size, test_start=test_start, test_end=test_end)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ) #make test data loader

        prediction['answer'] += target_scaler.inverse_transform(trainer.test(test_loader)) #softmax applied output; accumulate test prediction of current fold model
        prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 
        
        stackking_input.loc[valid_index, output_index] = target_scaler.inverse_transform(trainer.test(valid_loader)) #use the validation data(hold out dataset) to make input for Stacking Ensemble model(out of fold prediction)
        stackking_input.to_csv(os.path.join(result_path, f'for_stacking_input.csv'), index=False)

        '''np.savez_compressed(os.path.join(fold_result_path, 'test_prediction'), trainer.test(test_loader))
        np.savez_compressed(os.path.join(fold_result_path, 'valid_prediction'), trainer.test(valid_loader))
        np.savez(os.path.join(fold_result_path, 'valid_index'), valid_index)''' # case when output size is big

'''prediction['label'] = np.argmax(test_result, axis=-1) #use the most likely results as my final prediction
prediction.drop(columns=output_index).to_csv(os.path.join(result_path, 'prediction.csv'), index=False)''' #classification