import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from supervised.automl import AutoML


def custom_metric(y_true, y_predicted, sample_weight=None):
    v = 2 * abs(y_predicted - y_true) / ((abs(y_predicted) + abs(y_true)) + 1e-9)
    output = np.mean(v) * 100
    return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default= None)
    parser.add_argument("--mode", default= 'Compete', choices=['Optuna', 'Compete', 'Explain'])
    parser.add_argument("--time_limit", default= 5*60*60, type=int) #act diffrent for mode optuna and compete

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--submission", required=True)

    parser.add_argument("--target_col", required=True)
    parser.add_argument("--drop", nargs='*')

    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--eval_metric", default='mae')
    parser.add_argument("--use_custom_metric", action='store_true')

    parser.add_argument('--building_no', type=int, required=True)
    parser.add_argument('--n_jobs', type=int, required=True)


def run(index, args, data):
    automl = AutoML(mode=args.mode, eval_metric=custom_metric, total_time_limit=args.time_limit, optuna_time_budget=args.time_limit, random_state=args.seed, results_path=os.path.join(args.name, 'building'+str(index)), n_jobs=args.n_jobs)
    automl.fit(data['x'], data['y'])
    return automl

def process_data(args):
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)

    train_building = train[train['건물번호']==args.building_no]
    test_building = test[test['건물번호']==args.building_no]

    if args.drop is not None:
        train_x = train_building.drop(columns=args.drop + [args.target_col], axis = 1)
        test_x = test_building.drop(columns=args.drop, axis = 1)
    else:
        train_x = train_building.drop(columns=[args.target_col], axis = 1)
        test_x = test_building

    return {'x': train_x, 'y': train_building[args.target_col]}, {'x': test_x}

if __name__ == '__main__':
    args = get_args()

    train, test = process_data(args)
    models = run(args.building_no, args, train)

    pred = models.predict(test['x'])
    submission = pd.read_csv(args.submission)
    submission['answer'] = pred
    
    submission.to_csv(f"{args.name}_{args.building_no}_submission.csv", index=False)
