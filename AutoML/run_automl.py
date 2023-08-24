import os
import random
import argparse
import pandas as pd
import numpy as np

from supervised.automl import AutoML

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

def custom_metric(y_true, y_predicted, sample_weight=None):
    v = 2 * abs(y_predicted - y_true) / ((abs(y_predicted) + abs(y_true)) + 1e-9)
    output = np.mean(v) * 100
    return output

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
parser.add_argument("--use_custom_metric", action='store_ture')

args = parser.parse_args()

train_data = pd.read_csv(args.train_csv)
test_data = pd.read_csv(args.test_csv)

if args.drop is not None:
    train_x = train_data.drop(columns=args.drop + [args.target_col], axis = 1)
    test_x = test_data.drop(columns=args.drop, axis = 1)
else:
    train_x = train_data.drop(columns=[args.target_col], axis = 1)
    test_x = test_data
    
train_y = train_data[args.target_col]

if args.use_custom_metric: # https://github.com/mljar/mljar-supervised/issues/390#issuecomment-830049603
    automl = AutoML(mode=args.mode, eval_metric=custom_metric, total_time_limit=args.time_limit, optuna_time_budget=args.time_limit, eval_metric=args.eval_metric, random_state=args.seed, results_path=args.name)
else:
    automl = AutoML(mode=args.mode, total_time_limit=args.time_limit, optuna_time_budget=args.time_limit, eval_metric=args.eval_metric, random_state=args.seed, results_path=args.name)
automl.fit(train_x, train_y)

pred = automl.predict(test_x)
submission = pd.read_csv(args.submission)
submission[args.target_col] = pred
submission.to_csv(f"{automl.results_path}_submission.csv", index=False)