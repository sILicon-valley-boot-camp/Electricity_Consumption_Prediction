import os
import random
import argparse
import pandas as pd
import numpy as np

from supervised.automl import AutoML

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument("--name", default= None)

parser.add_argument("--train_csv", required=True)
parser.add_argument("--test_csv", required=True)
parser.add_argument("--submission", required=True)

parser.add_argument("--target_col", required=True)
parser.add_argument("--drop", nargs='*')

parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--eval_metric", default='mae')

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

automl = AutoML(mode="Optuna", optuna_time_budget=3600, eval_metric=args.eval_metric, random_state=args.seed, results_path=args.name)
automl.fit(train_x, train_y)

pred = automl.predict(test_x)
submission = pd.read_csv(args.submission)
submission[args.target_col] = pred
submission.to_csv(f"{automl.results_path}_submission.csv", index=False)