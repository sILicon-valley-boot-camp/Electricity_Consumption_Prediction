import os
import argparse
import pandas as pd
from glob import glob
from supervised.automl import AutoML


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default= None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    folders = glob(args.name+'/*')

    model_name, metric_value = [], []
    for folder in folders:
        leaderboard = pd.read_csv(os.path.join(folder, "leaderboard.csv"))
        best_model = leaderboard[leaderboard['metric_value']==leaderboard['metric_value'].min()]
        model_name.append(best_model['name'])
        metric_value.append(best_model['metric_value'])

    result = pd.DataFrame()
    result['folder'] = folders
    result['best_model'] = model_name
    result['smape'] = metric_value

    print(f"mean: {result['smape'].mean()}")
