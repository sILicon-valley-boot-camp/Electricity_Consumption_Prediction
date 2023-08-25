import os
import argparse
import pandas as pd
from glob import glob
from supervised.automl import AutoML

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default= None)
    parser.add_argument("--test_csv", default= None)
    parser.add_argument("--submission", default= None)
    parser.add_argument("--start_from", type=int)
    parser.add_argument("--end_to", type=int)
    parser.add_argument("--drop", nargs='*')
    return parser.parse_args()

def process_data(test, num, args):
    test_building = test[test['건물번호']==num]

    buildings = [f'building_num_{i}' for i in range(1, 101)]
    if args.drop is not None:
        test_x = test_building.drop(columns=args.drop + buildings, axis = 1)
    else:
        test_x = test_building.drop(columns=buildings, axis = 1)
    return test_x


if __name__ == '__main__':
    args = get_args()
    test = pd.read_csv(args.test_csv)
    submission = pd.read_csv(args.submission)

    for num in range(args.start_from, args.end_to+1):
        path = os.path.join(args.name, 'building'+str(num))
        automl = AutoML()
        automl.results_path = path
        automl.load(path)

        building_data = process_data(test, num, args)
        pred = automl.predict(building_data)
        submission = pd.read_csv(args.submission)
        submission.loc[building_data.index, 'answer'] = pred
        
        submission.to_csv(f"{args.name}_{args.start_from}_{args.end_to}_submission.csv", index=False)