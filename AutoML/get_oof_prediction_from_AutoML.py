import argparse
import pandas as pd
from supervised.automl import AutoML

parser = argparse.ArgumentParser()
parser.add_argument("--name", default= 'AutoML_1')
parser.add_argument("--mode", default= 'train')
parser.add_argument("--csv_data")
args = parser.parse_args()

automl = AutoML()
automl.results_path = f"./{args.name}"
automl.load(automl.results_path)
automl._perform_model_stacking()

if args.mode == "test":
    test_data = pd.read_csv(args.csv_data)
    test_data = automl._build_dataframe(test_data) #actually this code is not necessary, but it might do something usefull so I just copied it
    input_columns = test_data.columns.tolist()
    for column in automl._data_info["columns"]:
        if column not in input_columns:
                raise AutoMLException(f"Missing column: {column} in input data. Cannot predict")
    test_data = test_data[automl._data_info["columns"]]

all_oofs = []
for m in automl._stacked_models + [model for model in automl._models if model.get_name() == "Ensemble_Stacked"]:
    oof = None
    if args.mode=="train":
        oof = m.get_out_of_folds()
    else:
        if m.get_name() == "Ensemble_Stacked":
            oof = automl._base_predict(test_data)
        else:
            oof = m.predict(test_data)
        if automl._ml_task == "binary_classification":
            cols = [f for f in oof.columns if "prediction" in f]
            if len(cols) == 2:
                oof = pd.DataFrame({"prediction": oof[cols[1]]})

    cols = [f for f in oof.columns if "prediction" in f]
    oof = oof[cols]
    oof.columns = [f"{m.get_name()}_{c}" for c in cols]
    all_oofs += [oof]

oof_predictions = pd.concat(all_oofs, axis=1)
oof_predictions.to_csv(f"{args.name}_stacking_input_{args.mode}.csv", index=False)