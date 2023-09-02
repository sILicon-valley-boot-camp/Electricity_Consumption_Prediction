import argparse
import pandas as pd
from supervised.automl import AutoML

def process_oof(task, oof):
    if task == "binary_classification":
        cols = [f for f in oof.columns if "prediction" in f]
        if len(cols) == 2:
            oof = pd.DataFrame({"prediction": oof[cols[1]]})

    cols = [f for f in oof.columns if "prediction" in f]
    oof = oof[cols]
    oof.columns = [f"{m.get_name()}_{c}" for c in cols]
    return oof        

parser = argparse.ArgumentParser()
parser.add_argument("--name", default= 'AutoML_1')
parser.add_argument("--test_csv")
args = parser.parse_args()

automl = AutoML()
automl.results_path = f"./{args.name}"
automl.load(automl.results_path)
automl._perform_model_stacking()

test_data = pd.read_csv(args.test_csv)
test_data = automl._build_dataframe(test_data) #actually this code is not necessary, but it might do something usefull so I just copied it
input_columns = test_data.columns.tolist()
for column in automl._data_info["columns"]:
    if column not in input_columns:
            raise AutoMLException(f"Missing column: {column} in input data. Cannot predict")
test_data = test_data[automl._data_info["columns"]]
stacked_test_data = automl.get_stacked_data(test_data, mode="predict")

train_all_oofs = []
test_all_oofs = []
for m in automl._stacked_models + [model for model in automl._models if "Stacked" in model.get_name()]:
    train_oof = m.get_out_of_folds()
    
    if m._is_stacked:
        if m.get_type() == "Ensemble":
            test_oof = m.predict(test_data, stacked_test_data)
        else:
            test_oof = m.predict(stacked_test_data)
    else:
        test_oof = m.predict(test_data)

    train_all_oofs += [process_oof(automl._ml_task, train_oof)]
    test_all_oofs += [process_oof(automl._ml_task, test_oof)]

pd.concat(train_all_oofs, axis=1).to_csv(f"{args.name}_stacking_input_train.csv", index=False)
pd.concat(test_all_oofs, axis=1).to_csv(f"{args.name}_stacking_input_test.csv", index=False)