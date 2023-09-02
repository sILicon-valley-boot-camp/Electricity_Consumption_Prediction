# Using AutoML with GNN Stacking Ensemble meta model 

- running AutoML
```
cd AutoML
python run_automl.py --train_csv {train file} --test_csv {test file} --submission ../../data/sample_submission.csv --target_col '전력소비량(kWh)' --drop num_date_time 건물번호 day hour day_of_week 일시 --name smae_compete_v2 --use_custom_metric --mode Compete --time_limit 600
```
- getting automl's out-of-fold predictions
```
cd AutoML
python get_oof_prediction_from_AutoML.py --name mae_compete --mode test --csv_data ../../data/test_ver3_for_automl.csv
```
- Stacking Ensemble Hyperparameter tuning
```
python -u hyper_parameter_optuna.py --train ../data/ver13_train.csv --test ../data/ver13_test.csv --flat ../data/building_info_ver2.csv --gnn_input enc_out node_emb --outputs_after_gnn enc_out node_emb --loss smape --scaler_name MinMax --batch_size 10 --comment ver13_optuna_epoch_wise_test --test_ratio 0.3  --epochs 15 --timeout 84000
```
- 10-fold cv training
```
python main.py --config {hyperparameter tuned best model path}/config.json --target_args epochs base_epochs comment --comment run_best --base_epochs 100
```
