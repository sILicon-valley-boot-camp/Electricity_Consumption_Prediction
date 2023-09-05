# Using AutoML with GNN Stacking Ensemble meta model 
![스크린샷 2023-09-02 112739](https://github.com/sILicon-valley-boot-camp/Electricity_Consumption_Prediction/assets/67096173/d146fe98-957c-4509-b33c-a605f94c45a9)
- 위의 모델의 역할
  - AutoML 만을 사용했을때는 이전의 정보를 사용하지 않는다는 점과 동일한 시간대에 다른 빌딩의 정보를 사용하지 못한다는 한계점을 가지고 있기 때문에 RNN-GNN Stacking Ensemble을 통해 AutoML과 DeepLearning 둘 모두의 장점을 취했습니다. 
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

- features
  - time-series feature을 기반으로 weighted directed knn graph 생성 (빌딩과 빌딩과의 유사성을 나타내는 100개의 노드를 가진 그래프) → 해당 그래프를 바탕으로 GNN 수행
  - ts encoder (LSTM, GRU, RNN, Transformer), graph encoder(GCN, GAT, GraphSAGE, GIN, EdgeCNN), 그래프 생성 파라미터, 모델별 파라미터 → Hyper parameter tuning
- Stacking Ensemble의 성능
  - 기존 AutoML의 성과에서 추가적으로 0.05% ~ 0.125% 성능 Boosting

- 자세한 내용은 아래 블로그 참고해주세요!
  
  https://doun-lee.notion.site/Predicting-Electricity-Consumption-Using-Automl-with-Graph-representation-Learning-088b5f56acdf4a5091249ee23e44d850?pvs=4
