#!/bin/bash

for num in {1..50}
do
    python run_automl_per_building.py --name smape_automl_building --mode Compete --train_csv ../../data/train_ver3_for_automl.csv --test_csv ../../data/ver7_train.csv --submission ../../data/sample_submission.csv --target_col '전력소비량(kWh)' --drop 건물번호 일시 --use_custom_metric --n_jobs 1 --time_limit 7200 --building_no ${num} &
    if [[ $(jobs -r -p | wc -l) -ge 4 ]] # running process's pid of current shell | word count of lines is Greater Than or Equal $N
    then 
        wait -n # as soon as one task is done, refill it with another
    fi
done
