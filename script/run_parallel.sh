#!/bin/bash

N = $1
for k in 5 3 7
do
    for graph in knn EU_mean_knn EU_ts_knn ts_knn ts_all_knn node_emb
    do
        python -u main.py --train ../data/train_ver2.csv --test ../data/test_ver2.csv --flat ../data/building_info_ver2.csv --lr 1e-3 --model LSTM --GNN GCN --window_size 48 --gnn_input enc_out node_emb --outputs_after_gnn enc_out node_emb --loss smape --graph ${graph} --cv_k 3 --k ${$1} --comment graph=${$1}_${graph} &
        if [[ $(jobs -r -p | wc -l) -ge $N ]] # running process's pid of current shell | word count of lines is Greater Than or Equal $N
        then 
            wait -n # as soon as one task is done, refill it with another
        fi
    done
done