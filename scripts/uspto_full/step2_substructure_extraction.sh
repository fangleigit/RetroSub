#!/bin/bash

ROOT=.
chunk_id=$1
total_chunks=$2
dataset_dir=${ROOT}/data/uspto_full
sub_store_path=${ROOT}/data/uspto_full/$3/

dataset_prefix=${dataset_dir}/retrieval

mkdir -p $sub_store_path
for name in test val train
do
    python3 subext.py --input_file ${dataset_prefix}/$name.json \
                         --dataset $name \
                         --store_path $sub_store_path \
                         --reactions $dataset_dir/reaction.pkl \
                         --nprocessors 17 \
                         --total_chunks $total_chunks \
                         --chunk_id $chunk_id
    
    python3 data_utils/generate_training_data.py --dataset $name \
                         --store_path $sub_store_path \
                         --out_dir $sub_store_path \
                         --total_chunks $total_chunks \
                         --chunk_id $chunk_id \
                         --data_aug
    
    python data_utils/generate_stat.py --dataset $name \
                            --store_path $sub_store_path \
                            --out_dir $sub_store_path \
                            --total_chunks $total_chunks \
                            --chunk_id $chunk_id
done