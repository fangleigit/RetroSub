#!/bin/bash

ROOT=.
dataset=uspto_full
dataset_dir=${ROOT}/data/${dataset}
dataset_prefix=${dataset_dir}/retrieval
retrieval_model_dir=${ROOT}/ckpts/${dataset}/dual_encoder

ckpt_folder=$1

# retrievel top 20 candidates
top=20

# collect candidates from the target of train and val data
python3 data_utils/collect_candidates.py --train_target ${dataset_dir}/tgt-train.txt \
	--val_target ${dataset_dir}/tgt-val.txt \
	--candidate_file ${dataset_dir}/candidates.txt

# build the index 
python3 RetrievalModel/build_index.py \
        --input_file ${dataset_dir}/candidates.txt \
        --ckpt_path ${retrieval_model_dir}/${ckpt_folder}/response_encoder \
        --args_path ${retrieval_model_dir}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/tgt.vocab \
        --index_path ${retrieval_model_dir}/${ckpt_folder}/mips_index \
        --max_training_instances 30000000 \
        --batch_size 512

python3 RetrievalModel/build_index.py \
        --input_file ${dataset_dir}/candidates.txt\
        --ckpt_path ${retrieval_model_dir}/${ckpt_folder}/response_encoder \
        --args_path ${retrieval_model_dir}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/tgt.vocab \
        --index_path ${retrieval_model_dir}/${ckpt_folder}/mips_index \
        --max_training_instances 30000000 \
        --batch_size 512 \
        --only_dump_feat

# query the index
for prefix in train val test
do
    python3 -u RetrievalModel/search_index.py \
        --input_file ${dataset_prefix}/${prefix}.txt \
        --output_file ${dataset_prefix}/${prefix}.top${top}.txt \
        --ckpt_path ${retrieval_model_dir}/${ckpt_folder}/query_encoder \
        --args_path ${retrieval_model_dir}/${ckpt_folder}/args \
        --vocab_path ${dataset_prefix}/src.vocab \
        --index_file ${dataset_dir}/candidates.txt \
        --index_path ${retrieval_model_dir}/${ckpt_folder}/mips_index \
        --topk ${top} \
        --allow_hit \
        --batch_size 1024
    python3 data_utils/collect_topk.py --input ${dataset_prefix}/${prefix}.top${top}.txt \
        --dataset ${prefix} \
        --topk ${top} \
        --output_path ${dataset_prefix}
done