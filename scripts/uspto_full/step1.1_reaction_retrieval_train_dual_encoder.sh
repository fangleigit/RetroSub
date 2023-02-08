#!/bin/bash

ROOT=.
dataset=uspto_full
dataset_dir=${ROOT}/data/${dataset}
dataset_prefix=${dataset_dir}/retrieval
mkdir -p ${dataset_prefix}

# prepare data
python3 RetrievalModel/prepare.py --train_data_src ${dataset_dir}/src-train.txt \
	--train_data_tgt ${dataset_dir}/tgt-train.txt \
	--vocab_src ${dataset_prefix}/src.vocab \
	--vocab_tgt ${dataset_prefix}/tgt.vocab \
	--max_len 1024 \
	--ratio 1000 \
	--output_file ${dataset_prefix}/train.txt
paste -d '\t' ${dataset_dir}/src-val.txt ${dataset_dir}/tgt-val.txt > ${dataset_prefix}/val.txt
paste -d '\t' ${dataset_dir}/src-test.txt ${dataset_dir}/tgt-test.txt > ${dataset_prefix}/test.txt

# train retrieval model
retrieval_model_dir=${ROOT}/ckpts/${dataset}/dual_encoder
mkdir -p ${retrieval_model_dir}

python3 -u RetrievalModel/pretrain.py --train_data ${dataset_prefix}/train.txt \
        --dev_data  ${dataset_prefix}/val.txt \
        --src_vocab ${dataset_prefix}/src.vocab \
        --tgt_vocab ${dataset_prefix}/tgt.vocab \
        --ckpt ${retrieval_model_dir} \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 256 \
        --total_train_steps 500000 \
        --layers 3 \
        --per_gpu_train_batch_size 256 \
        --bow