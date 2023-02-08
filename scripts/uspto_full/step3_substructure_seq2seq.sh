#!/bin/bash

dataset=uspto_full
dataset_dir=./data/${dataset}/$1
transformer_model_dir=./ckpts/${dataset}/$1
layers=$2

python MolecularTransformer/preprocess.py -train_src ${dataset_dir}/src-train.txt \
                     -train_tgt ${dataset_dir}/tgt-train.txt \
                     -valid_src ${dataset_dir}/src-val.txt \
                     -valid_tgt ${dataset_dir}/tgt-val.txt \
                     -save_data ${dataset_dir}/$dataset \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
                     -shard_size 0


python  MolecularTransformer/train.py -data ${dataset_dir}/${dataset} \
                   -save_model ${transformer_model_dir}/${dataset} \
                   -seed 42 -world_size 8 -gpu_ranks 0 1 2 3 4 5 6 7 -save_checkpoint_steps 10000 -keep_checkpoint 50 \
                   -train_steps 500000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -batch_size 8192 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 2 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers ${layers} -rnn_size 512 -word_vec_size 512 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048