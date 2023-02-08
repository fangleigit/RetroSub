#!/bin/bash

# current directory
CURRENT_DIR=$PWD
# script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# change to script directory
cd $SCRIPT_DIR
# retrieve top 20 candidates
top=20

retrieval_model_dir="../ckpts/uspto_full/dual_encoder/epoch116_batch349999_acc0.79"
data_dir="../data/uspto_full"


# run the retrieval script
python_dir=`conda env list|grep retrieval| awk  '{print $2}'`
$python_dir/bin/python3 -u ../RetrievalModel/search_index.py \
        --input_file test_input_dual_encoder.txt \
        --output_file test_input_dual_encoder.top${top}.txt \
        --ckpt_path ${retrieval_model_dir}/query_encoder \
        --args_path ${retrieval_model_dir}/args \
        --vocab_path ${data_dir}/retrieval/src.vocab \
        --index_file ${data_dir}/candidates.txt \
        --index_path ${retrieval_model_dir}/mips_index \
        --topk ${top} \
        --allow_hit \
        --batch_size 1024
# change back to current directory
cd $CURRENT_DIR