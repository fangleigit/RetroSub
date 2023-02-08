#!/bin/bash

gpu=$1
bz=$2

# current directory
CURRENT_DIR=$PWD
# script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# change to script directory
cd $SCRIPT_DIR

python ../MolecularTransformer/translate.py \
            -model ../models/uspto_full_retrosub.pt \
            -src test_input_seq2seq.txt \
            -output predict_output.txt \
            -batch_size $bz -replace_unk -max_length 200 -fast -n_best 10 -beam_size 10  -gpu $gpu
cd $CURRENT_DIR