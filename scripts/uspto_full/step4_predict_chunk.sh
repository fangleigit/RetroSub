model_name=$1
chunk_id=$2
total_chunks=$3
subextr_path=$4
gpu=$5
result_dir=$6
data_split=test

echo ${result_dir}
mkdir -p ${result_dir}
test_input=${subextr_path}/src-${data_split}_${chunk_id}_${total_chunks}_unique.txt
subs_input=${subextr_path}/${data_split}_${chunk_id}_${total_chunks}_info.pkl

predict_output=${result_dir}/${data_split}_${chunk_id}_${total_chunks}_prediction.txt

rm -f ${predict_output}
echo using gpu $gpu

CUDA_VISIBLE_DEVICES=$gpu python MolecularTransformer/translate.py -model models/${model_name}.pt \
                                        -src ${test_input} \
                                        -output ${predict_output} \
                                        -batch_size 32 -replace_unk -max_length 200 -fast -n_best 10 -beam_size 10 -gpu 0

sleep 1m

t_file=${result_dir}/${data_split}_${chunk_id}_${total_chunks}_prediction_res_False.json
if [ ! -f ${t_file} ]
then
python evaluate_chunk.py --predictions ${predict_output} \
                              --substructures  ${subs_input} \
                              --n_best 10 \
                              --test_input ${test_input}
fi

t_file=${result_dir}/${data_split}_${chunk_id}_${total_chunks}_prediction_res_True.json
if [ ! -f ${t_file} ]
then
python evaluate_chunk.py --predictions ${predict_output} \
                              --substructures  ${subs_input} \
                              --n_best 10 \
                              --test_input ${test_input} \
                              --use_oracle
fi