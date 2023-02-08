import datetime
import os
import sys
from os.path import join

from tqdm import tqdm

from merge_training_data import training_data_loader

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.common_utils import parse_config

###
# python merge_training_data2.py --total_chunks 200 --out_dir ./data/uspto_full/subextraction
###

if __name__ == "__main__":
    start = datetime.datetime.now()
    args = parse_config()
    num_chunks = args.total_chunks
    saved_folder = args.out_dir

    with open(join(saved_folder, f'src-train.txt'), 'w') as src_f, \
                open(join(saved_folder, f'tgt-train.txt'), 'w') as tgt_f:
        total_instance = 0   
        for split in ['val', 'train']:
            src_list = [line for line in training_data_loader(saved_folder, split,
                                     num_chunks, 'src-', '.txt')]
            tgt_list = [line for line in training_data_loader(saved_folder, split,
                                     num_chunks, 'tgt-', '.txt')]
            assert len(src_list) == len(tgt_list)
            for src, tgt in tqdm(zip(src_list,tgt_list)):
                src_f.write(src)
                tgt_f.write(tgt)
                total_instance += 1
        print(f'train: {total_instance}')

    with open(join(saved_folder, f'src-val.txt'), 'w') as src_f, \
            open(join(saved_folder, f'tgt-val.txt'), 'w') as tgt_f:
        total_instance = 0   
        split = 'val'  
        src_list = [line for line in training_data_loader(saved_folder, split,
                                     num_chunks, 'src-', '.txt')]
        tgt_list = [line for line in training_data_loader(saved_folder, split,
                                    num_chunks, 'tgt-', '.txt')]
        assert len(src_list) == len(tgt_list)
        for src, tgt in tqdm(zip(src_list,tgt_list)):  
            src_f.write(src)
            tgt_f.write(tgt)
            total_instance += 1
        print(f'val: {total_instance}')

    end = datetime.datetime.now()
    print(f'Time cost: {(end - start).seconds}s')
