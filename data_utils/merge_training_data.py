import datetime
import os
import sys
from os.path import join

from tqdm import tqdm

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.common_utils import parse_config


def training_data_loader(store_path, datasplit, total_chunks, prefix, postfix):
    for i in range(total_chunks):
        f_path = join(
            store_path, f'{prefix}{datasplit}_{i}_{total_chunks}{postfix}')
        with open(f_path) as f:
            for line in f:
                yield line

###
# python data_utils/merge_training_data.py --total_chunks 200 --out_dir ./data/uspto_full/subextraction
###

if __name__ == "__main__":
    start = datetime.datetime.now()
    args = parse_config()
    num_chunks = args.total_chunks
    saved_folder = args.out_dir

    for split in ['val', 'train']:
        total_instance = 0        
        with open(join(saved_folder, f'src-{split}.txt'), 'w') as src_f, \
                open(join(saved_folder, f'tgt-{split}.txt'), 'w') as tgt_f:
            src_list = [line for line in training_data_loader(saved_folder, split,
                                     num_chunks, 'src-', '.txt')]
            tgt_list = [line for line in training_data_loader(saved_folder, split,
                                     num_chunks, 'tgt-', '.txt')]
            assert len(src_list) == len(tgt_list)
            for src, tgt in tqdm(zip(src_list,tgt_list)):
                src_f.write(src)
                tgt_f.write(tgt)
                total_instance += 1
        print(f'{split}: {total_instance}')
    end = datetime.datetime.now()
    print(f'Time cost: {(end - start).seconds}s')
