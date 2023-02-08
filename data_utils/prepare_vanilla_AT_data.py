import argparse
import os
import sys
from os.path import exists, join
from pathlib import Path

from tqdm import tqdm

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.smiles_utils import (canonicalize_smiles, get_random_smiles,
                                smi_tokenizer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    if not exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args


def get_augmented_data(args, split, naug=5):
    src_file = join(args.input_dir, f'src-{split}.txt')
    tgt_file = join(args.input_dir, f'tgt-{split}.txt')
    src_list = []
    tgt_list = []
    with open(src_file, 'r') as src_f, open(tgt_file, 'r') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        assert len(src_lines) == len(tgt_lines)
        for src, tgt in tqdm(zip(src_lines, tgt_lines)):
            src = src.strip()
            tgt = tgt.strip()
            src_smi = canonicalize_smiles(''.join(src.split(' ')))
            tgt_smi = smi_tokenizer(
                canonicalize_smiles(''.join(tgt.split(' '))))
            src_list.append(smi_tokenizer(src_smi))
            tgt_list.append(tgt_smi)
            for _ in range(naug):
                src_list.append(smi_tokenizer(get_random_smiles(src_smi)))
                tgt_list.append(tgt_smi)
    return src_list, tgt_list

#
# python data_utils/prepare_vanilla_AT_data.py --input_dir data/uspto_full \
#                       --output_dir data/uspto_full/vanilla_AT
#


if __name__ == "__main__":
    args = parse_args()

    src_list = []
    tgt_list = []
    for split in ['train', 'val']:
        cur_src_list, cur_tgt_list = get_augmented_data(args, split, naug=5)
        src_list.extend(cur_src_list)
        tgt_list.extend(cur_tgt_list)
    print('len', len(src_list))
    with open(join(args.output_dir, f'src-train.txt'), 'w') as src_f, open(join(args.output_dir, f'tgt-train.txt'), 'w') as tgt_f:
        for src, tgt in tqdm(zip(src_list, tgt_list)):
            src_f.write(src + '\n')
            tgt_f.write(tgt + '\n')

    for split in ['val', 'test']:
        src_list, tgt_list = get_augmented_data(args, split, naug=5)
        print('len', len(src_list))
        with open(join(args.output_dir, f'src-{split}.txt'), 'w') as src_f, open(join(args.output_dir, f'tgt-{split}.txt'), 'w') as tgt_f:
            for src, tgt in tqdm(zip(src_list, tgt_list)):
                src_f.write(src + '\n')
                tgt_f.write(tgt + '\n')
