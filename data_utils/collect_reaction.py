#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pickle
import sys
from collections import defaultdict
from os.path import join

from tqdm import tqdm

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.smiles_utils import canonicalize_smiles

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/uspto_full')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    data_dir = args.dir
    src, tgt = [], []

    with open(join(data_dir, 'src-train.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            src.append(line.rstrip())
    with open(join(data_dir, 'src-val.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            src.append(line.rstrip())

    with open(join(data_dir, 'tgt-train.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            tgt.append(line.rstrip())
    with open(join(data_dir, 'tgt-val.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            tgt.append(line.rstrip())

    assert len(src) == len(tgt)

    rxt_dict = defaultdict(set)
    rxt_dict2 = defaultdict(set)

    for s, t in tqdm(zip(src, tgt)):
        s = s.replace(' ', '')
        t = t.replace(' ', '')
        s_ = canonicalize_smiles(s)
        t_ = canonicalize_smiles(t)
        rxt_dict[t].add(s)
        rxt_dict[t_].add(s_)
        rxt_dict2[s].add(t)
        rxt_dict2[s_].add(t_)

    print(f'collected reaction: {len(rxt_dict)}')
    print(f'collected reaction s2t: {len(rxt_dict2)}')

    with open(join(data_dir, 'reaction.pkl'), 'wb') as f:
        pickle.dump(rxt_dict, f)
    with open(join(data_dir, 'reaction_s2t.pkl'), 'wb') as f:
        pickle.dump(rxt_dict2, f)
