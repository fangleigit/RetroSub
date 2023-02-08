#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from os.path import join

from rdkit import Chem
from tqdm import tqdm

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.mol_utils import remove_atom_mapping
from utils.smiles_utils import smi_tokenizer


def preprocess_smiles(smiles):
    if not smiles:
        return smiles, 0
    input_mol = Chem.MolFromSmiles(smiles)
    mol = remove_atom_mapping(input_mol)
    # TODO: we should use the following code to get the output SMILES,
    # otherwise the output SMILES might not be canonicalized. 
    # Note that this does not affect the final results.
    # We leave the code as it was in order to reproduce the results
    # in our paper.

    # from utils.smiles_utils import canonicalize_smiles
    # output_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)    
    # output_smiles = canonicalize_smiles(output_smiles)
    
    output_smiles = Chem.MolToSmiles(mol, isomericSmiles=True) 
    return output_smiles, len([_ for _ in mol.GetAtoms()]) if mol else 0


def preprocess(data):
    """canonicalize and tokenize reactant and product smiles

    Args:
        data (tuple): (reactant_smi, product_smi)

    Returns:
        tuple:  (preprocessed_reactant_smi, preprocessed_product_smi)
    """
    r_smi, p_smi, p_id = data
    can_r_smi, r_cnt = preprocess_smiles(r_smi)
    can_p_smi, p_cnt = preprocess_smiles(p_smi)
    can_r_smi, can_p_smi = smi_tokenizer(can_r_smi), smi_tokenizer(can_p_smi)

    return can_r_smi, can_p_smi, r_cnt, p_cnt, p_id


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='./data/uspto_full')
    parser.add_argument('--output_path', type=str,
                        default='./data/uspto_full')
    parser.add_argument('--nprocessors', type=int, default=20)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    for split in ['test', 'val', 'train']:

        src_file = join(args.output_path, "src-" + split + '.txt')
        tgt_file = join(args.output_path, "tgt-" + split + '.txt')
        idx_file = join(args.output_path, "idx-" + split + '.txt')

        all_data = []
        rxt_smiles, pdt_smiles, idx = [], [], []
        with open(join(args.input_dir, 'raw_' + split + '.csv'), encoding='utf-8') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                all_data.append(row[-1])
                idx.append(row[0])
        idx=idx[1:]
        for item in all_data[1:]:
            rxt_smiles.append(item.split(">>")[0])
            pdt_smiles.append(item.split(">>")[1])

        assert len(rxt_smiles) == len(pdt_smiles)

        invalid_rxt = 0
        no_rxt = 0
        collected = 0

        rxt_atom_less = 0

        with Pool(args.nprocessors) as p, \
            open(src_file, 'w', encoding='utf-8') as src_fo, \
                open(tgt_file, 'w', encoding='utf-8') as tgt_fo, \
                    open(idx_file, 'w', encoding='utf-8') as idx_fo:
            for res in tqdm(p.imap(preprocess, zip(rxt_smiles, pdt_smiles, idx)),
                            total=len(rxt_smiles), desc='preprocessing'):
                rxt, pdt, atom_num, _, p_id = res
                if rxt:
                    skipped = False
                    if len(rxt.strip().split()) < 2 or len(pdt.strip().split()) < 2:
                        invalid_rxt += 1
                        skipped = True                        

                    if atom_num < 5:
                        rxt_atom_less += 1
                        skipped = True

                    if (not skipped and split == 'train') or split in ['val', 'test']:
                        src_fo.write(pdt + '\n')
                        tgt_fo.write(rxt + '\n')
                        idx_fo.write(p_id+ '\n')
                        collected += 1
                else:
                    no_rxt += 1                

        print(f'Data split: {split}')
        print(f'The num of reactions from source csv file: {len(rxt_smiles)}')
        print(f'Collected instances: {collected}')
        print(f'Ratio of stored reactions: {100*collected/len(rxt_smiles): .3f}%')

        print(f'The num of invalid reaction: {invalid_rxt}')
        print(f'The num of SMILES w/o reactant : {no_rxt}')
        print(
            f'The num of reactants with less than five atoms: {rxt_atom_less}')
