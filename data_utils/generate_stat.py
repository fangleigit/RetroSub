import json
import os
import sys
from os.path import join
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.common_utils import parse_config
from utils.data_utils import extraction_result_loader
from utils.smiles_utils import smi_tokenizer

"""
python generate_stat.py --dataset test --store_path /home/leifa/teamdrive/bdmstorage/projects/cc_leifa/data/uspto_full/subsearch_0818 --total_chunks 200 --chunk_id 0 --out temp2
"""

if __name__ == "__main__":

    args = parse_config()
    assert args.total_chunks >= 1

    saved_folder = args.out_dir
    Path(saved_folder).mkdir(parents=True, exist_ok=True)

    reactions_total = 0

    reactions_with_substructures = 0
    reactions_with_substructures__ = 0
    reactions_with_correct_substructures = 0
    reactions_with_correct_substructures__ = 0
    reactions_with_incorrect_substructures = 0
    reactions_with_incorrect_substructures__=0

    reactions_with_all_correct_substructures = 0
    reactions_with_all_correct_substructures__ = 0
    reactions_with_all_incorrect_substructures = 0  
    reactions_with_all_incorrect_substructures__ = 0 


    total_substructures = 0
    total_unique_substructures = 0
    total_correct_substructures = 0

    data_loader = extraction_result_loader(
        args.store_path, args.dataset, args.total_chunks, args.chunk_id)    
    original_length= 0
    cur_length =0
    original_target_atoms = 0
    original_src_atoms = 0
    cur_target_atoms = 0
    cur_sub_atoms = 0
    cur_sub_rings, cur_sub_carbons, cur_sub_aromatics, cur_sub_inrings = 0, 0, 0, 0
    cur_sub_isotopes = 0


    for idx, result_item in enumerate(data_loader):      
        reactions_total += 1  
        exists_in_golden_list = []
        subsmiles_set = set()
        s, t, cans, cur_result = result_item
        
        for cand_id, can_res in cur_result.items():            
            if cand_id < 0:
                continue            
            if args.dataset != 'test':
                src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt = can_res
                cur_length += len(smi_tokenizer(Chem.MolToSmiles(tgt_frag)).split(' '))
                cur_target_atoms += tgt_frag.GetNumHeavyAtoms()
            else:
                src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt, exists_in_golden = can_res
            
            cur_sub_atoms += src_sub.GetNumHeavyAtoms()
            cur_sub_rings += CalcNumRings(src_sub)
            cur_sub_aromatics += len([atom for atom in src_sub.GetAtoms() if atom.GetIsAromatic()])
            cur_sub_carbons += len([atom for atom in src_sub.GetAtoms() if atom.GetAtomicNum() == 6])
            cur_sub_inrings += len([atom for atom in src_sub.GetAtoms() if atom.IsInRing()])
            cur_sub_isotopes += len([atom.GetIsotope() for atom in src_sub.GetAtoms() if atom.GetIsotope() != 0])
            subsmiles_set.add(Chem.MolToSmiles(src_sub))
            exists_in_golden_list.append(exists_in_golden if args.dataset == 'test' else True)
        
        total_substructures += len(exists_in_golden_list)
        total_unique_substructures += len(subsmiles_set)
        if len(exists_in_golden_list) > 0:
            original_length+=len(smi_tokenizer(t).split(' '))
            original_target_atoms += Chem.MolFromSmiles(t).GetNumHeavyAtoms()
            original_src_atoms += Chem.MolFromSmiles(s).GetNumHeavyAtoms()
            cur_correct_subs = sum(exists_in_golden_list)
            total_correct_substructures += cur_correct_subs
            reactions_with_substructures+=1
            reactions_with_substructures__ += cur_correct_subs
            if any(exists_in_golden_list):
                reactions_with_correct_substructures+=1
                reactions_with_correct_substructures__+=cur_correct_subs
                if all(exists_in_golden_list):
                    reactions_with_all_correct_substructures+=1
                    reactions_with_all_correct_substructures__+=cur_correct_subs
                else:
                    reactions_with_incorrect_substructures += 1
                    reactions_with_incorrect_substructures__ += cur_correct_subs
            else:
                reactions_with_incorrect_substructures+=1
                reactions_with_all_incorrect_substructures += 1
                reactions_with_all_incorrect_substructures__ += len(exists_in_golden_list)

    

    stat_dict = {'reactions_total': reactions_total,
                 'total_substructures': total_substructures,
                 'total_unique_substructures': total_unique_substructures,
                 'total_correct_substructures': total_correct_substructures,
                 'original_length': original_length,
                 'original_src_atoms': original_src_atoms,
                 'cur_length': cur_length,
                 'cur_sub_atoms': cur_sub_atoms,
                 'cur_sub_rings': cur_sub_rings,
                 'cur_sub_carbons': cur_sub_carbons,
                 'cur_sub_aromatics': cur_sub_aromatics,
                 'cur_sub_inrings': cur_sub_inrings,
                 'original_target_atoms': original_target_atoms,
                 'cur_target_atoms': cur_target_atoms,
                 'cur_sub_isotopes': cur_sub_isotopes,
                 'reactions_with_substructures': reactions_with_substructures,
                 'reactions_with_substructures__': reactions_with_substructures__,
                 'reactions_with_correct_substructures': reactions_with_correct_substructures,
                 'reactions_with_correct_substructures__': reactions_with_correct_substructures__,
                 'reactions_with_incorrect_substructures': reactions_with_incorrect_substructures,
                 'reactions_with_incorrect_substructures__': reactions_with_incorrect_substructures__,
                 'reactions_with_all_correct_substructures': reactions_with_all_correct_substructures,
                 'reactions_with_all_correct_substructures__': reactions_with_all_correct_substructures__,
                 'reactions_with_all_incorrect_substructures': reactions_with_all_incorrect_substructures,
                 'reactions_with_all_incorrect_substructures__': reactions_with_all_incorrect_substructures__,
                 }

    with open(join(saved_folder, f'{args.dataset}_{args.chunk_id}_{args.total_chunks}.stat2.json'), 'w', encoding='utf-8') as f:
        json.dump(stat_dict, f)
    
    print(stat_dict)

    