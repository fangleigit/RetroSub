import json
import os
import pickle
import sys
from os.path import join
from pathlib import Path

from rdkit import Chem, RDLogger

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.common_utils import parse_config
from utils.data_utils import extraction_result_loader
from utils.smiles_utils import (canonicalize_smiles, convert, get_isotopic,
                                get_random_smiles, smi_tokenizer)

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
RDLogger.DisableLog('rdApp.*')

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
    total_correct_substructures = 0






    data_loader = extraction_result_loader(
        args.store_path, args.dataset, args.total_chunks, args.chunk_id)

    data_info = []

    input_smiles = []
    output_smiles = []

    for idx, result_item in enumerate(data_loader):
        reactions_total += 1
        s, t, cans, cur_result = result_item
        cur_input_smiles_set = set()
        exists_in_golden_list = []        
        for cand_id, can_res in cur_result.items():
            # skip substructure extracted from golden target
            if cand_id < 0:
                continue
            if args.dataset != 'test':
                src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt = can_res
            else:
                src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt, exists_in_golden = can_res
            # sub_mol, _, src_frag_smi, tgt_frag_smi, src_mol, tgt_mol, query_sub, target_sub = can_res
            src_sub_smi = canonicalize_smiles(Chem.MolToSmiles(src_sub))
            tgt_sub_smi = canonicalize_smiles(Chem.MolToSmiles(tgt_sub))
            assert len(get_isotopic(smi_tokenizer(src_sub_smi))) >= len(get_isotopic(smi_tokenizer(tgt_sub_smi)))
            if not src_sub_smi or not tgt_sub_smi: 
                continue
            src_frag_smi = canonicalize_smiles(Chem.MolToSmiles(src_frag))
            tgt_frag_smi = canonicalize_smiles(Chem.MolToSmiles(tgt_frag))
            src_smi, tgt_smi = convert(
                src_sub_smi, src_frag_smi, tgt_frag_smi)
            if not get_isotopic(smi_tokenizer(src_sub_smi)):
                continue

            exists_in_golden_list.append(exists_in_golden if args.dataset == 'test' else True)

            # deduplicate for train and val
            if src_smi in cur_input_smiles_set and args.dataset != 'test':
                continue
            cur_input_smiles_set.add(src_smi)
            input_smiles.append(src_smi)
            output_smiles.append(tgt_smi)

            item = {'id': f'{args.chunk_id}_{idx}',
                    'src_sub': src_sub,
                    'tgt_sub':tgt_sub,
                    'src': Chem.MolFromSmiles(s),
                    'tgt': Chem.MolFromSmiles(t),
                    'labeled_src':labeled_src,
                    'labeled_tgt':labeled_tgt
                    }
            if args.dataset == 'test':
                item['exists_in_golden']=exists_in_golden
            data_info.append(item)
            if args.data_aug:
                for _ in range(2):
                    try:
                        src_sub_smi = get_random_smiles(src_sub_smi)
                        tgt_sub_smi = get_random_smiles(tgt_sub_smi)
                        src_frag_smi = get_random_smiles(canonicalize_smiles(Chem.MolToSmiles(src_frag))) 
                        if not src_sub_smi or not tgt_sub_smi: 
                            raise Exception('Empyt input')                    
                    except:
                        # TODO fix this??
                        print('error during random simles')
                        print(f'src_sub_smi: {src_sub_smi}')
                        print(f'tgt_sub_smi: {tgt_sub_smi}')
                        continue
                    
                    src_smi, tgt_smi = convert(
                        src_sub_smi, src_frag_smi, tgt_frag_smi)
                    input_smiles.append(src_smi)
                    output_smiles.append(tgt_smi)
                    data_info.append(item)

        total_substructures += len(exists_in_golden_list)
        if len(exists_in_golden_list) > 0:
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
                 'total_correct_substructures': total_correct_substructures,
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

    with open(join(saved_folder, f'src-{args.dataset}_{args.chunk_id}_{args.total_chunks}.txt'), 'w', encoding='utf-8') as f:
        for i, line in enumerate(input_smiles):
            f.write(line + '\n')
    with open(join(saved_folder, f'tgt-{args.dataset}_{args.chunk_id}_{args.total_chunks}.txt'), 'w', encoding='utf-8') as f:
        for i, line in enumerate(output_smiles):
            f.write(line + '\n')

    if args.dataset == 'test':
        with open(join(saved_folder, f'src-{args.dataset}_{args.chunk_id}_{args.total_chunks}_unique.txt'), 'w', encoding='utf-8') as f:
            input_smiles_set = set(input_smiles)
            for i, line in enumerate(input_smiles_set):
                f.write(line + '\n')


    with open(join(saved_folder, f'{args.dataset}_{args.chunk_id}_{args.total_chunks}_info.pkl'), 'wb') as f:
        pickle.dump(data_info, f)

    with open(join(saved_folder, f'{args.dataset}_{args.chunk_id}_{args.total_chunks}.stat.json'), 'w', encoding='utf-8') as f:
        json.dump(stat_dict, f)
    print(json.dumps(stat_dict, indent=4))
