import json
import os
import sys
from collections import defaultdict
from os.path import exists, join

from tqdm import tqdm

# hack to import parent packages
sys.path.append(os.getcwd())
from utils.common_utils import parse_config

###
# python data_utils/merge_stat.py --total_chunks 200 --out_dir ./data/uspto_full/subextraction/
###

if __name__ == "__main__":
    args = parse_config()
    num_chunks = args.total_chunks
    saved_folder = args.out_dir
    
    stat_dict = defaultdict(int)
    total_instance = 0
    for chunk_id in tqdm(range(num_chunks)):
        json_path = join(saved_folder, f'test_{chunk_id}_{num_chunks}.stat.json')
        if not exists(json_path):
            print(f'{json_path} does not exist')
            continue
        cur_stat_dict = json.load(open(json_path))
        for k, v in cur_stat_dict.items():
            stat_dict[k] += v
    
    print('reactions_total', stat_dict['reactions_total'])
    print('total_substructures', stat_dict['total_substructures'])
    print('total_correct_substructures', stat_dict['total_correct_substructures'])
    print('reactions_with_substructures', stat_dict['reactions_with_substructures'])
    print('reactions_with_substructures__', stat_dict['reactions_with_substructures__']/stat_dict['reactions_with_substructures'])
    print('reactions_with_correct_substructures', stat_dict['reactions_with_correct_substructures'])
    print('reactions_with_correct_substructures__', stat_dict['reactions_with_correct_substructures__']/stat_dict['reactions_with_correct_substructures'])
    print('reactions_with_incorrect_substructures', stat_dict['reactions_with_incorrect_substructures'])
    print('reactions_with_incorrect_substructures__', stat_dict['reactions_with_incorrect_substructures__']/stat_dict['reactions_with_incorrect_substructures'])
    print('reactions_with_all_correct_substructures', stat_dict['reactions_with_all_correct_substructures'])
    print('reactions_with_all_correct_substructures__', stat_dict['reactions_with_all_correct_substructures__']/stat_dict['reactions_with_all_correct_substructures'])
    print('reactions_with_all_incorrect_substructures', stat_dict['reactions_with_all_incorrect_substructures'])
    print('reactions_with_all_incorrect_substructures__', stat_dict['reactions_with_all_incorrect_substructures__']/stat_dict['reactions_with_all_incorrect_substructures'])
