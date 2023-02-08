import traceback
import argparse
from os import path

def log_traceback(ex):
    ex_traceback = ex.__traceback__
    tb_lines = [line.rstrip('\n') for line in
                traceback.format_exception(ex.__class__, ex, ex_traceback)]
    return tb_lines

class CommonConfig:
    Max_FP_Radius = 6
    Min_FP_Radius = 2
    Sim_FP_Radius = 2
    Sim_Threshold = 0.2
    

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='data/uspto_full/test.json')
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--store_path', type=str,
                        default='data/uspto_full/')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--nprocessors', type=int, default=8)
    parser.add_argument('--reactions', type=str,
                        default='data/uspto_full/reaction.pkl')
    parser.add_argument('--total_chunks', type=int, default=1)
    parser.add_argument('--chunk_id', type=int, default=0)    
    parser.add_argument('--out_dir', type=str,
                        default='reserved_for_other_usage')
    parser.add_argument('--test_submol', action='store_true')
    parser.add_argument('--data_aug', action='store_true')
    
    args = parser.parse_args()
    assert args.total_chunks >=1
    if not args.store_path.endswith(path.sep):
        args.store_path += path.sep 
    return args


class PropNames(object):
    Chiral_Tag = 'chiral_tag'
    Bond_Type = 'bond_type'