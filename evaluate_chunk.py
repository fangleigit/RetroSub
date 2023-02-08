#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import pickle
from collections import defaultdict, Counter
from multiprocessing import Pool
import json

from rdkit import Chem, RDLogger
from tqdm import tqdm

from utils.mol_utils import test_merge_sub_frag

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
RDLogger.DisableLog('rdApp.*')


def merge_worker(data):
    (rank, pred), merge_info = data
    golden = merge_info['golden']
    sub = merge_info['sub']
    merge_flag, merge_res = test_merge_sub_frag(sub, pred, Chem.MolToSmiles(golden))
    if merge_flag is None:
        # merge failed 
        return pickle.dumps((merge_info, None, False, rank))
    return pickle.dumps((merge_info, merge_res, merge_flag, rank))


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_input', type=str)
    parser.add_argument('--predictions',
                        type=str,
                        help="Path to file of the predictions")

    parser.add_argument('--substructures',
                        type=str,
                        help="Path to file of the substructures")

    parser.add_argument('--n_best',
                        type=int,
                        default=10)

    parser.add_argument('--nprocessors', type=int, default=10)

    parser.add_argument('--use_oracle', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    import datetime
    start = datetime.datetime.now()
    args = parse_config()
    reaction_all = []
    with open(args.substructures, 'rb') as f:
        reaction_all = pickle.load(f)
    to_merge = []

    id2info = {}
    for item in reaction_all:
        merge_item = {'rxt_id': item['id'],
                      'golden': item['tgt'],
                      'sub': item['src_sub'], 
                      'exists_in_golden': item['exists_in_golden']}
        id2info[item['id']] = (Chem.MolToSmiles(item['src']), Chem.MolToSmiles(item['tgt']))
        for _ in range(args.n_best):
            to_merge.append(merge_item)
    
    line2line_id = {}
    with open(args.test_input, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f.readlines()):
            line2line_id[line.strip()] = line_id    
     
    unique_pred_frag_list = []
    with open(args.predictions, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            try:
                line = line.split('| ')[1].replace(' ', '')
                unique_pred_frag_list.append((i%args.n_best,line))
            except IndexError:
                unique_pred_frag_list.append((i%args.n_best,None))
    unique_pred_frag_list = [unique_pred_frag_list[i:i + args.n_best] for i in range(0, len(unique_pred_frag_list), args.n_best)]
    
    pred_frag_list = []

    with open(args.test_input.replace('_unique',''), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            pred_frag_list.extend(unique_pred_frag_list[line2line_id[line]])
            
    assert len(pred_frag_list) == len(to_merge)
    
    merge_results = []

    with Pool(args.nprocessors) as p:
        it = p.imap(merge_worker, zip(pred_frag_list, to_merge))
        pbar = tqdm(total=len(pred_frag_list), desc='Merging')
        while True:
            try:
                tmp = it.next()
                tmp = pickle.loads(tmp)
                merge_results.append(tmp)
                pbar.update()
            except StopIteration:
                pbar.close()
                break
            except Exception as e:
                merge_results.append(None)
                print(e)

    reaction_rank = defaultdict(list)
    id2pred2sub = defaultdict(lambda : defaultdict(list))
    for res in merge_results:
        # (merge_info, merge_res, merge_flag, rank) = res
        if res and res[1] and res[1] != '[Error]':
            merge_item = res[0]
            rank = res[3]
            rxt_id, golden, sub = merge_item['rxt_id'], merge_item['golden'], merge_item['sub']
            exists_in_golden = merge_item['exists_in_golden']
            if (args.use_oracle and exists_in_golden) or not args.use_oracle:
                reaction_rank[rxt_id].append((res[1], res[2]))
                id2pred2sub[rxt_id][res[1]].append((Chem.MolToSmiles(sub), exists_in_golden, rank))

    results4dump =[]
    scores = {}
    for k in range(1, 1 + args.n_best):
        scores[k] = 0
    for rxt_id, item in reaction_rank.items():
        mol_list = []
        for m in item:
            mol_list.append((m[0], m[1]))

        mol_courts = Counter(mol_list).most_common(2*args.n_best)
        dump_arr = []
        for (pre, flag), cnt in mol_courts:
            dump_arr.append([pre, flag,cnt, id2pred2sub[rxt_id][pre]])
        results4dump.append((id2info[rxt_id], dump_arr))

        for k in range(1, 1 + args.n_best):
            cur_rank_flag = set()
            for i in mol_courts[:k]:
                cur_rank_flag.add(i[0][1])
            if True in cur_rank_flag:
                scores[k] += 1

    for k in range(1, 1 + args.n_best):
        print('Top-{}: {:.2f}%'.format(k, 100 *
              scores[k] / len(reaction_rank)))
    
    print(len(results4dump)/ len(reaction_rank))

    with open(args.predictions.replace('.txt', f'_res_{args.use_oracle}.json'), 'w') as f:
        json.dump((scores, len(reaction_rank)), f)
    
    with open(args.predictions.replace('.txt', f'_res_{args.use_oracle}_analysis.json'), 'w') as f:
        json.dump(results4dump, f, indent=4)

    end = datetime.datetime.now()
    print(f'Time cost: {(end - start).seconds}s')