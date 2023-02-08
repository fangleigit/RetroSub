#!/usr/bin/env python
# coding: utf-8


import multiprocessing
import pickle
from collections import defaultdict
from multiprocessing import Pool

import func_timeout
from rdkit import Chem, RDLogger
from tqdm import tqdm

from subextractor import SubMolExtractor
from utils.common_utils import CommonConfig, log_traceback, parse_config
from utils.data_utils import read_json
from utils.extract_utils import resplit
from utils.mol_utils import remove_isotope
from utils.smiles_utils import canonicalize_smiles

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
RDLogger.DisableLog('rdApp.*')

# global variables
all_reaction_t2s = []


def work(data):
    (s, t, cans, scores), args = data

    if s in cans:
        cans.remove(s)

    cur_result = {}
    # we use -1 as key for golden target, this is not used in both training and inference
    # TODO: use subs from -1 for training
    for candidate_idx in range(-1, len(cans)):
        try:
            temp_item = pickle.loads(work_within_timelimit(
                s, t, cans, args, candidate_idx))
            if temp_item and temp_item[0]:
                cur_result[candidate_idx] = temp_item
        except func_timeout.FunctionTimedOut as e:
            print('***FunctionTimedOut***')
            print(e.msg)
        except Exception as e:
            print('Exception')
            print(log_traceback(e))
    return pickle.dumps((s, t, cans, cur_result))


@func_timeout.func_set_timeout(60)
def work_within_timelimit(s, t, cans, args, candidate_idx=1000):
    # TODO: move some code out to avoid recompute the fingerprints
    # TODO: avoid using GetSubstructMatches(not efficient)
    test_flag = args.dataset == 'test'
    
    src_mol = Chem.MolFromSmiles(s)

    tgt_mol = Chem.MolFromSmiles(t)
    candidate_mols = [Chem.MolFromSmiles(s) for s in cans]

    reactions = defaultdict(set)

    for c in cans:
        c_smi = canonicalize_smiles(c)
        if c in all_reaction_t2s:
            for i in all_reaction_t2s[c]:
                reactions[c_smi].add(i.split('>')[0])
        else:
            print(
                'please generate the data using the script "data_utils/collect_reaction.py"')
            reactions[c_smi].add(c_smi)

    try:
        extractor = SubMolExtractor(query=src_mol,
                                    target=candidate_mols[candidate_idx] if candidate_idx >= 0 else tgt_mol,
                                    candidates=candidate_mols,
                                    reactions=reactions,
                                    max_fp_radius=CommonConfig.Max_FP_Radius,
                                    min_count=min(args.count, len(cans)),
                                    min_fp_radius=CommonConfig.Min_FP_Radius)
        split_res = extractor.extractor()
        if split_res != None:
            src_sub, src_frag, labeled_src, tgt_sub, tgt_frag, labeled_tgt = split_res

            sub_smi = canonicalize_smiles(Chem.MolToSmiles(tgt_sub))

            if sub_smi:
                if test_flag:
                    if candidate_idx >= 0:
                        # substructure from candidate should also remain unchanged in the candidate reaction
                        can_src_smi = list(all_reaction_t2s[cans[candidate_idx]])[
                            0].split('>')[0]
                        can_src_mol = Chem.MolFromSmiles(can_src_smi)
                        # TODO: consider allow adding isotope when resplit
                        if not resplit(can_src_mol, remove_isotope(src_sub), src_sub, None):
                            return pickle.dumps(None)
                    # check if the substurcture can resplit the golden target mol.
                    # for analysis purpose
                    splitted_golden = resplit(
                        tgt_mol, remove_isotope(src_sub), src_sub, None)
                    exists_in_golden = False
                    if splitted_golden:
                        exists_in_golden = True
                    return pickle.dumps((src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt, exists_in_golden))
                else:
                    # on dev and train data, use the substructure to resplit the golden target
                    # return None if failed
                    splitted_golden = resplit(
                        tgt_mol, remove_isotope(src_sub), src_sub, None)
                    if splitted_golden:
                        tgt_sub, tgt_frag, labeled_tgt = splitted_golden
                        return pickle.dumps((src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt))
                    return pickle.dumps(None)

    except func_timeout.FunctionTimedOut as e:
        print('FunctionTimedOut')
        print(e.msg)

    except AssertionError as e:
        print('AssertionError')
        print(log_traceback(e))

    except Exception as e:
        print('Exception')
        print(log_traceback(e))

    return pickle.dumps(None)


if __name__ == "__main__":
    args = parse_config()

    src, tgt, cands, scores = read_json(
        args.input_file, args.total_chunks, args.chunk_id)

    with open(args.reactions, 'rb') as f:
        all_reaction_t2s = pickle.load(f)

    res_all = []
    with Pool(args.nprocessors) as p:
        it = p.imap_unordered(work, ((item, args)
                              for item in zip(src, tgt, cands, scores)))
        pbar = tqdm(total=len(src), desc='extract substructure')
        interval = 5
        iter_id = -1
        while True:
            try:
                iter_id += 1
                if iter_id % interval == 0 and iter_id:
                    pbar.update(interval)
                tmp = it.next(timeout=1400)
                tmp = pickle.loads(tmp)
                res_all.append(tmp)
                # TODO: optimize the extraction process
                # save if we kill the process for running too long
                # For less then 1% data, the substructures extraction might take much longer time.
                # We may skip these data if necessary.
                if len(res_all)/len(src) > 0.99:
                    with open(args.store_path + args.dataset + f'_{args.chunk_id}_{args.total_chunks}_.pkl', 'wb') as f:
                        pickle.dump(res_all, f)

            except multiprocessing.TimeoutError:
                iter_id -= 1
                pbar.update(iter_id % interval)
                pbar.close()
                print(f'timeout')
                with open(args.store_path + args.dataset + f'_{args.chunk_id}_{args.total_chunks}_.pkl', 'wb') as f:
                    pickle.dump(res_all, f)
                exit(0)
            except StopIteration:
                pbar.update(iter_id % interval)
                pbar.close()
                break
            except Exception as e:
                res_all.append(None)
                print(e)

    with open(args.store_path + args.dataset + f'_{args.chunk_id}_{args.total_chunks}_.pkl', 'wb') as f:
        pickle.dump(res_all, f)
