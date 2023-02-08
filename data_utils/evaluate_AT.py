from collections import defaultdict
from multiprocessing import Pool
from os.path import join

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def eval_instance(inputs):
    src_smi, tgt_smi, a_id2preds, tid = inputs
    pred2score = defaultdict(int)

    smi2mol = {}

    for a_id, preds in a_id2preds.items():
        existed_set = set()
        for rank, p in enumerate(preds):
            mol = Chem.MolFromSmiles(''.join(p.split(' ')))
            if mol is None:
                continue
            canno_p = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canno_p in existed_set:
                continue
            smi2mol[canno_p] = mol
            pred2score[canno_p] += 1/(1+rank)
            existed_set.add(canno_p)
    sorted_preds = sorted(pred2score.items(), key=lambda x: x[1], reverse=True)
    rank = 10000
    for r, (p, score) in enumerate(sorted_preds):
        if p == tgt_smi:
            rank = r
            break
    return sorted_preds, rank, tid


def load_AT_predictions(prediction_file, n_best, datasplit, data_dir, total_aug, n_process=20):
    src_list = []
    tgt_list = []
    with open(join(data_dir, f'src-{datasplit}.txt')) as src_f, open(join(data_dir, f'tgt-{datasplit}.txt')) as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        cur_src_list = []
        cur_tgt_list = []
        for src, tgt in zip(src_lines, tgt_lines):
            cur_src_list.append(src.strip())
            cur_tgt_list.append(tgt.strip())
            if len(cur_tgt_list) == total_aug:
                src_list.append(''.join(cur_src_list[0].split(' ')))
                tgt_list.append(''.join(cur_tgt_list[0].split(' ')))
                tgt_set_debug = set(cur_tgt_list)
                assert len(tgt_set_debug) == 1
                cur_tgt_list = []
                cur_src_list = []

    with open(prediction_file) as pred_f:
        pred_lines = pred_f.readlines()
        assert len(pred_lines) == len(tgt_list) * n_best * total_aug
        t_id2a_id2preds = defaultdict(lambda: defaultdict(list))
        for pid, pred in enumerate(pred_lines):
            pred = pred.strip()
            t_id = pid // (n_best * total_aug)
            a_id = (pid % (n_best * total_aug)) // n_best
            t_id2a_id2preds[t_id][a_id].append(pred)

    assert len(tgt_list) == len(t_id2a_id2preds)

    accuracies = np.zeros([len(tgt_list), 10], dtype=np.float32)
    result_dict = {}
    with Pool(n_process) as p:
        for res in p.imap_unordered(eval_instance, ((src_list[t_id], tgt_list[t_id], a_id2preds, t_id) for t_id, a_id2preds in t_id2a_id2preds.items())):
            sorted_preds, rank, t_id = res
            tgt_smi = tgt_list[t_id]
            accuracies[t_id, rank:] = 1
            result_dict[(src_list[t_id], tgt_smi)] = [ele[0]
                                                      for ele in sorted_preds]
    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(10):
        print(f"Vanilla AT Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")

    return result_dict
