
from os.path import join

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from data_utils.evaluate_AT import load_AT_predictions
from ranker import calc_ranking_scores
from utils.rerank_utils import deduplicate_results, rerank_results_with_scores
from utils.smiles_utils import canonicalize_smiles


def load_valid_reactions(data_dir):
    """
    invalid reactions are filtered out during preprocessing
    """
    print('load vanilla AT predictions')
    src_list = []
    tgt_list = []
    with open(join(data_dir, f'src-test.txt')) as src_f, open(join(data_dir, f'tgt-test.txt')) as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        for src, tgt in zip(src_lines, tgt_lines):
            src_list.append(canonicalize_smiles(
                ''.join(src.strip().split(' '))))
            tgt_list.append(canonicalize_smiles(
                ''.join(tgt.strip().split(' '))))
    return src_list, tgt_list


def count_implicit_valence_N(input_mol, idset=None):
    total_N, v2_N, v1_N = 0, 0, 0
    amide = Chem.MolFromSmiles('NC=O')
    matches = input_mol.GetSubstructMatches(amide)
    excluded_atom_ids = [i for j in matches for i in j]
    for atom in input_mol.GetAtoms():
        if idset is not None and atom.GetIdx() not in idset:
            continue
        if atom.GetAtomicNum() == 7 and atom.GetIdx() not in excluded_atom_ids and 7 not in [a.GetAtomicNum() for a in atom.GetNeighbors()]:
            total_N += 1
            if atom.GetImplicitValence() == 2:
                v2_N += 1
            if atom.GetImplicitValence() == 1:
                v1_N += 1
            # those nitrogens with ImplicitValence == 0 are inactive by nature
    return total_N, v2_N, v1_N, len(matches)


def is_amidation_rxn(product, reactant):
    prd_mol = Chem.MolFromSmiles(product)
    ptotal_N, pv2_N, pv1_N, pamide_num = count_implicit_valence_N(prd_mol)
    r_mol = Chem.MolFromSmiles(reactant)
    rtotal_N, rv2_N, rv1_N, _ = count_implicit_valence_N(r_mol)
    if rtotal_N >= 2 and rv2_N > 0 and rv1_N > 0:
        # set rv1_N > 0 to discriminate those selective amidation reactions
        if count_implicit_valence_N(Chem.MolFromSmiles(reactant))[3] < pamide_num:
            return True
    return False


def get_amidation_rxns(src_tgts):
    rxns = []
    for src_tgt in tqdm(src_tgts):
        product, reactant = src_tgt
        if is_amidation_rxn(product, reactant):
            rxns.append(src_tgt)
    return rxns


def get_p_at_k(result_dict, n_best=10, total=-1):
    if total == -1:
        total = len(result_dict)
    accuracies = np.zeros([total, n_best], dtype=np.float32)
    for idx, (src_tgt, preds) in enumerate(result_dict.items()):
        _, tgt = src_tgt
        for j, pred in enumerate(preds[:n_best]):
            if pred == tgt:
                accuracies[idx, j:] = 1.0
    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(n_best):
        print(
            f"top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")


class Evaluator():
    def __init__(self, data_dir='../data/uspto_full') -> None:
        self.src_list, self.tgt_list = load_valid_reactions(data_dir)
        self.vanilla_AT_result_dict = load_AT_predictions(
            f'{data_dir}/vanilla_AT/predictions-no_sub.txt',
            n_best=10, datasplit='no_sub', data_dir=f'{data_dir}/vanilla_AT', total_aug=6)

    def ensemble_score(self, sub_result_dict, n_best=10):
        total = 101311
        accuracies = np.zeros([total, n_best], dtype=np.float32)
        num_with_sub = 0
        num_vanilla = 0
        for i, (smi_src, smi_tgt) in tqdm(enumerate(zip(self.src_list, self.tgt_list))):
            key = (smi_src, smi_tgt)
            if key in sub_result_dict:
                predicted_results = sub_result_dict[key]
                num_with_sub += 1
                for j, result in enumerate(predicted_results[:n_best]):
                    label = result[1]
                    if label:
                        accuracies[i, j:] = 1.0
                        break
            elif key in self.vanilla_AT_result_dict:
                num_vanilla += 1
                predicted_results = self.vanilla_AT_result_dict[key]
                for j, result in enumerate(predicted_results[:n_best]):
                    if result == smi_tgt:
                        accuracies[i, j:] = 1.0
                        break

        print('total:', total)
        print('num_with_sub:', num_with_sub)
        print('sub ratio:', num_with_sub/total)
        print('num_vanilla:', num_vanilla)
        print('our ratio:', (num_vanilla+num_with_sub)/total)
        print('Top-k accuracy')
        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(n_best):
            print(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")
        print(
            '##############################################################################')
        print('Top-k accuracy on valid reactions only')
        mean_accuracies = np.sum(accuracies, axis=0) / len(self.src_list)
        for n in range(n_best):
            print(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")

    @staticmethod
    def rank_with_model(predicted_results, ranker_model_path):
        id2score = calc_ranking_scores(
            predicted_results, model_path=ranker_model_path)
        sub_result_dict = rerank_results_with_scores(
            predicted_results, id2score)
        return sub_result_dict

    @staticmethod
    def rank_with_beamsearch_position(raw_predicted_results, hyper_para=3, n_best=10):
        predicted_results = deduplicate_results(raw_predicted_results)
        result_dict = {}
        accuracies = np.zeros(
            [len(predicted_results), n_best], dtype=np.float32)
        for idx, ((src, tgt), predictions) in tqdm(enumerate(predicted_results)):
            new_preds = []
            for i, pred in enumerate(predictions):
                predicted_smi, label, _, sub_exists_rankings = pred
                avg_ranking = sum(
                    [-1/(hyper_para*ele[2]+1) for ele in sub_exists_rankings])
                new_preds.append((pred, avg_ranking))
            new_preds.sort(key=lambda ele: ele[1])
            predictions = [ele[0] for ele in new_preds]

            for j, prediction in enumerate(predictions[:n_best]):
                predicted_smi, label, _, sub_exists_rankings = prediction
                if label:
                    accuracies[idx, j:] = 1.0
            result_dict[(src, tgt)] = predictions

        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(n_best):
            print(
                f"Partial sub data, top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")
        return result_dict
