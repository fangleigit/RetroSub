#!/usr/bin/env python
# coding: utf-8

'''
This file is mainly for demo purpose, please do not change the parameters 
because they are used in script './demo_data/retrieval_script.sh' and
'./demo_data/subseq2seq.sh'
'''

from collections import Counter, defaultdict

import torch
from IPython import display
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from torch.utils.data import DataLoader

from ranker import (RankingModel, RankingTestDataset, get_all_features,
                    get_all_ranking_info)
from subextractor import SubMolExtractor
from utils.extract_utils import resplit
from utils.mol_utils import remove_isotope, test_merge_sub_frag
from utils.smiles_utils import (canonicalize_smiles, get_random_smiles,
                                smi_tokenizer)

RDLogger.DisableLog('rdApp.*')


def save_input_for_dual_encoder(tokenized_product_smi, save_file='demo_data/test_input_dual_encoder.txt'):
    with open(save_file, 'w') as f:
        f.write(tokenized_product_smi + '\t' + 'TGT_PLACEHOLDER')


def save_input_for_subseq2seq(input_smiles, save_file='demo_data/test_input_seq2seq.txt'):
    with open(save_file, 'w') as f:
        for line in input_smiles:
            f.write(line + '\n')


def load_retrieved_candidates(s, file='demo_data/test_input_dual_encoder.top20.txt'):
    with open(file, 'r') as f:
        line = f.readline()
        _, _, *can = line.strip().split('\t')
        candidates = [ele.replace(' ', '') for ele in can[:-1:2]]
        # remove the input src from candidates if exists
        if s in candidates:
            candidates.remove(s)
        return candidates


def extract_all(s, cans, all_reaction_t2s, min_count=5):
    """Extract substructures from all candidates

    Args:
        s (str): input SMILES of source (product in retrosynthesis)
        cans (List): list of SMILES of retrieved candidate targets (reactant in retrosynthesis)
        all_reaction_t2s (Dict): key: target (reactants), value: source (all possible products)
        min_count (int, optional): min fingerprint alignment count. Defaults to 5.

    Returns:
        extr_result (Dict): dictionary {candidate_id: extraction_result)
    """
    src_mol = Chem.MolFromSmiles(s)
    candidate_mols = [Chem.MolFromSmiles(s) for s in cans]

    reactions = defaultdict(list)
    for c in cans:
        c_smi = canonicalize_smiles(c)
        if c in all_reaction_t2s:
            reactions[c_smi].extend(all_reaction_t2s[c])
        else:
            reactions[c_smi].append(c_smi)

    extr_result = {}
    for candidate_idx in range(len(cans)):
        try:
            candidate_reaction = all_reaction_t2s[cans[candidate_idx]]
            extraction_result = extract(
                src_mol, candidate_mols[candidate_idx], candidate_mols, candidate_reaction, reactions, min_count)
            if extraction_result and extraction_result[0]:
                extr_result[candidate_idx] = extraction_result
        except:
            pass
    return extr_result


def extract(src_mol, can_mol, candidate_mols, can_src_set, reactions, min_count):
    """Extract substructures from one candidate

    Args:
        src_mol (Mol): Mol object of src (product in retrosynthesis)
        can_mol (Mol): Mol object of candidate to extract substructure
        candidate_mols (List<Mol>): List of Mol for all candidates
        can_src_set (List): List of SMILES of all possible product for current candidate
        reactions (Dict): Dictionary of reactions for all retrieved candidates
        min_count (_type_): min fingerprint alignment count.

    Returns:
        src_sub: substructure in src_mol (product in retrosynthesis)
        tgt_sub: substructure in can_mol (will not be used either in training or inference)
        src_frag: fragments in src_mol (product in retrosynthesis)        
        tgt_frag: fragments in can_mol (will not be used for inference, this is from can_mol)
        labeled_src: isotope labeled src_mol
        labeled_tgt: isotope labeled can_mol
    """
    extractor = SubMolExtractor(query=src_mol,
                            target=can_mol,
                            candidates=candidate_mols,
                            reactions=reactions,
                            max_fp_radius=6,
                            min_count=min_count,
                            min_fp_radius=2)
    split_res = extractor.extractor()
    if split_res != None:
        src_sub, src_frag, labeled_src, tgt_sub, tgt_frag, labeled_tgt = split_res
        sub_smi = canonicalize_smiles(Chem.MolToSmiles(tgt_sub))
        if sub_smi:
            # we only pick the first to verify the result
            can_src_smi = list(can_src_set)[0]
            can_src_mol = Chem.MolFromSmiles(can_src_smi)
            # we should be able to split the can_src_mol with src_sub, otherwise will return None
            if not resplit(can_src_mol, remove_isotope(src_sub), src_sub, None):
                return None
            return (src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt)
    return None


def view_sub_extractions(s, t, candidates, extraction_results):
    sub_smi2candidates = defaultdict(list)
    for cand_id, can_res in extraction_results.items():
        src_sub, tgt_sub, src_frag, tgt_frag, labeled_src, labeled_tgt = can_res
        sub_smi2candidates[Chem.MolToSmiles(src_sub)].append(
            (candidates[cand_id], labeled_tgt))
    src_mol = Chem.MolFromSmiles(s)
    tgt_mol = Chem.MolFromSmiles(t)
    for idx, (sub_smi, cans) in enumerate(sub_smi2candidates.items()):
        sub_mol = Chem.MolFromSmiles(sub_smi)
        sub_mol_isotope_removed = remove_isotope(sub_mol)
        can_mols = [cand[1] for cand in cans]
        highlights = [(), src_mol.GetSubstructMatch(
            sub_mol_isotope_removed), tgt_mol.GetSubstructMatch(sub_mol_isotope_removed)]
        highlights += [can_mol.GetSubstructMatch(sub_mol_isotope_removed)
                       for can_mol in can_mols]
        legends = [f'Substructure_{idx}', 'Product', 'Reactants'] + [
            f'Isotope labeled candidate #{candidates.index(cans[i][0])+1}' for i in range(len(can_mols))]

        plt = Draw.MolsToGridImage([sub_mol, src_mol, tgt_mol] + can_mols, molsPerRow=4, subImgSize=(300, 300), highlightAtomLists=highlights,
                                   legends=legends)
        display.display(plt)


def prepaire_sub_seq2seq_input(extraction_results, data_aug=2):
    """generate input for substructure-level sequence to sequence model

    Args:
        extraction_results (Dict): dictionary {candidate_id: extraction_result)
        data_aug (int, optional): add randomized SMILES as data augmentation. Defaults to 2.

    Returns:
        input_smiles (List): SMILES of input to the model
        subs_for_merge (List): substructures used for merging with predicted fragments
    """
    input_smiles = []
    subs_for_merge = []
    for cand_id, can_res in extraction_results.items():
        src_sub, _, src_frag, tgt_frag, labeled_src, labeled_tgt = can_res
        src_sub_smi = canonicalize_smiles(Chem.MolToSmiles(src_sub))
        src_frag_smi = canonicalize_smiles(Chem.MolToSmiles(src_frag))
        input_smiles.append(smi_tokenizer(src_sub_smi) +
                            ' | ' + smi_tokenizer(src_frag_smi))
        subs_for_merge.append(src_sub)
        for _ in range(data_aug):
            src_sub_smi = get_random_smiles(src_sub_smi)
            src_frag_smi = get_random_smiles(
                canonicalize_smiles(Chem.MolToSmiles(src_frag)))
            input_smiles.append(smi_tokenizer(src_sub_smi) +
                                ' | ' + smi_tokenizer(src_frag_smi))
            subs_for_merge.append(src_sub)
    return input_smiles, subs_for_merge


def load_sub_seq2seq_output(n_best=10):
    """ load model output, a list of tuple (rank, predicted_fragments)

    Args:
        n_best (int, optional): n_best model predictions for each input. Defaults to 10.

    Returns:
        rank_pred_frag_list (List): a list of tuple (rank, predicted_fragments)
    """
    rank_pred_frag_list = []
    with open('demo_data/predict_output.txt', 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f.readlines()):
            # each line starts with '| '
            rank_pred_frag_list.append(
                (line_id % n_best, line.strip().split('| ')[1].replace(' ', '')))
    return rank_pred_frag_list


def merge(pred_frag, sub, golden):
    """merge predicted fragments with substructure

    Args:
        pred_frag (str): predicted fragments
        sub (_type_): substructure
        golden (str): golden target (golden reactants)

    Returns:
        merge_res (str): SIMIES of merged molecules, None or '[Error]' if failed
        merge_flag (bool): if merge_res equals to the golden 
    """

    merge_flag, merge_res = test_merge_sub_frag(sub, pred_frag, golden)
    if merge_flag is None:
        # merge failed
        return None, False
    if Chem.MolFromSmiles(merge_res) is None:
        # merge failed
        return None, False
    return merge_res, merge_flag


def merge_predicted_frag_with_substructures(rank_pred_frag_list, subs_for_merge, golden_reactants_smiles, n_best=10):
    """merge predicted fragments with substructures

    Args:
        rank_pred_frag_list (List): a list of tuple (rank, predicted_fragments)
        subs_for_merge (List): substructures used for merging with predicted fragments
        golden_reactants_smiles (str): golden target (golden reactants)
        n_best (int, optional): n_best used in inference. Defaults to 10.

    Returns:
        predict2sub_rank (Dict): key is the SMILES of merged reactants, and value is
                a list of tuple (SMILES of substructure, PLACEHOLDER, rank), we 
                will extract features from value for reranking.
        all_predictions (List): a list of tuple (SMILES of merged reactants, flag) 
                and flag = [True, False]. There might have duplicate entries in 
                all_predictions, because different substructures might have same prediction.
    """
    to_merge = []
    for item in subs_for_merge:
        for _ in range(n_best):
            to_merge.append(item)

    assert len(rank_pred_frag_list) == len(to_merge)
    pred2sub_rank = defaultdict(list)
    all_predictions = []
    for rank_pred_frag, sub in zip(rank_pred_frag_list, to_merge):
        rank, pred_frag = rank_pred_frag
        merge_res, merge_flag = merge(pred_frag, sub, golden_reactants_smiles)
        if merge_res and merge_res != '[Error]':
            # None is a placehoder, while in our code we use it to indicate whether the substructure is correct
            pred2sub_rank[merge_res].append(
                (Chem.MolToSmiles(sub), None, rank))
            all_predictions.append((merge_res, merge_flag))
    return pred2sub_rank, all_predictions


def rerank_candidates(s, pred2sub_rank, all_predictions, rerank_top=20):
    """rerank top frequent candidates

    Args:
        s (str): input smiles of src (product)
        pred2sub_rank (Dict): key is the SMILES of merged reactants, and value is
                a list of tuple (SMILES of substructure, PLACEHOLDER, rank), we 
                will extract features from value for reranking.
        all_predictions (List): a list of tuple (SMILES of merged reactants, flag) 
                and flag = [True, False]. There might have duplicate entries in 
                all_predictions, because different substructures might have same prediction.
        rerank_top (int, optional): do re-ranking for most frequent entries. Defaults to 20.

    Returns:
        pred_smi2score (Dict): {predicted SMILES: ranking score}
    """
    predicted_smiles = []
    model_input = []
    for (predict_smi, label), _ in Counter(all_predictions).most_common(rerank_top):
        if predict_smi == s:
            continue
        features = get_all_features(
            get_all_ranking_info(pred2sub_rank[predict_smi]))
        predicted_smiles.append((predict_smi, label))
        model_input.append(features)

    model = RankingModel()
    model.load_state_dict(torch.load('./models/ranker/rank_model.pt', map_location='cpu'))
    model.eval()

    test_loader = DataLoader(RankingTestDataset(
        model_input), batch_size=1000, shuffle=False, num_workers=2)
    ranking_scores = []
    for data in test_loader:
        outputs = model(data)[0]
        ranking_scores.extend(outputs.detach().cpu().numpy())

    assert len(predicted_smiles) == len(ranking_scores)
    pred_smi2score = {k: v[1]
                      for k, v in zip(predicted_smiles, ranking_scores)}
    return pred_smi2score
