import pickle
from collections import defaultdict, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class RankingTrainDataset(Dataset):
    """ for training, each instance is a pair of input ranking features for postive instance and negtive instance    
    """

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        pos_ins, neg_ins = self.instances[idx]
        return torch.from_numpy(np.array(pos_ins, dtype=np.float32)), torch.from_numpy(np.array(neg_ins, dtype=np.float32))


class RankingTestDataset(Dataset):
    """for testing, each instance is the input ranking features   
    """

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        ins = self.instances[idx]
        return torch.from_numpy(np.array(ins, dtype=np.float32))


# TODO add more features like top3_ratio, top4_ratio to further improve the results
InFeature = namedtuple(
    'InFeature', ['sub_count', 'avg_ranking', 'top1_ratio', 'top2_ratio'])
AllFeature = namedtuple(
    'AllFeature', ['raw_feature', 'dedup_feature1', 'dedup_feature2'])


def deduplicate_sub_exists_rankings(sub_exists_rankings):
    """dedpulicate to get the highest ranking
    """
    sub_exists2rankings = defaultdict(list)
    for sub, exists, rank in sub_exists_rankings:
        sub_exists2rankings[tuple((sub, exists))].append(rank)
    result = []
    for (sub, exists), rankings in sub_exists2rankings.items():
        result.append([sub, exists, min(rankings)])
    return result


def deduplicate_results(predicted_results):
    result_arr = []
    for (src, tgt), predictions in tqdm(predicted_results):
        new_predictions = []
        for pred in predictions:
            predicted_smi, label, _, sub_exists_rankings = pred
            sub_exists_rankings = deduplicate_sub_exists_rankings(
                sub_exists_rankings)
            if predicted_smi == src:
                continue
            new_predictions.append(
                (predicted_smi, label, _, sub_exists_rankings))
        result_arr.append(((src, tgt), new_predictions))
    return result_arr


def get_ranking_info(sub_exists_rankings):
    """calculate the ranking features
    """
    avg_ranking = sum(
        [ele[2] for ele in sub_exists_rankings])/len(sub_exists_rankings)
    top1_ranking_ratio = sum(
        [ele[2] == 0 for ele in sub_exists_rankings])/len(sub_exists_rankings)
    top2_ranking_ratio = sum(
        [ele[2] <= 1 for ele in sub_exists_rankings])/len(sub_exists_rankings)
    return InFeature(len(sub_exists_rankings), avg_ranking, top1_ranking_ratio, top2_ranking_ratio)


def get_all_ranking_info(sub_exists_rankings):
    """calculate the ranking features for one instance
    """
    sub_exists_rankings2 = set([tuple(ele) for ele in sub_exists_rankings])
    sub_exists_rankings3 = deduplicate_sub_exists_rankings(
        sub_exists_rankings2)
    return AllFeature(get_ranking_info(sub_exists_rankings), get_ranking_info(sub_exists_rankings2), get_ranking_info(sub_exists_rankings3))


def get_all_features(all_feature: AllFeature):
    """collect the ranking features
    """
    def get_features(in_feature: InFeature):
        return [in_feature.sub_count, in_feature.avg_ranking, in_feature.top1_ratio, in_feature.top2_ratio]
    result = []
    result.extend(get_features(all_feature.raw_feature))
    result.extend(get_features(all_feature.dedup_feature1))
    result.extend(get_features(all_feature.dedup_feature2))
    return result


def should_rank_higher_fea(fea1: InFeature, fea2: InFeature):
    """rules to filter some instances that might mislead the ranking model at feature-level
    """
    if fea1.sub_count >= fea2.sub_count \
            and fea1.avg_ranking <= fea2.avg_ranking \
    and fea1.top1_ratio >= fea2.top1_ratio \
            and fea1.top2_ratio >= fea2.top2_ratio:
        return True
    return False


def should_rank_higher_ins(fea1: AllFeature, fea2: AllFeature):
    """rules to filter some instances that might mislead the ranking model at instance-level
    """
    if should_rank_higher_fea(fea1.raw_feature, fea2.raw_feature) \
        and should_rank_higher_fea(fea1.dedup_feature1, fea2.dedup_feature1) \
            and should_rank_higher_fea(fea1.dedup_feature2, fea2.dedup_feature2):
        return True
    return False


def build_ranking_instances(predicted_results):
    """build ranking instance, note that the predicted_results should be obtained 
    by using the model (trained on the TRAINING data only) to obtain predictions on
    the VAL data. 
    """
    intances_list = []
    for idx, ((src, tgt), predictions) in tqdm(enumerate(predicted_results)):
        other_instance_list = []
        correct_instance = None
        for i, pred in enumerate(predictions[:10]):
            predicted_smi, label, _, sub_exists_rankings = pred
            if label:
                correct_instance = get_all_ranking_info(sub_exists_rankings)
            else:
                other_instance_list.append(
                    get_all_ranking_info(sub_exists_rankings))
        if correct_instance:
            intances_list.append((correct_instance, other_instance_list))
    return intances_list


def select_ranking_data(predicted_results, save_path="rank_training_data.pkl"):
    """build the training data for the ranking model
    """
    all_ranking_instances = build_ranking_instances(predicted_results)
    print('total ranking instances:', len(all_ranking_instances))
    print('ratio:', len(all_ranking_instances)/len(predicted_results))
    incorrect_top1 = 0
    training_data = []
    for pos_instance, neg_instances in all_ranking_instances:
        has_neg_rank_higher = False
        if len(neg_instances) == 0:
            continue
        for neg_instance in neg_instances:
            if should_rank_higher_ins(neg_instance, pos_instance):
                has_neg_rank_higher = True
                continue
            training_data.append((pos_instance, neg_instance))
        if has_neg_rank_higher:
            incorrect_top1 += 1
    print(f"Incorrect top1: {incorrect_top1}")
    print(f"Incorrect top1 ratio: {incorrect_top1/len(predicted_results):.2%}")
    print(
        f"upbound top1: {(len(all_ranking_instances)-incorrect_top1)/len(predicted_results):.2%}")
    print(f"Training data size: {len(training_data)}")

    rank_training_data = []
    for pos_instance, neg_instance in training_data:
        pos_features = get_all_features(pos_instance)
        neg_features = get_all_features(neg_instance)
        rank_training_data.append((pos_features, neg_features))
    print(f"Rank training data size: {len(rank_training_data)}")
    pickle.dump(rank_training_data, open(save_path, "wb"))


def get_ranking_score_features(predicted_results):
    """build ranking features on the predictions

    Args:
        predicted_results: the predictions on test data

    Returns:
        Tuple: a tuple of two list, entries in the first list are the key to locate the predictions,
        entries in the second list are the input features for re-ranking.
    """
    id_list = []
    model_input = []
    for idx, ((src, tgt), predictions) in tqdm(enumerate(predicted_results)):
        for i, pred in enumerate(predictions[:20]):
            predicted_smi, label, _, sub_exists_rankings = pred
            if predicted_smi == src:
                continue
            features = get_all_features(
                get_all_ranking_info(sub_exists_rankings))
            model_input.append(features)
            id_list.append((idx, i))
    return id_list, model_input


def rerank_results_with_scores(predicted_results, id2rankscore, rerank_topk=20, n_best=10):
    """re-rank results with the ranking model.

    Args:
        predicted_results: the predictions on test data
        id2rankscore: a dictionary of key to calculated ranking score, the key is a tuple to locate the instance in predicted_results 
        rerank_topk (int, optional): number of most common predictions to re-rank. Defaults to 20.
        n_best (int, optional): show re-ranked results to n_best accuracies. Defaults to 10.

    Returns:
        Dict: a dictionary of (src, tgt) to re-ranked predictions.
    """
    accuracies = np.zeros([len(predicted_results), n_best], dtype=np.float32)
    result_dict = {}
    for idx, ((src, tgt), predictions) in tqdm(enumerate(predicted_results)):
        new_predictions = []
        for i, pred in enumerate(predictions[:rerank_topk]):
            predicted_smi, label, _, sub_exists_rankings = pred
            #sub_exists_rankings = list(set([tuple(ele) for ele in sub_exists_rankings]))
            if predicted_smi == src:
                continue
            score = id2rankscore[(idx, i)]
            new_predictions.append((pred, score))
        new_predictions = sorted(
            new_predictions, key=lambda x: x[1], reverse=True)
        new_predictions = [ele[0] for ele in new_predictions]

        if len(predictions) > rerank_topk:
            new_predictions += predictions[rerank_topk:]

        for j, prediction in enumerate(new_predictions[:n_best]):
            predicted_smi, label, _, sub_exists_rankings = prediction
            if label:
                accuracies[idx, j:] = 1.0
                break
        result_dict[(src, tgt)] = new_predictions

    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(n_best):
        print(
            f"Partial sub data, top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")

    return result_dict


def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, target):
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * \
            (self.smoothing / inputs.size(-1))
        weight.scatter_(-1, target.unsqueeze(-1), (1. -
                        self.smoothing) + self.smoothing / (inputs.size(-1)))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss.mean()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
