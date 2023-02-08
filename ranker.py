import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.rerank_utils import *


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_training', action='store_true')
    parser.add_argument('--val_preds', type=str)
    parser.add_argument('--data_save_path', type=str)
    parser.add_argument('--model_save_dir', type=str)
    return parser.parse_args()


def calc_ranking_scores(predicted_results, model_path):
    id_list, model_input = get_ranking_score_features(predicted_results)
    model = RankingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(RankingTestDataset(
        model_input), batch_size=1000, shuffle=False, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ranking_scores = []
    for data in tqdm(test_loader):
        data = data.to(device)
        outputs = model(data)[0]
        ranking_scores.extend(outputs.detach().cpu().numpy())
    id2score = {k: v[1] for k, v in zip(id_list, ranking_scores)}
    return id2score


class RankingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(12, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 2)
        )
        self.num_labels = 2
        self.loss_fct = LabelSmoothingCrossEntropy(0.01)
        self.scorer.apply(init_weights)

    def forward(self, x1, x2=None):
        pos_logits = self.scorer(x1)
        output = (pos_logits,)
        if x2 is None:
            return output

        batch_size = x1.shape[0]
        labels = torch.ones(batch_size, 1, dtype=torch.long, device=x1.device)

        neg_logits = self.scorer(x2)
        logits_diff = pos_logits - neg_logits
        loss = self.loss_fct(
            logits_diff.view(-1, self.num_labels), labels.view(-1))

        return (loss,) + output


if __name__ == '__main__':
    args = parse_config()
    if not args.do_training:
        predicted_results = json.load(open(args.val_preds))
        select_ranking_data(predicted_results, save_path=args.data_save_path)
        exit(0)

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    rank_training_data = pickle.load(open(args.data_save_path, 'rb'))
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankingModel().to(device)
    learning_rate = 1e-3
    batch_size = 64
    print('size of training data: ', len(rank_training_data))
    training_loader = DataLoader(RankingTrainDataset(
        rank_training_data[:300000]), batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(RankingTrainDataset(
        rank_training_data[300000:]), batch_size=batch_size*10, shuffle=False, num_workers=2)

    num_epochs = 2000
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)

    for _ in range(num_epochs):
        running_loss = 0.
        train_correct = 0
        for data in training_loader:
            data = move_to_device(data, device)
            optimizer.zero_grad()
            pos_ins, neg_ins = data
            loss, outputs = model(pos_ins, neg_ins)
            train_correct += (outputs.argmax(1) ==
                              1).type(torch.float).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print("Learning Rate = ", optimizer.param_groups[0]["lr"])

        size = len(validation_loader.dataset)
        num_batches = len(validation_loader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in validation_loader:
                data = move_to_device(data, device)
                pos_ins, neg_ins = data
                loss, outputs = model(pos_ins, neg_ins)
                test_loss += loss.item()
                correct += (outputs.argmax(1) ==
                            1).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f'epoch {_} loss: {(running_loss/len(training_loader)):>5f}, Train acc: {(100*train_correct/len(training_loader.dataset)):>0.1f}%, Test acc: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
        torch.save(model.state_dict(),
                   f'{args.model_save_dir}/rank_model_{_}.pt')
