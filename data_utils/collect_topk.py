import json
import argparse


def read_file(file_path):
    candidates, scores  = [], []
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            try:
                src, tgt, *can =line.strip().split('\t')
            except:
                continue
            src = src.replace(' ', '')
            tgt = tgt.replace(' ', '')
            candidates = [pro.replace(' ', '') for pro in can[:-1:2]]
            scores = [float(score) for score in can[1::2]]

            yield src, tgt, candidates, scores

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/uspto_full/retrieval/')
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--output_path', type=str, default='data/uspto_full/retrieval')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    all_data = []
    res = read_file(args.input)
    count = 0
    for src, tgt, candidates, scores in res:
        item = {'id': count,
                'src': src,
                'tgt': tgt,
                'cans': [(can, score) for can, score in zip(candidates,scores)]
        }
        count += 1
        all_data.append(item)
    json.dump(all_data, open(args.output_path + '/' + args.dataset + '.json', 'w', encoding='utf-8'))
    print(f'Collect items: {count}')