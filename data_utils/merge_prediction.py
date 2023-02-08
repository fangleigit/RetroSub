import argparse
import json
from collections import defaultdict
from os.path import join, exists

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', type=str)
    parser.add_argument('--total_chunks', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    total_chunks = args.total_chunks
    dir_path = args.dir

    for post_fix in ['False', 'True']:
        scores = defaultdict(int)
        num_reactions = 0
        dump_result = []
        for i in tqdm(range(total_chunks)):
            result_path = join(
                dir_path, f'test_{i}_{total_chunks}_prediction_res_{post_fix}.json')
            if not exists(result_path):
                print(result_path)
                continue
            with open(result_path) as f:
                cur_score, cur_num_reactions = json.load(f)
                for k, v in cur_score.items():
                    scores[k] += v
                num_reactions += cur_num_reactions
            dump_path = join(
                dir_path, f'test_{i}_{total_chunks}_prediction_res_{post_fix}_analysis.json')
            if not exists(dump_path):
                print(dump_path)
                continue
            dump_result.extend(json.load(open(dump_path)))
        print('Top-k acc based on frequency')
        print(num_reactions)
        for k in range(1, 1 + 10):
            print('Top-{}: {:.2f}%'.format(k, 100 *
                                           scores[str(k)] / num_reactions))

        with open(join(dir_path, f'dump_res_{post_fix}_analysis.json'), 'w') as f:
            json.dump(dump_result, f, indent=4)
