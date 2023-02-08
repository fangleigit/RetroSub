import json
import pickle
from os.path import join


def read_json(file_path, total_chunks=1, chunk_id=0):
    src, tgt, candidates, scores = [], [], [], []
    with open(file_path, encoding='utf-8') as f:
        all_data = json.loads(f.read())

    assert chunk_id >= 0 and chunk_id < total_chunks

    if total_chunks > 1:
        all_data = all_data[chunk_id::total_chunks]

    for item in all_data:
        src.append(item['src'])
        tgt.append(item['tgt'])
        candidates.append([m[0] for m in item['cans']])
        scores.append([m[1] for m in item['cans']])

    return src, tgt, candidates, scores


# data loader for one chunk
def extraction_result_loader(store_path, datasplit, total_chunks, chunk_id):
    f_path = join(
        store_path, f'{datasplit}_{chunk_id}_{total_chunks}_.pkl')
    with open(f_path, 'rb') as f:
        for item in pickle.load(f):
            yield item  
