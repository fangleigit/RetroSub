import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import GPUtil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mem', type=float, default=0.5)
    args = parser.parse_args()
    return args


# python -u data_utils/dispatch_infer.py --model uspto_full_retrosub --dir subextraction
if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    test_data_dir_name = args.dir
    print(datetime.now())
    total_chunks = 200
    test_data_dir = f'./data/uspto_full/{test_data_dir_name}'

    log_dir = f'logs/{model_name}_{test_data_dir_name}'
    result_dir = f'data/result_{model_name}_{test_data_dir_name}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    pro_list = []
    for chunk_id in range(total_chunks):
        device_id = GPUtil.getFirstAvailable(
            order='memory', maxLoad=0.8, maxMemory=args.mem, attempts=100, interval=60, verbose=False)[0]
        log_file = f'{log_dir}/{chunk_id}_{total_chunks}.log'
        bash_command = f'bash scripts/uspto_full/step4_predict_chunk.sh  {model_name} {chunk_id} {total_chunks} {test_data_dir} {device_id} {result_dir}'
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                bash_command, shell=True, stdout=f, stderr=f)
            print(datetime.now(), process.args)
            pro_list.append(process)
        # wait some time for the model to be loaded to GPU
        time.sleep(60)

    for process in pro_list:
        process.wait()
        if process.returncode != 0:
            print(process.args)
