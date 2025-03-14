import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import sys
sys.path.append('.')
from utils.config import Config
import json

from train import *


def parse_args():
    parser = argparse.ArgumentParser(description='rsseg: Benchmark a model')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'), default=None)
    parser.add_argument('--repeat-times', type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False

    benchmark_dict = dict(config=args.config, dataset=cfg.dataset, unit='img / s')
    overall_fps_list = []
    cfg.dataset_config.val_mode.loader.batch_size = 1

    # build the model and load checkpoint 
    if osp.exists(args.ckpt):
        model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg)
    else:
        model = myTrain(cfg)
    model = model.cuda()
    model.eval()

    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        # build the dataloader
        data_loader = build_dataloader(cfg.dataset_config, mode='val')

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0
        total_iters = 100

        # benchmark with 200 batches and take the average
        for i, input in enumerate(data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(input[0].cuda(), True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s\n')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                overall_fps_list.append(fps)
                break
    print(overall_fps_list)
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    
    json_str = json.dumps(benchmark_dict, indent=0)
    file_name = cfg.exp_name + "/eval_metric.txt"
    with open(file_name, 'a') as f:
        f.write(json_str+'\n')

if __name__ == '__main__':
    main()