import torch
import argparse
import time
from timm import create_model
from mmcv import Config

from mmseg.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
from mmseg.models import build_segmentor

torch.autograd.set_grad_enabled(False)

import sys
sys.path.append('.')
from train import *

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

T0 = 5
T1 = 10

def throughput(model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    while time.time() - start < T0:
        model(inputs)
    timing = []
    torch.cuda.synchronize()
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

device = "cuda:0"

from argparse import ArgumentParser

parser = ArgumentParser()

def parse_args():
    parser = argparse.ArgumentParser(description='rsseg: Benchmark a model')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = myTrain(cfg)

    batch_size = args.batch_size
    resolution = args.resolution
    torch.cuda.empty_cache()
    inputs = torch.randn(batch_size, 3, resolution,
                            resolution, device=device)
    replace_batchnorm(model)

    model.to(device)
    model.eval()
    throughput(model, device, batch_size, resolution=resolution)
