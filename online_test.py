import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import time
import os
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser('description=online test')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tta", type=str, default="d4")
    parser.add_argument("--masks_output_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)

    if args.masks_output_dir is not None:
        masks_output_dir = args.masks_output_dir
    else:
        masks_output_dir = cfg.exp_name + '/online_figs'

    model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg)
    model = model.to('cuda')

    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    results = []
    mask2RGB = False
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        print(len(test_loader))
        for input in tqdm(test_loader):
            raw_predictions, img_id = model(input[0].cuda(), True), input[2]
            pred = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask_pred = pred[i].cpu().numpy()
                mask_name = str(img_id[i])
                results.append((mask2RGB, mask_pred, cfg.dataset, masks_output_dir, mask_name))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    print("masks_save_dir: ", masks_output_dir)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(mask_save, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))