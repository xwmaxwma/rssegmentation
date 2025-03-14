import ttach as tta
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam import GradCAM
import random
import os
import torch.nn.functional as F

import sys
sys.path.append('.')
from train import *
from utils.config import Config

class SemanticSegmentationTarget:
    """wrap the model.

    requirement: pip install grad-cam

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.to("cuda:0")

    def __call__(self, model_output):
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear')
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()

def parse_args():
    parser = argparse.ArgumentParser(description='rsseg: cam map')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tar_layer", type=str, default="model.net.seghead.catconv2[-2]")
    parser.add_argument("--tar_category", type=int, default=1)
    parser.add_argument("--cam_output_dir", default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.dataset_config.val_mode.loader.batch_size = 1
    model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    model.to(device)

    if args.cam_output_dir is not None:
        cam_output_dir = args.cam_output_dir
    else:
        cam_output_dir = cfg.exp_name + '/cam_figs/'
        if not os.path.exists(cam_output_dir):
            os.makedirs(cam_output_dir)
    
    test_loader = build_dataloader(cfg.dataset_config, mode='test') 
    model.eval()

    for input in tqdm(test_loader):
        masks, gts, img_ids = model(input[0].to(device)), input[1].cuda(), input[2]
        masks = nn.Softmax(dim=1)(masks[0]) 
        masks = masks.argmax(dim=1)           
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            gt = gts[i].cpu().numpy()
            mask_name = img_ids[i]

            tar_layer = [eval(args.tar_layer)]
            category = args.tar_category

            height, width = gt.shape[-2:]
            mask_float = np.float32(mask == category)

            targets = [
                SemanticSegmentationTarget(category, mask_float, (height, width))
            ]

            data_cfg = cfg.dataset_config

            test_dataset = test_loader.dataset
            img_path = os.path.join(test_dataset.data_root, test_dataset.img_dir, mask_name + test_dataset.img_suffix)

            ori_img = cv2.imread(img_path)
            rgb_img = ori_img.astype(np.float32) / 255.0

            cam = GradCAM(model=model, target_layers=tar_layer)
            grayscale_cam = cam(input_tensor=input[0][i].unsqueeze(0).to(device), targets=targets)[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            Image.fromarray(cam_image).save(cam_output_dir + mask_name + '.png')

if __name__ == "__main__":
    main()
