from PIL import Image
import numpy as np
import argparse
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
from tqdm import tqdm
import os 
import sys
sys.path.append('.')
from train import *
from utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='rsseg: tsne map')
    parser.add_argument("-c", "--config", type=str, default="configs/vaihingen/logcanplus.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/logcanplus_vaihingen/epoch=64.ckpt")
    parser.add_argument("--tar_size", type=tuple, default=(64, 64))
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=45)
    parser.add_argument("--tsne_output_dir", default=None)
    args = parser.parse_args()
    return args

def color_trans(tsne_color):
    if tsne_color.all() == np.array([255, 255, 255]).all():
        tsne_color = np.array([50, 100, 150])
    return tsne_color / 255.0

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.dataset_config.val_mode.loader.batch_size = 1
    model = myTrain.load_from_checkpoint(args.ckpt, cfg = cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    model.to(device)
    
    if args.tsne_output_dir is not None:
        tsne_output_dir = args.tsne_output_dir
    else:
        tsne_output_dir = cfg.exp_name + '/tsne_figs/'
        if not os.path.exists(tsne_output_dir):
            os.makedirs(tsne_output_dir)
    
    test_loader = build_dataloader(cfg.dataset_config, mode='test') 
    model.eval()

    mytsne = TSNE(n_components=args.n_components, random_state=args.random_state)
    
    for input in tqdm(test_loader):
        images, gts, img_ids = input[0].to(device), input[1].to(device), input[2]
        img_id = img_ids[0]
        features = model.net.backbone(images)
        features = model.net.seghead(features)[0]

        features = F.interpolate(features, args.tar_size, mode='nearest')
        gts = F.interpolate(gts.float().unsqueeze(1), size=args.tar_size, mode='nearest').squeeze(1)

        features = features.flatten(2).transpose(1,2)
        features = features.cpu().detach().numpy()[0]
        gts = gts.cpu().numpy()[0].reshape(-1)
        class_name = cfg.class_name[cfg.eval_label_id_left:cfg.eval_label_id_right]

        tsne = mytsne.fit_transform(features)
        color_map = test_loader.dataset.color_map

        for j in range(0,len(class_name)):
            plt.scatter(tsne[:, 0][gts == j], tsne[:, 1][gts == j], c=color_trans(color_map[class_name[j]]).reshape(1, -1),
                        s=1, label=class_name[j])
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=6, borderpad=0.15, markerscale=5)
        plt.savefig(tsne_output_dir + '{}'.format(img_id) + '.pdf')
        plt.close()

if __name__ == "__main__":
    main()
