import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import prettytable
import time
import glob
import os
import os.path as osp
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser(description='rsseg: test model')
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
        masks_output_dir = cfg.exp_name + '/figs'

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

    metric_cfg1 = cfg.metric_cfg1
    metric_cfg2 = cfg.metric_cfg2

    test_oa=torchmetrics.Accuracy(**metric_cfg1).to('cuda')
    test_prec = torchmetrics.Precision(**metric_cfg2).to('cuda')
    test_recall = torchmetrics.Recall(**metric_cfg2).to('cuda')
    test_f1 = torchmetrics.F1Score(**metric_cfg2).to('cuda')
    test_iou=torchmetrics.JaccardIndex(**metric_cfg2).to('cuda')

    results = []
    mask2RGB = True
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='val')
        for input in tqdm(test_loader):
            raw_predictions, mask, img_id = model(input[0].cuda(), True), input[1].cuda(), input[2]
            pred = raw_predictions.argmax(dim=1)

            test_oa(pred, mask)
            test_iou(pred, mask)
            test_prec(pred, mask)
            test_f1(pred, mask)
            test_recall(pred, mask)

            for i in range(raw_predictions.shape[0]):
                mask_pred = pred[i].cpu().numpy()
                mask_name = str(img_id[i])
                results.append((mask2RGB, mask_pred, cfg.dataset, masks_output_dir, mask_name))

    metrics = [test_prec.compute(),
               test_recall.compute(),
               test_f1.compute(),
               test_iou.compute()]

    total_metrics = [test_oa.compute().cpu().numpy(),
                     np.mean([item.cpu() for item in metrics[0][cfg.eval_label_id_left: cfg.eval_label_id_right] if item > 0]),
                     np.mean([item.cpu() for item in metrics[1][cfg.eval_label_id_left: cfg.eval_label_id_right] if item > 0]),
                     np.mean([item.cpu() for item in metrics[2][cfg.eval_label_id_left: cfg.eval_label_id_right] if item > 0]),
                     np.mean([item.cpu() for item in metrics[3][cfg.eval_label_id_left: cfg.eval_label_id_right] if item > 0])]

    result_table = prettytable.PrettyTable()
    result_table.field_names = ['Class', 'OA', 'Precision', 'Recall', 'F1_Score', 'IOU']

    for i in range(len(metrics[0])):
        item = [i, '--']
        for j in range(len(metrics)):
            item.append(np.round(metrics[j][i].cpu().numpy(), 4))
        result_table.add_row(item)

    total = [np.round(v, 4) for v in total_metrics]
    total.insert(0, 'Total')
    result_table.add_row(total)

    print(result_table)

    file_name = cfg.exp_name + "/eval_metric.txt"
    f = open(file_name,"a")
    current_time = time.strftime('%Y_%m_%d %H:%M:%S',time.localtime(time.time()))
    f.write(current_time+' test\n')
    f.write(str(result_table)+'\n')

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    print("masks_save_dir: ", masks_output_dir)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(mask_save, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))