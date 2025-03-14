import os
import numpy as np
import argparse
import torch
from PIL import Image
import random
import glob
import time
import multiprocessing.pool as mpp
import multiprocessing as mp
import cv2
import re
import albumentations as albu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", type=str, default="potsdam")
    parser.add_argument("--img-dir", default="data/potsdam/train_images")
    parser.add_argument("--mask-dir", default="data/potsdam/train_masks")
    parser.add_argument("--output-img-dir", default="data/potsdam/train/images_1024")
    parser.add_argument("--output-mask-dir", default="data/potsdam/train/masks_1024")
    parser.add_argument("--mode", type=str, default='train')

    parser.add_argument("--split-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()

def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop['image'], crop['mask']
    return img_crop, mask_crop

def aug(image, mask, img_filename, mask_filename, k):
    # print(mask)
    h,w = image.shape[:2]
    car_mask = np.zeros((h,w),dtype=np.uint8)
    veg_mask = np.zeros((h,w),dtype=np.uint8)
    mask = np.array(mask)
    car_mask[np.all(mask==np.array([0, 255, 255]), axis=-1)] = 1
    veg_mask[np.all(mask==np.array([255, 255, 0]), axis=-1)] = 1
    count_car = np.count_nonzero(car_mask)
    count_veg = np.count_nonzero(veg_mask)

    if count_car / (h*w) < 0.08:
        return

    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())

    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']

    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    for i in range(len(image_list)):
        image, mask = image_list[i], mask_list[i]
        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, k, i))
        cv2.imwrite(out_img_path, image)
        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, k, i))
        cv2.imwrite(out_mask_path, mask)    
    return

def split(img,mask_rgb,img_filename,mask_filename):
    assert  img.shape[0] == mask_rgb.shape[0] \
           and img.shape[1] == mask_rgb.shape[1]
    img_W = img.shape[1]
    img_H = img.shape[0]
    k = 0

    v_flip = albu.VerticalFlip(p=1.0)(image=img.copy(), mask=mask_rgb.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=img.copy(), mask=mask_rgb.copy())
    img_mask = [(img, mask_rgb)]
    if dataset_type == 'vaihingen' and mode == "train":
        img_mask += [(v_flip['image'],v_flip['mask']), (h_flip['image'],h_flip['mask'])]

    for i in range(len(img_mask)):
        img, mask_rgb = img_mask[i]

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                x_str = x
                x_end = x + split_size
                y_str = y
                y_end = y + split_size

                if x_end > img_W:
                    diff_x = x_end - img_W
                    x_str -= diff_x
                    x_end = img_W

                if y_end > img_H:
                    diff_y = y_end - img_H
                    y_str -= diff_y
                    y_end = img_H


                img_tile = img[y_str:y_end, x_str:x_end]
                mask_rgb_tile = mask_rgb[y_str:y_end, x_str:x_end]

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_rgb_tile.shape[0] == split_size and mask_rgb_tile.shape[1] == split_size:


                    out_img_path = os.path.join(imgs_output_dir, "{}_{}.tif".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(masks_output_dir, "{}_{}.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_rgb_tile)

                    if dataset_type == "vaihingen" and mode == "train":
                        img_crop, mask_crop = randomsizedcrop(img_tile, mask_rgb_tile)
                        aug(img_crop, mask_crop, img_filename, mask_filename, k)

                k += 1


def vaihingen_split(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = img_filename

    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')

    image_width, image_height = image.size[1], mask.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width

    img = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
    mask_rgb = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    split(img,mask_rgb,img_filename,mask_filename)

def get_vaihingen_file(imgs_dir,masks_dir,imgs_output_dir,masks_output_dir,split_size,stride,mode):
    train_seq_list = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
    test_seq_list = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

    seq_list = train_seq_list if mode == 'train' else test_seq_list

    img_paths_ori = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths_ori = glob.glob(os.path.join(masks_dir, "*.tif"))

    img_num = []
    img_paths = []
    mask_paths = []
    mask_num = []

    for file_name in img_paths_ori:
        match = re.search(r'\d+', file_name[::-1])
        if match:
            number = int(match.group()[::-1])
            if number in seq_list:
                img_paths.append(file_name)
                img_num.append(number)
    for file_name in mask_paths_ori:
        match = re.search(r'\d+', file_name[::-1])
        if match:
            number = int(match.group()[::-1])
            if number in seq_list:
                mask_paths.append(file_name)
                mask_num.append(number)

    img_paths = sorted(img_paths, key=lambda x: img_num[img_paths.index(x)])
    mask_paths = sorted(mask_paths, key=lambda x: mask_num[mask_paths.index(x)])



    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_split, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))

def potsdam_split(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, split_size, stride) = inp

    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = img_filename

    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')


    image_width, image_height = image.size[1], mask.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width

    img = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
    mask_rgb = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    split(img,mask_rgb,img_filename,mask_filename)


def get_potsdam_file(imgs_dir,masks_dir,imgs_output_dir,masks_output_dir,split_size,stride,mode):

    train_seq_list =  ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7', '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9']
    test_seq_list = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13']

    seq_list = train_seq_list if mode == 'train' else test_seq_list


    img_paths_ori = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths_ori = glob.glob(os.path.join(masks_dir, "*.tif"))

    img_paths = []
    mask_paths = []

    for file_name in img_paths_ori:
        match = re.search(r'(\d+_\d+)', file_name)
        if match:
            number = match.group()
            if number in seq_list:
                img_paths.append(file_name)
    for file_name in mask_paths_ori:
        match = re.search(r'(\d+_\d+)', file_name)
        if match:
            number = match.group()
            if number in seq_list:
                mask_paths.append(file_name)


    img_paths.sort()
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir,
            split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(potsdam_split, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


SEED = 66
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    seed_everything(66)
    args = parse_args()
    dataset_type = args.dataset_type
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    mode = args.mode

    split_size = args.split_size
    stride = args.stride

    if dataset_type == "vaihingen":
        get_vaihingen_file(imgs_dir,masks_dir,imgs_output_dir,masks_output_dir,split_size,stride,mode)
    elif dataset_type == "potsdam":
        get_potsdam_file(imgs_dir,masks_dir,imgs_output_dir,masks_output_dir,split_size,stride,mode)
    else:
        print("dataset_type error [vaihingen,potsdam]")