from torch.utils.data import Dataset
from .transform import *
import albumentations as albu
from PIL import Image
import numpy as np
import os
import torch
class BaseDataset(Dataset):
    def __init__(self, transform=None,mode="train"):
        self.mode = mode
        aug_list = []
        for k,v in transform.items():
            if v != None:
                aug_list.append(eval(k)(**v))
            else: aug_list.append(eval(k)())

        self.transform = Compose(aug_list)

        self.normalize = albu.Compose([
            albu.Normalize()])


    def __getitem__(self, item):
        pass
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img, mask, img_id = self.load_img_and_mask(index)

        if len(self.transform.transforms) != 0:
            img, mask = self.transform(img, mask)
        img,mask = np.array(img), np.array(mask)

        aug = self.normalize(image=img.copy(), mask=mask.copy())
        img, mask = aug['image'], aug['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return [img,mask,img_id]

    def get_path(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(os.path.join(data_root, img_dir))
        mask_filename_list = os.listdir(os.path.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids
    
    def load_img_and_mask(self, index):
        img_id = self.file_paths[index]
        img_name = os.path.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask_rgb = Image.open(mask_name).convert('RGB')
        mask = self.rgb2label(mask_rgb)
        return [img, mask, img_id]

