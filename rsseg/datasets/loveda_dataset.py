from .base_dataset import BaseDataset
from PIL import Image
import os
import numpy as np
class LoveDA(BaseDataset):
    def __init__(self,data_root='data/vaihingen', mode='train', transform=None,img_dir='images_png', mask_dir='masks_png_convert_rgb', img_suffix='.png', mask_suffix='.png', **kwargs):
        super(LoveDA, self).__init__(transform)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mode = mode

        if mode == "train":
            self.data_root = data_root + "/Train"
        elif mode == "val":
            self.data_root = data_root + "/Val"
        elif mode == "test":
            self.data_root = data_root + "/Test"

        self.file_paths = self.get_path(self.data_root,img_dir,mask_dir)


        #RGB
        self.color_map = {
            'building' : np.array([255, 0, 0]),  # label 0
            'road' : np.array([255, 255, 0]),  # label 1
            'water' : np.array([0, 0, 255]),  # label 2
            'barren' : np.array([159, 129, 183]),  # label 3
            "forest" : np.array([0, 255, 0]),  # label 4
            'agricultural' : np.array([255, 195, 128]),  # label 5
            'background' : np.array([255, 255, 255]),  # label 6
        }

        self.num_classes = 7

    def get_path(self, data_root, img_dir, mask_dir):
        urban_img_filename_list = os.listdir(os.path.join(data_root, 'Urban', img_dir))
        if self.mode != 'test':
            urban_mask_filename_list = os.listdir(os.path.join(data_root, 'Urban', mask_dir))
            assert len(urban_img_filename_list) == len(urban_mask_filename_list)

        urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]

        rural_img_filename_list = os.listdir(os.path.join(data_root, 'Rural', img_dir))
        if self.mode != 'test':
            rural_mask_filename_list = os.listdir(os.path.join(data_root, 'Rural', mask_dir))
            assert len(rural_img_filename_list) == len(rural_mask_filename_list)
        rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]
        img_ids = urban_img_ids + rural_img_ids
        return img_ids


    def load_img_and_mask(self, index):
        img_id, img_type = self.file_paths[index]
        img_name = os.path.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        if self.mode == 'test':
            mask = np.array(img)
            mask = Image.fromarray(mask).convert('L')
            return [img, mask, img_id]

        mask_name = os.path.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        mask_rgb = Image.open(mask_name).convert('RGB')
        mask = self.rgb2label(mask_rgb)
        return [img, mask, img_id]

    def rgb2label(self,mask_rgb):

        mask_rgb = np.array(mask_rgb)
        _mask_rgb = mask_rgb.transpose(2, 0, 1)
        label_seg = np.zeros(_mask_rgb.shape[1:], dtype=np.uint8)
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['building'], axis=-1)] = 1
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['road'], axis=-1)] = 2
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['water'], axis=-1)] = 3
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['barren'], axis=-1)] = 4
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['forest'], axis=-1)] = 5
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['agricultural'], axis=-1)] = 6
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['background'], axis=-1)] = 0

        _label_seg = Image.fromarray(label_seg).convert('L')
        return _label_seg
