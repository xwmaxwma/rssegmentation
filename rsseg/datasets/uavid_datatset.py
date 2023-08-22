from .base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Uavid(BaseDataset):
    def __init__(self,data_root='data/uavid', mode='train', transform=None,img_dir='images', mask_dir='masks', img_suffix='.png', mask_suffix='.png', **kwargs):
        super(Uavid, self).__init__(transform)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.data_root = data_root + "/train" if mode == "train" else data_root + "/test"
        self.file_paths = self.get_path(self.data_root,img_dir,mask_dir)

        #RGB
        self.color_map = {
            'Building' : np.array([128, 0, 0]),  # label 0
            'Road' : np.array([128, 64, 128]),  # label 1
            'Tree' : np.array([0, 128, 0]),  # label 2
            'LowVeg' : np.array([128, 128, 0]),  # label 3
            "Moving_Car" : np.array([64, 0, 128]),  # label 4
            'Static_Car' : np.array([192, 0, 192]),  # label 5
            'Human' : np.array([64, 64, 0]),  # label 6
            'Clutter': np.array([0, 0, 0]), # label 7
            'Boundary': np.array([255, 255, 255]) # label 8
        }

        self.num_classes = 6

        num_classes = 8


    def rgb2label(self,mask_rgb):
        mask_rgb = np.array(mask_rgb)
        _mask_rgb = mask_rgb.transpose(2, 0, 1)
        label_seg = np.zeros(_mask_rgb.shape[1:], dtype=np.uint8)
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Building'], axis=-1)] = 0
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Road'], axis=-1)] = 1
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Tree'], axis=-1)] = 2
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['LowVeg'], axis=-1)] = 3
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Moving_Car'], axis=-1)] = 4
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Static_Car'], axis=-1)] = 5
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Human'], axis=-1)] = 6
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Clutter'], axis=-1)] = 7
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Boundary'], axis=-1)] = 8

        _label_seg = Image.fromarray(label_seg).convert('L')
        return _label_seg