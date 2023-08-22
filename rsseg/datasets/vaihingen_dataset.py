from PIL import Image
from .base_dataset import BaseDataset
import numpy as np
class Vaihingen(BaseDataset):
    def __init__(self, data_root='data/vaihingen', mode='train', transform=None, img_dir='images_1024', mask_dir='masks_1024', img_suffix='.tif', mask_suffix='.png', **kwargs):
        super(Vaihingen, self).__init__(transform, mode)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.data_root = data_root + "/train" if mode == "train" else data_root + "/test"
        self.file_paths = self.get_path(self.data_root, img_dir, mask_dir)

        #RGB
        self.color_map = {
            'ImSurf' : np.array([255, 255, 255]),  # label 0
            'Building' : np.array([0, 0, 255]),  # label 1
            'LowVeg' : np.array([0, 255, 255]),  # label 2
            'Tree' : np.array([0, 255, 0]),  # label 3
            "Car" : np.array([255, 255, 0]),  # label 4
            'Clutter' : np.array([255, 0, 0]),  # label 5
            'Boundary' : np.array([0, 0, 0]),  # label 6
        }

        self.num_classes = 6

    def rgb2label(self,mask_rgb):
        mask_rgb = np.array(mask_rgb)
        _mask_rgb = mask_rgb.transpose(2, 0, 1)
        label_seg = np.zeros(_mask_rgb.shape[1:], dtype=np.uint8)
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['ImSurf'], axis=-1)] = 0
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Building'], axis=-1)] = 1
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['LowVeg'], axis=-1)] = 2
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Tree'], axis=-1)] = 3
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Car'], axis=-1)] = 4
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Clutter'], axis=-1)] = 5
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Boundary'], axis=-1)] = 6

        _label_seg = Image.fromarray(label_seg).convert('L')
        return _label_seg


