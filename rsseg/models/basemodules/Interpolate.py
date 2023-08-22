import torch.nn as nn
import torch.nn.functional as F

class Interpolate(nn.Module):
    def __init__(self, scale=8, mode='bilinear'):
        super().__init__()
        self.scale_list = scale
        self.mode = mode
    
    def forward(self, x_list):
        for i in range(len(self.scale_list)):
            x_list[i] = F.interpolate(x_list[i], scale_factor = self.scale_list[i], mode = self.mode)
        return x_list