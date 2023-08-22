import torch
import torch.nn as nn
from torch.nn import functional as F

class Dice_Loss(nn.Module):
    def __init__(self, exp=2, smooth=1, ignore_index=None, reduction='mean'):
        super(Dice_Loss, self).__init__()
        self.exp = exp
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    # pred (B K H W) target (B H W) 
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        pred = pred.reshape(pred.shape[0], -1) # (B *)
        target = target.reshape(target.shape[0], -1) #
        
        num = torch.sum(torch.mul(pred, one_hot_target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth