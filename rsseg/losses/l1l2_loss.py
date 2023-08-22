import torch
import torch.nn as nn
from torch.nn import functional as F

class L1_Loss(nn.Module):
    def __init__(self, ignore_index = None, reduction = 'mean'):
        super(L1_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss = nn.L1Loss(reduction=reduction)
    
    # pred (B K H W) target (B H W) 
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1) # (B H W K)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred[mask]                   # (n, k)
            target = target[mask]               # (n)
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes) # (n k)
        
        l1_loss = self.loss(pred, one_hot_target)
        return l1_loss

class L2_Loss(nn.Module):
    def __init__(self, ignore_index = None, reduction = 'mean'):
        super(L2_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=reduction)
    
    # pred (B K H W) target (B H W) 
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1) # (B H W K)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred[mask]                   # (n, k)
            target = target[mask]               # (n)
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes) # (n k)
        
        l2_loss = self.loss(pred, one_hot_target)
        return l2_loss

class Smooth_L1_Loss(nn.Module):
    def __init__(self, beta = 1, ignore_index = None, reduction = 'mean'):
        super(Smooth_L1_Loss, self).__init__()

        self.beta = beta
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1) # (B H W K)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred[mask]                   # (n, k)
            target = target[mask]               # (n)
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes) # (n k)
        
        if self.beta < 1e-5:
            loss = torch.abs(pred - one_hot_target)
        else:
            n = torch.abs(pred - one_hot_target)
            cond = n < self.beta
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            loss = torch.where(cond, 0.5 * n**2 / self.beta, n - 0.5 * self.beta)

        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

if __name__ == "__main__":
    # lossmodel = Smooth_L1_Loss()
    lossmodel = Smooth_L1_Loss(ignore_index=6)
    pred = torch.randn(4, 6, 64, 64)
    target = torch.randint(low=0, high=7, size=(4, 64, 64))
    loss = lossmodel(pred, target)
    print(loss)
        
