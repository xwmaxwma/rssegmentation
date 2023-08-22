import torch
import torch.nn as nn
from torch.nn import functional as F

class sigmoid_focal_loss(nn.Module):
    def __init__(self, alpha=-1, gamma=0, reduction='mean', ignore_index=None):
        super(sigmoid_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    # pred (B K H W) target (B H W) 
    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1).float() # (B H W K)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred[mask]                   # (n, k)
            target = target[mask]               # (n)
            
        num_classes = pred.size(-1)
        one_hot_target = F.one_hot(target, num_classes) # (n k)
        one_hot_target = one_hot_target.float() 
        
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, one_hot_target, reduction="none")
    
            
        p_t = p * one_hot_target + (1 - p) * (1 - one_hot_target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - one_hot_target)
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

if __name__ == "__main__":
    lossmodel = sigmoid_focal_loss(ignore_index=6)
    pred = torch.randn(4, 6, 64, 64)
    target = torch.randint(low=0, high=7, size=(4, 64, 64))
    loss = lossmodel(pred, target)
    print(loss)
    
    
        
