import torch
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(CELoss, self).__init__()

        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")
    
    def forward(self, pred, target):
        loss = self.criterion(pred, target) 
        return loss
