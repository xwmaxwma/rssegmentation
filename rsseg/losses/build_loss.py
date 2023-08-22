import torch
import torch.nn as nn
from rsseg.losses.ce_loss import CELoss
class myLoss(nn.Module):
    def __init__(self, loss_name=['CELoss'], loss_weight=[1.0], ignore_index=255, reduction='mean', **kwargs):
        super(myLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.loss = list()
        for _loss in loss_name:
            self.loss.append(eval(_loss)(ignore_index,**kwargs))
    
    def forward(self, preds, target):
        #loss = self.loss[0](preds[0], target) * self.loss_weight[0]
        all_loss = dict()
        all_loss['total_loss'] = 0
        for i in range(0, len(self.loss)):
            loss = self.loss[i](preds[i], target) * self.loss_weight[i]
            if self.loss_name[i] in all_loss:
                all_loss[self.loss_name[i]] += loss
            else:
                all_loss[self.loss_name[i]] = loss
            all_loss['total_loss'] += loss
        return all_loss

def build_loss(cfg):
    loss_type = cfg.pop('type')
    obj_cls = eval(loss_type)
    obj = obj_cls(**cfg)
    return obj