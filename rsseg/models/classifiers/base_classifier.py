import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_Classifier(nn.Module):
    def __init__(self, transform_channel, num_class):
        super(Base_Classifier, self).__init__()
        self.classifier = nn.Conv2d(transform_channel, num_class, kernel_size=1, stride=1)

    def forward(self, out):
        pred = self.classifier(out[0])
        return [pred] + out[1:]