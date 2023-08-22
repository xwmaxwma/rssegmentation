import torch
from torch import nn
from utils.build import build_from_cfg
class myModel(nn.Module):
    def __init__(self, cfg):
        super(myModel, self).__init__()
        self.backbone = build_from_cfg(cfg.backbone)
        self.seghead = build_from_cfg(cfg.seghead)
        self.classifier = build_from_cfg(cfg.classifier)
        self.upsample = build_from_cfg(cfg.upsample)
        
    
    def forward(self, x):
        backbone_outputs = self.backbone(x)
        x_list = self.seghead(backbone_outputs)   # 考虑到辅助损失
        x_list = self.classifier(x_list)
        x_list = self.upsample(x_list)

        return x_list

"""
对于不满足该范式的模型可在backbone部分进行定义, 并在此处导入
"""

# model_config
def build_model(cfg):
    c = myModel(cfg)
    return c


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    target = torch.randint(low=0,high=6,size=[2, 512, 512])
    file_path = "/home/xwma/rssegmentation/configs/docnet.py"
    import sys
    sys.path.append('/home/xwma/rssegmentation')
    sys.path.append('/home/xwma/rssegmentation/rsseg')
    from utils.config import Config
    from rsseg.losses import build_loss


    cfg = Config.fromfile(file_path)
    net = build_model(cfg.model_config)
    res = net(x)
    loss = build_loss(cfg.loss_config)

    compute = loss([res],target)
    print(compute)