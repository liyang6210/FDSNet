from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
from .RCCAModule import RCCAModule


class CCNet(nn.Module):
    def __init__(self, num_classes):
        super(CCNet, self).__init__()
        # self.backbone = resnet50(pretrained=True)
        # del self.backbone.avgpool
        # self.backbone.fc = RCCAModule(2048, 512, num_classes)

        self.backbone = RCCAModule(2048, 512, num_classes)


    def forward(self, x):
        # print(x.shape)
        x = self.backbone(x)
        # x = F.interpolate(x, scale_factor=32, mode="bilinear", align_corners=True)
        # print(x.shape)
