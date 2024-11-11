# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/11/2 9:24
# Author     ：XuJ1E
# version    ：python 3.8
# File       : model.py
"""
import math
import torch
import torch.nn as nn
from .convnext import convnext_base
from .mstn import MsTN


class Model(nn.Module):
    def __init__(self, in_chans=3, stage=3, pretrained=False, num_classes=7, drop_path_rate=0.25):
        super().__init__()
        self.backbone = convnext_base(in_chans=in_chans, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path_rate)
        self.mstn = MsTN(stage=stage, num_classes=num_classes, drop_path_rate=drop_path_rate)

    def forward(self, x):
        x1, x2, x3, out1 = self.backbone(x)

        # ---------transpose the dim--------
        x1 = x1.flatten(2).transpose(-1, -2)
        x2 = x2.flatten(2).transpose(-1, -2)
        x3 = x3.flatten(2).transpose(-1, -2)
        # ---------transpose the dim--------

        out2 = self.mstn(x1, x2, x3)
        return out1, out2


if __name__ == '__main__':
    from thop import profile
    input = torch.rand((2, 3, 224, 224))
    models = Model()
    print(models.backbone)
