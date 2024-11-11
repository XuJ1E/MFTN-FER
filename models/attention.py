# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/11/2 17:31
# Author     ：XuJ1E
# version    ：python 3.8
# File       : cswin.py
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath


class SePeAttention(nn.Module):
    """ Static enhanced  Positional encoding."""
    def __init__(self, dim, num_heads=8, act_layer=nn.GELU, qkv_bias=False, attn_drop=0.):
        super().__init__()
        self.dim = dim
        hidden_dim = int(max(dim//num_heads, 32))
        self.query = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=qkv_bias),
            act_layer(),
            nn.Linear(hidden_dim, dim, bias=qkv_bias))
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.position = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=dim)
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        sepe = x.transpose(-1, -2).reshape(b, c, h, w)
        sepe = self.position(sepe).reshape(b, c, -1).transpose(-1, -2)
        kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        q = self.query(x)
        a = self.attn_drop(q*kv[0])
        a = self.softmax(a)*kv[1] + sepe
        a = self.proj(a)
        return a


class Mlp(nn.Module):
    """ MLP for ViT Block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SePeAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim), act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PyramidViT(nn.Module):
    def __init__(self, stage, base_dim=128, depth=3, num_classes=7, drop_path_rate=.0):
        super().__init__()

        self.stage = stage
        dpr = [x.item() for x in torch.linspace(1, drop_path_rate, depth)]
        self.branch = nn.Sequential(*[Block(dim=base_dim*8, num_heads=16, drop_path=dpr[j]) for j in range(depth)])

        self.branches = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        for i in range(stage):
            dim = base_dim*(2**(i+1))
            num_depth = depth - i

            dpr = [x.item() for x in torch.linspace(1, drop_path_rate, num_depth)]

            branch = nn.Sequential(*[Block(dim=dim, num_heads=16, drop_path=dpr[j]) for j in range(num_depth)])

            self.branches.append(branch)

            fuse_layer = nn.Linear(base_dim*(2**(i+1)), base_dim*4, bias=False)
            self.fuse_layers.append(fuse_layer)

        self.head = nn.Linear(base_dim*4, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=.02)

    def forward(self, x1, x2, x3):
        x1 = self.branches[0](x1)
        x1 = self.fuse_layers[0](x1).mean(1)

        x2 = self.branches[1](x2)
        x2 = self.fuse_layers[1](x2).mean(1)

        x3 = self.branches[2](x3)
        x3 = self.fuse_layers[2](x3).mean(1)

        x = self.head(x1+x2+x3)
        return x


if __name__ == '__main__':
    x1 = torch.rand((1, 784, 256))
    x2 = torch.rand((1, 196, 512))
    x3 = torch.rand((1, 49, 1024))

    model = PyramidViT(stage=3)
    x3 = model(x1, x2, x3)
    # print(model)
    print(x3.shape)
