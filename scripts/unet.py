# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 5, depth = 5, first_conv_channels = 64, concat=True):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.first_conv_channels = first_conv_channels
        self.uppers = []
        self.downers = []
        self.concat = concat

        for i in range(self.depth):
          if i == 0:
            in_c = self.in_channels
            out_c = self.first_conv_channels
            pool = True
          else:
            in_c = out_c
            out_c = self.first_conv_channels * 2**i

            if i == (depth-1):
              pool = False
            else:
              pool = True

          self.downers.append(UnetDownConv(in_c, out_c, pooling=pool))

        for i in range(self.depth-1):
          in_c = out_c
          out_c = in_c // 2
          self.uppers.append(UnetUpConv(in_c, out_c))
        
        self.uppers = nn.ModuleList(self.uppers)
        self.downers = nn.ModuleList(self.downers)

        self.last_conv = nn.Sequential(
          nn.Conv2d(out_c, self.num_classes, kernel_size=1, stride=1)
        )

        # initialize weights as recommended in paper
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
              init.xavier_normal_(m.weight)
              init.constant_(m.bias, 0)

    def forward(self, x):
        down_sample_outs = []
        # down sampling
        for i, module in enumerate(self.downers):
            x, cache = module(x)
            down_sample_outs.append(cache)

        # up sampling
        for i, module in enumerate(self.uppers):
            cache = down_sample_outs[-(i+2)]
            x = module(cache, x)
        
        x = self.last_conv(x)
        return x

class UnetDownConv(nn.Module):
    def __init__(self, in_channels, out_channels,pooling=True):
        super(UnetDownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

    def forward(self, x):
      cache = self.conv_block(x)
      if self.pooling:
        x = self.pool(cache)
      else:
        x = cache
      return (x, cache)

class UnetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, concat=True):
        super(UnetUpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat

        if self.concat:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                

    def forward(self, high_res, low_res):
        x = self.conv_transpose(low_res)
        if self.concat:
            x = torch.cat((high_res, x),dim=1)
        else:
            x = high_res + x
        x = self.conv_block(x)
        
        return x