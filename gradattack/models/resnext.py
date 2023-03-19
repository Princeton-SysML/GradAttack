from __future__ import division

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
                 widen_factor):
        """Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels,
                                     D,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D,
            D,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "shortcut_conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
            )
            self.shortcut.add_module("shortcut_bn",
                                     nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        residual = self.shortcut(x)
        return F.relu(residual + bottleneck, inplace=True)

    def train(self, mode=True, freeze_DP=True):
        super(ResNeXtBottleneck, self).train(mode)

        if freeze_DP:
            self.conv_conv.eval()
            self.conv_conv.weight.requires_grad = False
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self,
                 cardinality,
                 depth,
                 num_classes,
                 widen_factor=4,
                 dropRate=0):
        """Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [
            64,
            64 * self.widen_factor,
            128 * self.widen_factor,
            256 * self.widen_factor,
        ]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block("stage_1", self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block("stage_2", self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block("stage_3", self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(1024, num_classes)
        init.kaiming_normal_(self.classifier.weight)

        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key],
                                         mode="fan_out")
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = "%s_bottleneck_%d" % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(
                    name_,
                    ResNeXtBottleneck(
                        in_channels,
                        out_channels,
                        pool_stride,
                        self.cardinality,
                        self.widen_factor,
                    ),
                )
            else:
                block.add_module(
                    name_,
                    ResNeXtBottleneck(
                        out_channels,
                        out_channels,
                        1,
                        self.cardinality,
                        self.widen_factor,
                    ),
                )
        return block

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)

    def train(self, mode=True, freeze_DP=True):
        super(CifarResNeXt, self).train(mode)
        if freeze_DP:
            for m in self.modules():
                if isinstance(m, ResNeXtBottleneck):
                    m.train(mode, freeze_DP)

                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


def resnext(**kwargs):
    """Constructs a ResNeXt."""
    model = CifarResNeXt(**kwargs)
    return model


def resnext_8x64d(num_classes=100):
    return resnext(cardinality=8,
                   num_classes=num_classes,
                   depth=29,
                   widen_factor=4,
                   dropRate=0)
