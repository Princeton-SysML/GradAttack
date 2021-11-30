import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, 1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1)
        # self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1)
        # self.conv3_bn = nn.BatchNorm2d(128)
        self.n_size = self._get_conv_output((3, 32, 32))
        self.fc1 = nn.Linear(self.n_size, 1024)
        # self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        # x = self.conv1_bn(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        # x = self.conv2_bn(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        # x = self.conv3_bn(x)
        x = x.view(-1, self.n_size)
        x = F.relu(self.fc1(x))
        # x = self.bn1(x)
        x = self.fc2(x)
        return x

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        return x
