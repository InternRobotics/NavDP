# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

from configs.learning_config import TrainCfg

# visual-imperative-planner
from plannernet import PlannerNet
from rgb_encoder import PRE_TRAIN_POSSIBLE, RGBEncoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        x = x.expand(-1, 3, -1, -1)
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class DualAutoEncoder(nn.Module):
    def __init__(
        self,
        train_cfg: TrainCfg,
        m2f_cfg=None,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem and PRE_TRAIN_POSSIBLE:
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])

        if train_cfg.decoder_small:
            self.decoder = DecoderS(1024, train_cfg.in_channel, train_cfg.knodes)
        else:
            self.decoder = Decoder(1024, train_cfg.in_channel, train_cfg.knodes)
        return

    def forward(self, x_depth: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        # encode depth
        x_depth = x_depth.expand(-1, 3, -1, -1)
        x_depth = self.encoder_depth(x_depth)
        # encode sem
        x_sem = self.encoder_sem(x_sem)
        # concat
        x = torch.cat((x_depth, x_sem), dim=1)  # x.size = (N, 1024, 12, 20)
        # decode
        x, c = self.decoder(x, goal)
        return x, c


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels),
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3)

        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)

        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c


class DecoderS(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels),
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(64 * 48, 256)  # --> in that setting 33 million parameters
        self.fc2 = nn.Linear(256, k * 3)

        self.frc1 = nn.Linear(256, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)  # x.size = (N, 1024+16, 12, 20)
        # compute x
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)  --> (N, 512, 10, 18),
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)  --> (N, 256, 8, 16)
        x = self.relu(self.conv3(x))  # size = (N, 512, x.H/90, x.W/90)  --> (N, 128, 6, 14)
        x = self.relu(self.conv4(x))  # size = (N, 512, x.H/120, x.W/120) --> (N, 64, 4, 12)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.fc2(f)
        x = x.reshape(-1, self.k, 3)

        c = self.sigmoid(self.frc1(f))

        return x, c


# EoF