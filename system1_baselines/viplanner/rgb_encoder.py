# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pickle
from typing import Optional

import torch
import torch.nn as nn

# detectron2 and mask2former (used to load pre-trained models from Mask2Former)
try:
    from detectron2.config import get_cfg
    from detectron2.modeling.backbone import build_resnet_backbone
    from detectron2.projects.deeplab import add_deeplab_config

    PRE_TRAIN_POSSIBLE = True
except ImportError:
    PRE_TRAIN_POSSIBLE = False
    print("[Warning] Pre-trained ResNet50 models cannot be used since detectron2" " not found")

try:
    from viplanner.third_party.mask2former.mask2former import add_maskformer2_config
except ImportError:
    PRE_TRAIN_POSSIBLE = False
    print("[Warning] Pre-trained ResNet50 models cannot be used since" " mask2former not found")


def get_m2f_cfg(cfg_path: str):  # -> CfgNode:
    # load config from file
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


class RGBEncoder(nn.Module):
    def __init__(self, cfg, weight_path: Optional[str] = None, freeze: bool = True) -> None:
        super().__init__()

        # load pre-trained resnet
        input_shape = argparse.Namespace()
        input_shape.channels = 3
        self.backbone = build_resnet_backbone(cfg, input_shape)

        # load weights
        if weight_path is not None:
            with open(weight_path, "rb") as file:
                model_file = pickle.load(file, encoding="latin1")

            model_file["model"] = {k.replace("backbone.", ""): torch.tensor(v) for k, v in model_file["model"].items()}

            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_file["model"], strict=False)
            if len(missing_keys) != 0:
                print(f"[WARNING] Missing keys: {missing_keys}")
                print(f"[WARNING] Unexpected keys: {unexpected_keys}")
            print(f"[INFO] Loaded pre-trained backbone from {weight_path}")

        # freeze network
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # layers to get correct output shape --> modifiable
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)["res5"]  # size = (N, 2048, 12, 20) (height and width same as ResNet18)
        x = self.conv1(x)  # size = (N, 512,  12, 20)
        return x


# EoF