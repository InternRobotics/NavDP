# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# python

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation import INSTANCE_OFFSET
from configs.coco_sem_meta import get_class_for_id_mmdet
from configs.viplanner_sem_meta import VIPlannerSemMetaHandler

class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation."""
    debug: bool = True
    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        checkpoint_file="model_final.pth",
    ) -> None:
        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]

    def predict(self, image: np.ndarray) -> np.ndarray:
        result = inference_detector(self.model, image)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        # create output
        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            curr_label = curr_sem_class % INSTANCE_OFFSET
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_label]
            except KeyError:
                if curr_sem_class != len(self.model.dataset_meta["classes"]):
                    print(f"Category {curr_label} not found in" " coco_viplanner_cls_mapping.")
                panoptic_mask[result == curr_sem_class] = self.viplanner_sem_class_color_map["static"]
        if self.debug:
            cv2.imwrite("panoptic_mask.png", panoptic_mask)
        return panoptic_mask

if __name__ == "__main__":
    m2f = Mask2FormerInference(
        config_file="/home/PJLAB/caiwenzhe/miniconda3/envs/habitat/lib/python3.9/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",
        checkpoint_file="/home/PJLAB/caiwenzhe/Desktop/system1_toolbox/viplanner/checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth",
    )
    image = cv2.imread("/home/PJLAB/caiwenzhe/Desktop/system1_toolbox/viplanner/assets/input_image.jpg")
    batch_image = np.array([image,image,image])
    batch_mask = [m2f.predict(img) for img in batch_image]
    #m2f.predict(image)