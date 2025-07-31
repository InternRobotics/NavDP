import os
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import yaml
from PIL import Image

class NoMadBaseAgent():
    def __init__(
        self,
        image_intrinsic: torch.Tensor,
        model_path: str,
        model_config_path: Optional[str] = None,
        robot_config_path: Optional[str] = None,
        device="cuda:0",
    ):
        self.image_intrinsic = image_intrinsic
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.device = device

        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.RATE = robot_config["frame_rate"]
        self.DT = 1 / self.RATE
        self.EPS = 1e-8

        with open(model_config_path, "r") as f:  # type: ignore
            self.cfg = yaml.safe_load(f)
        self.memory_size = self.cfg["context_size"]
        self.image_size = self.cfg["image_size"]
        self.normalize = self.cfg["normalize"]

    def process_image(self, image):
        """Process for image."""
        return Image.fromarray(image)
    
    def reset(self,batch_size):
        self.batch_size = batch_size
        self.memory_queue = [[] for i in range(batch_size)]
        
    def reset_env(self,i):
        self.memory_queue[i] = []

    def project_trajectory(self, image, n_trajectories):
        trajectory_mask = np.array(image)
        for waypoints in n_trajectories:
            if n_trajectories.shape[0] == 1:
                color = (0, 255, 0)
            else:
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
            input_points = np.zeros((waypoints.shape[0], 3)) - 0.15
            input_points[:, 0:2] = waypoints
            camera_x = (
                self.image_intrinsic[0][0] * -input_points[:, 1] / (input_points[:, 0] + 1e-8)
                + self.image_intrinsic[0][2]
            )
            camera_z = (
                image.shape[0]
                - 1
                - self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8)
                - self.image_intrinsic[1][2]
            )

            for i in range(camera_x.shape[0] - 1):
                try:
                    if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i + 1] > 0 and camera_z[i + 1] > 0:
                        trajectory_mask = cv2.line(
                            trajectory_mask,
                            (int(camera_x[i]), int(camera_z[i])),
                            (int(camera_x[i + 1]), int(camera_z[i + 1])),
                            color,
                            5,
                        )
                except (IndexError, ValueError, cv2.error) as e:
                    print(f"An error occurred: {e}")
                    pass
        return trajectory_mask

    def callback_obs(self, imgs):
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size + 1:
                self.memory_queue[i] = [self.process_image(imgs[i])] * (self.memory_size + 1)
            else:
                self.memory_queue[i].pop(0)
                self.memory_queue[i].append(self.process_image(imgs[i]))