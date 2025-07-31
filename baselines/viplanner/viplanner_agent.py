from typing import Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as transforms
from configs.learning_config import TrainCfg
from autoencoder import AutoEncoder,DualAutoEncoder
from m2f_inference import Mask2FormerInference
import traj_opt

class VIPlannerAgent():
    def __init__(
        self,
        image_intrinsic: torch.Tensor,
        m2f_path: str,
        m2f_config_path: str,
        model_path: str,
        model_config_path: str,
        device="cuda:0",
    ):
        self.image_intrinsic = image_intrinsic
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.device = device
        self.traj_generate = traj_opt.TrajOpt()
        self.m2f_inference = Mask2FormerInference(
            config_file=m2f_config_path,
            checkpoint_file=m2f_path,
        )
        self.train_config: TrainCfg = None
        self.load_model(self.model_path, self.model_config_path)
        self.transform = transforms.Resize(self.img_input_size, antialias=None)  # type: ignore
        self.traj_generate = traj_opt.TrajOpt()

    def load_model(self, model_path: str, model_config_path: str):
        self.train_config: TrainCfg = TrainCfg.from_yaml(model_config_path)
        self.img_input_size = self.train_config.img_input_size
        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth
        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)
        try:
            model_state_dict, _ = torch.load(model_path, weights_only=True)
        except ValueError:
            model_state_dict = torch.load(model_path, weights_only=True)
        self.net.load_state_dict(model_state_dict, strict=True)
        self.net.eval()
        self.net.to(self.device)

    def process_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = self.transform(depth).expand(1, 3, -1, -1)
        depth[depth > self.max_depth] = 0.0
        depth[~torch.isfinite(depth)] = 0  # set all inf or nan values to 0
        return depth

    def plan(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        # transform input
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.process_depth(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear

    def step_pointgoal(
        self,
        image: torch.Tensor,
        dep_image: torch.Tensor,
        goal_robot_frame: torch.Tensor,
    ):
        with torch.no_grad():
            tensor_dep_image = torch.as_tensor(dep_image[:,:,:,0], device=self.device, dtype=torch.float32)
            tensor_goal_robot_frame = torch.as_tensor(goal_robot_frame[:,0:3], device=self.device, dtype=torch.float32)
            semantic_image = np.array([self.m2f_inference.predict(img) for img in image])
            sem_image = torch.tensor(semantic_image).permute(0, 3, 1, 2).float().to(self.device)
            keypoints, traj, fear = self.plan(tensor_dep_image, sem_image, tensor_goal_robot_frame)
            #print(keypoints.shape,traj.shape)
            return keypoints, traj, fear

    