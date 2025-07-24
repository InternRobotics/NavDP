# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import yaml

class Loader(yaml.SafeLoader):
    pass

def construct_datacfg(loader, node):
    add_dicts = {}
    for node_entry in node.value:
        if isinstance(node_entry[1], yaml.MappingNode):
            add_dicts[node_entry[0].value] = loader.construct_mapping(node_entry[1])
            node.value.remove(node_entry)

    return DataCfg(**loader.construct_mapping(node), **add_dicts)

Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.learning_cfg.DataCfg",
    construct_datacfg,
)

@dataclass
class DataCfg:
    """Config for data loading"""
    # real world data used --> images have to be rotated by 180 degrees
    real_world_data: bool = False
    # from carla dataset (exclude certain spaces)
    carla: bool = False

    # identification suffix of the cameras for semantic and depth images
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"

    # data processing
    max_depth: float = 15.0
    "maximum depth for depth image"

    # odom (=start) point selection
    max_goal_distance: float = 15.0
    min_goal_distance: float = 0.5
    "maximum and minimum distance between odom and goal"
    distance_scheme: dict = field(default_factory=lambda: {1: 0.2, 3: 0.35, 5: 0.25, 7.5: 0.15, 10: 0.05})
    # select goal points for the samples according to the scheme:
    # {distance: percentage of goals}, distances have to be increasing
    # and max distance has to be equal to max_goal_distance
    obs_cost_height: float = 1.5
    "all odom points with cost of more than obs_cost_height are discarded (negative cost of cost_map will be automatically added)"
    fov_scale: float = 1.0
    "scaling of the field of view (only goals within fov are considered)"
    depth_scale: float = 1000.0
    "scaling of the depth image"

    # train val split
    ratio: float = 0.9
    "ratio between train and val dataset"
    max_train_pairs: Optional[int] = None
    pairs_per_image: int = 4
    "maximum number of train pairs (can be used to limit training time) can be set, otherwise number of recorded images times pairs_per_image is used"
    ratio_fov_samples: float = 1.0
    ratio_front_samples: float = 0.0
    ratio_back_samples: float = 0.0
    "samples distribution -> either within the robots fov, in front of the robot but outside the fov or behind the robot"

    # edge blur (real world RealSense difficulties along edges)  --> will be also visible in rgb/sem images due to warp
    noise_edges: bool = False  # not activate for CARLA yet
    edge_threshold: int = 100
    extend_kernel_size: Tuple[int, int] = field(default_factory=lambda: [5, 5])

    # noise augmentation --> will be applied to a scaled image with range between [0, 1]
    depth_salt_pepper: Optional[float] = None  # Proportion of image pixels to replace with noise on range [0, 1]
    depth_gaussian: Optional[float] = None  # Standard deviation of the noise to add (no clipping applied)
    depth_random_polygons_nb: Optional[int] = None  # Number of random polygons to add
    depth_random_polygon_size: int = 10  # Size of the random polygons in pixels

    sem_rgb_pepper: Optional[float] = None  # Proportion of pixels to randomly set to 0
    sem_rgb_black_img: Optional[float] = None  # Randomly set this proportion of images to complete black images  -->
    sem_rgb_random_polygons_nb: Optional[int] = None  # Number of random polygons to add
    sem_rgb_random_polygon_size: int = 20  # Size of the random polygons in pixels

@dataclass
class TrainCfg:
    """Config for multi environment training"""

    # high level configurations
    sem: bool = True
    rgb: bool = False
    "use semantic/ rgb image"
    file_name: Optional[str] = None
    "appendix to the model filename if needed"
    seed: int = 0
    "random seed"
    gpu_id: int = 0
    "GPU id"
    file_path: str = "${USER_PATH_TO_MODEL_DATA}"
    "file path to models and data directory, can be overwritten by environment variable EXPERIMENT_DIRECTORY (e.g. for cluster)"
    # NOTE: since the environment variable is intended for cluster usage, some visualizations will be automatically switched off

    # data and dataloader configurations
    cost_map_name: str = "cost_map_sem"  # "cost_map_sem"
    "cost map name"
    env_list: List[str] = field(
        default_factory=lambda: [
            "2azQ1b91cZZ",
            "JeFG25nYj2p",
            "Vvot9Ly1tCj",
            "ur6pFq6Qu1A",
            "B6ByNegPMKs",
            "8WUmhLawc2A",
            "E9uDoFAP3SH",
            "QUCTc6BB5sX",
            "YFuZgdQ5vWj",
            "2n8kARJN3HM",
        ]
    )
    test_env_id: int = 9
    "the test env id in the id list"
    data_cfg: Union[DataCfg, List[DataCfg]] = DataCfg()
    "further data configuration (can be individualized for every environment)"
    multi_epoch_dataloader: bool = False
    "load all samples into RAM s.t. do not have to be reloaded for each epoch"
    num_workers: int = 4
    "number of workers for dataloader"
    load_in_ram: bool = False
    "if true, all samples will be loaded into RAM s.t. do not have to be reloaded for each epoch"

    # loss configurations
    fear_ahead_dist: float = 2.5
    "fear lookahead distance"
    w_obs: float = 0.25
    w_height: float = 1.0
    w_motion: float = 1.5
    w_goal: float = 4.0
    "weights for the loss components"
    obstacle_thread: float = 1.2
    "obstacle threshold to decide if fear path or not (neg reward for semantic cost-maps is added automatically)"

    # network configurations
    img_input_size: Tuple[int, int] = field(default_factory=lambda: [360, 640])
    "image size (will be cropped if larger or resized if smaller)"
    in_channel: int = 16
    "goal input channel numbers"
    knodes: int = 5
    "number of max waypoints predicted"
    pre_train_sem: bool = True
    pre_train_cfg: Optional[str] = "m2f_model/coco/panoptic/maskformer2_R50_bs16_50ep.yaml"
    pre_train_weights: Optional[str] = "m2f_model/coco/panoptic/model_final_94dc52.pkl"
    pre_train_freeze: bool = True
    "loading of a pre-trained rgb encoder from mask2former (possible is ResNet 50 or 101)"
    # NOTE: `pre_train_cfg` and `pre_train_weights` are assumed to be found under `file_path/models` (see above)
    decoder_small: bool = False
    "small decoder with less parameters"

    # training configurations
    resume: bool = False
    "resume training"
    epochs: int = 100
    "number of training epochs"
    batch_size: int = 64
    "number of minibatch size"
    hierarchical: bool = False
    hierarchical_step: int = 50
    hierarchical_front_step_ratio: float = 0.02
    hierarchical_back_step_ratio: float = 0.01
    "hierarchical training with an adjusted data structure"

    # optimizer and scheduler configurations
    lr: float = 2e-3
    "learning rate"
    factor: float = 0.5
    "ReduceLROnPlateau factor"
    min_lr: float = 1e-5
    "minimum lr for ReduceLROnPlateau"
    patience: int = 3
    "patience of epochs for ReduceLROnPlateau"
    optimizer: str = "sgd"  # either adam or sgd
    "optimizer"
    momentum: float = 0.1
    "momentum of the optimizer"
    w_decay: float = 1e-4
    "weight decay of the optimizer"

    # visualization configurations
    camera_tilt: float = 0.15
    "camera tilt angle for visualization only"
    n_visualize: int = 15
    "number of trajectories that are visualized"

    # logging configurations
    wb_project: str = "Matterport"
    wb_entity: str = "viplanner"
    wb_api_key: str = "enter_your_key_here"

    # functions
    def get_model_save(self, epoch: Optional[int] = None):
        input_domain = "DepSem" if self.sem else "Dep"
        cost_name = "Geom" if self.cost_map_name == "cost_map_geom" else "Sem"
        optim = "SGD" if self.optimizer == "sgd" else "Adam"
        name = f"_{self.file_name}" if self.file_name is not None else ""
        epoch = epoch if epoch is not None else self.epochs
        hierarch = "_hierarch" if self.hierarchical else ""
        return f"plannernet_env{self.env_list[0]}_ep{epoch}_input{input_domain}_cost{cost_name}_optim{optim}{hierarch}{name}"

    @property
    def all_model_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "models")

    @property
    def curr_model_dir(self):
        return os.path.join(self.all_model_dir, self.get_model_save())

    @property
    def data_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "data")

    @property
    def log_dir(self):
        return os.path.join(os.getenv("EXPERIMENT_DIRECTORY", self.file_path), "logs")

    @classmethod
    def from_yaml(cls, yaml_path: str):
        # open yaml file and load config
        with open(yaml_path) as f:
            cfg_dict = yaml.load(f, Loader=Loader)

        return cls(**cfg_dict["config"])


# EoF