# Copyright (c) 2025 X-NavDP contributors
# Humanoid Velocity Tracking Controller using trained IsaacLab checkpoint
"""G1 humanoid velocity controller backed by an exported IsaacLab policy."""

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController


class G1VelocityController(BaseController):
    """
    G1 humanoid velocity tracking controller using trained RSL-RL policy.

    This controller uses a pretrained policy from IsaacLab's Isaac-Velocity-Flat-G1-v0 task
    to convert high-level velocity commands into low-level joint position targets.

    The policy expects observations in the following format:
    - Base linear velocity (3)
    - Base angular velocity (3)
    - Projected gravity (3)
    - Velocity commands (3: lin_vel_x, lin_vel_y, ang_vel_z)
    - Joint positions (num_joints, default: 37 for G1)
    - Joint velocities (num_joints, default: 37 for G1)
    - Previous actions (num_joints, default: 37 for G1)

    Args:
        name (str): Controller name
        policy_path (str): Path to trained policy checkpoint (.pt file)
        env: Environment object (for extracting robot states)
        device (str): Device to run policy on ('cuda' or 'cpu')
        num_joints (int): Number of controlled joints (default: 23 for G1)
        control_dt (float): Control timestep in seconds
        max_linear_speed (float): Maximum linear velocity (m/s)
        max_angular_speed (float): Maximum angular velocity (rad/s)
        action_scale (float): Action scaling factor
        clip_actions (float): Action clipping range
    """
    def __init__(
        self,
        name: str,
        policy_path: str = os.path.join(
            os.path.dirname(__file__), "checkpoints", "humanoid_g1", "policy.pt"
        ),
        device: str = 'cuda',
        num_joints: int = 37,
        control_dt: float = 0.02,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        action_scale: float = 0.5,
        clip_actions: float = 100.0,
        **kwargs,
    ) -> None:
        """Load the exported G1 tracking policy and controller limits."""
        super().__init__(name)
        self.device = device
        self.num_joints = num_joints
        self.control_dt = control_dt
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.action_scale = action_scale
        self.clip_actions = clip_actions
        self.debug_mode = True
        # Load policy checkpoint
        print(f"[G1VelocityController] Loading policy from: {policy_path}")
        assert os.path.exists(policy_path), f"Policy path {policy_path} does not exist!"
        # Load the exported policy (TorchScript format)
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        print(f"[G1VelocityController] Loaded exported policy (TorchScript)")
        print(f"[G1VelocityController] Initialized successfully")
        print(f"  Device: {device}")
        print(f"  Num joints: {num_joints}")
        print(f"  Action scale: {action_scale}")

    def forward(self, command: np.ndarray) -> ArticulationAction:
        """
        Convert velocity command to joint position targets (single environment).
        Args:
            command: [linear_velocity, angular_velocity] in m/s and rad/s
        Returns:
            ArticulationAction with joint position targets
        """
        raise NotImplementedError(
            "G1VelocityController requires batch processing. Use forward_batch() instead."
        )

    def forward_batch(
        self,
        obs,
        commands
    ) -> torch.Tensor:
        """
        Convert batch of velocity commands to joint position targets.
        Args:
            commands: (batch_size, 2) array of [linear_velocity, angular_velocity]
            robot_states: Optional dict with robot states. If None, will extract from self.env
        Returns:
            (batch_size, num_joints) tensor of joint position targets
        """
        if isinstance(commands, np.ndarray):
            commands = torch.from_numpy(commands).float().to(self.device)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        batch_size = commands.shape[0]
        # Clip commands to limits

        commands = torch.clamp(
            commands,
            min=torch.tensor([-self.max_linear_speed, -self.max_angular_speed], device=self.device),
            max=torch.tensor([self.max_linear_speed, self.max_angular_speed], device=self.device),
        )
        vel_commands = torch.zeros((batch_size, 3), device=self.device)
        vel_commands[:, 0] = commands[:, 0]  # forward velocity
        vel_commands[:, 2] = commands[:, 1]  # yaw rate

        obs = torch.cat([obs[:, :9], vel_commands, obs[:, 9:]], dim=1)
        with torch.no_grad():
            actions = self.policy(obs)
        # Clip actions
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        return actions
