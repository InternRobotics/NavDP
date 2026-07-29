# Copyright (c) 2025 X-NavDP contributors
# Quadruped Velocity Tracking Controller using trained IsaacLab checkpoint
"""Go2 quadruped velocity controller backed by an exported IsaacLab policy."""

import os

import numpy as np
import torch
from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController


def _default_go2_policy_path() -> str:
    """Return the first supported Go2 policy checkpoint path that exists."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    candidates = [
        os.path.join(checkpoint_dir, "quadruped_go2", "policy.pt"),
        os.path.join(checkpoint_dir, "go2", "policy.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


class Go2VelocityController(BaseController):
    """
    Go2 quadruped velocity tracking controller using trained IsaacLab / RSL-RL policy.

    This controller follows the same usage pattern as `G1VelocityController`:
    it receives a batch of high-level commands `[vx, wz]`, inserts them into the
    policy observation, and outputs low-level joint actions for the 12 Go2 joints.

    The policy expects observations in the following format:
    - Base linear velocity / angular velocity / projected gravity (9)
    - Velocity commands (3: lin_vel_x, lin_vel_y, ang_vel_z)
    - Joint positions (12)
    - Joint velocities (12)
    - Previous actions (12)

    This project may provide either:
    - an exported TorchScript policy, or
    - a raw RSL-RL / IsaacLab checkpoint containing actor weights.
    """

    def __init__(
        self,
        name: str,
        policy_path: str = _default_go2_policy_path(),
        device: str = 'cuda',
        obs_dim: int = 48,
        num_joints: int = 12,
        control_dt: float = 0.02,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        action_scale: float = 0.25,
        clip_actions: float = 100.0,
        **kwargs,
    ) -> None:
        """Load the exported Go2 tracking policy and controller limits."""
        super().__init__(name)
        self.device = device
        self.obs_dim = obs_dim
        self.num_joints = num_joints
        self.control_dt = control_dt
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.action_scale = action_scale
        self.clip_actions = clip_actions

        print(f"[Go2VelocityController] Loading policy from: {policy_path}")
        assert os.path.exists(policy_path), f"Policy path {policy_path} does not exist!"
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        print(f"[Go2VelocityController] Loaded exported policy (TorchScript)")
        print(f"[Go2VelocityController] Initialized successfully")
        print(f"  Device: {device}")
        print(f"  Obs dim: {obs_dim}")
        print(f"  Num joints: {num_joints}")
        print(f"  Action scale: {action_scale}")

    def forward(self, command: np.ndarray) -> ArticulationAction:
        """Reject single-env control because Go2 policy inference is batched."""
        raise NotImplementedError(
            "Go2VelocityController requires batch processing. Use forward_batch() instead."
        )

    def forward_batch(self, obs, commands) -> torch.Tensor:
        """Inject velocity commands into policy observations and return joint actions."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs = obs.to(self.device)

        if isinstance(commands, np.ndarray):
            commands = torch.from_numpy(commands).float().to(self.device)
        else:
            commands = commands.to(self.device)

        commands = torch.clamp(
            commands,
            min=torch.tensor([-self.max_linear_speed, -self.max_angular_speed], device=self.device),
            max=torch.tensor([self.max_linear_speed, self.max_angular_speed], device=self.device),
        )

        batch_size = commands.shape[0]
        vel_commands = torch.zeros((batch_size, 3), device=self.device)
        vel_commands[:, 0] = commands[:, 0]
        vel_commands[:, 2] = commands[:, 1]

        if obs.shape[1] == self.obs_dim - 3:
            policy_obs = torch.cat([obs[:, :9], vel_commands, obs[:, 9:]], dim=1)
        elif obs.shape[1] == self.obs_dim:
            policy_obs = obs.clone()
            policy_obs[:, 9:12] = vel_commands
        else:
            raise ValueError(
                f"Unexpected Go2 observation dimension: {obs.shape[1]}. Expected {self.obs_dim - 3} or {self.obs_dim}."
            )

        with torch.no_grad():
            actions = self.policy(policy_obs)

        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        return actions
