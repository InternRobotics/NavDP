"""Differential-drive velocity controller for the Dingo base."""

import numpy as np
import torch
from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController


class DifferentialController(BaseController):
    """Convert base linear/angular velocity commands into wheel speeds."""

    def __init__(
        self,
        name: str,
        wheel_radius: float,
        wheel_base: float,
        max_linear_speed: float = float("inf"),
        max_angular_speed: float = float("inf"),
        max_wheel_speed: float = float("inf"),
    ) -> None:
        super().__init__(name)
        if wheel_radius <= 0 or wheel_base <= 0:
            raise ValueError("wheel_radius and wheel_base must be positive")
        if min(max_linear_speed, max_angular_speed, max_wheel_speed) < 0:
            raise ValueError("controller speed limits must be non-negative")

        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.max_wheel_speed = float(max_wheel_speed)

    def _wheel_speeds(self, commands: np.ndarray) -> np.ndarray:
        commands = np.asarray(commands, dtype=np.float32)
        if commands.shape[-1] != 2:
            raise ValueError(f"commands must have shape (..., 2), got {commands.shape}")

        clipped = np.clip(
            commands,
            [-self.max_linear_speed, -self.max_angular_speed],
            [self.max_linear_speed, self.max_angular_speed],
        )
        linear_velocity = clipped[..., 0]
        angular_velocity = clipped[..., 1]
        left = (linear_velocity - 0.5 * angular_velocity * self.wheel_base) / self.wheel_radius
        right = (linear_velocity + 0.5 * angular_velocity * self.wheel_base) / self.wheel_radius
        wheel_speeds = np.stack((left, right), axis=-1)
        return np.clip(wheel_speeds, -self.max_wheel_speed, self.max_wheel_speed)

    def forward(self, command: np.ndarray) -> ArticulationAction:
        """Convert one `[linear, angular]` command into wheel targets."""
        wheel_speeds = self._wheel_speeds(command)
        if wheel_speeds.ndim != 1:
            raise ValueError(f"forward expects one command, got shape {wheel_speeds.shape}")
        return ArticulationAction(joint_velocities=wheel_speeds.tolist())

    def forward_batch(self, obs: np.ndarray, commands: np.ndarray) -> torch.Tensor:
        """Convert a batch of `[linear, angular]` commands into wheel targets."""
        del obs
        joint_velocities = np.zeros((commands.shape[0], 2))
        joint_velocities[:, 0] = (
            (2 * commands[:, 0]) - (commands[:, 1] * self.wheel_base)
        ) / (2 * self.wheel_radius)
        joint_velocities[:, 1] = (
            (2 * commands[:, 0]) + (commands[:, 1] * self.wheel_base)
        ) / (2 * self.wheel_radius)
        return torch.tensor(joint_velocities, dtype=torch.float32)
