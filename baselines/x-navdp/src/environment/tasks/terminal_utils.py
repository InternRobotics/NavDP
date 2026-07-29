"""Termination terms for point-goal navigation episodes."""

import torch
import numpy as np
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from .curriculum_utils import (
    ENABLE_CURRICULUM_LEARNING,
    DEFAULT_STOP_DELAY,
    scaled_arrival_stop_delay_steps,
)


def stuck_terminal_check_duration(
    env: ManagerBasedEnv,
    stuck_duration_s: float = 12.0,
) -> torch.Tensor:
    """``stuck_reward_long_window`` 置位 ``_is_stuck_long`` 后，连续达 ``stuck_duration_s`` 则终止。"""
    device = env._arrival_timer_started.device
    decimation = int(getattr(env.cfg, "decimation", None) or getattr(env, "_decimation", 10))
    dt = float(getattr(env.cfg.sim, "dt", 0.01))
    step_duration = decimation * dt
    max_steps = max(1, int(np.ceil(stuck_duration_s / step_duration)))

    is_stuck = (
        env._is_stuck_long
        if hasattr(env, "_is_stuck_long")
        else torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    )
    if not hasattr(env, "_stuck_consecutive_count"):
        env._stuck_consecutive_count = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
    env._stuck_consecutive_count = torch.where(
        is_stuck,
        env._stuck_consecutive_count + 1,
        torch.zeros_like(env._stuck_consecutive_count),
    )
    is_stuck_long = env._stuck_consecutive_count >= max_steps
    return is_stuck_long & (~env._arrival_timer_started)


def arrival_terminal_check(env: ManagerBasedEnv,
                           robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Terminate when the arrival stop timer reaches the configured delay."""

    enable_curriculum = getattr(env, '_enable_curriculum_learning', ENABLE_CURRICULUM_LEARNING)
    if enable_curriculum:
        arrival_result = env._arrival_stop_timer >= env._arrival_stop_delay
    else:
        default_stop_delay = int(DEFAULT_STOP_DELAY * (10 / env._decimation))
        arrival_result = env._arrival_stop_timer >= default_stop_delay
    if hasattr(env, '_has_arrived_goal'):
        arrival_mask = arrival_result.cpu().numpy()
        env._has_arrived_goal[arrival_mask] = True
    return arrival_result

def arrival_terminal_check_eval(env: ManagerBasedEnv):
    """Evaluation arrival termination using embodiment-scaled delay steps."""

    delay_steps = scaled_arrival_stop_delay_steps(env)
    arrival_result = env._arrival_stop_timer >= delay_steps
    return arrival_result


def humanoid_low_height_fall_terminal(
    env: ManagerBasedEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_duration_s: float = 4.0,
) -> torch.Tensor:
    """Terminate G1 after an externally latched fall persists for ``fall_duration_s``."""
    robot_asset = env.scene[robot_asset_cfg.name]
    device = robot_asset.data.root_pos_w.device
    if getattr(env, "_embodiment", None) != "unitree_g1":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    decimation = int(getattr(env.cfg, "decimation", None) or getattr(env, "_decimation", 4))
    dt = float(getattr(env.cfg.sim, "dt", 0.01))
    step_duration = decimation * dt
    max_steps = max(1, int(np.ceil(fall_duration_s / step_duration)))

    if not hasattr(env, "_humanoid_fell_latched"):
        env._humanoid_fell_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_humanoid_fall_terminate_timer"):
        env._humanoid_fall_terminate_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=device)

    env._humanoid_fall_terminate_timer = torch.where(
        env._humanoid_fell_latched,
        env._humanoid_fall_terminate_timer + 1,
        torch.zeros_like(env._humanoid_fall_terminate_timer),
    )
    return env._humanoid_fall_terminate_timer >= max_steps


@configclass
class NavigationOffTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(
        func=stuck_terminal_check_duration,
        params={"stuck_duration_s": 12.0},
    )
    humanoid_low_height_fall = DoneTerm(
        func=humanoid_low_height_fall_terminal,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "fall_duration_s": 4.0,
        },
    )

@configclass
class EvalTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check_eval)
