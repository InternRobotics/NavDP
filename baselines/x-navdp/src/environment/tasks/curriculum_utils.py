"""Curriculum thresholds and per-environment progress tracking utilities."""

import torch
import numpy as np
from collections import deque
from isaaclab.envs import ManagerBasedEnv
from typing import Any, Optional

_EMBODIMENT_DECIMATION_CFG = {'dingo': 10, 'unitree_g1': 4, 'unitree_go2': 4}
# Curriculum learning configuration
ENABLE_CURRICULUM_LEARNING = False # 课程学习开关，设置为 False 则使用固定阈值

CURRICULUM_STAGES = [
    {"episode_threshold": 350, "success_rate_threshold": 0.8, "distance": 0.5, "velocity": 0.25, "stop_delay": 40, "random_reset_prob": 0.0},
    {"episode_threshold": 300, "success_rate_threshold": 0.75, "distance": 0.5, "velocity": 0.25, "stop_delay": 40, "random_reset_prob": 0.2},
    {"episode_threshold": 250, "success_rate_threshold": 0.75, "distance": 0.5, "velocity": 0.25, "stop_delay": 40, "random_reset_prob": 0.5},
]

CURRICULUM_SUCCESS_WINDOW_SIZE = 200  # 滑动窗口大小，用于计算成功率
CURRICULUM_SUCCESS_EMA_ALPHA = 0.95  # 成功率移动指数平均数的平滑系数

# 模块级课程统计：导入本模块时初始化；环境销毁重建后 _init 会再次同步到 env，数据不丢
GLOBAL_CURRICULUM_SUCCESS_HISTORY: deque[bool] = deque(maxlen=CURRICULUM_SUCCESS_WINDOW_SIZE)
GLOBAL_CURRICULUM_EPISODE_COUNT: int = 0
GLOBAL_CURRICULUM_SUCCESS_EMA: float = 0.0
GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA: float = 0.0
GLOBAL_CURRICULUM_CURRENT_STAGE_INDEX: int = 0
GLOBAL_CURRICULUM_STAGE_START_EPISODE_COUNT: int = 0


def _sync_curriculum_globals_to_env(env) -> None:
    """将模块全局课程状态绑定到 env（供 event_utils / reward 等读取 env._curriculum_*）。"""
    idx = min(max(0, GLOBAL_CURRICULUM_CURRENT_STAGE_INDEX), len(CURRICULUM_STAGES) - 1)
    env._curriculum_success_history = GLOBAL_CURRICULUM_SUCCESS_HISTORY
    env._curriculum_episode_count = GLOBAL_CURRICULUM_EPISODE_COUNT
    env._curriculum_success_ema = GLOBAL_CURRICULUM_SUCCESS_EMA
    env._curriculum_stage_success_ema = GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA
    env._curriculum_current_stage_index = idx
    env._curriculum_stage_start_episode_count = GLOBAL_CURRICULUM_STAGE_START_EPISODE_COUNT
    env._curriculum_current_stage = CURRICULUM_STAGES[idx]


# 固定阈值（当课程学习关闭时使用）
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_VELOCITY_THRESHOLD = 0.25
DEFAULT_STOP_DELAY = 40


def scaled_arrival_stop_delay_steps(env) -> int:
    """
    到达所需停留步数（sim 步）：stop_delay 以「控制周期」计，乘以 (10 / decimation) 与 embodiment 控制频率对齐。
    与 terminal_utils.arrival_terminal_check、_init_curriculum_attributes 中 _arrival_stop_delay 一致。
    """
    dec = getattr(env, "_decimation", 10)
    enable_curriculum = getattr(env, "_enable_curriculum_learning", ENABLE_CURRICULUM_LEARNING)
    if enable_curriculum and hasattr(env, "_curriculum_current_stage"):
        return int(env._curriculum_current_stage["stop_delay"] * (10 / dec))
    return int(DEFAULT_STOP_DELAY * (10 / dec))

DEFAULT_RANDOM_RESET_PROB = 0.5

def _get_current_stage(current_stage_index, stage_start_episode_count, total_episode_count, success_rate, stages):
    """根据当前阶段持续的episode数量和成功率返回应该切换到的课程学习阶段
    episode_threshold 是每个阶段至少要学习的episode数目，需要乘以环境数
    """
    # 计算当前阶段持续的episode数目
    stage_episode_count = total_episode_count - stage_start_episode_count

    # 获取当前阶段（current_stage_index 保证在有效范围内）
    current_stage = stages[current_stage_index]

    # 检查是否可以切换到下一个阶段
    # 需要满足：1) 当前阶段持续的episode数目 >= 阈值 2) 成功率满足要求 3) 不是最后一个阶段
    threshold = current_stage["episode_threshold"]
    next_stage_index = current_stage_index + 1
    if (stage_episode_count >= threshold and
        success_rate >= current_stage["success_rate_threshold"] and
        next_stage_index < len(stages)):
        # 可以切换到下一个阶段
        return stages[next_stage_index], next_stage_index

    # 保持当前阶段
    return current_stage, current_stage_index

def _get_random_reset_prob(env):
    """获取随机reset概率（课程学习阶段或默认值）"""
    enable_curriculum = getattr(env, '_enable_curriculum_learning', ENABLE_CURRICULUM_LEARNING)
    if enable_curriculum and hasattr(env, '_curriculum_current_stage'):
        return env._curriculum_current_stage.get("random_reset_prob", DEFAULT_RANDOM_RESET_PROB)
    return DEFAULT_RANDOM_RESET_PROB

def _reset_arrival_timer(env, env_ids):
    """重置到达计时器相关属性"""
    if hasattr(env, '_arrival_stop_timer'):
        env._arrival_stop_timer[env_ids] = 0
    if hasattr(env, '_arrival_timer_started'):
        env._arrival_timer_started[env_ids] = False
    for env_id in env_ids:
        env._has_arrived_goal[env_id] = False

def _reset_reward_related_state(env, env_ids, device):
    """重置reward/terminal相关的状态（stuck检测、距离、接触力等）"""
    if hasattr(env, '_recent_positions'):
        for idx in env_ids:
            env._recent_positions[idx].clear()
    if not hasattr(env, '_is_stuck'):
        env._is_stuck = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._is_stuck[env_ids] = False
    if hasattr(env, '_stuck_consecutive_count'):
        env._stuck_consecutive_count[env_ids] = 0
    if hasattr(env, '_recent_positions_long'):
        for idx in env_ids:
            env._recent_positions_long[idx].clear()
    if not hasattr(env, '_frame_counter_long'):
        env._frame_counter_long = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
    if not hasattr(env, '_is_stuck_long'):
        env._is_stuck_long = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._frame_counter_long[env_ids] = 0
    env._is_stuck_long[env_ids] = False
    if hasattr(env, '_prev_distance_to_goal'):
        env._prev_distance_to_goal[env_ids] = -1
    if hasattr(env, '_prev_euclidean_distance_to_goal'):
        env._prev_euclidean_distance_to_goal[env_ids] = -1
    if hasattr(env, '_prev_contact_force_mag'):
        env._prev_contact_force_mag[env_ids] = -1
    if hasattr(env, '_humanoid_fell_latched'):
        env._humanoid_fell_latched[env_ids] = False
    if hasattr(env, '_humanoid_fall_terminate_timer'):
        env._humanoid_fall_terminate_timer[env_ids] = 0

def _init_curriculum_attributes(env, embodiment: Optional[str] = None):
    """初始化课程学习相关的环境属性"""
    if embodiment is None:
        embodiment = str(getattr(env, "_embodiment", "dingo"))

    # 检查是否启用课程学习
    enable_curriculum = getattr(env, '_enable_curriculum_learning', ENABLE_CURRICULUM_LEARNING)

    # 如果启用课程学习，获取阶段1的配置用于初始化默认值
    stage1_config = CURRICULUM_STAGES[0] if enable_curriculum else None
    if not hasattr(env, '_embodiment'):
        env._embodiment = embodiment
        env._decimation = _EMBODIMENT_DECIMATION_CFG[str(embodiment)]

    _sync_curriculum_globals_to_env(env)

    if not hasattr(env, '_arrival_stop_timer'):
        env._arrival_stop_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.scene.device)
    if not hasattr(env, '_arrival_stop_delay'):
        # 如果启用课程学习，使用当前全局阶段对应的 stop_delay；否则使用DEFAULT_STOP_DELAY
        if enable_curriculum and stage1_config is not None:
            default_stop_delay = env._curriculum_current_stage["stop_delay"]
        else:
            default_stop_delay = DEFAULT_STOP_DELAY
        env._arrival_stop_delay = torch.full(
            (env.num_envs,),
            default_stop_delay * (10 / env._decimation),
            dtype=torch.int32,
            device=env.scene.device
        )
    if not hasattr(env, '_arrival_timer_started'):
        env._arrival_timer_started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.scene.device)  # 标志计时器是否已开始

    if not hasattr(env, '_episode_metadata'):
        env._episode_metadata = [deque[Any](maxlen=2) for _ in range(env.num_envs)]
    if not hasattr(env, '_episodes_num_list'):
        env._episodes_num_list = np.zeros(env.num_envs, dtype=np.int32)
    if not hasattr(env, '_eval_interval'):
        env._eval_interval = 3
    if not hasattr(env, '_is_eval'):
        env._is_eval = np.ones(env.num_envs, dtype=bool)
    if not hasattr(env, '_has_arrived_goal'):
        env._has_arrived_goal = np.zeros(env.num_envs, dtype=bool)  # 是否到达目标的标志
    if not hasattr(env, '_is_last_success'):
        env._is_last_success = np.zeros(env.num_envs, dtype=bool)


def update_curriculum_success(env, success, episodes_num, record_success=True):
    """更新课程学习的成功状态并更新统计信息（合并了原来的两个函数）
    返回: (总成功率, 阶段成功率)

    record_success: 为 False 时仍更新 GLOBAL_CURRICULUM_EPISODE_COUNT 与阶段/同步逻辑，
    但不写入成功历史、不更新成功率 EMA（用于仅训练回合不计入成功率统计）。
    """
    global GLOBAL_CURRICULUM_SUCCESS_HISTORY, GLOBAL_CURRICULUM_EPISODE_COUNT, GLOBAL_CURRICULUM_SUCCESS_EMA
    global GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA, GLOBAL_CURRICULUM_CURRENT_STAGE_INDEX
    global GLOBAL_CURRICULUM_STAGE_START_EPISODE_COUNT

    # 检查是否启用课程学习
    enable_curriculum = getattr(env, '_enable_curriculum_learning', ENABLE_CURRICULUM_LEARNING)
    alpha = CURRICULUM_SUCCESS_EMA_ALPHA

    GLOBAL_CURRICULUM_EPISODE_COUNT = episodes_num

    if record_success:
        GLOBAL_CURRICULUM_SUCCESS_HISTORY.append(success)
        GLOBAL_CURRICULUM_SUCCESS_EMA = alpha * GLOBAL_CURRICULUM_SUCCESS_EMA + (1 - alpha) * float(success)
    total_success_rate = GLOBAL_CURRICULUM_SUCCESS_EMA

    # 如果不开启课程学习，阶段成功率直接等于总成功率
    if not enable_curriculum:
        _sync_curriculum_globals_to_env(env)
        return total_success_rate, total_success_rate

    old_stage_index = GLOBAL_CURRICULUM_CURRENT_STAGE_INDEX
    if record_success:
        GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA = alpha * GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA + (1 - alpha) * float(success)
    current_stage_success_rate = GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA

    # 获取应该切换到的阶段（基于当前阶段持续的episode数目和当前阶段的成功率）
    new_stage, new_stage_index = _get_current_stage(
        old_stage_index,
        GLOBAL_CURRICULUM_STAGE_START_EPISODE_COUNT,
        GLOBAL_CURRICULUM_EPISODE_COUNT,
        current_stage_success_rate,  # 使用当前阶段的成功率
        CURRICULUM_STAGES,
    )

    stage_changed = (new_stage_index != old_stage_index)

    # 如果阶段切换，更新阶段开始的episode计数并设置标志（不清空历史记录，因为需要保留全局成功率）
    if stage_changed:
        GLOBAL_CURRICULUM_STAGE_START_EPISODE_COUNT = GLOBAL_CURRICULUM_EPISODE_COUNT
        GLOBAL_CURRICULUM_CURRENT_STAGE_INDEX = new_stage_index
        GLOBAL_CURRICULUM_STAGE_SUCCESS_EMA = 0.0

    _sync_curriculum_globals_to_env(env)

    # 为所有环境设置相同的延迟时间（按 embodiment 的 decimation 缩放到 sim 步）
    delay_steps = int(env._curriculum_current_stage["stop_delay"] * (10 / env._decimation))
    for env_idx in range(env.num_envs):
        env._arrival_stop_delay[env_idx] = delay_steps

    return total_success_rate, current_stage_success_rate
