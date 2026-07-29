"""Distributed X-NavDP post-training entry point."""

import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

import argparse
import datetime
import torch
import torch.distributed as dist
import numpy as np
import imageio
import random
from collections import deque
from multiprocessing import get_context
from scipy.spatial.transform import Rotation as R
from argparse import Namespace
from gymnasium.spaces import Dict, Box
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from src.x_navdp import XNavDPPolicy, DummyOffPolicyBuffer
from src.x_navdp.trainer import GQRMTrainer
from src.utils import BatchMPCController
from src.training.constants import (
    EMBODIMENT_CON_STEP,
    EMBODIMENT_SIM_TIME,
    EMBODIMENT_TO_IDX,
    MAX_CON_STEP,
)
from src.training.observation import parse_x_navdp_observations
from src.training.scene_assets import load_and_mix_scene_data, scene_output_tag
from src.training.worker import configure_mdl_system_path, run_env_worker, wait_worker_result
from src.utils.config import load_config
from src.utils.video import video_fps_for_embodiment
from src.utils.visualization import (
    get_grid_img_from_metadata,
    visualize_grid_with_path_on_image,
    visualize_mpc_controller,
)


def parse_args():
    """Parse CLI options for distributed X-NavDP training."""
    parser = argparse.ArgumentParser(description="X-NavDP GQRM distributed training, env in subprocess")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "scenes"),
        help="Root dir of the NavDP-style scene dataset, including scene assets, scene_split.json, and navigation_metadata.",
    )
    parser.add_argument(
        "--scene_split_file",
        type=str,
        default=None,
        help="Scene split json. Defaults to <scene_dir>/scene_split.json when present.",
    )
    parser.add_argument("--scene_split", type=str, default="train", choices=("train", "eval", "all"))
    parser.add_argument(
        "--usd_variant",
        type=str,
        default="navigation",
        choices=("navigation", "raw", "interaction", "auto"),
        help="Preferred InternScenes USD file variant to load.",
    )
    parser.add_argument(
        "--uniform_scene_mix",
        action="store_true",
        help="Evenly interleave clutter scenes into the commercial/home scene list to balance per-rank memory load.",
    )
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--config_file", type=str, default=os.path.join(_REPO_ROOT, "config", "x-navdp_config.yaml"))
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Optional policy checkpoint override, useful for continuing a previous run.",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=0,
        help="Initial global environment step for a continued run.",
    )
    parser.add_argument(
        "--scene_scale",
        type=float,
        default=None,
        help="Optional uniform scene-scale override; defaults to the scene/embodiment preset.",
    )
    parser.add_argument("--log_dir", type=str, default="./logs/x-navdp_more/")
    parser.add_argument("--video_dir", type=str, default="./video/x-navdp_more/")
    parser.add_argument("--txt_dir", type=str, default="./txt/x-navdp_more/")
    parser.add_argument("--checkpoint_dir", type=str, default="./ckpt/x-navdp_more/")
    parser.add_argument("--checkpoint_interval", type=int, default=6000)
    parser.add_argument(
        "--embodiment_switch_interval",
        "--scene_switch_interval_base",
        dest="embodiment_switch_interval",
        type=int,
        default=24000,
        help="Steps between embodiment switches; the old scene_switch name is kept as an alias.",
    )
    parser.add_argument(
        "--embodiments",
        type=str,
        default=None,
        help="Comma-separated embodiment override, e.g. dingo or dingo,unitree_g1.",
    )
    parser.add_argument(
        "--worker_timeout",
        type=float,
        default=3600.0,
        help="Seconds to wait for environment worker responses.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop cleanly after this many environment steps (useful for smoke tests).",
    )
    return parser.parse_args()


def init_ddp_process():
    """Initialize one torch.distributed worker and seed all local RNGs."""

    # Keep each rank's CPU thread usage bounded before launching GPU work.
    num_cpu_cores = os.cpu_count()
    num_threads_per_process = num_cpu_cores // int(os.getenv("WORLD_SIZE", 1))
    print(f"num_cpu_cores: {num_cpu_cores}, num_threads_per_process: {num_threads_per_process}")

    # DDP rank information is supplied by torch.distributed.run.
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=7200)
    )

    seed = 1234 * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return local_rank, rank, device


def update_embodiment_switch_state(
    switch_count: int,
    last_switch_step: int,
    step_num: int,
    switch_interval: float,
    rank: int,
) -> tuple[int, int]:
    """Update counters after the active embodiment is replaced."""
    switch_count += 1
    last_switch_step = step_num
    print(f"[Rank {rank}] embodiment switch done, interval -> {switch_interval}")
    return switch_count, last_switch_step

def run():
    """Main distributed off-policy training loop."""
    # Load CLI/config, material search paths, and the ordered mixed scene list.
    args_cli = parse_args()
    configure_mdl_system_path(args_cli.scene_dir)
    training_config = load_config(args_cli.config_file)
    data_root = args_cli.scene_dir
    scene_split_file = args_cli.scene_split_file or os.path.join(data_root, "scene_split.json")
    scene_list = load_and_mix_scene_data(
        data_root,
        scene_split_file=scene_split_file,
        split=args_cli.scene_split,
        usd_variant=args_cli.usd_variant,
        uniform_scene_mix=args_cli.uniform_scene_mix,
    )
    num_scenes = len(scene_list)
    if num_scenes == 0:
        raise ValueError(f"No valid scene directories found in {args_cli.scene_dir}")
    home_count = len([s for s in scene_list if s['scene_type'] == 'home'])
    commercial_count = len([s for s in scene_list if s['scene_type'] == 'commercial'])
    cluttered_easy_count = len([s for s in scene_list if s['scene_type'] == 'cluttered_easy'])
    cluttered_hard_count = len([s for s in scene_list if s['scene_type'] == 'cluttered_hard'])
    print(
        f"Loaded {home_count} home, {commercial_count} commercial, "
        f"{cluttered_easy_count} cluttered_easy, {cluttered_hard_count} cluttered_hard scenes"
    )
    print(f"Total {num_scenes} scenes after cross-mixing")

    # Initialize one DDP rank per process and create rank-local output directories.
    local_rank, rank, device = init_ddp_process()
    print(f"[Rank {rank}] DDP initialized, device={device}")

    # Embodiment switching keeps each rank on its assigned scene and rotates robot type.
    embodiment_switch_interval = args_cli.embodiment_switch_interval
    embodiment_switch_count = 0

    last_switch_step = 0

    # Rank 0 creates one run timestamp and broadcasts it so all ranks share a log directory.
    base_log_dir = args_cli.log_dir
    run_time_list = [datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None]
    dist.broadcast_object_list(run_time_list, src=0)
    current_time = run_time_list[0]
    if not isinstance(current_time, str) or not current_time:
        raise RuntimeError("broadcast_object_list failed to deliver run timestamp from rank 0")
    args_cli.log_dir = os.path.join(base_log_dir, current_time)
    os.makedirs(args_cli.log_dir, exist_ok=True)
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)
    writer_log_dir = os.path.join(args_cli.log_dir, f"rank{rank}")
    os.makedirs(writer_log_dir, exist_ok=True)
    summary = SummaryWriter(log_dir=writer_log_dir)
    os.makedirs(args_cli.video_dir, exist_ok=True)

    # Run Isaac Lab in a spawned subprocess so SimulationApp lifecycle is isolated.
    ctx = get_context('spawn')
    scene_index = rank % num_scenes
    cmd_queue = ctx.Queue()
    result_queue = ctx.Queue()
    if args_cli.embodiments:
        embodiment_list = [item.strip() for item in args_cli.embodiments.split(",") if item.strip()]
        invalid_embodiments = [item for item in embodiment_list if item not in EMBODIMENT_TO_IDX]
        if invalid_embodiments:
            raise ValueError(
                f"Invalid --embodiments values {invalid_embodiments}; "
                f"valid values are {list(EMBODIMENT_TO_IDX)}"
            )
    else:
        embodiment_list = training_config.embodiment if isinstance(getattr(training_config, 'embodiment', None), list) else [getattr(training_config, 'embodiment', 'dingo')]
    # Ranks rotate embodiments; embodiment switches advance the same rotation.
    embodiment = embodiment_list[(rank + embodiment_switch_count) % len(embodiment_list)]
    worker = ctx.Process(
        target=run_env_worker,
        args=(cmd_queue, result_queue, scene_list, scene_index, args_cli.num_envs, args_cli.scene_scale, device, embodiment_switch_count + 1, embodiment),
    )
    worker.start()

    # Initial reset and short zero-action warmup fill camera/history buffers.
    cmd_queue.put(('reset',))
    init_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
    if init_result.get('type') == 'error':
        raise RuntimeError(f"Worker error: {init_result.get('error', 'unknown')}")

    obs_raw_np = init_result['obs_raw']
    metadata = init_result['metadata']
    obs_raw = {
        'obs_rgb': torch.from_numpy(obs_raw_np['obs_rgb']).float().to(device),
        'obs_depth': torch.from_numpy(obs_raw_np['obs_depth']).float().to(device),
        'goal_pose': torch.from_numpy(obs_raw_np['goal_pose']).float().to(device),
        'raw_rgb': torch.from_numpy(obs_raw_np['raw_rgb']).to(device),
    }
    if 'birdeye_rgb' in obs_raw_np:
        obs_raw['birdeye_rgb'] = torch.from_numpy(obs_raw_np['birdeye_rgb']).to(device)
    obs = parse_x_navdp_observations(obs_raw)

    for _ in range(10):
        cmd_queue.put(('step', np.zeros((args_cli.num_envs, 2), dtype=np.float32), obs_raw_np.get('policy')))
        step_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
        if step_result.get('type') == 'error':
            raise RuntimeError(f"Worker error: {step_result.get('error')}")
        obs_raw_np = step_result['obs_raw']
        metadata = step_result['metadata']
        obs_raw = {
            'obs_rgb': torch.from_numpy(obs_raw_np['obs_rgb']).float().to(device),
            'obs_depth': torch.from_numpy(obs_raw_np['obs_depth']).float().to(device),
            'goal_pose': torch.from_numpy(obs_raw_np['goal_pose']).float().to(device),
            'raw_rgb': torch.from_numpy(obs_raw_np['raw_rgb']).to(device),
        }
        if 'birdeye_rgb' in obs_raw_np:
            obs_raw['birdeye_rgb'] = torch.from_numpy(obs_raw_np['birdeye_rgb']).to(device)
        obs = parse_x_navdp_observations(obs_raw)

    # Replay buffer shapes are inferred from the first real observation batch.
    observation_space = Dict({
        'rgb': Box(low=0, high=1.0, shape=obs['rgb'].shape[1:]),
        'depth': Box(low=0, high=5.0, shape=obs['depth'].shape[1:]),
        'pointgoal': Box(low=-np.inf, high=np.inf, shape=obs['pointgoal'].shape[1:])
    })
    action_space = Box(low=-np.inf, high=np.inf, shape=(24, 3))
    memory = DummyOffPolicyBuffer(
        observation_space=observation_space,
        action_space=action_space,
        n_envs=args_cli.num_envs,
        horizon_size=training_config.buffer.horizon_size
    )

    # Trainer hyperparameters are flattened into the trainer Namespace.
    config = {
        'learning_rate': training_config.trainer.learning_rate,
        'end_factor_lr_decay': training_config.trainer.end_factor_lr_decay,
        'total_iters': training_config.trainer.total_iters,
        'depth_mask_prob': getattr(training_config.trainer, 'depth_mask_prob', 0.0),
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'device': device,
        'delay_update': 2,
    }
    config = Namespace(**config)

    # Build the high-level diffusion policy used to sample navigation trajectories.
    policy = XNavDPPolicy(
        image_size=training_config.policy.image_size,
        memory_size=training_config.policy.memory_size,
        predict_size=training_config.policy.predict_size,
        temporal_depth=training_config.policy.temporal_depth,
        ft_denoise_step=training_config.policy.ft_denoise_step,
        distinguish_embodiment=training_config.policy.distinguish_embodiment,
        device=device
    )

    if rank == 0:
        # Load pretrained NavDP weights once, then initialize RL-specific heads from base heads.
        ckpt_path = args_cli.pretrained_model or getattr(training_config.paths, 'pretrained_model', None)
        if not ckpt_path:
            raise ValueError("Set paths.pretrained_model in config or provide a checkpoint path.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Pretrained policy checkpoint not found: {ckpt_path}. "
                "Set paths.pretrained_model in config/x-navdp_config.yaml."
            )
        policy.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        policy.decoder_ft.load_state_dict(policy.decoder.state_dict())
        policy.layernorm_ft.load_state_dict(policy.layernorm.state_dict())
        policy.action_head_ft.load_state_dict(policy.action_head.state_dict())
        policy.input_embed_ft.load_state_dict(policy.input_embed.state_dict())
        policy.input_embed_q.load_state_dict(policy.input_embed.state_dict())
        policy.layernorm_q.load_state_dict(policy.layernorm.state_dict())
        policy.decoder_q.load_state_dict(policy.decoder.state_dict())

    policy.to(device)
    # Broadcast rank-0 initialization so every DDP rank starts from identical weights.
    for param in policy.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in policy.buffers():
        dist.broadcast(buffer.data, src=0)

    ddp_policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    ddp_policy.train()

    trainer = GQRMTrainer(config, ddp_policy, local_rank, device)

    # MPC horizon is scaled per embodiment to keep simulated time comparable.
    con_step = int(training_config.mpc.N * EMBODIMENT_CON_STEP[embodiment])
    mpc_controller = BatchMPCController(
        batch=args_cli.num_envs,
        N=con_step,
        desired_v=training_config.mpc.desired_v,
        v_max=training_config.mpc.v_max,
        w_max=training_config.mpc.w_max,
        v_min_high_curvature=training_config.mpc.v_min_high_curvature,
        ref_gap=training_config.mpc.ref_gap,
        T=EMBODIMENT_SIM_TIME[embodiment]
    )

    # Per-env queues hold MPC controls generated from the latest sampled high-level plan.
    control_queues = [deque() for _ in range(args_cli.num_envs)]
    is_gap = [False for _ in range(args_cli.num_envs)]
    plan_cache = [None for _ in range(args_cli.num_envs)]
    pending_samples = [deque() for _ in range(args_cli.num_envs)]
    plan_rewards = np.zeros(args_cli.num_envs, dtype=np.float32)
    smooth_rewards = np.zeros(args_cli.num_envs, dtype=np.float32)
    plan_steps = np.zeros(args_cli.num_envs, dtype=np.int32)

    # Episode buffers are only for periodic rollout videos and metric visualization.
    episode_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_birdeye_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_point_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_pos_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_quat_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_action_buffer = [[] for _ in range(args_cli.num_envs)]
    episode_score_buffer = [[] for _ in range(args_cli.num_envs)]

    episodes_num = 0
    episodes_eval_num = 0
    episodes_return = np.zeros(args_cli.num_envs, dtype=np.float32)
    episodes_tra_return = np.zeros(args_cli.num_envs, dtype=np.float32)
    step_num = args_cli.initial_step
    update_num = 0
    warmup_stage1_num = 100 * training_config.mpc.N

    last_checkpoint_step = 0
    last_save_tra_step = 0
    latest_path_reward_np = np.zeros(args_cli.num_envs, dtype=np.float32)

    def obs_to_numpy(obs_tensor_dict):
        """Move one observation dict from GPU tensors to NumPy arrays for replay storage."""
        return {key: value.detach().cpu().numpy() for key, value in obs_tensor_dict.items()}

    def reset_plan_state(env_idx: int, terminal_flag) -> None:
        """Clear queued controls and cached rollout state for one vectorized env."""
        is_gap[env_idx] = terminal_flag
        if terminal_flag:
            control_queues[env_idx] = deque([np.zeros(2) for _ in range(5)])
            plan_rewards[env_idx] = 0.0
            smooth_rewards[env_idx] = 0.0
            plan_steps[env_idx] = 0
            return
        control_queues[env_idx].clear()
        plan_cache[env_idx] = None
        plan_rewards[env_idx] = 0.0
        plan_steps[env_idx] = 0

    def fetch_path_reward(env_indices: list) -> np.ndarray:
        """Request path-progress rewards from the Isaac worker for selected envs."""
        cmd_queue.put(('path_reward', env_indices))
        dir_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
        if dir_result.get('type') == 'error':
            raise RuntimeError(f"Worker error: {dir_result.get('error')}")
        return np.asarray(dir_result['reward'], dtype=np.float32)

    def generate_plans_batch(env_indices: list, metadata: dict) -> None:
        """Sample X-NavDP plans, convert them through MPC, and cache per-env controls."""
        if not env_indices:
            return
        obs_batch = {key: obs[key][env_indices] for key in obs.keys()}
        embodiment_batch = torch.full(
                (len(env_indices),), EMBODIMENT_TO_IDX[embodiment],
                dtype=torch.long, device=obs_batch['rgb'].device
            )
        _is_eval = metadata['_is_eval'][env_indices]
        with torch.no_grad():
            is_eval = torch.tensor(_is_eval, device=obs_batch['rgb'].device)
            use_q_select = step_num > warmup_stage1_num
            rollout_k = 8 if use_q_select else 1
            actions, perturb_actions = ddp_policy.module.generate_action_mix(
                obs_batch, num_samples=rollout_k, is_nogoal_perturb=True,
                is_old_policy=False, embodiment=embodiment_batch,
            )
            if use_q_select:
                origin_best = ddp_policy.module.select_best_action_by_q(
                    obs_batch, actions, num_samples=rollout_k,
                    embodiment=embodiment_batch, is_target=False,
                )
                if random.random() < 0.5:
                    perturb_best = ddp_policy.module.select_best_action_by_q(
                        obs_batch, perturb_actions, num_samples=rollout_k,
                        embodiment=embodiment_batch, is_target=False,
                    )
                else:
                    bsz = len(env_indices)
                    cand = perturb_actions.reshape(bsz, rollout_k, *perturb_actions.shape[1:])
                    rand_idx = torch.randint(0, rollout_k, (bsz,), device=cand.device)
                    batch_idx = torch.arange(bsz, device=cand.device)
                    perturb_best = cand[batch_idx, rand_idx]
                sim_actions = perturb_best.clone()
                sim_actions[is_eval] = origin_best[is_eval]
            else:
                sim_actions = perturb_actions.clone()
                sim_actions[is_eval] = actions[is_eval]

            final_actions = torch.cumsum(sim_actions / 4.0, dim=1)
            final_actions = final_actions.reshape(len(env_indices), 24, 3)
            actions_with_idle = torch.concat((torch.zeros_like(final_actions[:, 0:1]), final_actions), dim=1)

        mpc_controls_batch, mpc_states_batch, real_desired_v, _ = mpc_controller.solve(
            actions_with_idle.detach().cpu().numpy()
        )

        nonlocal last_save_tra_step
        if rank == 0 and (step_num - last_save_tra_step) >= 30:
            tra_dir = os.path.join(args_cli.video_dir, "tra_vis")
            os.makedirs(tra_dir, exist_ok=True)
            tra_save_path = os.path.join(tra_dir, f"tra_{step_num}_{embodiment}.jpg")
            visualize_mpc_controller(
                actions_with_idle.detach().cpu().numpy(),
                mpc_states_batch,
                mpc_controls_batch.shape[0],
                save_path=tra_save_path,
                real_desired_v=real_desired_v,
            )
            last_save_tra_step = step_num

        for i, env_idx in enumerate(env_indices):
            control_queues[env_idx] = deque([ctrl for ctrl in mpc_controls_batch[i]])
            obs_single = {key: obs_batch[key][i:i + 1] for key in obs_batch.keys()}
            actions_np = sim_actions.detach().cpu().numpy()
            smooth_rewards[env_idx] = 0.0
            plan_cache[env_idx] = {
                'obs': obs_to_numpy(obs_single),
                'actions': actions_np[i:i + 1],
            }
            plan_rewards[env_idx] = 0.0
            plan_steps[env_idx] = 0
            actions_vis = torch.cat((actions_with_idle[i, :, :2], torch.zeros_like(actions_with_idle[i, :, :1])), dim=1)
            episode_action_buffer[env_idx].append(actions_vis.cpu().numpy())
            episode_score_buffer[env_idx].append(0.0)

    def enqueue_sample(env_idx: int, terminal_flag: bool, metadata: dict) -> None:
        """Convert a finished control segment into one off-policy replay sample."""
        if is_gap[env_idx]:
            reset_plan_state(env_idx, terminal_flag)
        else:
            plan = plan_cache[env_idx]
            if plan is None:
                return
            steps = max(plan_steps[env_idx], 1) / (EMBODIMENT_CON_STEP[embodiment] * 3)
            plan_rewards[env_idx] += smooth_rewards[env_idx]
            episodes_return[env_idx] += smooth_rewards[env_idx]
            episodes_tra_return[env_idx] += smooth_rewards[env_idx]
            norm_reward = np.clip(plan_rewards[env_idx] / steps, -100.0, 100.0).astype(np.float32)
            sample = {
                'obs': plan['obs'],
                'actions': plan['actions'],
                'rewards': np.array([norm_reward], dtype=np.float32),
                'terminals': np.array([1.0 if terminal_flag else 0.0], dtype=np.float32),
                'embodiment': EMBODIMENT_TO_IDX[embodiment],
                'done': terminal_flag,
            }
            pending_samples[env_idx].append(sample)
            reset_plan_state(env_idx, terminal_flag)

    def flush_ready_batches() -> None:
        """Store synchronized per-env samples once every env has one pending item."""
        while all(len(q) > 0 for q in pending_samples):
            batch_samples = [q[0] for q in pending_samples]
            obs_keys = batch_samples[0]['obs'].keys()
            obs_batch = {key: np.concatenate([s['obs'][key] for s in batch_samples], axis=0) for key in obs_keys}
            actions_batch = np.concatenate([s['actions'] for s in batch_samples], axis=0)
            rewards_batch = np.concatenate([s['rewards'] for s in batch_samples], axis=0)
            terminals_batch = np.concatenate([s['terminals'] for s in batch_samples], axis=0)
            embodiment_batch = np.array([s['embodiment'] for s in batch_samples], dtype=np.float32)
            memory.store(obs_batch, actions_batch, rewards_batch, terminals_batch, embodiment_batch)
            for q in pending_samples:
                q.popleft()

    # Initialize the path-progress state at the pre-action poses. The returned
    # values are only a baseline; training rewards are queried after each
    # matching cached plan finishes.
    latest_path_reward_np = fetch_path_reward(list(range(args_cli.num_envs)))

    while True:
        # Generate a new high-level plan wherever the previous MPC control queue is empty.
        needs_replan = [i for i in range(args_cli.num_envs) if plan_cache[i] is None]
        if needs_replan:
            generate_plans_batch(needs_replan, metadata)

        # Save pre-step frames and poses so completed episodes can be rendered as videos.
        prev_obs_raw = obs_raw
        for env_idx in range(args_cli.num_envs):
            if not is_gap[env_idx]:
                raw_rgb = prev_obs_raw['raw_rgb']
                episode_buffer[env_idx].append(raw_rgb[env_idx].cpu().numpy().astype(np.uint8))
                if 'birdeye_rgb' in prev_obs_raw:
                    birdeye_np = prev_obs_raw['birdeye_rgb'][env_idx].cpu().numpy()
                    if birdeye_np.max() <= 1.0:
                        birdeye_np = (birdeye_np * 255).astype(np.uint8)
                    episode_birdeye_buffer[env_idx].append(birdeye_np)
                episode_point_buffer[env_idx].append(prev_obs_raw['goal_pose'][env_idx].cpu().numpy())
                robot_pos = metadata['robot_pos'][env_idx] if env_idx < metadata['robot_pos'].shape[0] else np.zeros(2)
                episode_pos_buffer[env_idx].append(robot_pos[:2] if len(robot_pos) >= 2 else np.zeros(2))
                robot_quat = metadata['robot_quat'][env_idx] if env_idx < metadata['robot_quat'].shape[0] else np.array([1, 0, 0, 0])
                episode_quat_buffer[env_idx].append(robot_quat[[1, 2, 3, 0]])

        goal_prev = obs['pointgoal'].cpu().numpy()[:, 0:2]
        episode_length_prev = metadata['episode_length_buf']

        # Pop the next queued MPC command for each vectorized env and step Isaac once.
        current_control = np.zeros((args_cli.num_envs, 2), dtype=np.float32)
        for env_idx in range(args_cli.num_envs):
            if control_queues[env_idx]:
                current_control[env_idx] = control_queues[env_idx][0]

        dist.barrier()
        cmd_queue.put(('step', current_control, obs_raw_np.get('policy')))
        step_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
        if step_result.get('type') == 'error':
            raise RuntimeError(f"Worker error: {step_result.get('error')}")

        obs_raw_np = step_result['obs_raw']
        metadata = step_result['metadata']
        obs_raw = {
            'obs_rgb': torch.from_numpy(obs_raw_np['obs_rgb']).float().to(device),
            'obs_depth': torch.from_numpy(obs_raw_np['obs_depth']).float().to(device),
            'goal_pose': torch.from_numpy(obs_raw_np['goal_pose']).float().to(device),
            'raw_rgb': torch.from_numpy(obs_raw_np['raw_rgb']).to(device),
        }
        if 'birdeye_rgb' in obs_raw_np:
            obs_raw['birdeye_rgb'] = torch.from_numpy(obs_raw_np['birdeye_rgb']).to(device)
        obs = parse_x_navdp_observations(obs_raw)
        rewards = step_result['rewards']
        dones = step_result['dones']

        reward_np = np.array(rewards, dtype=np.float32)
        is_gap_float = np.array([float(not g) for g in is_gap])
        reward_np = reward_np * is_gap_float
        dones_np = np.array(dones)
        episodes_return += reward_np
        step_num += 1

        print(f"[Rank {rank}]: step_num = {step_num}")

        # A path reward belongs to the plan that just finished, not to the next
        # observation batch. First identify finished plans, then query and
        # write their path rewards back into the matching per-env plan cache.
        finished_envs = []
        terminal_flags = [False for _ in range(args_cli.num_envs)]
        for env_idx in range(args_cli.num_envs):
            if control_queues[env_idx]:
                control_queues[env_idx].popleft()
            plan_rewards[env_idx] += reward_np[env_idx]
            plan_steps[env_idx] += 1 if not is_gap[env_idx] else 0

            terminal_flag = bool(dones_np[env_idx])
            terminal_flags[env_idx] = terminal_flag
            plan_exhausted = len(control_queues[env_idx]) == 0
            if terminal_flag or plan_exhausted:
                finished_envs.append(env_idx)

        finished_env_set = set(finished_envs)
        reward_envs = [
            env_idx for env_idx in finished_envs
            if (not is_gap[env_idx]) and plan_cache[env_idx] is not None
        ]
        if reward_envs:
            path_rewards = fetch_path_reward(reward_envs)
            for env_idx in reward_envs:
                path_reward = float(path_rewards[env_idx])
                latest_path_reward_np[env_idx] = path_reward
                smooth_rewards[env_idx] = path_reward
                if episode_score_buffer[env_idx]:
                    episode_score_buffer[env_idx][-1] = path_reward

        # Convert exhausted plans or terminal episodes into replay samples.
        for env_idx in range(args_cli.num_envs):
            terminal_flag = terminal_flags[env_idx]
            if env_idx in finished_env_set:
                enqueue_sample(env_idx, terminal_flag, metadata)

            if terminal_flag:
                episodes_num += 1
                distance = np.sqrt(np.square(goal_prev[env_idx]).sum())

                # Ask the worker to update success/eval/curriculum state after terminal.
                cmd_queue.put(('episode_end', env_idx, episodes_num))
                ep_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
                print(f"[Rank {rank}]: episode_end")
                success = ep_result.get('success', False)
                total_success_rate = ep_result.get('total_success_rate', 0.0)
                stage_success_rate = ep_result.get('stage_success_rate', 0.0)

                is_eval_mode = metadata['_is_eval'][env_idx]
                scene_out_tag = scene_output_tag(scene_list[scene_index])
                if is_eval_mode:
                    # Write compact per-scene eval records alongside TensorBoard scalars.
                    episodes_eval_num += 1
                    eval_txt_dir = args_cli.txt_dir
                    os.makedirs(eval_txt_dir, exist_ok=True)
                    eval_txt_path = os.path.join(eval_txt_dir, f"{scene_out_tag}_{embodiment}_{rank}.txt")
                    with open(eval_txt_path, "a") as f:
                        f.write(f"{rank},{step_num},{episodes_eval_num},{success},{stage_success_rate}\n")

                if rank >= 0:
                    scene_embod_tag = f"{scene_out_tag}/{embodiment}"
                    if is_eval_mode:
                        summary.add_scalar(f"{scene_embod_tag}/eval/success", float(success), episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/success_eval", float(success), episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/distance", distance, episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/timestep", episode_length_prev[env_idx], episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/score", episodes_return[env_idx], episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/step_num_episodes", step_num, episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/tra_reward", episodes_tra_return[env_idx], episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/total_success_rate", total_success_rate, episodes_eval_num)
                        summary.add_scalar(f"{scene_embod_tag}/eval/stage_success_rate", stage_success_rate, episodes_eval_num)
                    else:
                        summary.add_scalar(f"{scene_embod_tag}/all/success", float(success), episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/all/distance", distance, episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/all/timestep", episode_length_prev[env_idx], episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/all/score", episodes_return[env_idx], episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/all/step_num_episodes", step_num, episodes_num)
                        summary.add_scalar(f"{scene_embod_tag}/all/tra_reward", episodes_tra_return[env_idx], episodes_num)

                eval_interval = metadata.get('_eval_interval', 10)
                episodes_num_list = metadata.get('_episodes_num_list', np.zeros(args_cli.num_envs, dtype=np.int64))
                if (episodes_num_list[env_idx]) % (int(eval_interval)*3) == 0:
                    # Render occasional episode videos with RGB, optional bird-eye, and occupancy-map overlay.
                    video_subdir = os.path.join(args_cli.video_dir, scene_out_tag, embodiment)
                    os.makedirs(video_subdir, exist_ok=True)
                    video_path = os.path.join(
                        video_subdir, f"rl_rank{rank}_{env_idx}_{episodes_eval_num}_{scene_out_tag}_{embodiment}.mp4"
                    )
                    video_fps = video_fps_for_embodiment(embodiment)
                    writer = imageio.get_writer(video_path, fps=video_fps, macro_block_size=None)
                    occ_min_bound = metadata['occ_min_bound']
                    grid_size = float(metadata['grid_size'])
                    grid_img = get_grid_img_from_metadata(metadata)
                    waypoint_index = np.zeros((0, 2), dtype=np.int32)
                    robot_pos_index = np.zeros(2, dtype=np.int32)
                    goal_point_index = np.zeros(2, dtype=np.int32)
                    score = 0.0
                    for i, image in enumerate(episode_buffer[env_idx]):
                        padded_image = np.array(image)
                        robot_pos = episode_pos_buffer[env_idx][i]
                        robot_quat = episode_quat_buffer[env_idx][i]
                        robot_pos_index = ((robot_pos - occ_min_bound[:2]) / grid_size).astype(np.int32)
                        goal_point = episode_point_buffer[env_idx][i]
                        tra_idx = i // con_step

                        if i % con_step == 0 and tra_idx < len(episode_action_buffer[env_idx]):
                            action = episode_action_buffer[env_idx][tra_idx]
                            score = episode_score_buffer[env_idx][tra_idx]
                            try:
                                rot = R.from_quat(robot_quat)
                                rot_matrix = rot.as_matrix()
                            except (ValueError, TypeError) as e:
                                print(f"[Rank {rank}] Quat conversion failed (env_idx={env_idx}, i={i}): {e}")
                                rot_matrix = np.eye(3)
                            action = action @ rot_matrix.T
                            waypoint = robot_pos[None, :] + action[:, :2]
                            waypoint_index = ((waypoint - occ_min_bound[:2]) / grid_size).astype(np.int32)
                            goal_point = goal_point @ rot_matrix.T
                            goal_point_index = ((robot_pos + goal_point[:2] - occ_min_bound[:2]) / grid_size).astype(np.int32)

                        birdeye_img = episode_birdeye_buffer[env_idx][i] if i < len(episode_birdeye_buffer[env_idx]) else None
                        padded_image = visualize_grid_with_path_on_image(
                            padded_image, grid_img, waypoint_index, robot_pos_index, goal_point_index, score,
                            birdeye_image=birdeye_img
                        )
                        if padded_image.shape[0] % 16 != 0:
                            pad_h = 16 - (padded_image.shape[0] % 16)
                            padded_image = np.pad(padded_image, ((0, pad_h), (0, 0), (0, 0)), mode="edge")
                        if padded_image.shape[1] % 16 != 0:
                            pad_w = 16 - (padded_image.shape[1] % 16)
                            padded_image = np.pad(padded_image, ((0, 0), (0, pad_w), (0, 0)), mode="edge")
                        writer.append_data(padded_image)
                    writer.close()

                episodes_return[env_idx] = 0.0
                episodes_tra_return[env_idx] = 0.0
                episode_buffer[env_idx] = []
                episode_birdeye_buffer[env_idx] = []
                episode_point_buffer[env_idx] = []
                episode_action_buffer[env_idx] = []
                episode_score_buffer[env_idx] = []
                episode_pos_buffer[env_idx] = []
                episode_quat_buffer[env_idx] = []

        flush_ready_batches()

        # After warmup, update the policy from replay in minibatches synchronized across ranks.
        if (step_num > warmup_stage1_num) and (step_num % int(training_config.mpc.N * MAX_CON_STEP) == 0):
            dist.barrier()

            num_samples = args_cli.num_envs * 24
            batchsize = num_samples // training_config.trainer.minibatch
            indexes = np.random.choice(memory.size * args_cli.num_envs, size=num_samples, replace=True)

            for i in range(training_config.trainer.minibatch):
                update_num += 1
                batch_indexes = indexes[i * batchsize:(i + 1) * batchsize]
                sample_dict = memory.sample(batch_indexes)
                loss_dict = trainer.update(update_num, sample_dict)

                if rank >= 0:
                    summary.add_scalar("train/value_loss", loss_dict['value_loss'], global_step=loss_dict['iteration'])
                    summary.add_scalar("train/actor_loss", loss_dict['policy_loss'], global_step=loss_dict['iteration'])
                    summary.add_scalar("train/value", loss_dict['value'], global_step=loss_dict['iteration'])
                    summary.add_scalar("train/log_alpha", loss_dict['log_alpha'], global_step=loss_dict['iteration'])
                    summary.add_scalar("train/step_num_update", step_num, global_step=loss_dict['iteration'])
                    summary.add_scalar("train/memory_size", memory.size, global_step=loss_dict['iteration'])

            torch.cuda.empty_cache()

        # Save rank-0 checkpoints at half-interval boundaries after the initial warmup window.
        half_interval = args_cli.checkpoint_interval // 2
        if step_num >= (args_cli.checkpoint_interval * 3) and step_num // half_interval > last_checkpoint_step // half_interval:
            last_checkpoint_step = (step_num // half_interval) * half_interval
            if rank == 0:
                checkpoint_path = os.path.join(args_cli.checkpoint_dir, f"x_navdp_rl_{last_checkpoint_step}.ckpt")
                torch.save(policy.state_dict(), checkpoint_path)

        # Periodically tear down and recreate Isaac to switch embodiment on the same scene.
        effective_interval = embodiment_switch_interval + (args_cli.checkpoint_interval if embodiment_switch_count == 0 else 0)
        if step_num > 0 and (step_num - last_switch_step) >= effective_interval:
            dist.barrier()
            print(f"[Rank {rank}] Embodiment switching starts")

            cmd_queue.put(('close',))
            worker.join(timeout=360)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)

            embodiment_switch_count, last_switch_step = update_embodiment_switch_state(
                embodiment_switch_count, last_switch_step, step_num,
                embodiment_switch_interval, rank
            )

            # Recreate queues and worker because SimulationApp/env objects are process-local.
            cmd_queue = ctx.Queue()
            result_queue = ctx.Queue()
            embodiment = embodiment_list[(rank + embodiment_switch_count) % len(embodiment_list)]
            worker = ctx.Process(
                target=run_env_worker,
                args=(cmd_queue, result_queue, scene_list, scene_index, args_cli.num_envs, args_cli.scene_scale, device, embodiment_switch_count + 1, embodiment),
            )
            worker.start()
            print(f"[Rank {rank}] Worker restarted with scene_index={scene_index}, embodiment={embodiment}")

            # Rebuild MPC when embodiment changes because the horizon and timestep may change.
            con_step = int(training_config.mpc.N * EMBODIMENT_CON_STEP[embodiment])
            mpc_controller = BatchMPCController(
                batch=args_cli.num_envs,
                N=con_step,
                desired_v=training_config.mpc.desired_v,
                v_max=training_config.mpc.v_max,
                w_max=training_config.mpc.w_max,
                v_min_high_curvature=training_config.mpc.v_min_high_curvature,
                ref_gap=training_config.mpc.ref_gap,
                T=EMBODIMENT_SIM_TIME[embodiment]
            )

            # Reset and warm up the recreated env before resuming policy rollouts.
            cmd_queue.put(('reset',))
            init_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
            if init_result.get('type') == 'error':
                raise RuntimeError(f"Worker error: {init_result.get('error', 'unknown')}")

            obs_raw_np = init_result['obs_raw']
            metadata = init_result['metadata']
            obs_raw = {
                'obs_rgb': torch.from_numpy(obs_raw_np['obs_rgb']).float().to(device),
                'obs_depth': torch.from_numpy(obs_raw_np['obs_depth']).float().to(device),
                'goal_pose': torch.from_numpy(obs_raw_np['goal_pose']).float().to(device),
                'raw_rgb': torch.from_numpy(obs_raw_np['raw_rgb']).to(device),
            }
            if 'birdeye_rgb' in obs_raw_np:
                obs_raw['birdeye_rgb'] = torch.from_numpy(obs_raw_np['birdeye_rgb']).to(device)
            obs = parse_x_navdp_observations(obs_raw)

            for _ in range(5):
                cmd_queue.put(('step', np.zeros((args_cli.num_envs, 2), dtype=np.float32), obs_raw_np.get('policy')))
                step_result = wait_worker_result(result_queue, timeout=args_cli.worker_timeout, rank=rank)
                if step_result.get('type') == 'error':
                    raise RuntimeError(f"Worker error: {step_result.get('error')}")
                obs_raw_np = step_result['obs_raw']
                metadata = step_result['metadata']
                obs_raw = {
                    'obs_rgb': torch.from_numpy(obs_raw_np['obs_rgb']).float().to(device),
                    'obs_depth': torch.from_numpy(obs_raw_np['obs_depth']).float().to(device),
                    'goal_pose': torch.from_numpy(obs_raw_np['goal_pose']).float().to(device),
                    'raw_rgb': torch.from_numpy(obs_raw_np['raw_rgb']).to(device),
                }
                if 'birdeye_rgb' in obs_raw_np:
                    obs_raw['birdeye_rgb'] = torch.from_numpy(obs_raw_np['birdeye_rgb']).to(device)
                obs = parse_x_navdp_observations(obs_raw)

            # Clear all per-worker rollout state after worker replacement.
            # The recreated worker owns fresh path-progress state, so establish
            # a new pre-action baseline before generating the next plans.
            latest_path_reward_np = fetch_path_reward(list(range(args_cli.num_envs)))
            for env_idx in range(args_cli.num_envs):
                reset_plan_state(env_idx, False)
                plan_cache[env_idx] = None
                episodes_return[env_idx] = 0.0
                episodes_tra_return[env_idx] = 0.0
                episode_buffer[env_idx] = []
                episode_birdeye_buffer[env_idx] = []
                episode_point_buffer[env_idx] = []
                episode_action_buffer[env_idx] = []
                episode_score_buffer[env_idx] = []
                episode_pos_buffer[env_idx] = []
                episode_quat_buffer[env_idx] = []

            dist.barrier()
            print(f"\n[Rank {rank}] Embodiment switched to {embodiment} on scene {scene_index}")

        if args_cli.max_steps is not None and step_num >= args_cli.max_steps:
            print(f"[Rank {rank}] Reached --max_steps={args_cli.max_steps}; shutting down cleanly")
            break

    cmd_queue.put(('close',))
    worker.join(timeout=args_cli.worker_timeout)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=10)
    summary.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
