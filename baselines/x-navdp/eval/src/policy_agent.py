"""
NavDP Policy Agent - High-level navigation policy wrapper.

This module provides the NavDP_Agent class that wraps the low-level policy
network and handles trajectory planning, visualization, and state management.
"""

import torch
import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import deque


class NavDP_Agent:
    """Navigation policy agent with trajectory planning capabilities."""

    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 embodiment=0,
                 is_real=False,
                 navi_model="./100.ckpt",
                 device='cuda:0'):
        try:
            from .policy_network_embodiment import NavDP_Policy_Embodiment
        except ImportError:
            from policy_network_embodiment import NavDP_Policy_Embodiment

        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.embodiment = embodiment
        self.is_real = is_real

        self.navi_former = NavDP_Policy_Embodiment(
            image_size, memory_size, predict_size, temporal_depth, heads, token_dim, device
        )
        self.navi_former.load_state_dict(
            torch.load(navi_model, map_location=self.device), strict=False
        )
        self.navi_former.to(self.device)
        self.navi_former.eval()

        self.current_scene_name = None
        self._occ_cache = {}
        self.save_counter = np.zeros(1000, dtype=np.int32)
        self.save_plan_dir = None

    def reset(self, batch_size, stuck_window=5, stuck_xy_threshold=0.25):
        """Reset the agent state for a new episode."""
        self.batch_size = batch_size
        self.stuck_window = int(stuck_window)
        self.stuck_xy_threshold = float(stuck_xy_threshold)
        self.history_window = 16
        if not self.is_real and self.embodiment in (1, 2):
            self.history_window = int(round(self.history_window * 2.5))
        self.frame_interval = 1
        self.memory_queue = np.zeros(
            (batch_size, self.history_window, self.image_size, self.image_size, 3),
            dtype=np.float32
        )
        self.frame_count = [0 for _ in range(batch_size)]
        self.sample_idx_list = list(range(batch_size))
        self.last_robot_pos = np.zeros((batch_size, 3), dtype=np.float64)
        self.last_robot_quat = np.zeros((batch_size, 4), dtype=np.float64)
        self.last_robot_quat[:, 3] = 1.0
        self.last_execute_trajectory = np.zeros((batch_size, self.predict_size, 3), dtype=np.float64)
        self.last_valid = np.zeros(batch_size, dtype=bool)
        self._guidance_inference_step_count = np.zeros(batch_size, dtype=np.int32)
        self._stuck_xy_history = [deque(maxlen=self.stuck_window) for _ in range(batch_size)]
        self.is_stuck = np.zeros(batch_size, dtype=bool)
        self._waypoint_direction_sleep_pending = False

    def reset_env(self, i):
        """Reset a specific environment in the batch."""
        self.memory_queue[i].fill(0)
        self.frame_count[i] = 0
        self.last_valid[i] = False
        self._guidance_inference_step_count[i] = 0
        self._stuck_xy_history[i].clear()
        self.is_stuck[i] = False
        self._waypoint_direction_sleep_pending = False

    def process_image(self, images):
        """Preprocess RGB images for the policy network."""
        assert len(images.shape) == 4
        H, W, C = images.shape[1], images.shape[2], images.shape[3]
        prop = self.image_size / max(H, W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_image.shape[0]) // 2, 0)
            pad_image = np.pad(
                resize_image,
                ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                mode='constant', constant_values=0
            )
            resize_image = cv2.resize(pad_image, (self.image_size, self.image_size))
            resize_image = np.array(resize_image).astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self, depths):
        """Preprocess depth images for the policy network."""
        assert len(depths.shape) == 4
        depths[depths == np.inf] = 0
        H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
        prop = self.image_size / max(H, W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1]) // 2, 0)
            pad_height = max((self.image_size - resize_depth.shape[0]) // 2, 0)
            pad_depth = np.pad(
                resize_depth,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant', constant_values=0
            )
            resize_depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
            resize_depth[resize_depth > 5.0] = 0
            resize_depth[resize_depth < 0.1] = 0
            return_depths.append(resize_depth[:, :, np.newaxis])
        return np.array(return_depths)

    def process_pointgoal(self, goals):
        """Clip point goals to a maximum distance."""
        norm_goals = np.linalg.norm(goals, axis=1)
        mask = (norm_goals > 25.0).astype(np.float32)
        scale_factor = mask * (25.0 / norm_goals) + (1.0 - mask)
        clip_goals = goals * np.expand_dims(scale_factor, axis=1)
        return clip_goals

    def _update_and_sample_history(self, process_images, num_samples=None):
        """Update memory queue with new frames and sample for policy input."""
        if num_samples is None:
            num_samples = self.memory_size - 1
        indices = np.linspace(0, self.history_window - 1, num_samples - 1).astype(np.int64)
        history_part = self.memory_queue[:, indices]
        sampled_frames = np.concatenate([history_part, process_images[:, np.newaxis]], axis=1)
        for i in range(self.batch_size):
            self.frame_count[i] += 1
            should_save = ((self.frame_count[i] - 1) % self.frame_interval == 0) and (self.frame_count[i] > 0)
            if should_save:
                self.memory_queue[i, 0:-1] = self.memory_queue[i, 1:].copy()
                self.memory_queue[i, -1] = process_images[i]
        return sampled_frames

    def project_trajectory_2d(self, images, n_trajectories, n_values):
        """Project trajectories onto 2D visualization panels."""
        trajectory_masks = []
        colormap = cm.get('jet')
        max_color = np.array(colormap(1.0)[0:3]) * 255.0

        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i, :, :, 0:2]
            n_value = n_values[i]
            top_indices = np.argsort(n_value)[::-1][:6]
            n_trajectory = n_trajectory[top_indices]
            n_value = n_value[top_indices]
            mean_value = np.mean(n_value)
            std_value = np.std(n_value)
            if std_value < 1e-8:
                std_value = 1.0

            img_height, img_width = trajectory_mask.shape[0], trajectory_mask.shape[1]
            max_value_idx = np.argmax(n_value)
            trajectory_colors = []
            for idx, (waypoints, value) in enumerate(zip(n_trajectory, n_value)):
                if idx == max_value_idx:
                    color = max_color
                else:
                    normalized = (value - mean_value) / std_value
                    norm_value = 1 / (1 + np.exp(-normalized))
                    color = np.array(colormap(norm_value)[0:3]) * 255.0
                trajectory_colors.append(color)

            left_image = trajectory_mask.copy()
            fig, ax = plt.subplots(1, 1, figsize=(img_width/100, img_height/100), dpi=100)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            all_waypoints = n_trajectory.reshape(-1, 2)
            if all_waypoints.shape[0] > 0:
                x_min, x_max = all_waypoints[:, 0].min(), all_waypoints[:, 0].max()
                y_min, y_max = all_waypoints[:, 1].min(), all_waypoints[:, 1].max()
                x_range = max(x_max - x_min, 1e-8)
                y_range = max(y_max - y_min, 1e-8)
                margin = 0.1
                x_min -= x_range * margin
                x_max += x_range * margin
                y_min -= y_range * margin
                y_max += y_range * margin

                for idx, (waypoints, value) in enumerate(zip(n_trajectory, n_value)):
                    x_coords = waypoints[:, 0]
                    y_coords = waypoints[:, 1]
                    color = trajectory_colors[idx] / 255.0
                    linewidth = 3 if idx == max_value_idx else 2
                    ax.plot(y_coords, x_coords, '-', color=color, linewidth=linewidth, alpha=0.8)
                    if len(x_coords) > 0:
                        ax.plot(y_coords[0], x_coords[0], 'o', color=color, markersize=2, alpha=0.9)
                        ax.plot(y_coords[-1], x_coords[-1], 's', color=color, markersize=2, alpha=0.9)
                        dx = 0.02 * (y_max - y_min)
                        dy = 0.02 * (x_max - x_min)
                        ax.text(y_coords[-1] + dx, x_coords[-1] + dy, f"{float(value):.2f}",
                                color=color, fontsize=8, ha="left", va="bottom", alpha=0.9)
                ax.set_xlim(y_min, y_max)
                ax.set_ylim(x_min, x_max)
            else:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

            ax.set_aspect("equal", adjustable="box")
            ax.invert_xaxis()
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("x [m]")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fig.canvas.draw()
            if hasattr(fig.canvas, "tostring_rgb"):
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            else:
                buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            right_image = cv2.resize(buf, (img_width, img_height))
            plt.close(fig)
            combined_mask = np.concatenate([left_image, right_image], axis=1)
            trajectory_masks.append(combined_mask)

        return np.concatenate(trajectory_masks, axis=1)

    def _trajectory_debug_panel(self, current_traj, guidance_traj, robot_pos, robot_quat, image_shape):
        """Render current output and state-guided previous trajectory in world frame."""
        img_height, img_width = image_shape[:2]
        if torch.is_tensor(current_traj):
            current_traj = current_traj.detach().cpu().numpy()
        if torch.is_tensor(guidance_traj):
            guidance_traj = guidance_traj.detach().cpu().numpy()
        current_traj = np.asarray(current_traj, dtype=np.float64)
        guidance_traj = None if guidance_traj is None else np.asarray(guidance_traj, dtype=np.float64)
        panels = []
        for i in range(current_traj.shape[0]):
            curr_local = np.asarray(current_traj[i], dtype=np.float64)
            if curr_local.ndim == 2:
                curr_local = np.concatenate([np.zeros((1, curr_local.shape[1])), curr_local], axis=0)
            if curr_local.shape[1] == 2:
                curr_local = np.concatenate([curr_local, np.zeros((curr_local.shape[0], 1))], axis=1)

            curr_rot = R.from_quat(np.asarray(robot_quat[i], dtype=np.float64))
            curr_pos = np.asarray(robot_pos[i], dtype=np.float64)
            curr_world = curr_rot.apply(curr_local) + curr_pos[None, :]

            lines = [(curr_world[:, :2], "current", "red", 2.5, "o")]
            guidance_valid = i < self.last_valid.shape[0] and bool(self.last_valid[i])
            if guidance_valid and guidance_traj is not None and i < guidance_traj.shape[0]:
                guide_local = np.asarray(guidance_traj[i], dtype=np.float64)
                if guide_local.shape[1] == 2:
                    guide_local = np.concatenate([guide_local, np.zeros((guide_local.shape[0], 1))], axis=1)
                guide_world = curr_rot.apply(guide_local) + curr_pos[None, :]
                lines.append((guide_world[:, :2], "guidance", "royalblue", 2.0, "s"))

            fig, ax = plt.subplots(1, 1, figsize=(img_width / 100, img_height / 100), dpi=100)
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            all_pts = np.concatenate([line[0] for line in lines if line[0].shape[0] > 0], axis=0)
            if all_pts.shape[0] > 0:
                x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
                y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
                span = max(x_max - x_min, y_max - y_min, 1.0)
                cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
                margin = span * 0.65
                ax.set_xlim(cx - margin, cx + margin)
                ax.set_ylim(cy - margin, cy + margin)
            else:
                ax.set_xlim(curr_pos[0] - 1.0, curr_pos[0] + 1.0)
                ax.set_ylim(curr_pos[1] - 1.0, curr_pos[1] + 1.0)

            for pts, label, color, linewidth, marker in lines:
                if pts.shape[0] == 0:
                    continue
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=linewidth, label=label)
                ax.plot(pts[0, 0], pts[0, 1], marker, color=color, markersize=4)
                ax.plot(pts[-1, 0], pts[-1, 1], marker, color=color, markersize=4, markerfacecolor="none")

            yaw = curr_rot.as_euler("xyz", degrees=False)[2]
            heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
            ax.arrow(
                curr_pos[0], curr_pos[1],
                heading[0] * 0.35, heading[1] * 0.35,
                width=0.025, color="black", length_includes_head=True,
            )
            ax.plot(curr_pos[0], curr_pos[1], "ko", markersize=3)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title("world-frame trajectory check", fontsize=8)
            ax.set_xlabel("world x [m]")
            ax.set_ylabel("world y [m]")
            plt.tight_layout()
            fig.canvas.draw()
            if hasattr(fig.canvas, "tostring_rgb"):
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            else:
                buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            panels.append(cv2.resize(buf, (img_width, img_height)))
            plt.close(fig)

        return np.concatenate(panels, axis=1)

    def _update_stuck_state(self, robot_pos):
        """Detect if robot is stuck based on recent position history."""
        for i in range(self.batch_size):
            xy = np.asarray(robot_pos[i, :2], dtype=np.float64)
            self._stuck_xy_history[i].append(xy.copy())
            h = self._stuck_xy_history[i]
            if len(h) < self.stuck_window:
                self.is_stuck[i] = False
                continue
            current = np.asarray(h[-1], dtype=np.float64)
            history = np.stack(list(h)[:-1], axis=0)
            max_dist = np.sqrt(np.square(history - current[None, :]).sum(axis=-1)).max()
            self.is_stuck[i] = max_dist < self.stuck_xy_threshold

    def _build_guidance_factor_batch(self, is_stuck, sample_num=8):
        """Build guidance factor mask for stuck detection."""
        is_stuck = np.asarray(is_stuck, dtype=bool)
        gf = np.zeros((is_stuck.shape[0], sample_num), dtype=np.float32)
        g_default = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05, 0.05], dtype=np.float32)
        if g_default.shape[0] != sample_num:
            raise ValueError("g_default length must equal sample_num.")
        gf[~is_stuck] = g_default
        return gf

    def _update_last_trajectory_state(self, traj, robot_pos, robot_quat):
        """Update the last executed trajectory for guidance."""
        traj_np = traj.detach().cpu().numpy() if torch.is_tensor(traj) else np.array(traj)
        if traj_np.ndim == 2:
            traj_np = traj_np[None, ...]
        traj_np = np.concatenate([np.zeros((traj_np.shape[0], 1, 3)), traj_np], axis=1)
        traj_np[:, :, 2] = 0.0
        self.last_robot_pos = np.array(robot_pos, dtype=np.float64)
        self.last_robot_quat = np.array(robot_quat, dtype=np.float64)
        self.last_execute_trajectory = traj_np.astype(np.float64)
        b = traj_np.shape[0]
        self._guidance_inference_step_count[:b] += 1
        self.last_valid[:b] = self._guidance_inference_step_count[:b] > 3

    def _first_waypoint_direction_inconsistent_with_last(self, current_traj):
        """Check if current trajectory direction is inconsistent with last."""
        cur = np.asarray(
            current_traj.detach().cpu().numpy() if torch.is_tensor(current_traj) else current_traj,
            dtype=np.float64,
        )
        if not self.last_valid[0]:
            return False
        cx = cur[0, 0, 0]
        px = self.last_execute_trajectory[0, 1, 0]
        return cx * px < 0 and self.is_stuck[0]

    def _make_ref_denser(self, ref_traj, ratio=50):
        """Interpolate trajectory to have more points."""
        if len(ref_traj) < 2:
            return ref_traj
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        return np.stack((interp_func_x(new_x), interp_func_y(new_x)), axis=1)

    def get_guidance_trajectory(self, robot_pos, robot_quat):
        """Compute guidance trajectory from last executed actions."""
        batch_size = len(robot_pos)
        valid = self.last_valid
        prev_actions = np.zeros((batch_size, self.predict_size, 3), dtype=np.float32)
        prev_paths = np.zeros((batch_size, self.predict_size + 1, 3), dtype=np.float32)
        valid_segment_len = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            if not valid[i]:
                prev_actions[i] = 0
                valid_segment_len[i] = 0
                continue

            last_traj_i = self.last_execute_trajectory[i]
            last_pos_i = self.last_robot_pos[i]
            last_quat_i = self.last_robot_quat[i]
            curr_pos_i = robot_pos[i]
            curr_quat_i = robot_quat[i]

            old_action_xy = last_traj_i.copy()
            old_rot = R.from_quat(last_quat_i)
            current_rot = R.from_quat(curr_quat_i)
            old_action_world = old_rot.apply(old_action_xy) + last_pos_i[None, :]
            old_action_current = current_rot.inv().apply(old_action_world - curr_pos_i[None, :])
            old_action_xy = old_action_current[:, :2]
            n_path = min(old_action_current.shape[0], prev_paths.shape[1])
            prev_paths[i, :n_path] = old_action_current[:n_path].astype(np.float32)

            old_action_xy_denser = self._make_ref_denser(old_action_xy, ratio=50)
            distances = np.sqrt(np.sum(old_action_xy_denser ** 2, axis=1))
            closest_idx_denser = distances.argmin()
            closest_distance = distances[closest_idx_denser]
            n_orig = old_action_xy.shape[0]
            num_denser = n_orig * 50
            closest_idx = int(np.round(closest_idx_denser * (n_orig - 1) / max(1, num_denser - 1)))
            closest_idx = int(np.clip(closest_idx, 0, n_orig - 1))
            si = closest_idx + 1

            if si >= n_orig:
                valid_segment_len[i] = 0
                continue

            zero = np.zeros((1, old_action_xy.shape[-1]), dtype=old_action_xy.dtype)
            segment = old_action_xy[si:, :]
            prev_action_pre = np.concatenate([zero, segment], axis=0)
            prev_action_pre = np.concatenate([prev_action_pre, np.zeros((prev_action_pre.shape[0], 1), dtype=prev_action_pre.dtype)], axis=1)
            prev_action_pre = prev_action_pre[None, :, :]

            segment_len = n_orig - si
            valid_segment_len[i] = segment_len

            if closest_distance > 0.02:
                predict_size = prev_action_pre.shape[1]
                weight = np.ones(predict_size)
                weight[:8] = 5.0
                weight[0] = 20.0
                vel_pre = (prev_action_pre[:, 1:, :] - prev_action_pre[:, :-1, :]) * 4.0
                vel_pre_t = torch.from_numpy(vel_pre.astype(np.float32)).to(self.device)
                num_points = vel_pre_t.shape[1] + 1
                smooth_out = self.navi_former.smooth_trajectory(
                    vel_pre_t, num_points=num_points, smooth_factor=0.5, weight=weight
                )
                prev_action_np = smooth_out.detach().cpu().numpy()[0]
            else:
                cid = closest_idx_denser
                segment = old_action_xy_denser[cid:, :]
                num_sample = 25 - closest_idx
                inds = np.round(np.linspace(0, len(segment) - 1, num=num_sample, endpoint=True)).astype(int)
                sampled = segment[inds][:]
                velocities_xy = (sampled[1:] - sampled[:-1]) * 4.0
                prev_action_np = np.concatenate([velocities_xy, np.zeros((velocities_xy.shape[0], 1), dtype=np.float32)], axis=1)

            n = prev_action_np.shape[0]
            prev_actions[i, :n] = prev_action_np[:]

        return torch.from_numpy(prev_actions).float().to(self.device), valid_segment_len, prev_paths

    def step_pointgoal_with_guidance(self, goals, images, depths, robot_pos, robot_quat):
        """Perform point goal navigation step with robot state guidance."""
        has_robot_state = robot_pos is not None and robot_quat is not None
        if self.batch_size == 1 and self._waypoint_direction_sleep_pending:
            print("waypoint direction was inconsistent with last; sleep 1s before this predict !!!!!!!!!!!!!!!!!!")
            time.sleep(1.5)
            self._waypoint_direction_sleep_pending = False

        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._update_and_sample_history(process_images, num_samples=self.memory_size)
        input_goals = self.process_pointgoal(goals)
        if has_robot_state:
            self._update_stuck_state(robot_pos)
        else:
            self.is_stuck[:] = False
        input_depth = process_depths
        start_index = 0
        end_index = self.predict_size - 1
        sample_num = 8

        if has_robot_state:
            prev_action, valid_segment_len, guidance_paths = self.get_guidance_trajectory(robot_pos, robot_quat)
        else:
            prev_action = torch.zeros(
                (self.batch_size, self.predict_size, 3),
                dtype=torch.float32,
                device=self.device,
            )
            valid_segment_len = np.zeros(self.batch_size, dtype=np.int32)
            guidance_paths = None
        guidance_factor = self._build_guidance_factor_batch(self.is_stuck, sample_num=sample_num)

        predict_start_time = time.time()
        all_trajectory, all_values, good_trajectory, _ = (
            self.navi_former.predict_pointgoal_action_with_guidance(
                input_goals, input_image, input_depth,
                sample_num=sample_num,
                valid_segment_len=valid_segment_len,
                prev_action=prev_action,
                start_index=start_index,
                end_index=end_index,
                guidance_factor=guidance_factor,
                guidance_step=5,
                prefix_attention_schedule="exp",
                embodiment=self.embodiment
            )
        )
        print(f"predict time: {time.time() - predict_start_time:.3f}s")
        print(f"Q-values: max={all_values.max():.3f}, min={all_values.min():.3f}")

        stuck = np.asarray(self.is_stuck, dtype=bool)
        if stuck.any():
            n_samples = all_trajectory.shape[1]
            for i in range(good_trajectory.shape[0]):
                if i < stuck.shape[0] and stuck[i]:
                    r = int(np.random.randint(0, n_samples))
                    picked = np.copy(all_trajectory[i, r])
                    good_trajectory[i, 0] = picked

        trajectory_mask = self.project_trajectory_2d(images, all_trajectory, all_values)
        if has_robot_state:
            debug_panel = self._trajectory_debug_panel(
                good_trajectory[:, 0], guidance_paths, robot_pos, robot_quat, images.shape[1:3]
            )
            trajectory_mask = np.concatenate([trajectory_mask, debug_panel], axis=1)

        if has_robot_state and self.batch_size == 1 and self._first_waypoint_direction_inconsistent_with_last(good_trajectory[:, 0]):
            self._waypoint_direction_sleep_pending = True
        if has_robot_state:
            self._update_last_trajectory_state(good_trajectory[:, 0], robot_pos, robot_quat)

        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask
