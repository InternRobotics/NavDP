"""Diffusion navigation policy with embodiment-conditioned actor and critics."""

import torch
import torch.nn as nn
import math
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.interpolate import splprep, splev

from .x_navdp_backbone import (
    ImageGoalBackbone,
    LearnablePositionalEncoding,
    PixelGoalBackbone,
    RGBDBackbone,
    SinusoidalPosEmb,
)

class XNavDPPolicy(nn.Module):
    """X-NavDP diffusion actor with twin-Q critics and embodiment modulation."""
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 ft_denoise_step=5,
                 off_policy=False,
                 distinguish_embodiment=True,
                 device='cuda:0'):
        """Initialize encoders, diffusion decoders, critics, and sampling knobs."""
        super().__init__()
        self.device = device
        self.distinguish_embodiment = distinguish_embodiment
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        self.ft_denoise_step = ft_denoise_step
        self.off_policy = off_policy
        self.embodiment_embedding = nn.Embedding(3, token_dim)  # dingo=0, unitree_g1=1, unitree_go2=2
        self.rgbd_encoder = RGBDBackbone(image_size,token_dim,memory_size=memory_size,rgb_training=False,depth_training=False,fusion_training=False,device=device)
        self.point_encoder = nn.Linear(3,self.token_dim)
        self.pixel_encoder = PixelGoalBackbone(image_size,token_dim,device=device)
        self.image_encoder = ImageGoalBackbone(image_size,token_dim,device=device)
        # fusion layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        dropout = 0.0,
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer, num_layers = self.temporal_depth)
        self.input_embed = nn.Linear(3,token_dim) # encode the actions for denoise/critic
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,
                                       prediction_type='epsilon')

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask = torch.zeros((predict_size, 6 + memory_size * 16))
        self.cond_critic_mask[:,0:1] = float('-inf') # mask time embedding
        self.cond_critic_mask_no_embod = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask_no_embod[:,0:1] = float('-inf')
        self.cond_critic_mask_one_embod = torch.zeros((predict_size, 5 + memory_size * 16))
        self.cond_critic_mask_one_embod[:,0:1] = float('-inf')

        self.decoder_ft = nn.TransformerDecoder(decoder_layer = self.decoder_layer, num_layers = self.temporal_depth)
        self.layernorm_ft = nn.LayerNorm(token_dim)
        self.action_head_ft = nn.Linear(token_dim, 3)
        self.input_embed_ft = nn.Linear(3,token_dim)
        # Action-side embodiment: identity at init (pretrained cross-attn memory unchanged).
        self.embody_tgt_delta = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        _embody_delta_last = self.embody_tgt_delta[2]
        assert isinstance(_embody_delta_last, nn.Linear)
        nn.init.zeros_(_embody_delta_last.weight)
        nn.init.zeros_(_embody_delta_last.bias)
        self.embody_out_film = nn.Linear(token_dim, 2 * token_dim)
        nn.init.zeros_(self.embody_out_film.weight)
        nn.init.zeros_(self.embody_out_film.bias)
        actor_params_base = (
            list(self.input_embed_ft.parameters())
            + list(self.decoder_ft.parameters())
            + list(self.layernorm_ft.parameters())
            + list(self.action_head_ft.parameters())
        )
        if distinguish_embodiment:
            self.actor_params = (
                list(self.rgbd_encoder.parameters())
                + list(self.embodiment_embedding.parameters())
                + list(self.embody_tgt_delta.parameters())
                + list(self.embody_out_film.parameters())
                + actor_params_base
            )
        else:
            self.actor_params = actor_params_base

        self.decoder_q_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                nhead = heads,
                                dim_feedforward = 4 * token_dim,
                                activation = 'gelu',
                                dropout = 0.0,
                                batch_first = True,
                                norm_first = True)
        self.decoder_q = nn.TransformerDecoder(decoder_layer = self.decoder_q_layer, num_layers = self.temporal_depth)
        self.layernorm_q = nn.LayerNorm(token_dim)
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(predict_size * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        # Shared Q heads; embodiment via tgt delta + FiLM (same as actor ft path).
        self.q1_heads = nn.Linear(token_dim, 1)
        self.q2_heads = nn.Linear(token_dim, 1)
        self.input_embed_q = nn.Linear(3,token_dim)
        critic_params_base = (
            list(self.rgbd_encoder.parameters())
            + list(self.input_embed_q.parameters())
            + list(self.decoder_q.parameters())
            + list(self.layernorm_q.parameters())
            + list(self.q_pool_mlp.parameters())
            + list(self.q1_heads.parameters())
            + list(self.q2_heads.parameters())
        )
        if distinguish_embodiment:
            self.critic_params = (
                list(self.embodiment_embedding.parameters())
                + list(self.embody_tgt_delta.parameters())
                + list(self.embody_out_film.parameters())
                + critic_params_base
            )
        else:
            self.critic_params = critic_params_base

        self.q1_target_heads = nn.Linear(token_dim, 1)
        self.q2_target_heads = nn.Linear(token_dim, 1)
        self.q1_target_heads.load_state_dict(self.q1_heads.state_dict())
        self.q2_target_heads.load_state_dict(self.q2_heads.state_dict())
        for param in self.q1_target_heads.parameters():
            param.requires_grad = False
        for param in self.q2_target_heads.parameters():
            param.requires_grad = False

        self.ft_step = 6
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(1), dtype=torch.float32))

        self.reverse_prob_x = 0.25
        self.reverse_prob_y = 0.25
        self.keep_origin_prob = 0.2
        self.keep_line_prob = 0.25
        self.keep_nogoal_prob = 0.15

        self.keep_line_len_min = 1.5
        self.keep_line_len_max = 3.5
        self.truncate_prob = 1.0

        self.weight = np.ones(self.predict_size + 1)
        self.weight[:9] = 5.0
        self.weight[:5] = 50.0
        self.weight[0] = 100.0

        # 运动学参考：总轨迹长度 2.0 m 对应期望速度 0.5 m/s；更短线性降速，最低 0.1 m/s
        self.ref_traj_length_m = 2.0
        self.ref_desired_v = 0.5
        self.min_desired_v = 0.05
        self.omega_max = 0.5
        self.curvature_horizon = 12

    def desired_v_from_trajectory_length(self, length_m: torch.Tensor) -> torch.Tensor:
        """轨迹总长度越小期望速度越低；>= ref_traj_length_m 时不超过 ref_desired_v，不低于 min_desired_v。"""
        length_m = length_m.clamp(min=1e-6)
        scale = (length_m / self.ref_traj_length_m).clamp(max=1.0)
        return (self.ref_desired_v * scale).clamp(min=self.min_desired_v, max=self.ref_desired_v)

    def min_turn_radius_from_desired_v(self, desired_v: torch.Tensor) -> torch.Tensor:
        """Convert desired linear velocity to minimum feasible turn radius."""
        return desired_v / self.omega_max

    @staticmethod
    def _calculate_curvature(traj):
        """Estimate smoothed planar curvature along one trajectory."""
        dx = np.gradient(traj[:, 0])
        dy = np.gradient(traj[:, 1])
        dy[0] = 0.0

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** 1.5
        denominator[denominator < 1e-6] = 1e-6

        curvature = numerator / denominator
        curvature = np.convolve(curvature, np.ones(3) / 3, mode="same")
        return curvature

    def calculate_max_curvature(self, points, num_points=None):
        """Return maximum curvature over the early part of each trajectory."""
        if num_points is None:
            num_points = self.curvature_horizon
        batch = points.shape[0]
        points_with_zero = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points_np = points_with_zero.cpu().numpy()
        max_curvatures = []
        for i in range(batch):
            traj = points_np[i, :num_points, :2]
            curv = self._calculate_curvature(traj)
            max_curvatures.append(float(np.max(curv)))
        return torch.tensor(max_curvatures, dtype=torch.float32, device=points.device)

    def desired_v_from_curvature(self, curvature: torch.Tensor) -> torch.Tensor:
        """Map curvature to a bounded desired velocity."""
        k = curvature.clamp(min=1e-6)
        v_curvature = self.omega_max / k
        return v_curvature.clamp(min=self.min_desired_v, max=self.ref_desired_v)

    def compute_desired_v(self, traj: torch.Tensor) -> torch.Tensor:
        """按轨迹总长度与曲率分别算期望速度，取两者较小值。"""
        desired_v_length = self.desired_v_from_trajectory_length(self.calculate_length(traj))
        desired_v_curvature = self.desired_v_from_curvature(self.calculate_max_curvature(traj))
        return torch.minimum(desired_v_length, desired_v_curvature)

    def apply_trajectory_kinematic_correction(self, traj: torch.Tensor) -> torch.Tensor:
        """按轨迹总长度与曲率算期望速度与最小转弯半径，并投影到圆外。"""
        desired_v = self.compute_desired_v(traj)
        return self.correct_trajectory_outside_circles(traj, desired_v=desired_v)

    def forward(self, observations, predict_trajectory, embodiment, t=None, is_critic=True):
        """Dispatch to critic Q prediction or actor denoising."""
        if is_critic:
            q1, q2 = self.generate_q_value(observations, predict_trajectory, embodiment, is_target=False)
            return q1, q2
        else:
            noise_pred = self.predict_noise_obs(observations, predict_trajectory, t, embodiment)
            return noise_pred

    def _encode_rgbd(self, observations, rgb_images, depth_images):
        """``depth_mask`` 仅为 ``True`` / ``False`` / 缺省：为 ``True`` 时整批在 fusion 中屏蔽 depth 段（RGB 与 depth 的 patch 数可不同，例如 8 帧 RGB + 单张 depth）。"""
        depth_mask = observations.get("depth_mask")
        if depth_mask is not True:
            return self.rgbd_encoder(rgb_images, depth_images, memory_mask=None)

        patches_per_image = 256
        if len(rgb_images.shape) == 4:
            num_rgb_tokens = patches_per_image
        else:
            num_rgb_tokens = rgb_images.shape[1] * patches_per_image
        if len(depth_images.shape) == 4:
            num_depth_tokens = patches_per_image
        else:
            num_depth_tokens = depth_images.shape[1] * patches_per_image

        num_fusion_queries = self.rgbd_encoder.memory_size * 16
        num_fusion_keys = num_rgb_tokens + num_depth_tokens
        fusion_memory_mask = rgb_images.new_zeros((num_fusion_queries, num_fusion_keys))
        fusion_memory_mask[:, num_rgb_tokens:] = float("-inf")
        return self.rgbd_encoder(
            rgb_images, depth_images, memory_mask=fusion_memory_mask,
        )

    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed):
        """Predict diffusion noise with the base decoder path."""
        action_embeds = self.input_embed(last_actions)
        if last_actions.shape[0] != timestep.shape[0]:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        else:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1)
        cond_embedding = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_noise_ft(self,last_actions,timestep,goal_embed,rgbd_embed,embodiment):
        """Predict diffusion noise with the fine-tuned embodiment-aware decoder."""
        action_embeds = self.input_embed_ft(last_actions)
        if last_actions.shape[0] != timestep.shape[0]:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        else:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1)
        base_cond = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        embody_vec = None
        cond_embedding = base_cond
        if self.distinguish_embodiment:
            embody_idx = self._align_embody_idx(embodiment, input_embedding.shape[0])
            embody_vec = self.embodiment_embedding(embody_idx)
            embody_embed = embody_vec.unsqueeze(1)
            cond_embedding = torch.cat([cond_embedding, embody_embed], dim=1)
            input_embedding = input_embedding + self.embody_tgt_delta(embody_vec).unsqueeze(1)
        output = self.decoder_ft(tgt = input_embedding, memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm_ft(output)
        if embody_vec is not None:
            dgamma, dbeta = self.embody_out_film(embody_vec).chunk(2, dim=-1)
            output = (1.0 + dgamma.unsqueeze(1)) * output + dbeta.unsqueeze(1)
        output = self.action_head_ft(output)
        return output

    def calculate_length(self, points):
        """Compute planar path length after prepending the origin."""
        batch = points.shape[0]
        points_with_zero = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points_np = points_with_zero.cpu().numpy()
        lengths = []
        for i in range(batch):
            points_t = points_np[i, :, :2].T
            lengths.append(float(np.sum(np.hypot(np.diff(points_t[0]), np.diff(points_t[1])))))
        return torch.tensor(lengths, dtype=torch.float32, device=self.device)

    def _make_line_trajectory(
        self,
        reference_action: torch.Tensor,
        length_min: float | None = None,
        length_max: float | None = None,
    ) -> torch.Tensor:
        """沿 x 轴生成无 y 偏差的累积路径直线，总弧长在 [length_min, length_max] 米。"""
        batch_size, horizon, _ = reference_action.shape
        device = reference_action.device
        dtype = reference_action.dtype
        if length_min is None:
            length_min = self.keep_line_len_min
        if length_max is None:
            length_max = self.keep_line_len_max

        lengths = torch.empty(batch_size, device=device, dtype=dtype).uniform_(length_min, length_max)
        t = torch.linspace(1.0 / horizon, 1.0, horizon, device=device, dtype=dtype).view(1, horizon)
        line_actions = torch.zeros(batch_size, horizon, 3, device=device, dtype=dtype)
        line_actions[:, :, 0] = lengths.view(batch_size, 1) * t
        return line_actions

    def smooth_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        """对轨迹做样条平滑；可选返回 xy 平面上的路径长度（平滑后折线弧长，与 x/y 同量纲）。"""
        batch = points.shape[0]
        points_with_zero = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points_np = points_with_zero.cpu().numpy()
        data = []
        for i in range(batch):
            points_t = points_np[i, :, :2].T
            try:
                tck, u = splprep(points_t, w=weight, s=smooth_factor, k=3)
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                res = np.column_stack((x_new, y_new, points_np[i, :, 2:]))[1:]
            except ValueError:
                print('smooth failed')
                res = points_np[i, 1:, :]
            data.append(res)
        out = torch.tensor(np.stack(data, axis=0), dtype=torch.float32).to(self.device)

        return out

    def correct_trajectory_outside_circles(self, traj, desired_v=None):
        """Project trajectory points outside embodiment-derived turning circles."""

        corrected_traj = traj.clone()
        xy = corrected_traj[:, :, :2]
        batch_size = traj.shape[0]
        device = traj.device
        dtype = traj.dtype

        if desired_v is None:
            desired_v = self.compute_desired_v(traj)
        else:
            desired_v = torch.as_tensor(desired_v, device=device, dtype=dtype)
            if desired_v.dim() == 0:
                desired_v = desired_v.expand(batch_size)
            elif desired_v.shape[0] != batch_size:
                raise ValueError(
                    f"desired_v batch {desired_v.shape[0]} != traj batch {batch_size}"
                )

        radius = self.min_turn_radius_from_desired_v(desired_v) * 0.85
        center1 = torch.stack(
            [torch.zeros(batch_size, device=device, dtype=dtype), radius], dim=-1
        ).unsqueeze(1)
        center2 = torch.stack(
            [torch.zeros(batch_size, device=device, dtype=dtype), -radius], dim=-1
        ).unsqueeze(1)
        r_view = radius.view(batch_size, 1, 1)

        dist1 = torch.norm(xy - center1, dim=-1)
        inside_circle1 = dist1 < radius.unsqueeze(-1)
        if inside_circle1.any():
            direction1_unit = (xy - center1) / dist1.unsqueeze(-1).clamp(min=1e-8)
            corrected_points1 = center1 + direction1_unit * r_view
            corrected_traj[:, :, :2] = torch.where(
                inside_circle1.unsqueeze(-1), corrected_points1, corrected_traj[:, :, :2]
            )
            xy = corrected_traj[:, :, :2]

        dist2 = torch.norm(xy - center2, dim=-1)
        inside_circle2 = dist2 < radius.unsqueeze(-1)
        if inside_circle2.any():
            direction2_unit = (xy - center2) / dist2.unsqueeze(-1).clamp(min=1e-8)
            corrected_points2 = center2 + direction2_unit * r_view
            corrected_traj[:, :, :2] = torch.where(
                inside_circle2.unsqueeze(-1), corrected_points2, corrected_traj[:, :, :2]
            )

        return corrected_traj

    @staticmethod
    def _path_cumulative_to_action_deltas(pos: torch.Tensor) -> torch.Tensor:
        """``pos = cumsum(seg / 4, dim=1)`` 时还原每步 ``seg``，形状与 ``pos`` 相同。"""
        seg = torch.zeros_like(pos)
        seg[:, 0, :] = pos[:, 0, :] * 4.0
        seg[:, 1:, :] = (pos[:, 1:, :] - pos[:, :-1, :]) * 4.0
        return seg

    @staticmethod
    def _action_deltas_to_path_cumulative(seg: torch.Tensor) -> torch.Tensor:
        """``torch.cumsum(seg / 4, dim=1)``，与训练里路径表示一致。"""
        return torch.cumsum(seg / 4.0, dim=1)

    def _truncate_origin_action_after_first_x_segment(
        self,
        origin_action: torch.Tensor,
        *,
        batch_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
        scale_tail_vs_segment: float = 0.1,
        enable_negative_traj_length_scale: bool = True,
    ) -> torch.Tensor:
        """
        **总程度缩放**（仅当 ``enable_negative_traj_length_scale`` 为真）：凡首步 ``delta_x`` 为负
        的轨迹，在 **当前** xy delta 上算折线总长 ``L = sum(||dxy||)/4``（米制近似）。**仅当**
        ``L>0.9`` 米时，在 ``(0.1,0.9)`` 米上均匀采样目标 ``T``，将 xy 整体乘以 ``T/L``（z 不变）；
        ``L <= 0.9`` 米则不缩放。

        逻辑在 **每步 delta（段）** 上执行。先将 ``cumsum(seg/4)`` 路径转为 delta，处理后再
        转回累积路径供 ``smooth_trajectory``。若 ``batch_mask`` 非空，仅对掩码为 ``True``
        的 batch 维度处理，其余轨迹不变。
        """
        out = self._path_cumulative_to_action_deltas(origin_action)
        batch, horizon, _ = out.shape
        if batch_mask is None:
            process = None
        else:
            batch_mask = batch_mask.to(device=out.device, dtype=torch.bool)
            assert batch_mask.shape == (batch,), (
                f"batch_mask must be shape ({batch},), got {tuple(batch_mask.shape)}"
            )
            process = batch_mask
        for batch_idx in range(batch):
            if process is not None and not bool(process[batch_idx].item()):
                continue
            delta_x = out[batch_idx, :, 0]
            first_x_sign = int(torch.sign(delta_x[0]).item())
            if first_x_sign >= 0:
                continue
            reverse_at = None
            for t in range(1, horizon):
                step_x_sign = int(torch.sign(delta_x[t]).item())
                if step_x_sign != first_x_sign:
                    reverse_at = t
                    break
            if reverse_at is not None:
                first_segment_xy = out[batch_idx, :reverse_at, :2]
                segment_sum_xy = first_segment_xy.sum(0)
                sum_norm = torch.linalg.norm(segment_sum_xy)
                if sum_norm > eps:
                    unit_xy = segment_sum_xy / sum_norm
                else:
                    unit_xy = delta_x.new_tensor([first_x_sign, 0.0])

                mean_step_len = torch.linalg.norm(first_segment_xy, dim=-1).mean().clamp(min=eps)
                tail_step_len = scale_tail_vs_segment * mean_step_len
                small_step_xy = unit_xy * tail_step_len
                out[batch_idx, reverse_at:, 0] = small_step_xy[0]
                out[batch_idx, reverse_at:, 1] = small_step_xy[1]

            if enable_negative_traj_length_scale:
                dxy = out[batch_idx, :, :2]
                path_len = torch.linalg.norm(dxy, dim=-1).sum().clamp(min=eps) / 4.0
                if path_len > 1.5:
                    target_len = torch.empty(1, device=out.device, dtype=out.dtype).uniform_(0.1, 1.5)
                    scale = target_len / path_len
                    out[batch_idx, :, 0] *= scale
                    out[batch_idx, :, 1] *= scale

        out = self._action_deltas_to_path_cumulative(out)
        return out

    def generate_action_mix(self, observations, num_samples=1, is_nogoal_perturb=False, is_old_policy=False, embodiment='dingo'):
        """Generate goal-conditioned actions and optional no-goal perturbations."""

        rgb_images = torch.as_tensor(observations['rgb'],device=self.device)
        depth_images = torch.as_tensor(observations['depth'],device=self.device)
        rgbd_embed = self._encode_rgbd(observations, rgb_images, depth_images)
        if 'pointgoal' in observations.keys():
            goal = torch.as_tensor(observations['pointgoal'],device=self.device)
            goal_embed = self.point_encoder(goal).unsqueeze(1)
        rgbd_embed = rgbd_embed.detach()
        goal_embed = goal_embed.detach()

        bsz = rgb_images.shape[0]
        rgbd_embed = torch.repeat_interleave(rgbd_embed, num_samples, dim=0)
        goal_embed = torch.repeat_interleave(goal_embed, num_samples, dim=0)
        embody_for_ft = torch.as_tensor(embodiment, device=self.device, dtype=torch.long)
        if embody_for_ft.dim() == 0:
            embody_for_ft = embody_for_ft.expand(bsz)
        if embody_for_ft.shape[0] == bsz:
            embody_for_ft = torch.repeat_interleave(embody_for_ft, num_samples, dim=0)
        elif embody_for_ft.shape[0] != bsz * num_samples:
            raise ValueError(
                f"embodiment batch {embody_for_ft.shape[0]} must be {bsz} or {bsz * num_samples} "
                f"(rgb batch={bsz}, num_samples={num_samples})"
            )
        naction = torch.randn((bsz * num_samples, self.predict_size, 3), device=self.device)
        for k in self.noise_scheduler.timesteps[:]:
            if k >= self.ft_step or is_old_policy:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),goal_embed,rgbd_embed)
            else:
                noise_pred = self.predict_noise_ft(naction,k.unsqueeze(0),goal_embed,rgbd_embed,embodiment=embody_for_ft)
            naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
        origin_action = naction.clone()

        origin_action = torch.cumsum(origin_action/4.0, dim=1)
        origin_action = self._truncate_origin_action_after_first_x_segment(origin_action, enable_negative_traj_length_scale=False)
        origin_action = self.smooth_trajectory(origin_action, num_points=25, smooth_factor=0.2, weight=self.weight)
        origin_action = self._path_cumulative_to_action_deltas(origin_action)

        if not is_nogoal_perturb:
            return origin_action.detach(), None

        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        nogoal_embed = nogoal_embed.detach()
        noisy_action = torch.randn((bsz * num_samples, self.predict_size, 3), device=self.device)
        naction = noisy_action.clone()
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.predict_noise(naction,k.unsqueeze(0),nogoal_embed,rgbd_embed)
            naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
        nogoal_actions = naction.clone()

        origin_action = torch.cumsum(origin_action/4.0, dim=1)
        nogoal_actions = torch.cumsum(nogoal_actions/4.0, dim=1)
        nogoal_actions = self.smooth_trajectory(nogoal_actions, num_points=25, smooth_factor=0.2, weight=self.weight)

        # 第8个点的x方向相反时，反转nogoal_actions
        nogoal_dir_x = nogoal_actions[:, 1, 0]  # (B*num_samples,)
        origin_dir_x = origin_action[:, 1, 0]  # 与首轮 num_samples 条独立 goal 轨迹一一对应
        is_opposite_x = (origin_dir_x * nogoal_dir_x) < 0
        nogoal_actions[is_opposite_x, :, 0] = -nogoal_actions[is_opposite_x, :, 0]

        len_origin = self.calculate_length(origin_action)
        len_nogoal = self.calculate_length(nogoal_actions)
        final_actions = origin_action.clone()

        line_actions = self._make_line_trajectory(origin_action)
        is_keep_line = torch.rand(final_actions.shape[0], device=final_actions.device) < self.keep_line_prob
        final_actions = torch.where(
            is_keep_line.unsqueeze(-1).unsqueeze(-1),
            line_actions,
            final_actions,
        )

        length_ratio = (len_origin / len_nogoal.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)
        u_sign = torch.rand(final_actions.shape[0], device=final_actions.device)
        u_mag = torch.rand(final_actions.shape[0], device=final_actions.device)
        if is_old_policy:
            mag = u_mag * 0.5
        else:
            mag = u_mag * 0.5
        sign = torch.where(u_sign < 0.5, -1.0, 1.0)
        ratio_origin = sign * mag
        ratio_origin = ratio_origin.unsqueeze(1).unsqueeze(2)  # (B*num_samples, 1, 1)
        final_actions = final_actions + ratio_origin * length_ratio * nogoal_actions

        is_keep_nogoal = torch.rand(final_actions.shape[0], device=final_actions.device) < self.keep_nogoal_prob
        final_actions = torch.where(
            is_keep_nogoal.unsqueeze(-1).unsqueeze(-1),
            nogoal_actions,
            final_actions,
        )

        is_reverse_x = (torch.rand(final_actions.shape[0]) < self.reverse_prob_x).to(final_actions.device)
        is_reverse_y = (torch.rand(final_actions.shape[0]) < self.reverse_prob_y).to(final_actions.device)
        final_actions[is_reverse_x, :, 0] = -final_actions[is_reverse_x, :, 0]
        final_actions[is_reverse_y, :, 1] = -final_actions[is_reverse_y, :, 1]

        _sx = torch.empty(
            final_actions.shape[0],
            device=final_actions.device,
            dtype=final_actions.dtype,
        ).uniform_(0.75, 1.25)
        _sy = torch.empty(
            final_actions.shape[0],
            device=final_actions.device,
            dtype=final_actions.dtype,
        ).uniform_(0.75, 1.25)
        final_actions[..., :, 0] = _sx.unsqueeze(-1) * final_actions[..., :, 0]
        final_actions[..., :, 1] = _sy.unsqueeze(-1) * final_actions[..., :, 1]

        is_keep_origin = torch.rand(final_actions.shape[0], device=final_actions.device) < self.keep_origin_prob
        final_actions = torch.where(
            is_keep_origin.unsqueeze(-1).unsqueeze(-1),
            origin_action,
            final_actions,
        )

        do_truncate = torch.rand(final_actions.shape[0], device=final_actions.device) < self.truncate_prob
        final_actions = self._truncate_origin_action_after_first_x_segment(final_actions, batch_mask=do_truncate, enable_negative_traj_length_scale=True)
        final_actions = self.apply_trajectory_kinematic_correction(final_actions)
        final_actions = self.smooth_trajectory(final_actions, num_points=25, smooth_factor=0.2, weight=self.weight)
        final_actions = self._path_cumulative_to_action_deltas(final_actions)
        origin_action = self._path_cumulative_to_action_deltas(origin_action)

        return origin_action.detach(), final_actions.detach()

    def generate_q_value(self, observations, predict_trajectory, embodiment, is_target=False):
        """Evaluate trajectories with the online or target twin-Q heads."""
        rgb_images = torch.as_tensor(observations['rgb'],device=self.device)
        depth_images = torch.as_tensor(observations['depth'],device=self.device)
        rgbd_embed = self._encode_rgbd(observations, rgb_images, depth_images)
        if 'pointgoal' in observations.keys():
            goal = torch.as_tensor(observations['pointgoal'],device=self.device)
            goal_embed = self.point_encoder(goal).unsqueeze(1)

        reverse_mc_num = int(predict_trajectory.shape[0] / rgbd_embed.shape[0])
        rgbd_embed = torch.repeat_interleave(rgbd_embed,reverse_mc_num,dim=0)
        goal_embed = torch.repeat_interleave(goal_embed,reverse_mc_num,dim=0)

        q1_value, q2_value = self.predict_pointgoal_q(predict_trajectory.detach(),rgbd_embed,goal_embed, is_target, embodiment)
        return q1_value, q2_value

    def select_best_action_by_q(
        self, observations, candidate_actions, num_samples, embodiment, is_target=False,
    ):
        """从 (B*num_samples, T, C) 候选中按 min(Q1,Q2) 为每个 batch 选 Q 最高的一条，返回 (B, T, C)。"""
        bsz = torch.as_tensor(observations['rgb']).shape[0]
        q1, q2 = self.generate_q_value(
            observations, candidate_actions, embodiment=embodiment, is_target=is_target,
        )
        q = torch.minimum(q1, q2).reshape(bsz, num_samples)
        best_idx = torch.argmax(q, dim=1)
        cand = candidate_actions.reshape(bsz, num_samples, *candidate_actions.shape[1:])
        batch_idx = torch.arange(bsz, device=cand.device)
        return cand[batch_idx, best_idx]

    def _align_embody_idx(self, embodiment, batch_size):
        """Broadcast embodiment ids to match a sampled batch size."""
        embody_idx = torch.as_tensor(embodiment, device=self.device).long()
        if embody_idx.dim() == 0:
            embody_idx = embody_idx.expand(batch_size)
        elif embody_idx.shape[0] != batch_size:
            reverse_mc_num = batch_size // embody_idx.shape[0]
            embody_idx = embody_idx.repeat_interleave(reverse_mc_num)
        return embody_idx.clamp(0, self.embodiment_embedding.num_embeddings - 1)

    def _pool_critic_output(self, critic_output: torch.Tensor) -> torch.Tensor:
        """用 MLP 汇聚整条轨迹 token，替代 mean pooling。"""
        return self.q_pool_mlp(critic_output.flatten(start_dim=1))

    def _q_heads_forward(self, pooled, is_target):
        """pooled: [B, token_dim]。所有 embodiment 共用同一组 Q 头。"""
        heads_q1 = self.q1_target_heads if is_target else self.q1_heads
        heads_q2 = self.q2_target_heads if is_target else self.q2_heads
        q1 = heads_q1(pooled).squeeze(-1)
        q2 = heads_q2(pooled).squeeze(-1)
        return q1, q2

    def predict_pointgoal_q(self, predict_trajectory, rgbd_embed, point_embed, is_target, embodiment):
        """Predict twin Q-values for point-goal conditioned trajectories."""
        action_embeddings = self.input_embed_q(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([point_embed,point_embed,point_embed,point_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([point_embed,point_embed,point_embed,point_embed,rgbd_embed],dim=1))
        embody_idx = self._align_embody_idx(embodiment, cond_embeddings.shape[0])
        cond_critic_mask = self.cond_critic_mask_no_embod
        if self.distinguish_embodiment:
            e = self.embodiment_embedding(embody_idx)
            embody_embed = e.unsqueeze(1)
            cond_embeddings = torch.cat([cond_embeddings, embody_embed], dim=1)
            cond_critic_mask = self.cond_critic_mask_one_embod
            action_embeddings = action_embeddings + self.embody_tgt_delta(e).unsqueeze(1)
        critic_output = self.decoder_q(tgt = action_embeddings, memory = cond_embeddings, memory_mask = cond_critic_mask.to(self.device))
        critic_output = self.layernorm_q(critic_output)
        if self.distinguish_embodiment:
            dgamma, dbeta = self.embody_out_film(e).chunk(2, dim=-1)
            critic_output = (1.0 + dgamma.unsqueeze(1)) * critic_output + dbeta.unsqueeze(1)

        pooled = self._pool_critic_output(critic_output)
        critic_output_q1, critic_output_q2 = self._q_heads_forward(pooled, is_target)

        return critic_output_q1, critic_output_q2

    def q_sample(self, t: int, x_start: torch.Tensor, noise: torch.Tensor):
        """Apply the forward diffusion process at timestep ``t``."""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod[t].to(x_start.device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).unsqueeze(1).unsqueeze(2)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

    def predict_noise_obs(self, observations, naction, t, embodiment):
        """Encode observations and predict fine-tuned diffusion noise."""
        rgb_images = torch.as_tensor(observations['rgb'],device=self.device)
        depth_images = torch.as_tensor(observations['depth'],device=self.device)
        rgbd_embed = self._encode_rgbd(observations, rgb_images, depth_images)
        if 'pointgoal' in observations.keys():
            goal = torch.as_tensor(observations['pointgoal'],device=self.device)
            goal_embed = self.point_encoder(goal).unsqueeze(1)

        reverse_mc_num = int(naction.shape[0] / rgbd_embed.shape[0])
        t = t.detach()
        rgbd_embed = torch.repeat_interleave(rgbd_embed, repeats=reverse_mc_num, dim=0).detach()
        goal_embed = torch.repeat_interleave(goal_embed,repeats=reverse_mc_num,dim=0).detach()
        embodiment_embed = torch.repeat_interleave(embodiment,repeats=reverse_mc_num,dim=0).detach()
        noise_pred = self.predict_noise_ft(naction,t,goal_embed,rgbd_embed,embodiment_embed)

        return noise_pred
