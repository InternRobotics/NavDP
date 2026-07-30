"""
NavDP Policy Network with Embodiment Modulation.

This module implements the diffusion-based policy network that predicts
navigation trajectories conditioned on RGB-D observations and goals.
Supports multiple embodiments (wheeled, humanoid, quadruped).
"""

import torch
import torch.nn as nn
import math
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.interpolate import splprep, splev

EMBODIMENT_NAME_TO_IDX = {"dingo": 0, "unitree_g1": 1, "unitree_go2": 2}


def get_prefix_weights(start: int, end: int, total: int, schedule: str, device='cpu') -> torch.Tensor:
    """Compute attention weights for prefix guidance scheduling."""
    n = end.shape[0]
    end = end.reshape(n, 1).float()
    idx = torch.arange(total, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)
    mid = (start + (end - start) / 3).long()

    if schedule == "ones":
        w = (idx < end).float()
    elif schedule == "zeros":
        w = torch.where(idx < end, (idx < start).float(), torch.zeros_like(idx))
    elif schedule in ("linear", "exp"):
        w = torch.zeros_like(idx)
        w = torch.where((idx >= start) & (idx < mid), 1.0, w)
        r = (end - mid).clamp(min=1e-8)
        decr = ((mid - idx) / r + 1).clamp(0.0, 1.0)
        if schedule == "exp":
            decr = decr * decr.expm1() / (torch.e - 1)
        w = torch.where((idx >= mid) & (idx < end), decr, w)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return torch.stack([w, w, torch.zeros_like(w)], dim=-1)


class NavDP_Policy_Embodiment(nn.Module):
    """Diffusion-based navigation policy with embodiment modulation."""

    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 ft_denoise_step=4,
                 is_original=False,
                 distinguish_embodiment=True,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        self.is_original = is_original
        self.distinguish_embodiment = distinguish_embodiment

        try:
            from .policy_backbone import (
                NavDP_RGBD_Backbone,
                SinusoidalPosEmb,
                LearnablePositionalEncoding,
            )
        except ImportError:
            from policy_backbone import (
                NavDP_RGBD_Backbone,
                SinusoidalPosEmb,
                LearnablePositionalEncoding,
            )

        self.rgbd_encoder = NavDP_RGBD_Backbone(
            image_size, token_dim, memory_size=memory_size, device=device
        )
        self.point_encoder = nn.Linear(3, token_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.temporal_depth
        )

        self.input_embed = nn.Linear(3, token_dim)
        self.cond_pos_embed = LearnablePositionalEncoding(
            token_dim, memory_size * 16 + 4
        )
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(
            self.tgt_mask == 0, float('-inf')
        ).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask_no_embod = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask_no_embod[:, 0:1] = float('-inf')
        self.cond_critic_mask_one_embod = torch.zeros((predict_size, 5 + memory_size * 16))
        self.cond_critic_mask_one_embod[:, 0:1] = float('-inf')

        self.embodiment_embedding = nn.Embedding(3, token_dim)
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

        self.decoder_ft = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.temporal_depth
        )
        self.layernorm_ft = nn.LayerNorm(token_dim)
        self.action_head_ft = nn.Linear(token_dim, 3)
        self.input_embed_ft = nn.Linear(3, token_dim)

        self.decoder_q_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            activation='gelu',
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.decoder_q = nn.TransformerDecoder(
            decoder_layer=self.decoder_q_layer,
            num_layers=self.temporal_depth
        )
        self.layernorm_q = nn.LayerNorm(token_dim)
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(predict_size * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.q1_heads = nn.Linear(token_dim, 1)
        self.q2_heads = nn.Linear(token_dim, 1)
        self.input_embed_q = nn.Linear(3, token_dim)

        self.q1_target_heads = nn.Linear(token_dim, 1)
        self.q2_target_heads = nn.Linear(token_dim, 1)
        self.q1_target_heads.load_state_dict(self.q1_heads.state_dict())
        self.q2_target_heads.load_state_dict(self.q2_heads.state_dict())
        for param in self.q1_target_heads.parameters():
            param.requires_grad = False
        for param in self.q2_target_heads.parameters():
            param.requires_grad = False

        self.ft_denoise_step = ft_denoise_step
        self.ft_step = 6
        self.weight = np.ones(self.predict_size + 1)
        self.weight[:9] = 5.0
        self.weight[:5] = 50.0
        self.weight[0] = 100.0

    def _align_embody_idx(self, embodiment, batch_size: int) -> torch.Tensor:
        """Align embodiment indices with batch dimension."""
        embody_idx = torch.as_tensor(embodiment, device=self.device).long()
        if embody_idx.dim() == 0:
            embody_idx = embody_idx.expand(batch_size)
        elif embody_idx.shape[0] != batch_size:
            reverse_mc_num = batch_size // embody_idx.shape[0]
            embody_idx = embody_idx.repeat_interleave(reverse_mc_num)
        return embody_idx.clamp(0, self.embodiment_embedding.num_embeddings - 1)

    def _resolve_embodiment_arg(self, embodiment):
        """Resolve embodiment argument to integer index."""
        if isinstance(embodiment, str):
            if embodiment not in EMBODIMENT_NAME_TO_IDX:
                raise ValueError(
                    f"embodiment should be one of {list(EMBODIMENT_NAME_TO_IDX.keys())}, got {embodiment!r}"
                )
            return EMBODIMENT_NAME_TO_IDX[embodiment]
        return embodiment

    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        """Predict noise for base diffusion step."""
        action_embeds = self.input_embed(last_actions)
        if last_actions.shape[0] != timestep.shape[0]:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0], 1, 1))
        else:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1)
        cond_embedding = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1) + \
            self.cond_pos_embed(torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_noise_ft(self, last_actions, timestep, goal_embed, rgbd_embed, embodiment):
        """Predict noise with fine-tuned decoder for specific embodiment."""
        action_embeds = self.input_embed_ft(last_actions)
        if last_actions.shape[0] != timestep.shape[0]:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0], 1, 1))
        else:
            time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1)
        base_cond = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1) + \
            self.cond_pos_embed(torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        embody_vec = None
        cond_embedding = base_cond
        if self.distinguish_embodiment:
            embody_idx = self._align_embody_idx(embodiment, input_embedding.shape[0])
            embody_vec = self.embodiment_embedding(embody_idx)
            embody_embed = embody_vec.unsqueeze(1)
            cond_embedding = torch.cat([cond_embedding, embody_embed], dim=1)
            input_embedding = input_embedding + self.embody_tgt_delta(embody_vec).unsqueeze(1)
        output = self.decoder_ft(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask.to(self.device))
        output = self.layernorm_ft(output)
        if embody_vec is not None:
            dgamma, dbeta = self.embody_out_film(embody_vec).chunk(2, dim=-1)
            output = (1.0 + dgamma.unsqueeze(1)) * output + dbeta.unsqueeze(1)
        output = self.action_head_ft(output)
        return output

    def _pool_critic_output(self, critic_output: torch.Tensor) -> torch.Tensor:
        """Pool critic output for Q-value computation."""
        return self.q_pool_mlp(critic_output.flatten(start_dim=1))

    def _q_heads_forward(self, pooled, is_target=False):
        """Compute Q-values from pooled critic output."""
        heads_q1 = self.q1_target_heads if is_target else self.q1_heads
        heads_q2 = self.q2_target_heads if is_target else self.q2_heads
        q1 = heads_q1(pooled).squeeze(-1)
        q2 = heads_q2(pooled).squeeze(-1)
        return q1, q2

    def predict_pointgoal_q(self, predict_trajectory, rgbd_embed, point_embed, is_target=False, embodiment=0):
        """Predict Q-values for point goal trajectories."""
        action_embeddings = self.input_embed_q(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat(
            [point_embed, point_embed, point_embed, point_embed, rgbd_embed], dim=1
        ) + self.cond_pos_embed(
            torch.cat([point_embed, point_embed, point_embed, point_embed, rgbd_embed], dim=1)
        )
        cond_critic_mask = self.cond_critic_mask_no_embod
        e = None
        if self.distinguish_embodiment:
            embody_idx = self._align_embody_idx(embodiment, cond_embeddings.shape[0])
            e = self.embodiment_embedding(embody_idx)
            embody_embed = e.unsqueeze(1)
            cond_embeddings = torch.cat([cond_embeddings, embody_embed], dim=1)
            cond_critic_mask = self.cond_critic_mask_one_embod
            action_embeddings = action_embeddings + self.embody_tgt_delta(e).unsqueeze(1)
        critic_output = self.decoder_q(
            tgt=action_embeddings,
            memory=cond_embeddings,
            memory_mask=cond_critic_mask.to(self.device),
        )
        critic_output = self.layernorm_q(critic_output)
        if e is not None:
            dgamma, dbeta = self.embody_out_film(e).chunk(2, dim=-1)
            critic_output = (1.0 + dgamma.unsqueeze(1)) * critic_output + dbeta.unsqueeze(1)
        pooled = self._pool_critic_output(critic_output)
        return self._q_heads_forward(pooled, is_target=is_target)

    def smooth_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        """Smooth trajectory using B-spline interpolation."""
        batch = points.shape[0]
        points = torch.cumsum(points / 4.0, dim=1)
        points = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points = points.cpu().numpy()
        data = []
        for i in range(batch):
            points_t = points[i, :, :2]
            n_pts = points_t.shape[0]
            points_t = points_t.T
            k = min(3, n_pts - 1)
            tck, u = splprep(points_t, w=weight, s=smooth_factor, k=k)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            res = np.column_stack((x_new, y_new, points[i, :, 2:]))
            res = (res[1:, :] - res[:-1, :]) * 4.0
            data.append(res)
        data = torch.tensor(np.stack(data, axis=0)).to(self.device)
        return data.to(dtype=torch.float32)

    def smooth_cumulative_trajectory(self, points, num_points=24, smooth_factor=0.5, weight=None):
        """Smooth cumulative trajectory and return cumulative coordinates."""
        batch = points.shape[0]
        points_with_zero = torch.concat((torch.zeros_like(points[:, 0:1]), points), dim=1)
        points_np = points_with_zero.cpu().numpy()
        data = []
        for i in range(batch):
            points_t = points_np[i, :, :2].T
            n_pts = points_t.shape[1]
            k = min(3, n_pts - 1)
            try:
                tck, u = splprep(points_t, w=weight, s=smooth_factor, k=k)
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                res = np.column_stack((x_new, y_new, points_np[i, :, 2:]))[1:]
            except ValueError:
                res = points_np[i, 1:, :]
            data.append(res)
        return torch.tensor(np.stack(data, axis=0), dtype=torch.float32, device=self.device)

    def pinv_corrected_velocity(self, pred_original_sample, x_t, y,
                                valid_segment_len,
                                inference_delay=0,
                                prefix_attention_horizon=8,
                                prefix_attention_schedule="exp"):
        """Compute guidance signal using pseudo-inverse correction."""
        x_0 = pred_original_sample
        weights = get_prefix_weights(
            start=inference_delay,
            end=valid_segment_len,
            total=self.predict_size,
            schedule=prefix_attention_schedule,
            device=x_t.device
        )
        diff = (y - x_0) * weights
        mat_x = (diff.detach() * x_0).sum()
        g = torch.autograd.grad(mat_x, x_t, retain_graph=False)[0].detach()
        return g

    def predict_pointgoal_action_with_guidance(
        self,
        goal_point,
        input_images,
        input_depths,
        sample_num,
        valid_segment_len,
        prev_action,
        start_index,
        end_index,
        guidance_factor,
        guidance_step=7,
        prefix_attention_schedule="exp",
        embodiment=0,
    ):
        """Predict point goal actions with trajectory guidance."""
        embodiment = self._resolve_embodiment_arg(embodiment)
        batch_size = goal_point.shape[0]

        if isinstance(guidance_factor, np.ndarray):
            if guidance_factor.shape != (sample_num,) and guidance_factor.shape != (batch_size, sample_num):
                raise ValueError(
                    f"guidance_factor shape {guidance_factor.shape} must match sample_num={sample_num} or (batch_size={batch_size}, sample_num={sample_num})"
                )
            guidance_mask = torch.as_tensor(guidance_factor, dtype=torch.float32, device=self.device)
            if guidance_mask.dim() == 1:
                guidance_mask = guidance_mask.unsqueeze(0).expand(batch_size, -1)
            guidance_mask = guidance_mask.reshape(batch_size * sample_num, 1, 1)
        else:
            guidance_mask = guidance_factor

        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            prev_action = torch.as_tensor(prev_action, device=self.device)
            valid_segment_len = torch.as_tensor(valid_segment_len.copy(), device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            goal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)
            prev_action = torch.repeat_interleave(prev_action, sample_num, dim=0).to(self.device)
            valid_segment_len = torch.repeat_interleave(valid_segment_len, sample_num, dim=0).to(self.device)

        naction = torch.randn((sample_num * goal_point.shape[0], self.predict_size, 3), device=self.device)
        for k in self.noise_scheduler.timesteps[:]:
            if k >= self.ft_step:
                naction = naction.requires_grad_(True)
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), goal_embed, rgbd_embed)
            else:
                naction = naction.requires_grad_(True)
                noise_pred = self.predict_noise_ft(
                    naction, k.unsqueeze(0), goal_embed, rgbd_embed, embodiment
                )

            if k <= guidance_step:
                prev_t = self.noise_scheduler.previous_timestep(k)
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[k]
                alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.noise_scheduler.one
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                current_alpha_t = alpha_prod_t / alpha_prod_t_prev
                current_beta_t = 1 - current_alpha_t
                pred_original_sample = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).pred_original_sample
                pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
                current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
                mut = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * naction

                g = self.pinv_corrected_velocity(
                    pred_original_sample, naction, prev_action,
                    valid_segment_len=valid_segment_len,
                    inference_delay=start_index,
                    prefix_attention_horizon=end_index,
                    prefix_attention_schedule=prefix_attention_schedule
                )
                g = g * guidance_mask

                z = torch.randn_like(noise_pred, device=self.device)
                variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
                variance = torch.clamp(variance, min=1e-20)
                naction = (mut + variance ** 0.5 * z + torch.sqrt(alpha_prod_t) * g).detach()
            else:
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample.detach()

        with torch.no_grad():
            final_actions = naction.detach()
            final_actions = self.smooth_trajectory(
                final_actions, num_points=25, smooth_factor=0.5, weight=self.weight
            )
            path_for_q = torch.cumsum(naction / 4.0, dim=1)
            path_for_q = self.smooth_cumulative_trajectory(
                path_for_q, num_points=25, smooth_factor=0.5, weight=self.weight
            )
            q1, q2 = self.predict_pointgoal_q(
                path_for_q, rgbd_embed, goal_embed, is_target=False, embodiment=embodiment
            )
            critic_values = ((q1 + q2) / 2).reshape(goal_point.shape[0], sample_num)

            all_trajectory = torch.cumsum(final_actions / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_point.shape[0], sample_num, self.predict_size, 3)
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), None
