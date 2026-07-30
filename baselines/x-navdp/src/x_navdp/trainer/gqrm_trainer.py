"""Offline GQRM trainer with critic, actor, and entropy updates."""

import torch
from torch import nn
from argparse import Namespace
from itertools import zip_longest
from collections.abc import Iterable
import numpy as np
import math
import random


def _random_zero_depth(obs, p):
    """按 batch 每条样本以概率 ``p`` 将 ``obs`` 中的 depth 置零。"""
    d = obs.get("depth")
    if d is None or p <= 0.0:
        return obs
    b = np.asarray(d).shape[0]
    mask = np.random.rand(b) < p
    d = np.asarray(d).copy()
    d[mask] = 0
    obs["depth"] = d
    return obs


class GQRMTrainer:
    """Trainer that updates the X-NavDP critic, actor denoiser, and entropy scale."""
    def __init__(self,
                 config:Namespace,
                 policy: nn.Module,
                 local_rank: 0,
                 device):
        """Create optimizers, schedules, and SAC-style training hyperparameters."""
        self.config = config
        self.policy = policy
        self.local_rank = local_rank
        self.critic_optimizer = torch.optim.Adam(self.policy.module.critic_params, self.config.learning_rate, eps=2e-5)
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.config.end_factor_lr_decay,
                                                           total_iters=self.config.total_iters)

        self.actor_optimizer = torch.optim.Adam(self.policy.module.actor_params, self.config.learning_rate, eps=2e-5)
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer,
                                                                 start_factor=1.0,
                                                                 end_factor=self.config.end_factor_lr_decay,
                                                                 total_iters=self.config.total_iters)
        self.ent_optimizer = torch.optim.Adam([self.policy.module.log_alpha], lr=2e-3)
        self.mse_loss = nn.MSELoss()

        self.device = device
        self.alpha_delay_update = 100
        self.target_delay_update = 2
        self.delay_update = 2
        self.start_learn = 100

        self.act_dim = 24
        self.target_entropy = -self.act_dim*2.0
        self.gamma = 0.99
        self.tau = 0.005

        self.num_samples = 64
        self.topk = 5
        self.ft_train = 4
        self.norm_scale = 2
        self.gradient_accumulate = 1

        self.iterations = 0


    def update(self, step, samples, is_stage2=False):
        """Run one off-policy update from replay samples."""

        self.iterations += 1
        obs_batch, next_obs_batch = samples['obs'], samples['next_obs']
        done_batch = torch.as_tensor(samples['terminals'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        reward_batch = torch.as_tensor(samples['rewards'], device=self.device)
        embodiment_batch = torch.as_tensor(samples['embodiment'], device=self.device)

        # Depth dropout: batch 内按样本 Bernoulli(p) 遮掩 obs 与 next_obs（各自独立采样）。
        p = float(getattr(self.config, "depth_mask_prob", 0.0))
        if p > 0.0 and obs_batch.get("rgb") is not None:
            obs_batch = _random_zero_depth(obs_batch, p)

        # update critic
        with torch.no_grad():
            next_act_batch, _ = self.policy.module.generate_action_mix(next_obs_batch, num_samples=1, is_nogoal_perturb=False, is_old_policy=False, embodiment=embodiment_batch)
            q1_target, q2_target = self.policy.module.generate_q_value(next_obs_batch, next_act_batch, embodiment=embodiment_batch, is_target=True)

            q_target = torch.minimum(q1_target, q2_target)
            q_backup = reward_batch + (1 - done_batch) * self.gamma * q_target

        q1, q2 = self.policy(obs_batch, act_batch, embodiment_batch)
        value_loss = self.mse_loss(q_backup, q1) + self.mse_loss(q_backup, q2)
        value_loss.backward()

        if step % self.gradient_accumulate == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.module.critic_params, 5.0)
            self.critic_optimizer.step()
            self.critic_scheduler.step()
            self.critic_optimizer.zero_grad()

        if step % (self.target_delay_update * self.gradient_accumulate) == 0:
            self.polyak_update(
                self.policy.module.q1_heads.parameters(),
                self.policy.module.q1_target_heads.parameters(),
                self.tau,
            )
            self.polyak_update(
                self.policy.module.q2_heads.parameters(),
                self.policy.module.q2_target_heads.parameters(),
                self.tau,
            )

        value = torch.mean(torch.minimum(q1, q2))

        # update actor
        policy_loss = torch.tensor(0.0).to(self.device)
        policy_loss_output = torch.tensor(0.0).to(self.device)
        log_alpha_loss = torch.tensor(0.0).to(self.device)

        if ((step - 1) // self.gradient_accumulate) % (self.delay_update) == 0 and step > (self.start_learn * self.gradient_accumulate):
            with torch.no_grad():
                _, perturb_act_batch = self.policy.module.generate_action_mix(next_obs_batch, num_samples=self.num_samples, is_nogoal_perturb=True, is_old_policy=False, embodiment=embodiment_batch)
                q1_next, q2_next = self.policy.module.generate_q_value(next_obs_batch, perturb_act_batch, embodiment=embodiment_batch, is_target=False)
                q_next = torch.minimum(q1_next, q2_next)
                q_next = q_next.reshape(-1, self.num_samples)
                _, topk_indices = torch.topk(q_next, k=self.topk, dim=1)

                # q_mean = q_next.mean(dim=1, keepdim=True)
                # q_std = q_next.std(dim=1, keepdim=True).clamp_min(1e-6)

                q_mean = q_next.mean(dim=-1).unsqueeze(1) # better
                q_std = q_next.std()

                norm_q = self.norm_scale * (q_next - q_mean) / q_std
                scaled_q = norm_q.clip(-3., 3.) / torch.exp(self.policy.module.log_alpha)
                q_weights = torch.exp(scaled_q)
                q_weights[norm_q <= 0] = 0

                q_weights = q_weights.gather(dim=1, index=topk_indices)
                perturb_act_batch = perturb_act_batch.reshape(-1, self.num_samples, 24, perturb_act_batch.shape[-1])
                perturb_act_batch = perturb_act_batch.gather(dim=1, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 24, perturb_act_batch.shape[-1]))
                perturb_act_batch = perturb_act_batch.reshape(-1, 24, perturb_act_batch.shape[-1])
                weights = q_weights.reshape(-1, 1, 1)

                x_noisy = []
                noise_list = []
                noise = torch.normal(mean=0.0, std=1.0, size=perturb_act_batch.shape).to(self.device)
                for t in range(0, self.policy.module.ft_step):
                    noise_list.append(noise)
                    k = torch.full((perturb_act_batch.shape[0],), t, dtype=torch.int64)
                    x_noisy.append(self.policy.module.q_sample(k, perturb_act_batch, noise))

            p = float(getattr(self.config, "depth_mask_prob", 0.0))
            if p > 0.0 and obs_batch.get("rgb") is not None:
                next_obs_batch = _random_zero_depth(next_obs_batch, p)

            for t in range(0, self.policy.module.ft_step):
                k = torch.full((perturb_act_batch.shape[0],), t, dtype=torch.int64)
                noise_pred = self.policy(next_obs_batch, x_noisy[t], embodiment_batch, k, is_critic=False)
                loss_t = (weights * torch.square(noise_pred - noise_list[t])).mean()
                loss_t.backward()
                policy_loss_output += loss_t.detach()

            if step % self.gradient_accumulate == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.module.actor_params, 5.0)
                self.actor_optimizer.step()
                self.actor_scheduler.step()
                self.actor_optimizer.zero_grad()

        if step % (self.alpha_delay_update * self.gradient_accumulate) == 0 and step > (self.start_learn * self.gradient_accumulate):
            approx_entropy = 0.5 * self.act_dim * torch.log( 2 * torch.pi * torch.exp(torch.tensor(1)) * (0.1 * torch.exp(self.policy.module.log_alpha)) ** 2)
            log_alpha_loss = -1 * self.policy.module.log_alpha * (-1 * approx_entropy.detach() + self.target_entropy)
            self.ent_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.ent_optimizer.step()

        return {'iteration':self.iterations,
                'policy_loss':policy_loss_output.item(),
                'value_loss':value_loss.item(),
                'value':value,
                'log_alpha':self.policy.module.log_alpha,
                'log_alpha_loss':log_alpha_loss.item()}


    def polyak_update(self,params: Iterable[torch.Tensor],target_params: Iterable[torch.Tensor],tau: float,) -> None:
        """
        Perform a Polyak average update on ``target_params`` using ``params``:
        target parameters are slowly updated towards the main parameters.
        ``tau``, the soft update coefficient controls the interpolation:
        ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
        The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
        or a computation graph, reducing memory cost and improving performance.  We scale the target params
        by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
        params (in place).
        See https://github.com/DLR-RM/stable-baselines3/issues/93

        :param params: parameters to use to update the target params
        :param target_params: parameters to update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
        """

        def zip_strict(*iterables: Iterable) -> Iterable:
            r"""
            ``zip()`` function but enforces that iterables are of equal length.
            Raises ``ValueError`` if iterables not of equal length.
            Code inspired by Stackoverflow answer for question #32954486.

            :param \*iterables: iterables to ``zip()``
            """
            # As in Stackoverflow #32954486, use
            # new object for "empty" in case we have
            # Nones in iterable.
            sentinel = object()
            for combo in zip_longest(*iterables, fillvalue=sentinel):
                if sentinel in combo:
                    raise ValueError("Iterables have different lengths")
                yield combo

        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in zip_strict(params, target_params):
                target_param.data.mul_(1 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
