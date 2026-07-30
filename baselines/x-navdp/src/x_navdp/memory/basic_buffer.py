"""Numpy replay buffers used by the X-NavDP trainer."""

import random
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict,Box,Discrete
from collections import deque
from abc import ABC, abstractmethod
from typing import Optional, Union
import scipy.signal
import os
import shutil

def space2shape(observation_space):
    """Convert gym.space variable to shape
    Args:
        observation_space: the space variable with type of gym.Space.

    Returns:
        The shape of the observation_space.
    """
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape

def discount_cumsum(x, discount=0.99):
    """Get a discounted cumulated summation.
    Args:
        x: The original sequence. In DRL, x can be reward sequence.
        discount: the discount factor (gamma), default is 0.99.

    Returns:
        The discounted cumulative returns for each step.

    Examples:
        >>> x = [0, 1, 2, 2]
        >>> y = discount_cumsum(x, discount=0.99)
        [4.890798, 4.9402, 3.98, 2.0]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def create_memory(shape: Optional[Union[tuple, dict]],
                  n_envs: int,
                  n_size: int,
                  dtype: type = np.float32):
    """
    Create a numpy array for memory data.

    Args:
        shape: data shape.
        n_envs: number of parallel environments.
        n_size: length of data sequence for each environment.
        dtype: numpy data type.

    Returns:
        An empty memory space to store data. (initial: numpy.zeros())
    """
    if shape is None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in shape.items():
            if value is None:  # save an object type
                memory[key] = np.zeros([n_envs, n_size], dtype=object)
            else:
                memory[key] = np.zeros([n_envs, n_size] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([n_envs, n_size] + list(shape), dtype)
    else:
        raise NotImplementedError

def store_element(data: Optional[Union[np.ndarray, dict, float]],
                  memory: Union[dict, np.ndarray],
                  ptr: int):
    """
    Insert a step of data into current memory.

    Args:
        data: target data that to be stored.
        memory: the memory where data will be stored.
        ptr: pointer to the location for the data.
    """
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in data.items():
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data

def sample_batch(memory: Optional[Union[np.ndarray, dict]],
                 index: Optional[Union[np.ndarray, tuple]]):
    """
    Sample a batch of data from the selected memory.

    Args:
        memory: memory that contains experience data.
        index: pointer to the location for the selected data.

    Returns:
        A batch of data.
    """
    if memory is None:
        return None
    elif isinstance(memory, dict):
        batch = {}
        for key, value in memory.items():
            batch[key] = value[index]
        return batch
    else:
        return memory[index]


class Buffer(ABC):
    """
    Basic buffer single-agent DRL algorithms.

    Args:
        observation_space: the space for observation data.
        action_space: the space for action data.
        auxiliary_info_shape: the shape for auxiliary data if needed.
    """
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_info_shape: Optional[dict]):
        """Store gym spaces and reset the shared buffer cursor state."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.auxiliary_shape = auxiliary_info_shape
        self.size, self.ptr = 0, 0
    def full(self):
        """Return whether the buffer has enough data to sample."""
        pass
    @abstractmethod
    def store(self, *args):
        """Append one transition or rollout step to the buffer."""
        raise NotImplementedError
    @abstractmethod
    def clear(self, *args):
        """Reset buffer storage and cursors."""
        raise NotImplementedError
    @abstractmethod
    def sample(self, *args):
        """Sample training batches from stored transitions."""
        raise NotImplementedError
    def finish_path(self, *args):
        """Finalize per-env trajectory statistics when needed."""
        pass

class DummyOnPolicyBuffer(Buffer):
    """
    Replay buffer for on-policy DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        horizon_size: max length of steps to store for one environment.
        use_gae: if use GAE trick.
        use_advnorm: if use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
    """

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 horizon_size: int,
                 use_gae: bool = True,
                 use_advnorm: bool = True,
                 gamma: float = 0.99,
                 gae_lam: float = 0.95):
        """Allocate rollout storage and advantage/return buffers."""
        super(DummyOnPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
        self.n_envs, self.horizon_size = n_envs, horizon_size
        self.n_size = self.horizon_size
        self.buffer_size = self.n_size * self.n_envs
        self.use_gae, self.use_advnorm = use_gae, use_advnorm
        self.gamma, self.gae_lam = gamma, gae_lam
        self.start_ids = np.zeros(self.n_envs, np.int64)
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.returns = create_memory((), self.n_envs, self.n_size)
        self.values = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.advantages = create_memory((), self.n_envs, self.n_size)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

    @property
    def full(self):
        """Return True once one full on-policy horizon is collected."""
        return self.size >= self.n_size

    def clear(self):
        """Reset all on-policy rollout arrays."""
        self.ptr, self.size = 0, 0
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.returns = create_memory((), self.n_envs, self.n_size)
        self.values = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.advantages = create_memory((), self.n_envs, self.n_size)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

    def store(self, obs, acts, rews, value, terminals, aux_info=None):
        """Store one vectorized on-policy transition at the current cursor."""
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(value, self.values, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(aux_info, self.auxiliary_infos, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def finish_path(self, val, i):
        """Compute returns and advantages for one env trajectory slice."""
        if self.full:
            path_slice = np.arange(self.start_ids[i], self.n_size).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
        vs = np.append(np.array(self.values[i, path_slice]), [val], axis=0)
        if self.use_gae:  # use gae
            rewards = np.array(self.rewards[i, path_slice])
            advantages = np.zeros_like(rewards)
            dones = np.array(self.terminals[i, path_slice])
            last_gae_lam = 0
            step_nums = len(path_slice)
            for t in reversed(range(step_nums)):
                delta = rewards[t] + (1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                advantages[t] = last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lam * last_gae_lam
            returns = advantages + vs[:-1]
        else:
            rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
            returns = discount_cumsum(rewards, self.gamma)[:-1]
            advantages = rewards[:-1] + self.gamma * vs[1:] - vs[:-1]

        self.returns[i, path_slice] = returns
        self.advantages[i, path_slice] = advantages
        self.start_ids[i] = self.ptr

    def sample(self, indexes):
        """Sample flattened env/time indexes from a completed on-policy horizon."""
        assert self.full, "Not enough transitions for on-policy buffer to random sample"
        env_choices, step_choices = divmod(indexes, self.n_size)
        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'returns': sample_batch(self.returns, tuple([env_choices, step_choices])),
            'values': sample_batch(self.values, tuple([env_choices, step_choices])),
            'aux_batch': sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices])),
            'batch_size': len(indexes),
        }
        adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
        if self.use_advnorm:
            adv_batch = (adv_batch - np.mean(adv_batch)) / (np.std(adv_batch) + 1e-8)
        samples_dict.update({
            'advantages': adv_batch
        })
        return samples_dict


class DummyOffPolicyBuffer(Buffer):
    """
    Replay buffer for off-policy DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        n_envs: number of parallel environments.
        horizon_size: max length of steps to store for one environment.
    """

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 n_envs: int,
                 horizon_size: int,
                 env_idx: int = 0,
                 auxiliary_shape = None):
        """Allocate circular replay storage for off-policy training."""
        super(DummyOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
        self.n_envs, self.horizon_size = n_envs, horizon_size
        self.n_size = self.horizon_size
        self.buffer_size = self.n_size * self.n_envs
        self.start_ids = np.zeros(self.n_envs, np.int64)
        self.env_idx = env_idx

        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.returns = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.embodiment = create_memory((), self.n_envs, self.n_size)

    @property
    def full(self):
        """Return True once the circular replay buffer has wrapped."""
        return self.size >= self.n_size

    def clear(self):
        """Reset all off-policy replay arrays."""
        self.ptr, self.size = 0, 0
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.returns = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.embodiment = create_memory((), self.n_envs, self.n_size)

    def store(self, obs, acts, rews, terminals, embodiment, aux_info=None):
        """Store one synchronized vectorized off-policy transition."""
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(embodiment, self.embodiment, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def finish_path(self, val, i):
        """No-op hook kept for interface compatibility."""
        pass

    def sample(self, indexes):
        """Sample transitions and matching next observations from replay."""
        env_choices, step_choices = divmod(indexes, self.size)
        if self.full:
            latest_step = (self.ptr - 1) % self.size
            step_choices[step_choices == latest_step] = self.ptr
            next_step_choices = step_choices + 1
            next_step_choices = next_step_choices % self.size
        else:
            step_choices = (step_choices % (self.size - 1))
            next_step_choices = step_choices + 1

        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'next_obs': sample_batch(self.observations, tuple([env_choices, next_step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'rewards': sample_batch(self.rewards, tuple([env_choices, step_choices])),
            'terminals': sample_batch(self.terminals, tuple([env_choices, step_choices])),
            'embodiment': sample_batch(self.embodiment, tuple([env_choices, step_choices])),
            'batch_size': len(indexes),
        }
        return samples_dict

if __name__ == '__main__':
    obs_space = Dict({'rgb':Box(low=0,high=255,shape=(8,224,224,3)),
                    'depth':Box(low=0,high=10,shape=(224,224,1)),
                    'pointgoal':Box(low=-np.inf,high=np.inf,shape=(3,))})
    act_space = Box(low=-np.inf,high=np.inf,shape=(5,24,3))
    aux_shape = {'old_logp':(4,24,3)}
    memory_buffer = DummyOnPolicyBuffer(observation_space=obs_space,
                                        action_space=act_space,
                                        auxiliary_shape=aux_shape,
                                        n_envs=4,
                                        horizon_size=256,
                                        use_gae=True,
                                        use_advnorm=True,
                                        gamma=0.99,
                                        gae_lam=0.95)
