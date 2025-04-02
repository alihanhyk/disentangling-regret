import copy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.func import stack_module_state
from torch.func import functional_call

from skrl.envs.wrappers.torch import GymnasiumWrapper

from ..models.base import init_params_orthogonal_

class Distractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 7 * 7),
            nn.Unflatten(-1, (1, 7, 7)),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2),
            nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4),
            nn.Flatten())
        
        self.apply(init_params_orthogonal_)
        # orthogonal initialization is critical for distractors to be informative

        self.requires_grad_(False)
        
    def forward(self, x):
        return self.net(x)

class AddDistractions(GymnasiumWrapper):
    def __init__(self, env, distractors, hide_distractions=False):
        super().__init__(env)
        assert len(distractors) == self.num_envs
        self.hide_distractions = hide_distractions

        distractors = [dist.to(self.device) for dist in distractors]
        self._dist_params, self._dist_buffers = stack_module_state(distractors)
        self._dist_base = copy.deepcopy(distractors[0]).to('meta')
        
        self._dist_forward = lambda pars, bufs, x: functional_call(self._dist_base, (pars, bufs), (x,))
        self.distractors = lambda xs: torch.vmap(self._dist_forward)(self._dist_params, self._dist_buffers, xs)
        
        obs_shape = self._observation_space.shape
        self.obs_space_withdist = gym.spaces.Box(-np.inf, np.inf, (2 * obs_shape[0], *obs_shape[1:]))
        self.num_channels = obs_shape[0]

    @property
    def observation_space(self):
        return self.obs_space_withdist
        
    def add_distractions(self, obs, info):
        _obs = np.stack(info['states'], axis=0)
        _obs = torch.tensor(_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _obs = self.distractors(_obs)
            if self.hide_distractions:
                _obs = torch.zeros_like(_obs)
        obs = obs.reshape(self.num_envs, self.num_channels, -1)
        obs = torch.concatenate((obs, _obs), axis=1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def reset(self):
        obs, info = super().reset()
        return self.add_distractions(obs, info), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        return self.add_distractions(obs, info), reward, terminated, truncated, info
