import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from ..models.base import BasePolicyValue

class _ActionNoise(nn.Module):
    def __init__(self, p_noise, action_space):
        super().__init__()
        self.loginvnumact = np.log(1. / action_space.n)
        self.logits = torch.nn.Parameter(torch.tensor([0., 0.]))
        self.requires_grad_(False)
        self.fill_p_noise_(p_noise)
        
    def forward(self, x):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        logits = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return torch.logaddexp(logits[0] + x, logits[1] + self.loginvnumact + torch.zeros_like(x))
    
    def is_active(self):
        # the module is considered active if p_noise is not exactly zero
        return self.logits[1] != -torch.inf

    def fill_p_noise_(self, p_noise):
        self.logits[0].fill_(-np.inf if p_noise == 1. else np.log(1. - p_noise))
        self.logits[1].fill_(-np.inf if p_noise == 0. else np.log(p_noise))

class _ObsNoise(nn.Module):
    def __init__(self, p_noise, observation_space):
        super().__init__()
        self.probs = nn.Parameter(torch.tensor(0.))
        self.requires_grad_(False)
        self.fill_p_noise_(p_noise)
        
    def forward(self, x):
        dist = Bernoulli(probs=self.probs)
        mask = dist.sample(x.size()).to(bool)
        return torch.where(mask, x, torch.zeros_like(x))
    
    def is_active(self):
        # the module is considered active if p_noise is not exactly zero
        return self.probs != 1.

    def fill_p_noise_(self, p_noise):
        self.probs.data.fill_(1. - p_noise)

class AddActionObsNoise(BasePolicyValue):
    def __init__(self, model: BasePolicyValue, p_action, p_obs):
        super().__init__(model.observation_space, model.action_space, model.device)

        self.noise_action = _ActionNoise(p_action, model.action_space)
        self.noise_obs = _ObsNoise(p_obs, model.observation_space)

        self.encoder = nn.Sequential(self.noise_obs, model.encoder)
        self.policy = nn.Sequential(model.policy, self.noise_action)
        self.value = model.value
    