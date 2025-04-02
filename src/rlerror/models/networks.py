import torch.nn as nn
import numpy as np

from .base import BasePolicyValue
# from .base import init_params_normal_
# from .base import init_params_orthogonal_

class SimplePolicyValue(BasePolicyValue):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)
        self.encoder = nn.Sequential(nn.Linear(self.num_observations, 64), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(64, action_space.n))
        self.value = nn.Sequential(nn.Linear(64, 1))

class GridPolicyValue(BasePolicyValue):
    def __init__(
            self,
            observation_space,
            action_space,
            device,
            num_rep_channels):
        
        super().__init__(observation_space, action_space, device)
        self.num_rep_channels = num_rep_channels

        conv_size = np.array(observation_space.shape[1:])
        conv_size = (conv_size - 1) // 2 -2
        conv_size = np.prod(conv_size)

        self.encoder = nn.Sequential(
            nn.Unflatten(-1, observation_space.shape),
            nn.Conv2d(observation_space.shape[0], 16, kernel_size=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, self.num_rep_channels, kernel_size=2), nn.ReLU(),
            nn.Flatten())

        self.policy = nn.Sequential(
            nn.Linear(conv_size * self.num_rep_channels, 64), nn.Tanh(),
            nn.Linear(64, action_space.n))
        
        self.value = nn.Sequential(
            nn.Linear(conv_size * self.num_rep_channels, 64), nn.Tanh(),
            nn.Linear(64, 1))
        
        # self.apply(init_params_normal_)

class ImagePolicyValue(BasePolicyValue):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        conv_size = np.array(observation_space.shape[1:])
        conv_size = (conv_size - 8) // 4 + 1
        conv_size = (conv_size - 4) // 2 + 1
        conv_size = (conv_size - 3) // 1 + 1
        conv_size = np.prod(conv_size)

        # NatureCNN
        self.encoder = nn.Sequential(
            nn.Unflatten(-1, observation_space.shape),
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_size * 64, 512), nn.ReLU())
        
        self.policy = nn.Sequential(
            nn.Linear(512, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_space.n))
        
        self.value = nn.Sequential(
            nn.Linear(512, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1))
        
        # self.encoder.apply(lambda m: init_params_orthogonal_(m, gain=np.sqrt(2)))
        # self.policy.apply(lambda m: init_params_orthogonal_(m, gain=0.01))
        # self.value.apply(lambda m: init_params_orthogonal_(m, gain=1))

from math import prod

class IdentityPolicyValue(BasePolicyValue):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.encoder = nn.Identity()
        obs_size = prod(observation_space.shape)
        
        # observation_space.shape[0] * observation_space.shape[1]

        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_space.n))
        
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1))
