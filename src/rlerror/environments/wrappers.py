from collections import deque
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

class GetFromDict(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, key):
        super().__init__(env)
        self.key = key
        self.observation_space = env.observation_space[key]
    
    def observation(self, obs):
        return obs[self.key]

class MoveChannelAxis(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_shape = (self.observation_space.shape[2], *self.observation_space.shape[:2])
        self.observation_space = spaces.Box(-np.inf, np.inf, observation_shape)
    
    def observation(self, obs):
        return np.moveaxis(obs, (0,1,2), (1,2,0))

class FilterActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env, active_actions):
        super().__init__(env)
        self.active_actions = active_actions
        self.action_space = spaces.Discrete(len(active_actions))

    def action(self, action):
        return self.active_actions[action]

class ImageProcessor(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, window=None):
        super().__init__(env)

        self.window = window
        if self.window is None:
            self.window = dict(
                top=0,
                left=0,
                height=env.observation_space.shape[0],
                width=env.observation_space.shape[1])
        if 'stride' not in self.window:
            self.window['stride'] = 1
        
        observation_shape = (
            self.window['height'] // self.window['stride'],
            self.window['width'] // self.window['stride'])
        self.observation_space = spaces.Box(0., 1., observation_shape)

    def observation(self, obs):
        obs = obs[
            self.window['top']:self.window['top']+self.window['height']:self.window['stride'],
            self.window['left']:self.window['left']+self.window['width']:self.window['stride']]
        obs = obs.mean(axis=-1) / 255
        return obs

class FrameStateStack(gym.wrappers.FrameStack):
    def __init__(self, env, num_stack):
        super().__init__(env, num_stack)
        self.states = deque(maxlen=num_stack)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for _ in range(self.num_stack):
            self.states.append(info['state'])
        info['states'] = np.stack(list(self.states), axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.states.append(info['state'])
        info['states'] = np.stack(list(self.states), axis=0)
        return obs, reward, terminated, truncated, info

class MiniGridScaler(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, obs):
        return obs / np.array([10., 5., 2.])[:,None,None]
