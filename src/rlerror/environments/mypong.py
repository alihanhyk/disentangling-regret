import gymnasium as gym
import numpy as np

# 80 x 80 observations
# first dimension is vertical (not horizontal!)

# colors after ImageProcessor:
_COL_BACK = 233. / 3 / 255
_COL_PADR = 370. / 3 / 255
_COL_PADL = 417. / 3 / 255
_COL_BALL = 708. / 3 / 255

class Pong_SkipEmptyFramesAfterPoints(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(15):
            obs, *_, info = self.env.step(0)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward != 0.:
            for _ in range(16):
                obs, *_, info = self.env.step(0)
        return obs, reward, terminated, truncated, info

class Pong_ResetAfterFirstPoint(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_shift=[0]):
        super().__init__(env)
        self.frame_shift = frame_shift

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        shift = self.env.np_random.choice(self.frame_shift)
        for _ in range(shift):
            obs, *_, info = self.env.step(0)

        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward != 0.:
            terminated = True
        return obs, reward, terminated, truncated, info

class Pong_ExtractState(gym.Wrapper):
    def __init__(self, env: gym.Env, distractor=None, hide_distractions=False):
        super().__init__(env)
        self.distractor = distractor
        self.hide_distractions = hide_distractions

    def information(self, obs, info):
        i_padl = -1.
        inds = np.argwhere(obs == _COL_PADL)
        if inds.size > 0:
            i_padl = inds[0][0] / obs.shape[0]

        i_padr = -1.
        inds = np.argwhere(obs == _COL_PADR)
        if inds.size > 0:
            i_padr = inds[0][0] / obs.shape[0]

        i_ball = -1.
        j_ball = -1.
        inds = np.argwhere(obs == _COL_BALL)
        if inds.size > 0:
            i_ball = inds[0][0] / obs.shape[0]
            j_ball = inds[0][1] / obs.shape[1]

        info['state'] = np.array([i_padl, i_padr, i_ball, j_ball])
        return info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self.information(obs, info)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self.information(obs, info)

class Pong_AddFrameCounter(gym.Wrapper):
    def __init__(self, env: gym.Env, hide_count=False, hide_field=False):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0., 1., shape=(84, 84))
        self.count = 0
        self.hide_count = hide_count
        self.hide_field = hide_field
        
    def observation(self, obs):
        obs = np.pad(obs, ((0, 0), (2, 2)), mode='edge')
        obs = np.pad(obs, ((4, 0), (0, 0)))
        if not self.hide_count:
            obs[0:4,:min(84, self.count)] = 1.
        if self.hide_field:
            obs[4:,:] = _COL_BACK
            # ignores different background colors at the start!
            # must use with Pong_SkipEmptyFramesAfterPoints
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.count = 0
        info['count'] = self.count
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.count += 1
        if self.count > 84:
            truncated = True
        info['count'] = self.count
        return self.observation(obs), reward, terminated, truncated, info

class Pong_GetFromInfo(gym.Wrapper):
    def __init__(self, env: gym.Env, key):
        super().__init__(env)
        self.key = key
        if self.key == 'state':
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,))
        if self.key == 'count':
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
        
    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        return info[self.key], info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return info[self.key], reward, terminated, truncated, info
