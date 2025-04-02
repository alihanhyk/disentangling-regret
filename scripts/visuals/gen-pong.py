from gymnasium.vector import SyncVectorEnv
import matplotlib.image as image
import numpy as np
import torch

from skrl.utils import set_seed

from rlerror.environments import make_mypong
from rlerror.resources import Distractor, AddDistractions
from rlerror.environments.mypong import _COL_BACK

seed = 0
set_seed(seed, deterministic=True)

distractors = [Distractor()] * 2
vec_env = SyncVectorEnv([make_mypong for _ in range(2)])
vec_env.reset(seed=seed)
env = AddDistractions(vec_env, distractors)

for _ in range(20):
    obs, *_ = env.step(torch.zeros(2))
obs = obs.cpu().numpy().reshape(2, 4, 84, 84)

obs[0,3,...] -= obs[0,3,...].min()
obs[0,3,...] /= obs[0,3,...].max()
obs = np.concatenate((obs[0,0,...], obs[0,3,...]), axis=-1)

import seaborn as sns
sns.set_theme()

image.imsave("reports/gen-pong/filter-id.png", obs, vmin=0., vmax=1., cmap="cividis")

_obs = obs.copy()
_obs[0:4,:84] = _COL_BACK
image.imsave("reports/gen-pong/filter-dists.png", _obs, vmin=0., vmax=1.,  cmap="cividis")

_obs = obs.copy()
_obs[0:4,:84] = _COL_BACK
_obs[:,84:] = _COL_BACK
image.imsave("reports/gen-pong/filter-justfield.png", _obs, vmin=0., vmax=1.,  cmap="cividis")

_obs = obs.copy()
_obs[4:,:84] = _COL_BACK
_obs[:,84:] = _COL_BACK
image.imsave("reports/gen-pong/filter-justcount.png", _obs, vmin=0., vmax=1.,  cmap="cividis")
