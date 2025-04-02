import argparse
from gymnasium.vector import SyncVectorEnv
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from rlerror.environments import make_coloredkeys
from rlerror.models import GridPolicyValue

###

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', required=True)

args = parser.parse_args()
args.exp_directory = "results/gen-minigrid"

args_exp_name = {
    "id": {"filter": None},
    "hidecols": {"filter": "HideCols"},
    "hidedoor": {"filter": "HideDoor"},
    "hideboth": {"filter": "HideBoth"},
    "onehot": {"filter": "OneHot"}}

if args.exp_name not in args_exp_name: raise ValueError
args.__dict__.update(args_exp_name[args.exp_name])

###

logger = logging.getLogger('skrl')
logger.setLevel(logging.CRITICAL)
set_seed(0, deterministic=True)

vecs_in = dict()
for i_door in range(1, 4):
    for i_cols in range(4):
    
        id = f"Index{i_cols}{i_door}"
        make_env = lambda: make_coloredkeys(id, filter=args.filter)
        vec_env = SyncVectorEnv([make_env for _ in range(1)])
        env = wrap_env(vec_env)

        vecs_in[id] = list()
        vecs_in[id].append(env.reset()[0])
        vecs_in[id].append(env.step(torch.tensor(1))[0])
        vecs_in[id] = torch.concatenate(vecs_in[id], dim=0)

sim = np.zeros((5, 12, 12))
sim_cols = np.zeros((5, 4, 4))
sim_door = np.zeros((5, 3, 3))

for k, seed in enumerate([1, 2, 3, 4, 5]):

    model = GridPolicyValue(env.observation_space, env.action_space, env.device)
    agent = PPO(
        models=dict(policy=model, value=model),
        memory=None,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=PPO_DEFAULT_CONFIG)
    agent.load(f"{args.exp_directory}/{args.exp_name}{seed}/checkpoints/best_agent.pt")
    agent.set_mode('eval')

    vecs = dict()
    for id in vecs_in:
        with torch.no_grad():
            vecs[id] = agent.models['policy'].encoder(vecs_in[id])

    vecs_cols = dict()
    vecs_cols["Index0_"] = torch.concatenate((vecs["Index01"], vecs["Index02"], vecs["Index03"]), axis=0)
    vecs_cols["Index1_"] = torch.concatenate((vecs["Index11"], vecs["Index12"], vecs["Index13"]), axis=0)
    vecs_cols["Index2_"] = torch.concatenate((vecs["Index21"], vecs["Index22"], vecs["Index23"]), axis=0)
    vecs_cols["Index3_"] = torch.concatenate((vecs["Index31"], vecs["Index32"], vecs["Index33"]), axis=0)

    vecs_door = dict()
    vecs_door["Index_1"] = torch.concatenate((vecs["Index01"], vecs["Index11"], vecs["Index21"], vecs["Index31"]), axis=0)
    vecs_door["Index_2"] = torch.concatenate((vecs["Index02"], vecs["Index12"], vecs["Index22"], vecs["Index32"]), axis=0)
    vecs_door["Index_3"] = torch.concatenate((vecs["Index03"], vecs["Index13"], vecs["Index23"], vecs["Index33"]), axis=0)

    for i, vec0 in enumerate(vecs.values()):
        for j, vec1 in enumerate(vecs.values()):
            sim[k,i,j] = -(torch.mean((vec0 - vec1)**2)**.5).item()

    for i, vec0 in enumerate(vecs_cols.values()):
        for j, vec1 in enumerate(vecs_cols.values()):
            sim_cols[k,i,j] = -(torch.mean((vec0 - vec1)**2)**.5).item()

    for i, vec0 in enumerate(vecs_door.values()):
        for j, vec1 in enumerate(vecs_door.values()):
            sim_door[k,i,j] = -(torch.mean((vec0 - vec1)**2)**.5).item()

sim = sim.mean(axis=0)
sim_cols = sim_cols.mean(axis=0)
sim_door = sim_door.mean(axis=0)

sim /= -sim.min() if sim.min() < 0. else 1.
sim_cols /= -sim_cols.min() if sim_cols.min() < 0. else 1.
sim_door /= -sim_door.min() if sim_door.min() < 0. else 1.

###

sns.set_theme()

plt.figure(figsize=(5, 4))
ax = sns.heatmap(sim, vmin=-1, vmax=0, cmap='viridis')

ax.set_box_aspect(1.)
ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, labelrotation=90)
ax.xaxis.set_ticklabels([f"{config} - {color}" for config in ["N", "C", "S"] for color in ["Grey", "Red", "Green", "Blue"]])
ax.yaxis.set_tick_params(labelrotation=0)
ax.yaxis.set_ticklabels([f"{color} - {config}" for config in ["N", "C", "S"] for color in ["Grey", "Red", "Green", "Blue"]])

plt.tight_layout()
plt.savefig(f"reports/gen-minigrid/{args.exp_name}.pdf")

plt.figure(figsize=(5, 4))
ax = sns.heatmap(sim_cols, vmin=-1, vmax=0, cmap='viridis')

ax.set_box_aspect(1.)
ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
ax.xaxis.set_ticklabels(["Grey", "Red", "Green", "Blue"])
ax.yaxis.set_ticklabels(["Grey", "Red", "Green", "Blue"])

plt.tight_layout()
plt.savefig(f"reports/gen-minigrid/{args.exp_name}-cols.pdf")

plt.figure(figsize=(5, 4))
ax = sns.heatmap(sim_door, vmin=-1, vmax=0, cmap='viridis')

ax.set_box_aspect(1.)
ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
ax.xaxis.set_ticklabels(["North", "Center", "South"])
ax.yaxis.set_ticklabels(["North", "Center", "South"])

plt.tight_layout()
plt.savefig(f"reports/gen-minigrid/{args.exp_name}-door.pdf")
