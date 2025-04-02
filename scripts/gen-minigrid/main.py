import argparse
from gymnasium.vector import SyncVectorEnv
import logging

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from rlerror.environments import make_coloredkeys
from rlerror.models import GridPolicyValue
from rlerror.resources import RetrainablePPO

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--filter', default=None)
parser.add_argument('--rep-size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--retrain', default=None)

args = parser.parse_args()
args.num_envs = 64
args.timesteps = 50_000
args.exp_directory = "results/gen-minigrid"

###

if args.silent:
    logger = logging.getLogger('skrl')
    logger.setLevel(logging.CRITICAL)

set_seed(args.seed, deterministic=True)
# the environment still needs to be reset with a seed

###

make_env = lambda: make_coloredkeys(args.env, args.filter)
vec_env = SyncVectorEnv([make_env for _ in range(args.num_envs)])
obs, _ = vec_env.reset(seed=args.seed)
env = wrap_env(vec_env)

model = GridPolicyValue(
    env.observation_space,
    env.action_space,
    env.device,
    num_rep_channels = args.rep_size)

###

cfg_agent = PPO_DEFAULT_CONFIG.copy()

cfg_agent["rollouts"] = 4096 // args.num_envs
cfg_agent["entropy_loss_scale"] = 0.01
cfg_agent["value_loss_scale"] = 0.5

cfg_agent["value_preprocessor"] = RunningStandardScaler
cfg_agent["value_preprocessor_kwargs"] = dict(size=1, device=env.device)

cfg_agent["experiment"]["directory"] = args.exp_directory
cfg_agent["experiment"]["experiment_name"] = args.exp_name
cfg_agent["experiment"]["write_interval"] = 500

cfg_trainer = dict()
cfg_trainer["headless"] = True
cfg_trainer["timesteps"] = args.timesteps

agent = RetrainablePPO(
    models=dict(policy=model, value=model),
    memory=RandomMemory(memory_size=cfg_agent["rollouts"], num_envs=env.num_envs, device=env.device),
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    cfg=cfg_agent)

if args.retrain is not None:
    agent.load(f"{args.exp_directory}/{args.retrain}/checkpoints/best_agent.pt")
    agent.policy.encoder.requires_grad_(False)
    agent.reset_optimizer()

trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)
trainer.train()
