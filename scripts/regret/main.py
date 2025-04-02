from ale_py import ALEInterface, LoggerMode
import argparse
from gymnasium.vector import SyncVectorEnv
import logging
import os
import warnings

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from rlerror.environments import make_minigrid, make_pong
from rlerror.models import GridPolicyValue, ImagePolicyValue
from rlerror.resources import AddActionObsNoise, FreezableStandardScaler, RetrainablePPO

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--retrain', default=None)
parser.add_argument('--resave', default=None)
parser.add_argument('--p-action', type=float, default=None)
parser.add_argument('--p-obs', type=float, default=None)

args = parser.parse_args()

args_env = {
    "MiniGrid": {
        "exp_directory": "results/regret-minigrid",
        "make_env": make_minigrid,
        "model_class": GridPolicyValue,
        "num_envs": 64,
        "timesteps": 100_000,
        "adaptive_lr": False},  # not stable for MiniGrid!
    "Pong": {
        "exp_directory": "results/regret-pong",
        "make_env": make_pong,
        "model_class": ImagePolicyValue,
        "num_envs": 64,
        "timesteps": 100_000,
        "adaptive_lr": True}}

if args.env not in args_env: raise ValueError
args.__dict__.update(args_env[args.env])

###

if args.silent:
    logger = logging.getLogger('skrl')
    logger.setLevel(logging.CRITICAL)
    ALEInterface.setLoggerMode(LoggerMode.Error)
    warnings.filterwarnings('ignore', module=".*lr_scheduler")

set_seed(args.seed, deterministic=True)
# the environment still needs to be reset with a seed

###

vec_env = SyncVectorEnv([args.make_env for _ in range(args.num_envs)])
vec_env.reset(seed=args.seed)
env = wrap_env(vec_env)

model = args.model_class(env.observation_space, env.action_space, env.device)
model = AddActionObsNoise(model,
    p_action=args.p_action if args.p_action is not None else 0.,
    p_obs=args.p_obs if args.p_obs is not None else 0.)

###

cfg_agent = PPO_DEFAULT_CONFIG.copy()

cfg_agent["rollouts"] = 4096 // args.num_envs
cfg_agent["entropy_loss_scale"] = 0.01
cfg_agent["value_loss_scale"] = 0.5

if args.adaptive_lr:
    cfg_agent["learning_rate_scheduler"] = KLAdaptiveLR

cfg_agent["state_preprocessor"] = FreezableStandardScaler
cfg_agent["state_preprocessor_kwargs"] = dict(size=env.observation_space, device=env.device)
cfg_agent["value_preprocessor"] = RunningStandardScaler
cfg_agent["value_preprocessor_kwargs"] = dict(size=1, device=env.device)

cfg_agent["experiment"]["directory"] = args.exp_directory
cfg_agent["experiment"]["experiment_name"] = args.exp_name
cfg_agent["experiment"]["write_interval"] = 1000

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

if args.resave is not None:
    agent.load(f"{args.exp_directory}/{args.resave}/checkpoints/best_agent.pt")
    
    if args.p_obs is not None:
        agent.policy.noise_obs.fill_p_noise_(args.p_obs)
    if args.p_action is not None:
        agent.policy.noise_action.fill_p_noise_(args.p_action)

    os.makedirs(f"{args.exp_directory}/{args.exp_name}/checkpoints", exist_ok=True)
    agent.save(f"{args.exp_directory}/{args.exp_name}/checkpoints/best_agent.pt")
    exit()

if args.retrain is not None:
    agent.load(f"{args.exp_directory}/{args.retrain}/checkpoints/best_agent.pt")
    
    agent._state_preprocessor.frozen = True
    agent.policy.encoder.requires_grad_(False)
    if agent.policy.noise_action.is_active():
        agent.policy.noise_action.requires_grad_(True)

    agent.reset_optimizer()

trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)
trainer.train()
