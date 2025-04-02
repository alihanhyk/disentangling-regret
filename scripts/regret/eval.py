from ale_py import ALEInterface, LoggerMode
import argparse
from gymnasium.vector import SyncVectorEnv
import logging

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from rlerror.environments import make_minigrid, make_pong
from rlerror.models import GridPolicyValue, ImagePolicyValue
from rlerror.resources import eval, AddActionObsNoise, FreezableStandardScaler, RetrainablePPO

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--output', default=None)

args = parser.parse_args()

args_env = {
    "MiniGrid": {
        "episodes":  1024,
        "exp_directory": "results/regret-minigrid",
        "make_env": make_minigrid,
        "model_class": GridPolicyValue,
        "num_envs": 32},
    "Pong": {
        "episodes": 256,
        "exp_directory": "results/regret-pong",
        "make_env": make_pong,
        "model_class": ImagePolicyValue,
        "num_envs": 4}}

if args.env not in args_env: raise ValueError
args.__dict__.update(args_env[args.env])

###

if args.silent:
    logger = logging.getLogger('skrl')
    logger.setLevel(logging.CRITICAL)
    ALEInterface.setLoggerMode(LoggerMode.Error)

set_seed(args.seed, deterministic=True)
# the environment still needs to be reset with a seed

###

vec_env = SyncVectorEnv([args.make_env for _ in range(args.num_envs)])
vec_env.reset(seed=args.seed)
env = wrap_env(vec_env)

model = args.model_class(env.observation_space, env.action_space, env.device)
model = AddActionObsNoise(model, p_action=0., p_obs=0.)

###

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["state_preprocessor"] = FreezableStandardScaler
cfg["state_preprocessor_kwargs"] = dict(size=env.observation_space, device=env.device)

agent = RetrainablePPO(
    models=dict(policy=model, value=model),
    memory=None,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    cfg=cfg)

agent.load(f"{args.exp_directory}/{args.exp_name}/checkpoints/best_agent.pt")
agent._state_preprocessor.frozen = True
agent.set_mode('eval')

scores = eval(env, agent, num_episodes=args.episodes)
score = scores.mean()

args.output = "" if args.output is None else f"_{args.output}"
with open(f"{args.exp_directory}/{args.exp_name}/score{args.output}.txt", 'w') as f:
    f.write(str(score))

env.close()
