import argparse
from gymnasium.vector import SyncVectorEnv
import logging

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from rlerror.environments import make_coloredkeys
from rlerror.models import GridPolicyValue
from rlerror.resources import eval

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--filter', default=None)
parser.add_argument('--rep-size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--output', default=None)

args = parser.parse_args()
args.num_envs = 32
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
vec_env.reset(seed=args.seed)
env = wrap_env(vec_env)

model = GridPolicyValue(
    env.observation_space,
    env.action_space,
    env.device,
    num_rep_channels = args.rep_size)

###

agent = PPO(
    models=dict(policy=model, value=model),
    memory=None,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
    cfg=PPO_DEFAULT_CONFIG)

agent.load(f"{args.exp_directory}/{args.exp_name}/checkpoints/best_agent.pt")
agent.set_mode('eval')

scores = eval(env, agent, num_episodes=1024)
score = scores.mean()

args.output = "" if args.output is None else f"_{args.output}"
with open(f"{args.exp_directory}/{args.exp_name}/score{args.output}.txt", 'w') as f:
    f.write(str(score))
