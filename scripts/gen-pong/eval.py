from ale_py import ALEInterface, LoggerMode
import argparse
from gymnasium.vector import SyncVectorEnv
import logging

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from rlerror.environments import make_mypong
from rlerror.models import ImagePolicyValue
from rlerror.resources import eval, Distractor, AddDistractions, FreezableStandardScaler, RetrainablePPO

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', default="Train", choices=["Train", "TestStochastic", "TestObservational"])
parser.add_argument('--filter', default=None, choices=["JustField", "JustCount", "FieldDistractions"])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--output', default=None)

args = parser.parse_args()
args.num_envs = 4
args.num_levels = 2  # must divide args.num_env
args.exp_directory = "results/gen-pong"

###

if args.silent:
    logger = logging.getLogger('skrl')
    logger.setLevel(logging.CRITICAL)
    ALEInterface.setLoggerMode(LoggerMode.Error)

set_seed(args.seed, deterministic=True)
# the environment still needs to be reset with a seed

###

make_env = lambda: make_mypong(args.env, args.filter)

distractors_train = [Distractor() for _ in range(args.num_levels)] * (args.num_envs // args.num_levels)
distractors_test = [Distractor() for _ in range(args.num_levels)] * (args.num_envs // args.num_levels)
distractors = distractors_test if args.env == "TestObservational" else distractors_train

vec_env = SyncVectorEnv([make_env for _ in range(args.num_envs)])
vec_env.reset(seed=args.seed)

# env = wrap_env(vec_env)
env = AddDistractions(vec_env, distractors, hide_distractions=(args.filter in ["JustField", "JustCount"]))

model = ImagePolicyValue(env.observation_space, env.action_space, env.device)

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

scores = eval(env, agent, num_episodes=256)
score = scores.mean()

args.output = "" if args.output is None else f"_{args.output}"
with open(f"{args.exp_directory}/{args.exp_name}/score{args.output}.txt", 'w') as f:
    f.write(str(score))

env.close()
