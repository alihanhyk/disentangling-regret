from ale_py import ALEInterface, LoggerMode
import argparse
from gymnasium.vector import SyncVectorEnv
import logging
import warnings

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from rlerror.environments import make_mypong
from rlerror.models import ImagePolicyValue
from rlerror.resources import Distractor, AddDistractions, FreezableStandardScaler, RetrainablePPO

###

parser = argparse.ArgumentParser()
parser.add_argument('--env', default="Train", choices=["Train", "TestStochastic", "TestObservational"])
parser.add_argument('--filter', default=None, choices=["JustField", "JustCount", "FieldDistractions"])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--silent', action='store_true')
parser.add_argument('--exp-name', default="")
parser.add_argument('--retrain', default=None)

args = parser.parse_args()
args.num_envs = 64
args.num_levels = 2  # must divide args.num_env
args.timesteps = 50_000
args.exp_directory = "results/gen-pong"

###

if args.silent:
    logger = logging.getLogger('skrl')
    logger.setLevel(logging.CRITICAL)
    ALEInterface.setLoggerMode(LoggerMode.Error)
    warnings.filterwarnings('ignore', module=".*lr_scheduler")

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

cfg_agent = PPO_DEFAULT_CONFIG.copy()

cfg_agent["rollouts"] = 4096 // args.num_envs
cfg_agent["entropy_loss_scale"] = 0.01
cfg_agent["value_loss_scale"] = 0.5

cfg_agent["learning_rate_scheduler"] = KLAdaptiveRL

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

if args.retrain is not None:
    agent.load(f"{args.exp_directory}/{args.retrain}/checkpoints/best_agent.pt")
    
    agent._state_preprocessor.frozen = True
    agent.policy.encoder.requires_grad_(False)
    agent.reset_optimizer()

trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg_trainer)
trainer.train()
