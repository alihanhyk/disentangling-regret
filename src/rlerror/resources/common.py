import numpy as np
import torch
import tqdm

from skrl.agents.torch.ppo import PPO
from skrl.resources.preprocessors.torch import RunningStandardScaler

# NOTE: for vectorized environments in skrl,
#       calling .reset() does *not* reset the base environments
#       the base environments are only reset if they are terminated or truncated

def eval(env, agent, num_episodes):
    scores = list()
    cum_rewards = np.zeros(env.num_envs)
    states, _ = env.reset()
    with tqdm.tqdm(total=num_episodes) as pbar:
        while len(scores) < num_episodes:
            with torch.no_grad():
                states = agent._state_preprocessor(states)
                actions, *_ = agent.policy.act(dict(states=states), role="policy")
            states, rewards, terminated, truncated, _ = env.step(actions)
            cum_rewards += rewards.cpu().numpy().flatten()
            dones = (terminated | truncated).cpu().numpy().flatten()
            for i_done in (i for i, done in enumerate(dones) if done):
                if len(scores) < num_episodes:
                    scores.append(cum_rewards[i_done])
                cum_rewards[i_done] = 0
            pbar.update(len(scores) - pbar.n)
    return np.array(scores)

class FreezableStandardScaler(RunningStandardScaler):
    def __init__(self, size, device=None):
        super().__init__(size=size, device=device)
        self.frozen = False

    def forward(self, x, train=False, inverse=False, no_grad=True):
        return super().forward(x, train and not self.frozen, inverse, no_grad)

class RetrainablePPO(PPO):
    def reset_optimizer(self):

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        # assumes that self.policy == self.value!
        
        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
        self.checkpoint_modules["optimizer"] = self.optimizer
