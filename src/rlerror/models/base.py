import torch.nn as nn

from skrl.models.torch import CategoricalMixin
from skrl.models.torch import DeterministicMixin
from skrl.models.torch import Model

class BasePolicyValue(Model, CategoricalMixin, DeterministicMixin):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, role='policy')
        DeterministicMixin.__init__(self, role='value')

    def act(self, inputs, role):
        if role == 'policy': return CategoricalMixin.act(self, inputs, role)
        elif role == 'value': return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        latent = self.encoder(inputs['states'])
        if role == 'policy': return self.policy(latent), {}
        elif role == 'value': return self.value(latent), {}

def init_params_normal_(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 1)
        module.weight.data /= module.weight.data.pow(2).sum(1, keepdim=True).sqrt()
        if module.bias is not None:
            module.bias.data.fill_(0)

def init_params_orthogonal_(module, gain=1):
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)