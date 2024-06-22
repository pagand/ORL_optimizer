import torch
import torch.nn as nn
from torch import Tensor
import math

from copy import deepcopy

def pytorch_init(tensor, fan_in):
    bound = math.sqrt(1 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
def uniform_init(tensor, bound):
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
def constant_init(tensor, value):
    with torch.no_grad():
        return tensor.fill_(value)
    
class DetActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, layernorm: bool, n_hiddens: int):
        super(DetActor, self).__init__()

        layers = []
        ll = nn.Linear(state_dim, hidden_dim)
        pytorch_init(ll.weight, state_dim)
        constant_init(ll.bias, 0.1)
        layers.append(ll)
        layers.append(nn.ReLU())
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_hiddens - 1):
            ll = nn.Linear(hidden_dim, hidden_dim)
            pytorch_init(ll.weight, hidden_dim)
            constant_init(ll.bias, 0.1)
            layers.append(ll)
            layers.append(nn.ReLU())
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
        ll = nn.Linear(hidden_dim, action_dim)
        uniform_init(ll.weight, 1e-3)
        uniform_init(ll.bias, 1e-3)
        layers.append(ll)
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, state: Tensor):
        return self.network(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, layernorm=True, n_hiddens=3):
        super(Critic, self).__init__()

        layers = []
        input_dim = state_dim + action_dim
        
        # Initial layer
        ll = nn.Linear(input_dim, hidden_dim)
        pytorch_init(ll.weight, input_dim)
        constant_init(ll.bias, 0.1)
        layers.append(ll)
        layers.append(nn.ReLU())
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(n_hiddens - 1):
            ll = nn.Linear(hidden_dim, hidden_dim)
            pytorch_init(ll.weight, hidden_dim)
            constant_init(ll.bias, 0.1)
            layers.append(ll)
            layers.append(nn.ReLU())
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        ll = nn.Linear(hidden_dim, 1)
        uniform_init(ll.weight, 3e-3)
        uniform_init(ll.bias, 3e-3)
        layers.append(ll)
        
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        out = self.network(state_action).squeeze(-1)
        return out

class EnsembleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_critics=10, layernorm=True, n_hiddens=3):
        super(EnsembleCritic, self).__init__()
        self.num_critics = num_critics
        self.critics = nn.ModuleList([
            Critic(state_dim, action_dim, hidden_dim, layernorm, n_hiddens)
            for _ in range(num_critics)
        ])

    def forward(self, state, action):
        q_values = torch.stack([critic(state, action) for critic in self.critics], dim=0)
        return q_values
    
class TrainState:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.target_model = deepcopy(model)

    def get_model(self):
        return self.model
    
    def get_target_model(self):
        return self.target_model
    
    def get_optimizer(self):
        return self.optimizer
    
    def soft_update(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def main():
    # test data for DetActor
    state_dim = 10
    action_dim = 5
    hidden_dim = 256
    layernorm = True
    n_hiddens = 3
    state = torch.randn(32, state_dim)
    actor = DetActor(state_dim, action_dim, hidden_dim, layernorm, n_hiddens)
    action = actor(state)
    assert action.shape == (32, action_dim)
    print(action)

    # test data for EnsembleCritic
    state_dim = 10
    action_dim = 5
    hidden_dim = 256
    num_critics = 3
    layernorm = True
    n_hiddens = 3
    state = torch.randn(32, state_dim)
    action = torch.randn(32, action_dim)
    critic = EnsembleCritic(state_dim, action_dim, hidden_dim, num_critics, layernorm, n_hiddens)
    q_values = critic(state, action)
    assert q_values.shape == (num_critics, 32)
    print(q_values)

if __name__ == "__main__":
    main()