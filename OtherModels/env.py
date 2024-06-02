import torch
import torch.nn as nn

from typing import TypeVar, List, Tuple, Dict, Any
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Env:
    states : Tensor
    actions : Tensor
    rewards : Tensor
    dones : Tensor
    config_dict : Dict[str, Any]
    dynamics_nn: nn.Module
    rewards_nn: nn.Module
    dones_nn: nn.Module

    def __init__(self, chkpt_path: str = None):
        self.load_from_chkpt(chkpt_path)

    def load_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path)
        self.config_dict = checkpoint['config']
        self.dynamics = Dynamics(self.config_dict['state_dim'], 
                                 self.config_dict['action_dim'], 
                                 self.config_dict['hidden_dim'], 
                                 self.config_dict['sequence_num'])
        self.rewards = Reward(self.config_dict['state_dim'],
                                self.config_dict['action_dim'],
                                self.config_dict['hidden_dim'],
                                self.config_dict['sequence_num'])
        self.dones = Dones(self.config_dict['state_dim'],
                            self.config_dict['action_dim'],
                            self.config_dict['hidden_dim'],
                            self.config_dict['sequence_num'])
        self.dynamics_nn.load_state_dict(checkpoint['dynamics_state_dict'])
        self.rewards_nn.load_state_dict(checkpoint['rewards_state_dict'])
        self.dones_nn.load_state_dict(checkpoint['dones_state_dict'])

    def reset(self, obs: ObsType) -> None:
        self.states = obs.unsqueeze(0)
        self.actions = torch.empty((0, self.config_dict['action_dim']))
        self.rewards = torch.empty((0))
        self.dones = torch.empty((0))
        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, float]:
        self.actions = torch.cat((self.actions, action.unsqueeze(0)), dim=0)
        steps = min(len(self.states), self.sequence_num)
        states_ = self.states[-steps:].unsqueeze(0)
        actions_ = self.actions[-steps:].unsqueeze(0)
        states_actions = torch.cat([states_, actions_], dim=-1)
        with torch.inference_mode():
            next_state = self.dynamics_nn(states_actions)
            reward = self.rewards_nn(states_actions)
            done = self.dones_nn(states_actions)
        self.states = torch.cat((self.states, next_state), dim=0)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.dones = torch.cat((self.dones, done), dim=0)
        return next_state, reward, done>=self.config_dict["dones_cutoff"], done


class Dynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, state_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
class Reward(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x.squeeze(-1)

class Dones(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x.squeeze(-1)

def main():
    pass

if __name__ == "__main__":
    main()
