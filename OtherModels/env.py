import torch
import torch.nn as nn

from typing import TypeVar, List, Tuple, Dict, Any
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Dynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, state_dim)
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
class Reward(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x.squeeze(-1)

class Dones(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x.squeeze(-1)

class Env:
    states : Tensor
    actions : Tensor
    rewards : Tensor
    dones : Tensor
    config_dict : Dict[str, Any]
    dynamics_nn: nn.Module
    rewards_nn: nn.Module
    dones_nn: nn.Module
    state_dim: int
    action_dim: int
    hidden_dim: int
    sequence_num: int
    dones_cutoff: float

    def __init__(self, chkpt_path: str = None):
        self.load_from_chkpt(chkpt_path)

    def load_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path)
        self.config_dict = checkpoint['config']
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.hidden_dim = self.config_dict['hidden_dim']
        self.sequence_num = self.config_dict['sequence_num']
        self.dones_cutoff = self.config_dict['dones_cutoff']
        self.dynamics_nn = Dynamics(self.state_dim, self.action_dim, self.hidden_dim, self.sequence_num)
        self.rewards_nn = Reward(self.state_dim, self.action_dim, self.hidden_dim, self.sequence_num)
        self.dones_nn = Dones(self.state_dim, self.action_dim, self.hidden_dim, self.sequence_num)
        self.dynamics_nn.load_state_dict(checkpoint['dynamics'])
        self.rewards_nn.load_state_dict(checkpoint['rewards'])
        self.dones_nn.load_state_dict(checkpoint['dones'])

    def reset(self, obs: ObsType) -> None:
        self.states = obs
        self.actions = torch.empty((0, self.action_dim))
        self.rewards = torch.empty((0))
        self.dones = torch.empty((0))
        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, float]:
        self.actions = torch.cat((self.actions, action), dim=0)
        steps = min(len(self.states), self.sequence_num)
        states_ = self.states[-steps:].unsqueeze(0)
        actions_ = self.actions[-steps:].unsqueeze(0)
        with torch.inference_mode():
            next_state = self.dynamics_nn(states_, actions_)
            reward = self.rewards_nn(states_, actions_)
            done = self.dones_nn(states_, actions_)
        self.states = torch.cat((self.states, next_state), dim=0)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.dones = torch.cat((self.dones, done), dim=0)
        return next_state, reward, done>=self.dones_cutoff, done

def main():
    chkpt_path = "/home/james/sfu/ORL_optimizer/OtherModels/chkpt/halfcheetah_v3.pt"
    env = Env(chkpt_path)
    obs = torch.randn(1, 17)
    print("init state", obs.numpy())
    env.reset(obs)
    for i in range(10):
        action = torch.randn(1, 6)
        next_state, reward, done, _ = env.step(action)
        print("step", i+1)
        print("action", action.numpy())
        print("next_state", next_state.numpy())
        print("reward", reward.numpy())
        print("done", done.numpy())

if __name__ == "__main__":
    main()
