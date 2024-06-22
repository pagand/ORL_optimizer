import torch
import torch.nn as nn

from typing import TypeVar, List, Tuple, Dict, Any
from torch import Tensor
from torch.nn.functional import pad

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Dynamics(nn.Module):
    '''
    input:
         state: (batch, sequence_num, state_dim), 
         action: (batch, sequence_num + future_num - 1, action_dim)

    output:
         next_state: (batch, future_num, state_dim * out_state_num)
    '''

    state_dim: int
    action_dim: int
    sequence_num: int
    future_num: int
    out_state_num: int
    use_future_act: bool
    state_action_dim: int
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, sequence_num: int, out_state_num: int, future_num: int,
                 use_future_act: bool = False):
        super().__init__()
        self.state_action_dim = state_dim + action_dim
        if use_future_act:
            if self.state_action_dim % 2 == 1:
                self.state_action_dim += 1            
        self.lstm = nn.LSTM(input_size=self.state_action_dim, hidden_size=hidden_dim, num_layers=sequence_num+out_state_num-1, 
                            batch_first=True)
        #self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_dim, state_dim*out_state_num)
        self.linear2 = nn.Linear(hidden_dim, out_state_num)
        self.future_num = future_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_num = sequence_num
        self.out_state_num = out_state_num
        self.use_future_act = use_future_act

    def forward(self, state, action, is_eval=False, is_ar=False):
        s_ = torch.empty((state.shape[0], 0, self.state_dim*self.out_state_num)).to(self.device)
        r_ = torch.empty((state.shape[0], 0, self.out_state_num)).to(self.device)
        state_= state[:,:self.sequence_num,:]
        for i in range(self.future_num):
            x = torch.cat((action[:,i:i+self.sequence_num,:], state_), dim=-1)
            if self.state_action_dim > self.state_dim + self.action_dim:
                x = pad(x, (0,1))
            if self.out_state_num > 1 and self.use_future_act:
                if is_eval:
                    future_act = torch.stack([action[:,i+self.sequence_num-1,:] for _ in range(self.out_state_num-1)], dim=1)
                else:
                    future_act = action[:,i+self.sequence_num:i+self.sequence_num+self.out_state_num-1,:]
                future_act = pad(future_act, (0, self.state_action_dim-self.action_dim))
                x = torch.cat((x, future_act), dim=1)
            x, _ = self.lstm(x)
            x = x[:,-1,:]
            #if not is_eval:
            #    x = self.dropout(x)
            s = self.linear(x)
            s = s.unsqueeze(1)
            r = self.linear2(x)
            r = r.unsqueeze(1)
            s_ = torch.cat((s_, s), dim=1)
            r_ = torch.cat((r_, r), dim=1)
            if is_ar:
                state_ = torch.cat((state_[:,1:,:], s[:,:,:self.state_dim]), dim=1)
            else:
                state_ = state[:,i+1:i+self.sequence_num+1,:]
        return (s_, r_)

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
