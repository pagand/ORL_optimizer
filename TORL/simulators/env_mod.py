import torch
import torch.nn as nn

from typing import TypeVar, List, Tuple, Dict, Any
from torch import Tensor
from torch.nn.functional import pad
from torch.distributions import Normal

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
    device: torch.device
    dynamics_nn: nn.Module

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, sequence_num: int, 
                 out_state_num: int, future_num: int, use_future_act: bool = False, 
                 device: torch.device = 'cuda', train_gamma = 0.99):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=sequence_num+out_state_num-1, 
                            batch_first=True)
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                ) for _ in range(1)])

        self.linear_state = nn.Linear(hidden_dim, state_dim * out_state_num)
        self.linear_reward = nn.Linear(hidden_dim, out_state_num)
        self.future_num = future_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_num = sequence_num
        self.out_state_num = out_state_num
        self.use_future_act = use_future_act
        self.device = device
        self.state_gammas = Tensor([[train_gamma**j for i in range(state_dim)] for j in range(future_num)]).to(device)
        self.reward_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
 
    def forward(self, state, action, next_state=None, next_reward=None, is_eval=False, is_ar=False):
        s_ = torch.empty((state.shape[0], 0, self.state_dim*self.out_state_num)).to(self.device)
        r_ = torch.empty((state.shape[0], 0, self.out_state_num)).to(self.device)
        state_= state[:,:self.sequence_num,:]
        for i in range(self.future_num):
            #print("action shape", action.shape, "state_ shape", state_.shape)
            #print("i", i, "action[] shape", action[:, i:i+self.sequence_num, :].shape)
            x = torch.cat((action[:,i:i+self.sequence_num,:], state_), dim=-1)
            x, _ = self.lstm(x)
            x = x[:,-1,:]
            for mlp in self.mlps:
                x = mlp(x) + x
            s = self.linear_state(x).unsqueeze(1)
            r = self.linear_reward(x).unsqueeze(1)
            s_ = torch.cat((s_, s), dim=1)
            r_ = torch.cat((r_, r), dim=1)
            if is_ar:
                state_ = torch.cat((state_[:,1:,:], s[:,:,:self.state_dim]), dim=1)
            else:
                state_ = state[:,i+1:i+self.sequence_num+1,:]

        if next_state is not None and next_reward is not None:
            next_state_diff = (((s_ - next_state) * self.state_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            reward_diff = (((r_ - next_reward) * self.reward_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            loss = next_state_diff + reward_diff
            return (s_, r_, loss)
        
        return (s_, r_)
    
class GRU_update(nn.Module):
    def __init__(self, input_size, state_dim, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5, device="cuda", train_gamma=0.99):
        super().__init__()
        future_num = prediction_horizon
        self.state_dim = state_dim
        self.future_num = prediction_horizon
        self.h = prediction_horizon
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential( nn.ReLU(),
                                  nn.Linear(hidden_size, 2048),
                                  nn.Dropout(0.2),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_size))
        self.hx_fc = nn.Linear(2*hidden_size, hidden_size)
        self.device = device
        self.train_gamma = train_gamma
        self.state_gammas = Tensor([[train_gamma**j for i in range(state_dim)] for j in range(future_num)]).to(device)
        self.reward_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)

    def forward(self, predicted_values, past_time_features, next_state = None, next_reward = None):
        # predicted_values (256, 5, 4)
        # past_time_features (256, 25, 12)
        xy = torch.zeros(size=(past_time_features.shape[0], 1, self.output_size)).float().to(self.device)
        hx = past_time_features.reshape(-1, 1, self.hidden_size)
        hx = hx.permute(1, 0, 2)
        out_wp = list()
        for i in range(self.h):
            ins = torch.cat([xy, predicted_values[:, i:i+1, :]], dim=1) # x
            # output hx (batch, 2, 69)
            hx, _ = self.gru(ins, hx.contiguous())
            hx = hx.reshape(-1, 2*self.hidden_size)
            hx = self.hx_fc(hx)
            d_xy = self.mlp(hx).reshape(-1, 1, self.output_size) #control v4
            hx = hx.reshape(1, -1, self.hidden_size)
            # print("dxy", d_xy)
            #(256,1,4) + (256,1,4)
            xy = xy + d_xy
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        # (256, 5, 4)
        if next_state is not None and next_reward is not None:
            next_state_diff = (((pred_wp[:,:,:self.state_dim] - next_state) * self.state_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            reward_diff = (((pred_wp[:,:,self.state_dim:] - next_reward) * self.reward_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            loss = next_state_diff + reward_diff
            return pred_wp, loss
        return pred_wp

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # for mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        recon_x = self.decoder(z)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        elbo_loss =  recon_loss + kld_loss   
        return elbo_loss, recon_x, mean, log_var
    
    def estimate(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        p_z = Normal(0, 1).log_prob(mean).sum(dim=-1) - log_var.sum(dim=-1)
        recon_x = self.decoder(mean)
        p_x_given_z = -nn.functional.mse_loss(recon_x, x, reduction='none').sum(dim=-1)
        elbo = p_x_given_z + p_z
        return elbo

class MyEnv:

    '''
    A simple environment that uses a dynamics model to simulate the next state and reward

    states: in shape (sequence_count, state_dim)
    actions: in shape (sequence_count, action_dim)
    '''
    states : Tensor
    actions : Tensor
    rewards : Tensor
    config_dict : Dict[str, Any]
    dynamics_nn: nn.Module
    state_dim: int
    action_dim: int
    sequence_num: int
    max_episode_steps: int = 996
    istep: int = 0

    def __init__(self, chkpt_path: str, state_dim: int, action_dim: int, 
                 device: torch.device, max_episode_steps: int = 980):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.load_from_chkpt(chkpt_path)
        self.dynamics_nn.to(device)

    def load_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path)
        self.config_dict = checkpoint['config']
        self.dynamics_nn = Dynamics(self.state_dim, self.action_dim,
                                    self.config_dict['hidden_dim'], self.config_dict['sequence_num'], 
                                    self.config_dict['out_state_num'], future_num=1,
                                    use_future_act=False, device=self.device)
        self.gru_nn = GRU_update(self.state_dim+1, (self.state_dim+self.action_dim)*self.config_dict['sequence_num'], 
                                 self.state_dim+1, 1, 1).to(self.device)
        self.dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        self.gru_nn.load_state_dict(checkpoint["gru_nn"])
        self.sequence_num = self.config_dict['sequence_num']

    def reset(self, obs: ObsType) -> None:
        self.states = obs
        self.actions = torch.empty((0, self.action_dim))
        self.rewards = torch.empty((0))
        self.istep = 0
        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.rewards = self.rewards.to(self.device)
        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        action = action.to(self.device)
        self.actions = torch.cat((self.actions, action), dim=0)
        steps = min(len(self.states), self.sequence_num)
        states_ = self.states[-steps:].unsqueeze(0)
        actions_ = self.actions[-steps:].unsqueeze(0)
        with torch.inference_mode():
            (1, 1, 17) , (1, 1, 1)
            next_state, reward = self.dynamics_nn(states_, actions_, is_eval=True, is_ar=True)
        
        if self.config_dict['use_gru_update'] and self.istep >= self.sequence_num:
            input = torch.cat((states_, actions_), dim=2)
            input = input[:, :self.sequence_num]
            pred_features = torch.cat((next_state.detach(), reward.detach()), dim=2)
            with torch.inference_mode():
                g_pred = self.gru_nn(pred_features, input)
            g_states_pred = g_pred[:, :, :self.state_dim]
            g_rewards_pred = g_pred[:, :, self.state_dim:]

            next_state = g_states_pred.squeeze(1)
            reward = g_rewards_pred.squeeze(1)
        else:
            next_state = next_state.squeeze(1)
            reward = reward.squeeze(1)

        #print("next_state", next_state.shape, "reward", reward.shape)
        #print("self.states", self.states.shape, "self.actions", self.actions.shape, "self.rewards", self.rewards.shape)

        # (N, state_dim)
        self.states = torch.cat((self.states, next_state), dim=0)
        # (N)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.istep += 1
        next_state = next_state.detach().cpu()
        reward = reward.detach().cpu()
        done = (self.istep >= self.max_episode_steps)
        return next_state, reward, done

def main():
    chkpt_path = "TORL/config/halfcheeth/halfcheetah_medium_v2_ar.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MyEnv(chkpt_path, 17, 6, device)
    obs = torch.randn(1, 17)
    print("init state", obs.numpy())
    env.reset(obs)
    for i in range(10):
        action = torch.randn(1, 6)
        next_state, reward, done = env.step(action)
        print("step", i+1)
        print("action", action.numpy())
        print("next_state", next_state.numpy())
        print("reward", reward.numpy())
        print("done", done)

if __name__ == "__main__":
    main()
