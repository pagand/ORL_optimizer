import torch
import torch.nn as nn

from typing import TypeVar, List, Tuple, Dict, Any
from torch import Tensor
from torch.nn.functional import pad
from torch.distributions import Normal
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

def str_to_floats(data: str) -> np.ndarray:
    return np.array([float(x) for x in data.split(",")])

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
        state_= state[:,:self.sequence_num,:].to(self.device)
        state = state.to(self.device)
        action = action.to(self.device)
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
    def __init__(self, input_size, state_dim, hidden_size=1, output_size=4, num_layers=1, prediction_horizon=5, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 train_gamma=0.99):
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
        enc = []
        enc.append(nn.Linear(input_dim, hidden_dim))
        enc.append(nn.ReLU())
        for _ in range(3):
            enc.append(nn.Linear(hidden_dim, hidden_dim))
            enc.append(nn.ReLU())
        enc.append(nn.Linear(hidden_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*enc)

        dec = []
        dec.append(nn.Linear(latent_dim, hidden_dim))
        dec.append(nn.ReLU())
        for _ in range(3):
            dec.append(nn.Linear(hidden_dim, hidden_dim))
            dec.append(nn.ReLU())
        dec.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*dec)
    
    def forward(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        #print("z", z.shape, z)
        recon_x = self.decoder(z)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        #print("recon_loss", recon_loss, "kld_loss", kld_loss)
        elbo_loss =  recon_loss + kld_loss   
        return elbo_loss, recon_x, mean, log_var
    
    def estimate(self, x):
        # Encode the input
        h = self.encoder(x)
        # Split the encoded representation into mean and log variance
        mean, log_var = torch.chunk(h, 2, dim=-1)
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Compute the KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        # Decode the mean of the latent variables to reconstruct the input
        recon_x = self.decoder(mean)
        # Compute the negative reconstruction loss using MSE
        p_x_given_z = nn.functional.mse_loss(recon_x, x, reduction='none').sum(dim=-1)
        # Compute the ELBO by combining the reconstruction loss and the KL divergence
        elbo = p_x_given_z + kl_div
        return elbo
    
def halfcheetah_reward(x, state, action, next_state):

    dt = 0.05
    x_before = x
    #x_after = next_state[8]*dt + x_before
    x_after = (state[8] + next_state[8])/2*dt + x_before
    x_velocity = (x_after - x_before) / dt

    control_cost = 0.1 * torch.sum(torch.square(action))
    forward_reward = 1.0 * x_velocity
    reward = forward_reward - control_cost
    return reward, x_after

def hopper_is_done(state_):
    state_ = state_.cpu().numpy()
    healthy_state_range=(-100.0, 100.0)
    healthy_z_range=(0.7, float("inf"))
    healthy_angle_range=(-0.2, 0.2)
    
    z, angle = state_[0:2]
    state = state_[1:]
    print("z", z, "angle", angle, "state", state)

    min_state, max_state = healthy_state_range
    min_z, max_z = healthy_z_range
    min_angle, max_angle = healthy_angle_range

    healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
    healthy_z = min_z < z < max_z
    healthy_angle = min_angle < angle < max_angle

    is_healthy = all((healthy_state, healthy_z, healthy_angle))
    return not is_healthy

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
                 device: torch.device, max_episode_steps: int = 980, vae_chkpt_path: str = None, kappa = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.chkpt_path = chkpt_path
        self.load_from_chkpt(chkpt_path)
        self.load_vae_from_chkpt(vae_chkpt_path)
        self.kappa = kappa
        self.dynamics_nn.to(device)
        self.x = 0

    def load_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path, map_location=self.device)
        self.config_dict = checkpoint['config']
        self.dynamics_nn = Dynamics(self.state_dim, self.action_dim,
                                    self.config_dict['hidden_dim'], self.config_dict['sequence_num'], 
                                    self.config_dict['out_state_num'], future_num=1,
                                    use_future_act=False, device=self.device)
        self.gru_nn = GRU_update(self.state_dim+1, self.state_dim, (self.state_dim+self.action_dim)*self.config_dict['sequence_num'], 
                                 self.state_dim+1, 1, 1).to(self.device)
        self.dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        self.gru_nn.load_state_dict(checkpoint["gru_nn"])
        self.sequence_num = self.config_dict['sequence_num']
        self.state_mean = Tensor(str_to_floats(self.config_dict['state_mean'])).to(self.device)
        #print("state_mean", self.state_mean)
        self.state_std = Tensor(str_to_floats(self.config_dict['state_std'])).to(self.device)
        self.reward_mean = self.config_dict['reward_mean']
        self.reward_std = self.config_dict['reward_std']

    def load_vae_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path, map_location=self.device)
        self.threshold = checkpoint["threshold"]
        self.vae_config = checkpoint["config"]        
        self.vae = VAE(self.state_dim+self.action_dim, self.vae_config["vae_latent_dim"], self.vae_config['vae_hidden_dim'])
        self.vae.load_state_dict(checkpoint["vae"])
        self.vae.to(self.device)


    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std
    
    def denormalize_state(self, state: Tensor) -> Tensor:
        return state * self.state_std + self.state_mean
    
    def normalize_reward(self, reward: Tensor) -> Tensor:
        return (reward - self.reward_mean) / self.reward_std
    
    def denormalize_reward(self, reward: Tensor) -> Tensor:
        return reward * self.reward_std + self.reward_mean

    def reset(self, obs: ObsType) -> None:
        self.states = self.normalize_state(obs)
        self.actions = torch.empty((0, self.action_dim))
        self.rewards = torch.empty((0))
        self.istep = 0
        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.x = 0

    def _cal_sensitivity(self, states: ObsType, actions: ActType, next_state: ObsType, reward, K=10, sigma=0.01):
        # generate perturbations of size (K, states.shape[0], states.shape[1])
        states_noise = torch.normal(mean=0, std=sigma, size=(K, states.shape[1], states.shape[2])).to(self.device)
        actions_noise = torch.normal(mean=0, std=sigma, size=(K, actions.shape[1], actions.shape[2])).to(self.device)
        noisy_states = states + states_noise
        noisy_actions = actions + actions_noise
        with torch.inference_mode():
            # (1, 1, 17) , (1, 1, 1)
            noisy_nstate, noisy_reward = self.dynamics_nn(noisy_states, noisy_actions, is_eval=True, is_ar=True)
        
        if self.config_dict['use_gru_update'] and self.istep >= self.sequence_num:
            input = torch.cat((noisy_states, noisy_actions), dim=-1)
            input = input[:, :self.sequence_num]
            pred_features = torch.cat((noisy_nstate.detach(), noisy_reward.detach()), dim=-1)
            with torch.inference_mode():
                g_pred = self.gru_nn(pred_features, input)
            g_states_pred = g_pred[:, :, :self.state_dim]
            g_rewards_pred = g_pred[:, :, self.state_dim:]

            noisy_nstate = g_states_pred.squeeze(1)
            noisy_reward = g_rewards_pred.squeeze(1)
        else:
            noisy_nstate = noisy_nstate.squeeze(1)
            noisy_reward = noisy_reward.squeeze(1)

        # calculate sensitivity
        s1 = torch.var(noisy_nstate - next_state, dim=0).mean()
        s2 = torch.var(noisy_reward - reward, dim=0).mean()
        return s1, s2
        
    def step(self, action: ActType, use_sensitivity: bool = False) -> Tuple[ObsType, float, bool]:
        action = action.to(self.device)
        self.actions = torch.cat((self.actions, action), dim=0)
        steps = min(len(self.states), self.sequence_num)
        states_ = self.states[-steps:].unsqueeze(0)
        actions_ = self.actions[-steps:].unsqueeze(0)
        with torch.inference_mode():
            # (1, 1, 17) , (1, 1, 1)
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

        with torch.inference_mode():
            elbo = self.vae.estimate(torch.cat((self.states[-1].unsqueeze(0), self.actions[-1].unsqueeze(0)), dim=1))

        #print("next_state", next_state.shape, "reward", reward.shape)
        #print("self.states", self.states.shape, "self.actions", self.actions.shape, "self.rewards", self.rewards.shape)
        '''
        _state = self.denormalize_state(self.states[-1])
        _next_state = self.denormalize_state(next_state[0])

        reward_, self.x = halfcheetah_reward(self.x, _state, self.actions[-1], _next_state)
        '''
        #print("reward_", reward_, "reward", reward)
        # calculate model sensitivity
        s_state, s_reward = (0.0, 0.0)
        if use_sensitivity:
            s_state, s_reward = self._cal_sensitivity(states_, actions_, next_state, reward, K=10, sigma=0.01)
            #print("s_state", s_state, "s_reward", s_reward)
        # (N, state_dim)
        self.states = torch.cat((self.states, next_state), dim=0)
        # (N)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.istep += 1
        next_state = next_state.detach()
        reward = reward.detach()

        next_state = self.denormalize_state(next_state)
        reward = self.denormalize_reward(reward)

        done = (self.istep >= self.max_episode_steps)
        elbo = elbo.detach()
        #print("threshold", self.threshold)
        prob = Tensor([1/(1+self.kappa*(e-self.threshold)) if e > self.threshold else 1 for e in elbo]).to(self.device)
        discounted_reward = reward * prob
        #print("next_state", next_state, "reward", reward, "done", done, "prob", prob, "elbo", elbo, "discounted_reward", discounted_reward)
        #if "hopper" in self.chkpt_path:
        #    done = hopper_is_done(next_state[0])                                                                                                                                                                                                                                                                 
        return next_state, reward, done, prob, elbo, discounted_reward, s_state, s_reward                                                                                                                                                                                                         

def main():
    chkpt_path = "/home/james/sfu/ORL_optimizer/TORL/config/halfcheetah/halfcheetah_medium_v2_ar.pt"
    vae_chkpt_path = "/home/james/sfu/ORL_optimizer/TORL/config/halfcheetah/halfcheetah_medium_v2_vae.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MyEnv(chkpt_path, 17, 6, device, vae_chkpt_path=vae_chkpt_path)
    obs = torch.randn(1, 17).to(device)
    print("init state", obs.cpu().numpy())
    env.reset(obs)
    for i in range(10):
        action = torch.randn(1, 6).to(device)
        next_state, reward, done, prob, elbo, discounted_reward = env.step(action)
        print("step", i+1)
        print("action", action.cpu().numpy())
        print("next_state", next_state.cpu().numpy())
        print("reward", reward.cpu().numpy())
        print("done", done)
        print("prob", prob.cpu().numpy())
        print("elbo", elbo.cpu().numpy())
        print("discounted_reward", discounted_reward.cpu().numpy())

if __name__ == "__main__":
    main()
