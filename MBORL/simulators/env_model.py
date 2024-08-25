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

loss_func = nn.MSELoss()

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
                 future_num: int, device: torch.device = 'cuda', train_gamma = 0.99):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.prep = nn.Sequential(
            nn.Flatten(1, -1),
            nn.BatchNorm1d(self.input_dim * sequence_num),
            nn.Unflatten(1, (sequence_num, self.input_dim)),
        )
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=sequence_num, 
                            batch_first=True)
        '''
        self.mlps =nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                )
        '''
        self.post = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()] * 3)
            
        self.linear_state = nn.Linear(hidden_dim, state_dim * future_num)
        self.linear_reward = nn.Linear(hidden_dim, future_num)
        #self.linear_done = nn.Linear(hidden_dim, future_num)
        self.future_num = future_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_num = sequence_num
        self.device = device
        # (future_num, state_dim)
        #self.state_gammas = Tensor([[train_gamma**j for i in range(state_dim)] for j in range(future_num)]).to(device)
        #self.reward_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
        #self.dones_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
        # print("state_gammas", self.state_gammas.shape, "reward_gammas", self.reward_gammas.shape, "dones_gammas", self.dones_gammas.shape)
 
    def forward(self, state, action, hc_in=None, next_state=None, next_reward=None):
        state_= state[:,:self.sequence_num,:].to(self.device)
        action_ = action[:,:self.sequence_num].to(self.device)
        #print("action shape", action.shape, "state_ shape", state_.shape)
        #print("i", i, "action[] shape", action[:, i:i+self.sequence_num, :].shape)
        x = torch.cat((action_, state_), dim=-1)
        x = self.prep(x)
        if hc_in is not None:
            x, (h,c) = self.lstm(x, hc_in)
        else:
            x, (h,c) = self.lstm(x)
        hc = (h.detach(), c.detach())
        x = x[:,-1,:]
        x = self.post(x)
        # (batch, 1, state_dim * future_num)
        s = self.linear_state(x).unsqueeze(1)
        r = self.linear_reward(x).unsqueeze(1)
        #d = torch.sigmoid(self.linear_done(x4)).unsqueeze(1)

        #s_ = torch.cat([s[:,:,i*self.state_dim:(i+1)*self.state_dim] for i in range(self.future_num)], dim=1).to(self.device)
        s_ = s.view(-1, self.future_num, self.state_dim)
        #r_ = torch.cat([r[:,:,i:i+1] for i in range(self.future_num)], dim=1).to(self.device)
        r_ = r.view(-1, self.future_num, 1)
        #d_ = torch.cat([d[:,:,i:i+1] for i in range(self.future_num)], dim=1).to(self.device)

        loss = None
        if next_state is not None and next_reward is not None:
            #next_state_diff = (((s_ - next_state) * self.state_gammas) ** 2).flatten(1, -1).sum(-1).mean()
            next_state_diff = ((s_ - next_state) ** 2).flatten(1, -1).sum(-1).mean()
            #print("r_", r_.shape, "next_reward", next_reward.shape)
            #reward_diff = (((r_ - next_reward) * self.reward_gammas) ** 2).flatten(1, -1).sum(-1).mean()
            reward_diff = ((r_ - next_reward) ** 2).flatten(1, -1).sum(-1).mean()
            #done_diff = (((d_ - next_done) * self.dones_gammas) ** 2).flatten(1, -1).sum(-1).mean()
            #next_state_diff = loss_func(s_, next_state)
            #reward_diff = loss_func(r_, next_reward)
            #done_diff = loss_func(d_, next_done)
            loss = next_state_diff + reward_diff
        return s_, r_, hc, loss
    
class GRU_update(nn.Module):

    # hidden_size: (state_dim+action_dim)*sequence_num
    # output_size: state_dim+1
    # input_size: state_dim+1
    # prediction_horizon: future_num
    def __init__(self, input_size, state_dim, hidden_size, output_size, num_layers, prediction_horizon, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 train_gamma=0.99):
        super().__init__()
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
        #self.state_gammas = Tensor([[train_gamma**j for i in range(state_dim)] for j in range(future_num)]).to(device)
        #self.reward_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
        #self.dones_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
    
    # predicted_values from LSTM (batchsize, future_num, state_dim+1)
    # past_time_features (batchsize, sequence_num, state_dim+action_dim)
    def forward(self, predicted_values, past_time_features, next_state = None, next_reward = None):
        # xy (batch, 1, state_dim+1)
        xy = torch.zeros(size=(past_time_features.shape[0], 1, self.output_size)).float().to(self.device)
        # hx: (batch_size, 1, (state_dim+action_dim)*sequence_num)
        #hx = past_time_features.reshape(-1, 1, self.hidden_size)
        hx = past_time_features.reshape(1, -1, self.hidden_size)
        # hx: (1, batch_size, (state_dim+action_dim)*sequence_num)
        #hx = hx.permute(1, 0, 2)
        out_wp = list()
        # h = future_num
        for i in range(self.h):
            ins = torch.cat([xy, predicted_values[:, i:i+1, :]], dim=1) # x
            # ins: (batch, 2, state_dim+1)
            # hx: (1, batch, (state_dim+action_dim)*sequence_num)            
            #hx, _ = self.gru(ins, hx.contiguous())
            #print("hx", hx.shape, "ins", ins.shape)
            hx, _ = self.gru(ins, hx)
            # hx: (batch, 2, (state_dim+action_dim)*sequence_num)
            hx = hx.reshape(-1, 2*self.hidden_size)
            hx = self.hx_fc(hx)
            # hx: (batch, (state_dim+action_dim)*sequence_num)
            d_xy = self.mlp(hx).reshape(-1, 1, self.output_size) #control v4
            # d_xy: (batch, 1, state_dim+1)
            hx = hx.reshape(1, -1, self.hidden_size)
            # print("dxy", d_xy)
            #(256,1,4) + (256,1,4)
            xy = xy + d_xy
            out_wp.append(xy)
        #pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        pred_wp = torch.cat(out_wp, dim=1)
        state_pred = pred_wp[:,:,:self.state_dim]
        reward_pred = pred_wp[:,:,self.state_dim:self.state_dim+1]
        #done_pred = pred_wp[:,:,self.state_dim+1:]
        # (256, 5, 4)
        if next_state is not None and next_reward is not None:
            #next_state_diff = (((state_pred - next_state) * self.state_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            next_state_diff = ((state_pred - next_state) ** 2).flatten(1,-1).sum(-1).mean()
            #reward_diff = (((reward_pred - next_reward) * self.reward_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            reward_diff = ((reward_pred - next_reward) ** 2).flatten(1,-1).sum(-1).mean()
            #done_diff = (((done_pred - next_done) * self.dones_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            loss = next_state_diff + reward_diff # + done_diff
            return state_pred, reward_pred, loss
        return state_pred, reward_pred, None
    '''
    # hidden_size: (state_dim+action_dim)*sequence_num
    # output_size: state_dim+1
    # input_size: state_dim+1
    # prediction_horizon: future_num
    def __init__(self, input_size, state_dim, hidden_size, output_size, num_layers, prediction_horizon, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 train_gamma=0.99):
        super().__init__()
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
        self.hx_fc = nn.Linear(self.future_num*hidden_size, hidden_size)
        self.device = device
        self.train_gamma = train_gamma
        #self.state_gammas = Tensor([[train_gamma**j for i in range(state_dim)] for j in range(future_num)]).to(device)
        #self.reward_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)
        #self.dones_gammas = Tensor([train_gamma**i for i in range(future_num)]).unsqueeze(1).to(device)

    # predicted_values from LSTM (batchsize, future_num, state_dim+1)
    # past_time_features (batchsize, sequence_num, state_dim+action_dim)
    def forward(self, predicted_values, past_time_features, next_state = None, next_reward = None):
        batch_size = past_time_features.shape[0]
        # xy (batch, 1, state_dim+1)
        xy = torch.zeros(size=(batch_size, 1, self.output_size)).float().to(self.device)
        # hx: (batch_size, 1, (state_dim+action_dim)*sequence_num)
        #hx = past_time_features.reshape(-1, 1, self.hidden_size)
        hx = past_time_features.reshape(1, -1, self.hidden_size)
        # hx: (1, batch_size, (state_dim+action_dim)*sequence_num)
        #hx = hx.permute(1, 0, 2)
        pred_wp = torch.empty((batch_size, 0, self.output_size)).to(self.device)
        # h = future_num
        for i in range(self.h):
            padding_len = self.future_num - i - 1
            padding = torch.zeros(size=(past_time_features.shape[0], padding_len, self.output_size)).float().to(self.device)
            ins = torch.cat([padding, pred_wp, predicted_values[:, i:i+1, :]], dim=1) # x

            #ins = torch.cat([xy, predicted_values[:, i:i+1, :]], dim=1) # x
            # ins: (batch, 2, state_dim+1)
            # hx: (1, batch, (state_dim+action_dim)*sequence_num)            
            #hx, _ = self.gru(ins, hx.contiguous())
            #print("hx", hx.shape, "ins", ins.shape)
            hx, _ = self.gru(ins, hx)
            #print("hx", hx.shape)
            # hx: (batch, 2, (state_dim+action_dim)*sequence_num)
            hx = hx.reshape(-1, self.future_num*self.hidden_size)
            hx = self.hx_fc(hx)
            # hx: (batch, (state_dim+action_dim)*sequence_num)
            d_xy = self.mlp(hx).reshape(-1, 1, self.output_size) #control v4
            # d_xy: (batch, 1, state_dim+1)
            hx = hx.reshape(1, -1, self.hidden_size)
            # print("dxy", d_xy)
            #(256,1,4) + (256,1,4)
            xy = xy + d_xy
            #xy = predicted_values[:, i:i+1, :] + d_xy
            pred_wp = torch.cat((pred_wp, xy), dim=1)
        #pred_wp = torch.stack(out_wp, dim=1).squeeze(2)
        #pred_wp = torch.cat(out_wp, dim=1)
        state_pred = pred_wp[:,:,:self.state_dim]
        reward_pred = pred_wp[:,:,self.state_dim:self.state_dim+1]
        #done_pred = pred_wp[:,:,self.state_dim+1:]
        # (256, 5, 4)
        if next_state is not None and next_reward is not None:
            #next_state_diff = (((state_pred - next_state) * self.state_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            next_state_diff = ((state_pred - next_state) ** 2).flatten(1,-1).sum(-1).mean()
            #reward_diff = (((reward_pred - next_reward) * self.reward_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            reward_diff = ((reward_pred - next_reward) ** 2).flatten(1,-1).sum(-1).mean()
            #done_diff = (((done_pred - next_done) * self.dones_gammas) ** 2).flatten(1,-1).sum(-1).mean()
            loss = next_state_diff + reward_diff # + done_diff
            return state_pred, reward_pred, loss
        return state_pred, reward_pred, None
    '''

class SeqModel(nn.Module):

    def __init__(self, use_gru_update: bool, state_dim: int, action_dim: int, hidden_dim: int, sequence_num: int, future_num: int, 
                 device: torch.device, train_gamma: float):
        super().__init__()
        '''
        self.prep = nn.Sequential(
            nn.Flatten(1, -1),
            nn.BatchNorm1d(state_dim * sequence_num),
            nn.Unflatten(1, (sequence_num, state_dim)),
        )
        '''
        self.dynamics_nn = Dynamics(state_dim, action_dim, hidden_dim, sequence_num, future_num, device, train_gamma)
        self.gru_nn = GRU_update(state_dim+1, state_dim, (state_dim+action_dim)*sequence_num, state_dim+1, 1, future_num, device, train_gamma)
        self.use_gru_update = use_gru_update
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_num = sequence_num
        self.future_num = future_num
        
    def forward(self, state, action, hc_in=None, next_state=None, next_reward=None, detach=False):
        next_state_pred, next_reward_pred, hc, loss1 = self.dynamics_nn(state, action, hc_in, next_state, next_reward)
        if detach:
            next_state_pred = next_state_pred.detach()
            next_reward_pred = next_reward_pred.detach()
        loss2 = None
        if self.use_gru_update:
            assert state.shape[1] >= self.sequence_num
            input = torch.cat((state[:,:self.sequence_num], action[:,:self.sequence_num]), dim=-1)
            pred_features = torch.cat((next_state_pred, next_reward_pred), dim=-1)
            next_state_pred, next_reward_pred, loss2 = self.gru_nn(pred_features, input, next_state, next_reward)
        return next_state_pred, next_reward_pred, hc, loss1, loss2

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
    #print("z", z, "angle", angle, "state", state)

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
        self.x = 0

    def load_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path, map_location=self.device)
        self.config_dict = checkpoint['config']
        self.seq_model = SeqModel(self.config_dict["use_gru_update"], self.state_dim, self.action_dim,
                                    self.config_dict['hidden_dim'], self.config_dict['sequence_num'], 
                                    future_num=self.config_dict['future_num'],
                                    device=self.device, train_gamma=self.config_dict['gamma']).to(self.device)
        self.seq_model.load_state_dict(checkpoint["seq_model"])
        self.sequence_num = self.config_dict['sequence_num']
        self.state_mean = Tensor(str_to_floats(self.config_dict['state_mean'])).to(self.device)
        self.state_std = Tensor(str_to_floats(self.config_dict['state_std'])).to(self.device)
        self.reward_mean = self.config_dict['reward_mean']
        self.reward_std = self.config_dict['reward_std']

    def load_vae_from_chkpt(self, chk_path: str) -> None:
        checkpoint = torch.load(chk_path, map_location=self.device)
        self.threshold = checkpoint["threshold"]
        self.vae_config = checkpoint["config"]        
        self.vae = VAE((self.state_dim+self.action_dim)*self.vae_config["vae_sequence_num"], self.vae_config["vae_latent_dim"], 
                       self.vae_config['vae_hidden_dim']).to(self.device)
        self.vae.load_state_dict(checkpoint["vae"])

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std
    
    def denormalize_state(self, state: Tensor) -> Tensor:
        return state * self.state_std + self.state_mean
    
    def normalize_reward(self, reward: Tensor) -> Tensor:
        return (reward - self.reward_mean) / self.reward_std
    
    def denormalize_reward(self, reward: Tensor) -> Tensor:
        return reward * self.reward_std + self.reward_mean

    def reset(self, obs: ObsType) -> None:
        self.states = torch.zeros((self.sequence_num-1, self.state_dim)).to(self.device)
        obs_ = self.normalize_state(obs).to(self.device)
        self.states = torch.cat((self.states, obs_), dim=0)
        self.actions = torch.zeros((self.sequence_num-1, self.action_dim)).to(self.device)
        self.rewards = torch.empty((0)).to(self.device)
        self.istep = 0
        self.elbos = []
        self.x = 0
        self.hc = None

    @torch.no_grad()
    def _cal_sensitivity(self, states: ObsType, actions: ActType, next_state: ObsType, reward, K=10, sigma=0.01):
        # generate perturbations of size (K, states.shape[0], states.shape[1])
        states_noise = torch.normal(mean=0, std=sigma, size=(K, states.shape[1], states.shape[2])).to(self.device)
        actions_noise = torch.normal(mean=0, std=sigma, size=(K, actions.shape[1], actions.shape[2])).to(self.device)
        noisy_states = states + states_noise
        noisy_actions = actions + actions_noise
        noisy_nstate, noisy_reward, *_ = self.seq_model(noisy_states, noisy_actions)
        
        noisy_nstate = noisy_nstate.squeeze(1)
        noisy_reward = noisy_reward.squeeze(1)

        # calculate sensitivity
        s1 = torch.var(noisy_nstate - next_state, dim=0).mean()
        s2 = torch.var(noisy_reward - reward, dim=0).mean()
        return s1, s2
        
    @torch.no_grad()
    def step(self, action: ActType, use_sensitivity: bool = False) -> Tuple[ObsType, float, bool]:
        action = action.to(self.device)
        self.actions = torch.cat((self.actions, action), dim=0)
        assert self.states.shape[0] == self.actions.shape[0]
        assert self.states.shape[0] >= self.sequence_num
        states_ = self.states[-self.sequence_num:].unsqueeze(0)
        actions_ = self.actions[-self.sequence_num:].unsqueeze(0)

        self.seq_model.eval()
        next_state, reward, self.hc, *_ = self.seq_model(states_, actions_, hc_in=self.hc)
        
        next_state = next_state[:, 0]
        reward = reward[:, 0]

        #calculate elbo
        elbo = 0.0
        vae_sequence_num = self.vae_config["vae_sequence_num"]
        if self.states.shape[0] >= vae_sequence_num:
            #print("states", self.states.shape, "actions", self.actions.shape)
            vae_states = self.states[-vae_sequence_num:].unsqueeze(0)
            vae_actions = self.actions[-vae_sequence_num:].unsqueeze(0)
            vae_input = torch.cat((vae_states, vae_actions), dim=-1)
            vae_input = vae_input.view(1, -1).to(self.device)
            self.vae.eval()
            elbo = self.vae.estimate(vae_input)

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
        #print("next_state", next_state.shape, "self.states", self.states.shape)
        self.states = torch.cat((self.states, next_state), dim=0)
        # (N)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.istep += 1

        next_state = self.denormalize_state(next_state)
        reward = self.denormalize_reward(reward)

        self.elbos.append(elbo)
        done = False
        if "hopper" in self.chkpt_path:
            done = hopper_is_done(next_state[0])
        
        '''
        if done:
            print("istep", self.istep, "d1", done1, "d2", done2, "d3", done3, "elbo", elbo, "cutoff", elbo_cutoff)
            print("elbos mean", torch.mean(torch.tensor(self.elbos)).item(), "elbos std", torch.std(torch.tensor(self.elbos)).item(),
                    "elbos max", torch.max(torch.tensor(self.elbos)).item(), "elbos min", torch.min(torch.tensor(self.elbos)).item())
        '''
        #print("threshold", self.threshold)
        prob = torch.tensor([1/(1+self.kappa*(e-self.threshold)) if e > self.threshold else 1.0 for e in elbo]).to(self.device)
        discounted_reward = reward * prob
        #print("next_state", next_state, "reward", reward, "done", done, "prob", prob, "elbo", elbo, "discounted_reward", discounted_reward)
        #if "hopper" in self.chkpt_path:
        #    done = hopper_is_done(next_state[0])                                                                                                                                                                                                                                                                 
        return next_state, reward, done, prob, elbo, discounted_reward, s_state, s_reward                                                                                                                                                                                                         

def test_hopper():
    chkpt_path = "MBORL/config/hopper/hopper_medium_v2_ar.pt"
    vae_chkpt_path = "MBORL/config/hopper/hopper_medium_v2_vae.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 11
    action_dim = 3
    env = MyEnv(chkpt_path, state_dim, action_dim, device, vae_chkpt_path=vae_chkpt_path)
    obs = torch.randn(1, state_dim).to(device)
    print("init state", obs.cpu().numpy())
    env.reset(obs)
    done = False
    i = 0
    while not done:
        action = torch.randn(1, action_dim).to(device)
        next_state, reward, done, prob, elbo, discounted_reward, s_state, s_reward = env.step(action)
        print("step", i+1)
        print("action", action.cpu().numpy())
        print("next_state", next_state.cpu().numpy())
        print("reward", reward.cpu().numpy())
        print("done", done)
        print("prob", prob.cpu().numpy())
        print("elbo", elbo.cpu().numpy())
        print("discounted_reward", discounted_reward.cpu().numpy())
        print("s_state", s_state, "s_reward", s_reward)
        print("")
        done = hopper_is_done(next_state[0]) or (i>1 and elbo > 50.0)
        i += 1

def main():
    test_hopper()


if __name__ == "__main__":
    main()
