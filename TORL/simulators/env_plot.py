import os

import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
from torch import Tensor
import torch
from dataclasses import dataclass, asdict
import pyrallis
import uuid
from tqdm.auto import trange

from env_util_offline import Config, qlearning_dataset2, get_env_info, sample_batch_offline, get_rsquare, str_to_floats
from env_mod import Dynamics, GRU_update


from mpl_interactions import ioff, panhandler, zoom_factory
import matplotlib.pyplot as plt


def plot_states(states, states_nar, states_ar, plot_idx, draw_cnt=50):
    with plt.ioff():
        figure, axis = plt.subplots()    
    plt.plot(states[0, :, plot_idx], label="True")
    plt.plot(states_nar[0, :, plot_idx], label="NAR")
    plt.plot(states_ar[0, :, plot_idx], label="AR")
    ymin = min(np.min(states[0, :, plot_idx]), np.min(states_nar[0, :, plot_idx]), np.min(states_ar[0, :, plot_idx]))
    ymax = max(np.max(states[0, :, plot_idx]), np.max(states_nar[0, :, plot_idx]), np.max(states_ar[0, :, plot_idx]))
    ymean = (ymin+ymax)/2
    ydiff = (ymax-ymin)/2
    plt.ylim(ymean-ydiff*5, ymean+ydiff*5)
    plt.xlabel("Time")
    plt.ylabel("State [" + str(plot_idx)+"]")
    plt.legend()
    disconnect_zoom = zoom_factory(axis)
    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(figure)
    plt.show()

def plot_rewards(rewards, rewards_nar, rewards_ar):
    with plt.ioff():
        figure, axis = plt.subplots()    
    plt.plot(rewards[0, :], label="True")
    plt.plot(rewards_nar[0, :], label="NAR")
    plt.plot(rewards_ar[0, :], label="AR")
    ymin = min(np.min(rewards[0, :]), np.min(rewards_nar[0, :]), np.min(rewards_ar[0, :]))
    ymax = max(np.max(rewards[0, :]), np.max(rewards_nar[0, :]), np.max(rewards_ar[0, :]))
    ymean = (ymin+ymax)/2
    ydiff = (ymax-ymin)/2
    plt.ylim(ymean-ydiff*5, ymean+ydiff*5)
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend()
    disconnect_zoom = zoom_factory(axis)
    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(figure)    
    plt.show()

@pyrallis.wrap()
def main(config: Config):
    np.random.seed(config.eval_seed)
    torch.manual_seed(config.eval_seed)
    dsnames = config.dataset_name.split(",")
    state_mean = str_to_floats(config.state_mean)
    state_std = str_to_floats(config.state_std)
    env = gym.make(dsnames[0])
    _, _, data_test, *_ = qlearning_dataset2(dsnames)
    N = data_test['state'].shape[0]
    state_dim, action_dim = get_env_info(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num, device=device)
    gru_nn = GRU_update(state_dim+1, state_dim, (state_dim+action_dim)*config.sequence_num, state_dim+1, 1, config.future_num)
    if os.path.exists(config.chkpt_path_nar):
        checkpoint = torch.load(config.chkpt_path_nar)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        gru_nn.load_state_dict(checkpoint["gru_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path_nar)
    else:
        print("No checkpoint found at", config.chkpt_path_nar)
        return

    dynamics_nn.to(device)
    gru_nn.to(device)
 
    states, actions, next_states, rewards = sample_batch_offline(
            data_test, 1, config.sequence_num, config.future_num, 
            config.out_state_num, randomize=config.eval_randomize)
    rewards = rewards.to(device)
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.cpu().detach().numpy()
    with torch.inference_mode():
        next_states_pred_nar, rewards_pred_nar = dynamics_nn(states, actions, is_eval=True, is_ar=False)
    if config.use_gru_update:
        input = torch.cat((states, actions), dim=2)
        input = input[:, :config.sequence_num]
        pred_features = torch.cat((next_states_pred_nar.detach(), rewards_pred_nar.detach()), dim=2)
        with torch.inference_mode():
            g_pred = gru_nn(pred_features, input)
        g_states_pred = g_pred[:, :, :state_dim]
        g_rewards_pred = g_pred[:, :, state_dim:]
        next_states_pred_nar = g_states_pred
        rewards_pred_nar = g_rewards_pred

    next_states_pred_nar = next_states_pred_nar.cpu().detach().numpy()
    rewards_pred_nar = rewards_pred_nar.cpu().detach().numpy()
    ############################################################## batch nar
    seq_idx = 0
    ss_pred_nar = torch.empty((0, config.future_num, state_dim)).to(device)
    sr_pred_nar = torch.empty((0, config.future_num, 1)).to(device)
    ns_pred_nar = torch.empty((0, config.future_num, state_dim))
    nr_pred_nar = torch.empty((0, config.future_num, 1))
    while seq_idx < N-1:
        states1, actions1, next_states1, rewards1, seq_idx = sample_batch_offline(
                data_test, 1, config.sequence_num, config.future_num, 
                config.out_state_num, randomize=config.eval_randomize, seq_idx=seq_idx)
        states1 = states1.to(device)
        actions1 = actions1.to(device)
        with torch.inference_mode():
            ss_pred_nar_, sr_pred_nar_ = dynamics_nn(states1, actions1, is_eval=True, is_ar=False)
        if config.use_gru_update:
            input = torch.cat((states1, actions1), dim=2)
            input = input[:, :config.sequence_num]
            pred_features = torch.cat((ss_pred_nar_.detach(), sr_pred_nar_.detach()), dim=2)
            with torch.inference_mode():
                g_pred = gru_nn(pred_features, input)
            g_states_pred = g_pred[:, :, :state_dim]
            g_rewards_pred = g_pred[:, :, state_dim:]
            ss_pred_nar_ = g_states_pred
            sr_pred_nar_ = g_rewards_pred
        ss_pred_nar = torch.cat((ss_pred_nar, ss_pred_nar_), dim=0)
        sr_pred_nar = torch.cat((sr_pred_nar, sr_pred_nar_), dim=0)
        ns_pred_nar = torch.cat((ns_pred_nar, next_states1), dim=0)
        nr_pred_nar = torch.cat((nr_pred_nar, rewards1), dim=0)
        #print("seq_idx", seq_idx, "N", N)
    
    #print("ss_pred_nar", ss_pred_nar.shape, "sr_pred_nar", sr_pred_nar.shape, "ns_pred_nar", ns_pred_nar.shape, "nr_pred_nar", nr_pred_nar.shape)
   
    ss_pred_nar = ss_pred_nar.cpu().detach().numpy()
    sr_pred_nar = sr_pred_nar.cpu().detach().numpy()
    ns_pred_nar = ns_pred_nar.cpu().detach().numpy()
    nr_pred_nar = nr_pred_nar.cpu().detach().numpy()
    s_r2_nar = get_rsquare(ns_pred_nar, ss_pred_nar)
    r_r2_nar = get_rsquare(nr_pred_nar, sr_pred_nar)
    print("State R^2 NAR:", s_r2_nar, "Reward R^2 NAR:", r_r2_nar)
    for i in range(0, state_dim):
        rsqr_nar = get_rsquare(ns_pred_nar[:,:,i:i+1], ss_pred_nar[:,:,i:i+1])
        print("state", i, "R^2 NAR:", rsqr_nar)

    ##############################################################3
    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num, device=device)
    gru_nn = GRU_update(state_dim+1, state_dim, (state_dim+action_dim)*config.sequence_num, state_dim+1, 1, config.future_num)
    if os.path.exists(config.chkpt_path_ar):
        checkpoint = torch.load(config.chkpt_path_ar)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        gru_nn.load_state_dict(checkpoint["gru_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path_ar)
    else:
        print("No checkpoint found at", config.chkpt_path_ar)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)
    gru_nn.to(device)

    with torch.inference_mode():
        next_states_pred_ar, rewards_pred_ar = dynamics_nn(states, actions, is_eval=True, is_ar=True)
    if config.use_gru_update:
        input = torch.cat((states, actions), dim=2)
        input = input[:, :config.sequence_num]
        pred_features = torch.cat((next_states_pred_ar.detach(), rewards_pred_ar.detach()), dim=2)
        with torch.inference_mode():
            g_pred = gru_nn(pred_features, input)
        g_states_pred = g_pred[:, :, :state_dim]
        g_rewards_pred = g_pred[:, :, state_dim:]

        next_states_pred_ar = g_states_pred
        rewards_pred_ar = g_rewards_pred

    next_states_pred_ar = next_states_pred_ar.cpu().detach().numpy()
    rewards_pred_ar = rewards_pred_ar.cpu().detach().numpy()

    rewards = rewards.cpu().detach().numpy()

    ############################################################## batch ar
    seq_idx = 0
    ss_pred_ar = torch.empty((0, config.future_num, state_dim)).to(device)
    sr_pred_ar = torch.empty((0, config.future_num, 1)).to(device)
    ns_pred_ar = torch.empty((0, config.future_num, state_dim))
    nr_pred_ar = torch.empty((0, config.future_num, 1))
    while seq_idx < N-1:
        states2, actions2, next_states2, rewards2, seq_idx = sample_batch_offline(
                data_test, 1, config.sequence_num, config.future_num, 
                config.out_state_num, randomize=config.eval_randomize, seq_idx=seq_idx)
        states2 = states2.to(device)
        actions2 = actions2.to(device)
        with torch.inference_mode():
            ss_pred_ar_, sr_pred_ar_ = dynamics_nn(states2, actions2, is_eval=True, is_ar=True)
        if config.use_gru_update:
            input = torch.cat((states2, actions2), dim=2)
            input = input[:, :config.sequence_num]
            pred_features = torch.cat((ss_pred_ar_.detach(), sr_pred_ar_.detach()), dim=2)
            with torch.inference_mode():
                g_pred = gru_nn(pred_features, input)
            g_states_pred = g_pred[:, :, :state_dim]
            g_rewards_pred = g_pred[:, :, state_dim:]
            ss_pred_ar_ = g_states_pred
            sr_pred_ar_ = g_rewards_pred
        ss_pred_ar = torch.cat((ss_pred_ar, ss_pred_ar_), dim=0)
        sr_pred_ar = torch.cat((sr_pred_ar, sr_pred_ar_), dim=0)
        ns_pred_ar = torch.cat((ns_pred_ar, next_states2), dim=0)
        nr_pred_ar = torch.cat((nr_pred_ar, rewards2), dim=0)
        #print("seq_idx", seq_idx)
    
    #print("ss_pred_nar", ss_pred_nar.shape, "sr_pred_nar", sr_pred_nar.shape, "ns_pred_nar", ns_pred_nar.shape, "nr_pred_nar", nr_pred_nar.shape)
   
    ss_pred_ar = ss_pred_ar.cpu().detach().numpy()
    sr_pred_ar = sr_pred_ar.cpu().detach().numpy()
    ns_pred_ar = ns_pred_ar.cpu().detach().numpy()
    nr_pred_ar = nr_pred_ar.cpu().detach().numpy()
    s_r2_ar = get_rsquare(ns_pred_ar, ss_pred_ar)
    r_r2_ar = get_rsquare(nr_pred_ar, sr_pred_ar)
    print("State R^2 AR:", s_r2_ar, "Reward R^2 AR:", r_r2_ar)
    for i in range(0, state_dim):
        r2_ar = get_rsquare(ns_pred_ar[:,:,i:i+1], ss_pred_ar[:,:,i:i+1])
        print("state", i, "R^2 AR:", r2_ar)

    ##############################################################3

    plot_rewards(rewards, rewards_pred_nar, rewards_pred_ar)

    #print("next_states", next_states.shape, "next_states_pred_nar", next_states_pred_nar.shape, "next_states_pred_ar", next_states_pred_ar.shape)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 0)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 5)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 10)


if __name__ == "__main__":
    main()
        



    