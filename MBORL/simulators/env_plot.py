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

from env_util import *
from env_model import SeqModel


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
    _, _, data_test, *_ = qlearning_dataset3(dsnames)
    N = len(data_test['state'])
    test_N = N
    state_dim, action_dim = get_env_info(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    tragectory_idx = N-2 # whic tragectory to plot
    tlen = len(data_test['state'][tragectory_idx])
    #print("reward", data_test['reward'][tragectory_idx])
    states_true = np.array(data_test['state'][tragectory_idx][config.sequence_num:tlen-config.future_num+1])
    rewards_true = np.array(data_test['reward'][tragectory_idx][config.sequence_num:tlen-config.future_num+1])
    states_true = states_true.reshape(1, -1, state_dim)
    rewards_true = rewards_true.reshape(1, -1)
    #print("states_true", states_true.shape, "rewards_true", rewards_true.shape)
    ############################ is_ar = False, single trajectory
    chkpt_path = config.chkpt_path_ar
    if os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=device)
        config_dict = checkpoint["config"]
        seqModel = SeqModel(config_dict["use_gru_update"], state_dim, action_dim, config_dict["hidden_dim"],
                            sequence_num=config_dict["sequence_num"], future_num=config_dict["future_num"], 
                            device=device, train_gamma=config_dict["gamma"]).to(device)
        seqModel.load_state_dict(checkpoint["seq_model"])        
        print("Checkpoint loaded from", chkpt_path)
    else:
        print("No checkpoint found at", chkpt_path)
        return

    states_plot_nar = np.empty((1, 0, state_dim))
    rewards_plot_nar = np.empty((1, 0))
    hc = None
    for u in range(0, tlen-config.sequence_num-config.future_num+1):
        states, actions, rewards = sample_batch_offline2(
                data_test, 1, config.sequence_num, config.future_num, 
                tragectory_idx=tragectory_idx, step_idx=u, device=device)
        rewards = rewards.to(device)
        states = states[:, :config.sequence_num].to(device)
        actions = actions[:, :config.sequence_num].to(device)
        with torch.inference_mode():
            seqModel.eval()
            next_states_pred_nar, rewards_pred_nar, hc, *_ = seqModel(states, actions, hc)

        next_states_pred_nar = next_states_pred_nar.cpu().detach().numpy()
        rewards_pred_nar = rewards_pred_nar.cpu().detach().numpy().reshape(1, -1)
        states_plot_nar = np.concatenate((states_plot_nar, next_states_pred_nar[:, 0:1]), axis=1)
        rewards_plot_nar = np.concatenate((rewards_plot_nar, rewards_pred_nar[:, 0:1]), axis=1)

    ############################################################## batch nar
    states_stat_true = np.empty((1, 0, state_dim))
    rewards_stat_true = np.empty((1, 0))
    states_stat_nar = np.empty((1, 0, state_dim))
    rewards_stat_nar = np.empty((1, 0))
    for tid in trange(test_N, desc="NAR", leave=False):
    #for tid in trange(N, desc="NAR", leave=False):
        #print("tid", tid, "N", N)
        tlen_ = len(data_test['state'][tid])
        states_stat_true = np.concatenate((states_stat_true, 
                        np.array(data_test['state'][tid][config.sequence_num:tlen_-config.future_num+1]).reshape(1,-1,state_dim)), axis=1)
        rewards_stat_true = np.concatenate((rewards_stat_true, 
                        np.array(data_test['reward'][tid][config.sequence_num:tlen_-config.future_num+1]).reshape(1,-1)), axis=1)
        hc = None
        for s in range(0, tlen_-config.sequence_num-config.future_num+1):
            states, actions, rewards = sample_batch_offline2(
                    data_test, 1, config.sequence_num, config.future_num, 
                    tragectory_idx=tid, step_idx=s, device=device)
            states = states[:, :config.sequence_num].to(device)
            actions = actions[:, :config.sequence_num].to(device)
            with torch.inference_mode():
                seqModel.eval()
                ss_pred_nar_, sr_pred_nar_, hc, *_ = seqModel(states, actions, hc)
            ss_pred_nar_ = ss_pred_nar_.cpu().detach().numpy()
            sr_pred_nar_ = sr_pred_nar_.cpu().detach().numpy().reshape(1, -1)
            states_stat_nar = np.concatenate((states_stat_nar, ss_pred_nar_[:, 0:1]), axis=1)
            rewards_stat_nar = np.concatenate((rewards_stat_nar, sr_pred_nar_[:, 0:1]), axis=1)
    
    s_r2_nar = get_rsquare(states_stat_nar, states_stat_true)
    r_r2_nar = get_rsquare(rewards_stat_nar, rewards_stat_true)
    print("State R^2 NAR:", s_r2_nar, "Reward R^2 NAR:", r_r2_nar)
    for i in range(0, state_dim):
        rsqr_nar = get_rsquare(states_stat_nar[:,:,i:i+1], states_stat_true[:,:,i:i+1])
        print("state", i, "R^2 NAR:", rsqr_nar)

    ############################################################## is_ar = True, one trajectory
    test_is_ar = True
    chkpt_path = config.chkpt_path_ar
    if os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=device)
        config_dict = checkpoint["config"]
        seqModel = SeqModel(config_dict["use_gru_update"], state_dim, action_dim, config_dict["hidden_dim"],
                            sequence_num=config_dict["sequence_num"], future_num=config_dict["future_num"], 
                            device=device, train_gamma=config_dict["gamma"]).to(device)
        seqModel.load_state_dict(checkpoint["seq_model"])
        print("Checkpoint loaded from", chkpt_path)
    else:
        print("No checkpoint found at", chkpt_path)
        return

    states_plot_ar = np.empty((1, 0, state_dim))
    rewards_plot_ar = np.empty((1, 0))
    hc = None
    for u in range(0, tlen-config.sequence_num-config.future_num+1):
        states, actions, rewards = sample_batch_offline2(
                data_test, 1, config.sequence_num, config.future_num, 
                tragectory_idx=tragectory_idx, step_idx=u, device=device)
        if u == 0:
            past_states = states[:, :config.sequence_num].to(device)
        if test_is_ar:
            states = past_states[:, -config.sequence_num:]
        else:
            states = states[:, :config.sequence_num].to(device)
        actions = actions[:, :config.sequence_num].to(device)
        
        with torch.inference_mode():
            seqModel.eval()
            next_states_pred_ar, rewards_pred_ar, hc, *_ = seqModel(states, actions, hc)

        past_states = torch.cat((past_states, next_states_pred_ar[:, 0:1]), dim=1)

        next_states_pred_ar = next_states_pred_ar.cpu().detach().numpy()
        rewards_pred_ar = rewards_pred_ar.cpu().detach().numpy().reshape(1, -1)
        states_plot_ar = np.concatenate((states_plot_ar, next_states_pred_ar[:, 0:1]), axis=1)
        rewards_plot_ar = np.concatenate((rewards_plot_ar, rewards_pred_ar[:, 0:1]), axis=1)

    ############################################################## batch ar
    states_stat_true = np.empty((1, 0, state_dim))
    rewards_stat_true = np.empty((1, 0))
    states_stat_ar = np.empty((1, 0, state_dim))
    rewards_stat_ar = np.empty((1, 0))
    for tid in trange(test_N, desc="AR", leave=False):
    #for tid in trange(N, desc="NAR", leave=False):
        #print("tid", tid, "N", N)
        tlen_ = len(data_test['state'][tid])
        if tlen_ < config.future_num + 1:
            continue
        states_stat_true = np.concatenate((states_stat_true, 
                        np.array(data_test['state'][tid][1:tlen_-config.future_num+1]).reshape(1,-1,state_dim)), axis=1)
        rewards_stat_true = np.concatenate((rewards_stat_true, 
                        np.array(data_test['reward'][tid][1:tlen_-config.future_num+1]).reshape(1,-1)), axis=1)

        past_states, past_actions, indices = sample_batch_init(data_test, 1, config.sequence_num,
                                                            tragectory_idx=tid, left_align=True, device=device)
        #print("past_states", past_states, "past_actions", past_actions, "indices", indices)
        hc = None
        for s in range(0, tlen_-config.future_num):
            next_states, next_actions, next_rewards = sample_batch_offline3(
                    data_test, 1, config.future_num, 
                    tragectory_idx=tid, indices=indices, device=device)

            states = past_states[:, -config.sequence_num:]
            actions = past_actions[:, -config.sequence_num:]
            with torch.inference_mode():
                seqModel.eval()
                ss_pred_ar_, sr_pred_ar_, hc, *_ = seqModel(states, actions, hc)

            past_states = torch.cat((past_states, ss_pred_ar_[:, 0:1]), dim=1)
            past_actions = torch.cat((past_actions, next_actions[:, 0:1]), dim=1)

            ss_pred_ar_ = ss_pred_ar_.cpu().detach().numpy()
            sr_pred_ar_ = sr_pred_ar_.cpu().detach().numpy().reshape(1, -1)
            states_stat_ar = np.concatenate((states_stat_ar, ss_pred_ar_[:, 0:1]), axis=1)
            rewards_stat_ar = np.concatenate((rewards_stat_ar, sr_pred_ar_[:, 0:1]), axis=1)
            indices += 1
    
    s_r2_ar = get_rsquare(states_stat_ar, states_stat_true)
    r_r2_ar = get_rsquare(rewards_stat_ar, rewards_stat_true)
    print("State R^2 AR:", s_r2_ar, "Reward R^2 AR:", r_r2_ar)
    for i in range(0, state_dim):
        rsqr_ar = get_rsquare(states_stat_ar[:,:,i:i+1], states_stat_true[:,:,i:i+1])
        print("state", i, "R^2 AR:", rsqr_ar)
    
    plot_rewards(rewards_true, rewards_plot_nar, rewards_plot_ar)

    #print("next_states", next_states.shape, "next_states_pred_nar", next_states_pred_nar.shape, "next_states_pred_ar", next_states_pred_ar.shape)

    plot_states(states_true, states_plot_nar, states_plot_ar, 0)

    plot_states(states_true, states_plot_nar, states_plot_ar, 5)


if __name__ == "__main__":
    main()
        



    