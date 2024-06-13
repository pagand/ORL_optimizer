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

from env_util_offline import Config, qlearning_dataset, get_env_info, sample_batch_offline, get_rsquare
from env import Dynamics


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
    plt.ylim(ymin*5, ymax*5)
    plt.xlabel("Time")
    plt.ylabel("State [" + str(plot_idx)+"]")
    plt.legend()
    disconnect_zoom = zoom_factory(axis)
    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(figure)
    plt.show()

@pyrallis.wrap()
def main(config: Config):
    np.random.seed(config.eval_seed)
    env = gym.make(config.dataset_name)
    dataset = qlearning_dataset(env)
    state_dim, action_dim = get_env_info(env)

    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num)
    if os.path.exists(config.chkpt_path_nar):
        checkpoint = torch.load(config.chkpt_path_nar)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path_nar)
    else:
        print("No checkpoint found at", config.chkpt_path_nar)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)
 
    states, actions, next_states, rewards = sample_batch_offline(
            dataset, 1, config.sequence_num, config.future_num, 
            config.out_state_num, is_eval=True, randomize=config.eval_randomize)
    rewards = rewards.to(device)
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.cpu().detach().numpy()
    with torch.inference_mode():
        next_states_pred_nar, rewards_pred = dynamics_nn(states, actions, is_eval=True, is_ar=False)

    next_states_pred_nar = next_states_pred_nar.cpu().detach().numpy()
    rsqr_nar = get_rsquare(next_states_pred_nar[:,:,:state_dim], next_states[:,:,:state_dim])

    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num)
    if os.path.exists(config.chkpt_path_ar):
        checkpoint = torch.load(config.chkpt_path_ar)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path_ar)
    else:
        print("No checkpoint found at", config.chkpt_path_ar)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)

    with torch.inference_mode():
        next_states_pred_ar, rewards_pred = dynamics_nn(states, actions, is_eval=True, is_ar=True)

    next_states_pred_ar = next_states_pred_ar.cpu().detach().numpy()
    rsqr_ar = get_rsquare(next_states_pred_ar[:,:,:state_dim], next_states[:,:,:state_dim])

    print("R^2 NAR:", rsqr_nar, "R^2 AR:", rsqr_ar)

    for i in range(0, state_dim):
        rsqr_nar = get_rsquare(next_states_pred_nar[0,:,i], next_states[0,:,i])
        rsqr_ar = get_rsquare(next_states_pred_ar[0,:,i], next_states[0,:,i])
        print("state", i, "R^2 NAR:", rsqr_nar, "R^2 AR:", rsqr_ar)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 0)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 5)

    plot_states(next_states, next_states_pred_nar, next_states_pred_ar, 12)

if __name__ == "__main__":
    main()
        



    