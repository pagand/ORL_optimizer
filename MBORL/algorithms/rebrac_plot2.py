from rebrac_model import DetActor, Critic, EnsembleCritic, TrainState
from rebrac_util import make_env, get_env_info, get_d4rl_dataset, sample_batch_d4rl, Config, Metrics

from rebrac_update import update_actor, update_critic 

from typing import Dict, Tuple
import wandb
import torch
from torch import Tensor
import pyrallis
from torch import nn
from dataclasses import dataclass, asdict
from copy import deepcopy

import numpy as np

import uuid
from tqdm.auto import trange
import gym

import sys
import os

from sklearn.metrics import r2_score
from rebrac_eval import evaluate, evaluate_simulator, augment_replay_buffer
from replay_buffer import ReplayBuffer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from simulators import env_model

from mpl_interactions import ioff, panhandler, zoom_factory
import matplotlib.pyplot as plt


def plot_simgym(sim_states, gym_states, desc, dones):
    with plt.ioff():
        figure, axis = plt.subplots()    
    # do not link the data with lines
    plt.plot(sim_states, label="Simulator", linestyle='none', marker='o')
    plt.plot(gym_states, label="Gym", linestyle='none', marker='x')
    # use different background color for dones
    for i in range(len(dones)):
        if dones[i]:
            plt.axvspan(i-0.5, i+0.5, color='gray', alpha=0.5)
    #plt.plot(dones, label="Dones")
    ymin = min(np.min(sim_states), np.min(gym_states))
    ymax = max(np.max(sim_states), np.max(gym_states))
    ymean = (ymin+ymax)/2
    ydiff = (ymax-ymin)/2
    plt.ylim(ymean-ydiff*5, ymean+ydiff*5)
    plt.xlabel("Time")
    plt.ylabel(desc)
    plt.legend()
    disconnect_zoom = zoom_factory(axis)
    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(figure)
    plt.show()

def plot_sim(sim_values, desc, dones):
    with plt.ioff():
        figure, axis = plt.subplots()    
    # do not link the data with lines
    plt.plot(sim_values, label=desc, linestyle='none', marker='o')
    # use different background color for dones
    for i in range(len(dones)):
        if dones[i]:
            plt.axvspan(i-0.5, i+0.5, color='gray', alpha=0.5)
    ymin = np.min(sim_values)
    ymax = np.max(sim_values)
    ymean = (ymin+ymax)/2
    ydiff = (ymax-ymin)/2
    plt.ylim(ymean-ydiff*5, ymean+ydiff*5)
    plt.xlabel("Time")
    plt.ylabel(desc)
    plt.legend()
    disconnect_zoom = zoom_factory(axis)
    # Enable scrolling and panning with the help of MPL
    # Interactions library function like panhandler.
    pan_handler = panhandler(figure)
    plt.show()


def get_rsquare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true-y_mean)**2)
    ss_res = np.sum((y_true-y_pred)**2)
    return 1 - ss_res/ss_tot

@torch.no_grad()
def run_gym(
    env: gym.Env,
    actor: nn.Module,
    init_obs: np.ndarray,
    done_cnt: int, # how many dones to wait before stopping
    device: torch.device
):
    states = []
    actions = []
    rewards = []
    dones = []
    total_reward = 0.0
    done_accum = 0
    obs = init_obs
    step = 0
    healthy_state_range=(-100.0, 100.0)
    while done_accum < done_cnt:
        action = actor(Tensor(obs).unsqueeze(0).to(device))
        action = action.squeeze(0).cpu().numpy()
        obs, reward, done, _ = env.step(action)
        if done:
            done_accum += 1
            # check the index of the obs that is unhealthy
            if np.any((obs < healthy_state_range[0]) | (obs > healthy_state_range[1])):
                unhealthy_idx = np.where((obs < healthy_state_range[0]) | (obs > healthy_state_range[1]))[0]
                print("Unhealthy state", unhealthy_idx)
            
        total_reward += reward
        step += 1
        states.append(obs)
        actions.append(action)
        rewards.append(total_reward)
        dones.append(done)
    return np.array(states), np.array(actions), np.array(rewards), np.array(dones)

# run simulated env for one trajectory
@torch.no_grad()
def run_simulator(
    env: env_model.MyEnv,
    actions: np.ndarray,
    init_obs: np.ndarray,
    step_limit: int,
    elbo_cutoff: float,
    device: torch.device,
):
    rewards = []
    states = []
    elbos = []

    step = 0
    total_reward = 0.0
    obs = Tensor(init_obs).unsqueeze(0).to(device)
    env.reset(obs)
    for step in range(len(actions)):
        action = Tensor(actions[step]).unsqueeze(0).to(device)
        #print("action", action.shape)
        obs, reward, done,  prob, elbo, discounted_reward, *_ = env.step(action, 
                            use_sensitivity=False)
        total_reward += float(reward.cpu().item())
        rewards.append(total_reward)
        states.append(obs.squeeze(0).cpu().numpy())
        elbos.append(elbo.cpu().item())
        step+=1
        # let step > ??? to stabilize the elbo values
        #if (step>15) and (done or step >= step_limit or elbo > elbo_cutoff):
        #    break
    return np.array(states), np.array(rewards), np.array(elbos)

@pyrallis.wrap(config_path="MBORL/config/hopper/rebrac_hopper_medium_v2.yaml")
def main(config: Config):

    epoch_plot = 5 # which epoch to plot
    min_plot_steps = 2000 # minimum steps to plot

    config.name = "rebrac_plot"
    config.refresh_name()
    dict_config = asdict(config)

    np.random.seed(config.train_seed)
    torch.manual_seed(config.train_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get basic info about the environment
    eval_env = make_env(config.dataset_name, config.eval_seed)
    state_dim, action_dim = get_env_info(eval_env)

    # simulated environment
    myenv = env_model.MyEnv(config.chkpt_path, state_dim, action_dim, device, config.eval_step_limit, 
                          vae_chkpt_path=config.vae_chkpt_path, kappa=config.sim_kappa)

    actor = DetActor(state_dim=state_dim, action_dim=action_dim, 
                     hidden_dim=config.hidden_dim, layernorm=config.actor_ln, 
                     n_hiddens=config.actor_n_hiddens)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate,
                                   betas=(0.9, 0.999), eps=1e-8)
    actor_target_model = deepcopy(actor)

    critic = EnsembleCritic(state_dim=state_dim, action_dim=action_dim, 
                            hidden_dim=config.hidden_dim, num_critics=2,
                            layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens)
    critic_target_model = deepcopy(critic)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate,
                                    betas=(0.9, 0.999), eps=1e-8)
    
    rebrac_chk_path = f"{config.save_chkpt_path}_{epoch_plot}.pt"
    if os.path.exists(rebrac_chk_path):
        checkpoint = torch.load(rebrac_chk_path, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        actor_target_model.load_state_dict(checkpoint["actor_target"])
        critic.load_state_dict(checkpoint["critic"])
        critic_target_model.load_state_dict(checkpoint["critic_target"])
        print("Checkpoint loaded from", rebrac_chk_path)
    else:
        print("Checkpoint not found at", rebrac_chk_path)
        return

    actor_state = TrainState(actor, actor_target_model, actor_optim)
    critic_state = TrainState(critic, critic_target_model, critic_optim)
    actor_state.to(device)
    critic_state.to(device)

    p = 0
    sim_states = np.empty((0, state_dim))
    gym_states = np.empty((0, state_dim))
    sim_dones = np.empty(0)
    sim_rewards = np.array([])
    gym_rewards = np.array([])
    sim_elbos = np.array([])
    while p < min_plot_steps:
        obs, done = eval_env.reset(), False
        g_states, actions, g_rewards, g_dones = run_gym(eval_env, actor_state.get_model(), obs, done_cnt=10, device=device)
        states, rewards, elbos = run_simulator(
            myenv,
            actions,
            obs,
            config.eval_step_limit,
            config.elbo_cutoff,
            device,
        )

        p += len(states)
        print("states", states.shape, "rewards", rewards.shape, "elbos", elbos.shape, "g_states", g_states.shape, "g_rewards", g_rewards.shape, "g_dones", g_dones.shape)
        sim_states = np.vstack((sim_states, states))
        gym_states = np.vstack((gym_states, g_states))
        sim_rewards = np.concatenate((sim_rewards, rewards))
        gym_rewards = np.concatenate((gym_rewards, g_rewards))        
        sim_dones = np.concatenate((sim_dones, g_dones))
        sim_elbos = np.concatenate((sim_elbos, elbos))

    print("Sim states", sim_states.shape)
    print("Gym states", gym_states.shape)
    print("Sim dones", sim_dones.shape)
    plot_simgym(sim_states[:, 0], gym_states[:, 0], "State [0]", sim_dones)
    plot_simgym(sim_states[:, 1], gym_states[:, 1], "State [1]", sim_dones)
    plot_simgym(sim_rewards, gym_rewards, "Rewards", sim_dones)
    plot_sim(sim_elbos, "ELBO", sim_dones)

if __name__ == "__main__":
    main()


