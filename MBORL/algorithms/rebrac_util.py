import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility
os.environ["WANDB_MODE"] = "online"  # For cloud sync

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import wandb
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch import Tensor
import time

@dataclass
class Config:
    # wandb params
    project: str 
    group: str 
    name: str
    # model params
    actor_learning_rate: float
    critic_learning_rate: float
    hidden_dim: int 
    actor_n_hiddens: int
    critic_n_hiddens: int
    gamma: float
    tau: float 
    actor_bc_coef: float 
    critic_bc_coef: float 
    actor_ln: bool 
    critic_ln: bool 
    policy_noise: float
    noise_clip: float 
    policy_freq: int 
    normalize_q: bool 
    # training params
    dataset_name: str
    batch_size: int
    num_epochs: int 
    num_updates_on_epoch: int
    normalize_reward: bool 
    normalize_states: bool 
    # evaluation params
    eval_episodes: int 
    eval_every: int 
    # general params
    train_seed: int
    eval_seed: int 
    #myenv params
    chkpt_path: str 
    vae_chkpt_path: str
    use_gym_env: bool 
    eval_step_limit: int 
    eval_total_steps: int
    augment_step_limit: int
    augment_total_steps: int    
    save_chkpt_path: str 
    save_chkpt_per: int
    sim_kappa: float
    replay_buffer_size: int
    highreward_buffer_size: int
    lowreward_buffer_size: int
    d4rl_ratio: float
    myenv_ratio: float
    highreward_ratio: float
    lowreward_ratio: float
    elbo_cutoff: float
    elbo_threshold: float
    use_augment_data: bool 
    augment_per: int 
    augment_episode: int 
    sensitivity_threshold: float 
    reward_penalize: bool 

    def refresh_name(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:4]}"


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def get_env_info(env: gym.Env) -> Tuple[int, int]:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim

def get_d4rl_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        next_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(next_action)
        reward_.append(reward)
        done_.append(done_bool)

    #print("state", np.array(obs_), "action", np.array(action_), "next_state", np.array(next_obs_), "reward", np.array(reward_), "done", np.array(done_))
    '''
    x = 0
    for i in range(100):
        x = halfcheetah_reward(x, np.array(obs_)[i], np.array(action_)[i], np.array(next_obs_)[i], np.array(reward_)[i])
    '''
    return {
        "state": np.array(obs_),
        "action": np.array(action_),
        "next_state": np.array(next_obs_),
        "next_action": np.array(next_action_), 
        "reward": np.array(reward_),
        "done": np.array(done_),
    }

def sample_batch_d4rl(
    dataset: Dict,
    batch_size: int,
    randomize: bool = False,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    '''
    output:
    state: (batch_size, state_dim)
    action: (batch_size, action_dim)
    next_state: (batch_size, state_dim)
    reward: (batch_size, )
    done: (batch_size, )
    '''
    if randomize:
        current_time = int(time.time())
        print("current_time", current_time)
        np.random.seed(current_time)
    
    N = dataset["state"].shape[0]
    idx = np.random.randint(0, N, batch_size)
    states = dataset["state"][idx]
    actions = dataset["action"][idx]
    next_states = dataset["next_state"][idx]
    next_actions = dataset["next_action"][idx]
    rewards = dataset["reward"][idx]
    dones = dataset["done"][idx]

    return {
        "state": torch.tensor(states, dtype=torch.float32).to(device),
        "action": torch.tensor(actions, dtype=torch.float32).to(device),
        "next_state": torch.tensor(next_states, dtype=torch.float32).to(device),
        "next_action": torch.tensor(next_actions, dtype=torch.float32).to(device),
        "reward": torch.tensor(rewards, dtype=torch.float32).to(device),
        "done": torch.tensor(dones, dtype=torch.float32).to(device),
    }

class Metrics:
    accumulators: Dict[str, Tuple[float, int]]

    def __init__(self, accumulators: Dict[str, Tuple[float, int]]):
        self.accumulators = accumulators

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (0.0, 0) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            acc, steps = self.accumulators[key]
            self.accumulators[key] = (acc + value, steps + 1)

    def compute(self) -> Dict[str, float]:
        # cumulative_value / total_steps
        return {k: (v[0] / v[1]) for k, v in self.accumulators.items()}
    
    def reset(self):
        for key in self.accumulators:
            self.accumulators[key] = (0.0, 0)

def halfcheetah_reward(x, state, action, next_state, true_reward):

    dt = 0.05
    x_before = x
    #x_after = next_state[8]*dt + x_before
    x_after = (state[8] + next_state[8])/2*dt + x_before
    x_velocity = (x_after - x_before) / dt

    control_cost = 0.1 * np.sum(np.square(action))
    forward_reward = 1.0 * x_velocity
    reward = forward_reward - control_cost
    print("reward", reward, "true_reward", true_reward)
    return x_after


def test():
    dataset_name = "halfcheetah-medium-v2"
    train_seed = 0
    batch_size = 1024
    env = make_env(dataset_name, train_seed)
    dataset = get_d4rl_dataset(env)
    batch = sample_batch_d4rl(dataset, batch_size, True)
    for key, value in batch.items():
        print(key, value.shape)



def main():
    test()
    
if __name__ == "__main__":
    main()