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
    project: str = "TORL"
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 0.005
    actor_bc_coef: float = 0.001
    critic_bc_coef: float = 0.01
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    # evaluation params
    eval_episodes: int = 50
    eval_every: int = 10
    # general params
    train_seed: int = 0
    eval_seed: int = 42

    def refresh_name(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


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