import uuid

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from typing import TypeVar, List, Tuple, Dict, Any, Sequence, Union
from dataclasses import asdict, dataclass
import pyrallis
from torch import Tensor
from tqdm.auto import trange

from env import Env, Dynamics, Reward, Dones, ObsType, ActType

@dataclass
class Config:
    project: str = "ORL_optimizer"
    group: str = "env"
    name: str = "env_train"
    env_name: str = "HalfCheetah-v3"
    chkpt_path: str = "/home/james/sfu/ORL_optimizer/OtherModels/chkpt/halfcheetah_medium_v2.pt"
    load_chkpt: bool = True    
    save_chkpt_per: int = 1000
    sequence_num: int = 5
    hidden_dim: int = 256
    dones_cutoff: float = 0.6
    num_epochs: int = 10000
    eval_episodes: int = 100
    batch_size: int = 1024
    dynamics_lr: float = 1e-3
    rewards_lr: float = 1e-3
    dones_lr: float = 1e-3
    eval_seed: int = 0
    train_seed: int = 0
    eval_randomize: bool = False
    train_randomize: bool = True

    def refresh_name(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"

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

def sample_batch_online(
    env: gym.Env, batch_size: int, sequence_num: int, randomize: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    if randomize:
        seed = np.random.randint(0, 1e18)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    samples_left = batch_size
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    states___ = torch.empty((0, sequence_num, state_dim))
    actions___ = torch.empty((0, sequence_num, action_dim))
    rewards___ = torch.empty((0, sequence_num))
    dones___ = torch.empty((0, sequence_num))
    next_states___ = torch.empty((0, sequence_num, state_dim))

    while samples_left > 0:
        obs = env.reset()
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for _ in range(samples_left+sequence_num-1):
            action = env.action_space.sample()
            states.append(obs)
            obs, reward, done, _ = env.step(action)
            next_states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            if done:
                break
        samples_cnt = len(states)-sequence_num+1
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        states_ = [states[i:i+samples_cnt] for i in range(sequence_num)]
        states__ = torch.stack(states_, dim=1)
        states___ = torch.cat((states___, states__), dim=0)
        actions_ = [actions[i:i+samples_cnt] for i in range(sequence_num)]
        actions__ = torch.stack(actions_, dim=1)
        actions___ = torch.cat((actions___, actions__), dim=0)
        rewards_ = [rewards[i:i+samples_cnt] for i in range(sequence_num)]
        rewards__ = torch.stack(rewards_, dim=1)
        rewards___ = torch.cat((rewards___, rewards__), dim=0)
        dones_ = [dones[i:i+samples_cnt] for i in range(sequence_num)]
        dones__ = torch.stack(dones_, dim=1)
        dones___ = torch.cat((dones___, dones__), dim=0)
        next_states_ = [next_states[i:i+samples_cnt] for i in range(sequence_num)]
        next_states__ = torch.stack(next_states_, dim=1)
        next_states___ = torch.cat((next_states___, next_states__), dim=0)
        samples_left -= samples_cnt

    batch = {
        "states": states___,
        "actions": actions___,
        "rewards": rewards___,
        "dones": dones___,
        "next_states": next_states___,}
    return batch
