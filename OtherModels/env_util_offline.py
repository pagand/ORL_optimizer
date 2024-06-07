import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
from torch import Tensor
import torch
from dataclasses import dataclass
import uuid
import time

@dataclass
class Config:
    project: str = "ORL_optimizer"
    group: str = "env"
    name: str = "env_train_offline"
    dataset_name: str = "halfcheetah-medium-v2"
    chkpt_path: str = "/home/james/sfu/ORL_optimizer/OtherModels/chkpt/halfcheetah_medium_v2.pt"
    load_chkpt: bool = True
    save_chkpt_per: int = 1000
    sequence_num: int = 5
    future_num: int = 20
    out_state_num: int = 3
    hidden_dim: int = 256
    num_epochs: int = 10000
    eval_episodes: int = 100
    batch_size: int = 512
    dynamics_lr: float = 1e-3
    dynamics_weight_decay: float = 1e-6
    eval_seed: int = 0
    train_seed: int = 0
    eval_randomize: bool = True
    train_randomize: bool = True

    def refresh_name(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


def qlearning_dataset(
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
    reward_ = []
    done_ = []
    episode_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
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
        reward_.append(reward)
        done_.append(done_bool)
        episode_.append(episode_step)
        episode_step += 1

    start_ = []
    stop_ = []

    start = 0
    for i in range(len(episode_)):
        #print("i", i, "episode_[i]", episode_[i])
        if episode_[i] == 0:
            start = i
        start_.append(start)
    stop = len(episode_)
    for i in range(len(episode_)-1, -1, -1):
        if episode_[i] == 0:
            stop = i
        stop_.append(stop)
    stop_.reverse()

    return {
        "state": np.array(obs_),
        "action": np.array(action_),
        "next_state": np.array(next_obs_),
        "reward": np.array(reward_),
        "done": np.array(done_),
        "episode": np.array(episode_),
        "start": np.array(start_),
        "stop": np.array(stop_),
    }

def get_env_info(env: gym.Env) -> Tuple[int, int]:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim

def sample_batch_offline(
    dataset: Dict,
    batch_size: int,
    sequence_num: int,
    future_num: int,
    out_state_num: int,
    randomize: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    '''
    output:
    states: (batch_size, sequence_num, state_dim)
    actions: (batch_size, sequence_num+future_num-1+out_state_num-1, action_dim)
    next_states: (batch_size, future_num, out_state_num * state_dim)
    '''
    if randomize:
        current_time = int(time.time())
        np.random.seed(current_time)
    N = dataset["state"].shape[0]
    samples_left = batch_size
    state_dim = dataset["state"].shape[1]
    action_dim = dataset["action"].shape[1]
    states = torch.empty((0, sequence_num, state_dim))
    actions = torch.empty((0, sequence_num+future_num+out_state_num-2, action_dim))
    next_states = torch.empty((0, future_num, out_state_num*state_dim))
    while samples_left > 0:
        index = np.random.randint(N)
        start = dataset["start"][index]
        stop_ = dataset["stop"][index]
        if stop_-start < sequence_num+future_num+out_state_num-1:
            continue
        step = min(samples_left, stop_-start-sequence_num-future_num-out_state_num+2)
        stop = start + step
        #print("start", start, "stop", stop, "stop_", stop_, "samples_left", samples_left)
        states_ = np.stack([dataset["state"][i:i+sequence_num] for i in range(start, stop)], axis=0)
        states = torch.cat((states, torch.tensor(states_, dtype=torch.float32)), dim=0)

        actions_ = np.stack([dataset["action"][i:i+sequence_num+future_num+out_state_num-2] for i in range(start, stop)], axis=0)
        actions = torch.cat((actions, torch.tensor(actions_, dtype=torch.float32)), dim=0)
        
        next_states__ = np.array([np.concatenate(dataset["next_state"][i:i+out_state_num])
                                        for i in range(start+sequence_num-1, stop+sequence_num+future_num-1-1)])
        next_states_ = np.stack([next_states__[i:i+future_num] for i in range(0, step)], axis=0)        
        next_states = torch.cat((next_states, torch.tensor(next_states_, dtype=torch.float32)), dim=0)
        samples_left -= step
    return states, actions, next_states
