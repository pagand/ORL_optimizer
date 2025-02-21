import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from typing import Any, Callable, Dict, Sequence, Tuple, Union, List
import numpy as np
from torch import Tensor
import torch
from dataclasses import dataclass
import uuid
import time
from copy import deepcopy

@dataclass
class Config:
    project: str 
    group: str 
    name: str 
    dataset_name: str 
    chkpt_path_nar: str
    chkpt_path_ar: str 
    vae_chkpt_path: str 
    load_chkpt: bool 
    save_chkpt_per: int
    sequence_num: int 
    future_num: int
    out_state_num: int 
    hidden_dim: int 
    num_epochs: int 
    eval_episodes: int
    batch_size: int 
    dynamics_lr: float 
    dynamics_weight_decay: float 
    eval_seed: int 
    train_seed: int
    eval_randomize: bool 
    train_randomize: bool 
    is_ar: bool
    gamma: float 
    holdout_per: int
    holdout_num: int
    holdout_randomize: bool 
    use_gru_update: bool 
    state_mean: str 
    state_std: str 
    reward_mean: float 
    reward_std: float 
    vae_hidden_dim: int 
    vae_latent_dim: int
    vae_num_epochs: int 
    vae_num_updates_per_epoch: int 
    vae_save_chkpt_per: int 

    def refresh_name(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"

def str_to_floats(data: str) -> np.ndarray:
    return np.array([float(x) for x in data.split(",")])

def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state - mean) / std

def denormalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return state * std + mean

def qlearning_dataset2(
        dsnames: List[str],
        use_normalize_state: bool = True,
        ext_mean: np.ndarray = None,
        ext_std: np.ndarray = None,
        ext_mean_reward: float = None,
        ext_std_reward: float = None,
        use_ext_mean_std: bool = False,
        verbose: bool = False,
        terminate_on_end: bool = False,
        **kwargs,
):
    perc_train = 0.8
    perc_holdout = 0.1
    data_train = dict()
    data_holdout = dict()
    data_test = dict()
    init = True
    data = None
    for dsname in dsnames:
        data_ = qlearning_dataset_(dsname, terminate_on_end, **kwargs)
        if data is None:
            data = data_
        else:
            for key in data_.keys():
                data[key] = np.concatenate((data[key], data_[key]), axis=0)

        N = data_["state"].shape[0]
        for i in range(N):
            if data_["episode"][i] == 0 and i>N*perc_train:
                N_holdout = i
                break
        for i in range(N):
            if data_["episode"][i] == 0 and i>N*(perc_train+perc_holdout):
                N_test = i
                break

        if init:
            for key in data_.keys():
                data_train[key] = data_[key][:N_holdout]
                data_holdout[key] = data_[key][N_holdout:N_test]
                data_test[key] = data_[key][N_test:]
            init = False
        else:
            for key in data_.keys():
                data_train[key] = np.concatenate((data_train[key], data_[key][:N_holdout]), axis=0)
                data_holdout[key] = np.concatenate((data_holdout[key], data_[key][N_holdout:N_test]), axis=0)
                data_test[key] = np.concatenate((data_test[key], data_[key][N_test:]), axis=0)

    mean = np.mean(data["state"], axis=0)
    std = np.std(data["state"], axis=0)
    mean_reward = np.mean(data["reward"], axis=0)
    std_reward = np.std(data["reward"], axis=0)
    if verbose:
        print_array(mean, "state mean")
        print_array(std, "state std")
        print("mean reward", mean_reward, "std reward", std_reward)

    for dt in [data_train, data_holdout, data_test]:                
        start_ = []
        stop_ = []

        episode_ = dt["episode"]
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
        dt["start"] = np.array(start_)
        dt["stop"] = np.array(stop_)
        if use_normalize_state:
            if use_ext_mean_std:
                dt["state"] = normalize_state(dt["state"], ext_mean, ext_std)
                dt["next_state"] = normalize_state(dt["next_state"], ext_mean, ext_std)
                dt["reward"] = (dt["reward"] - ext_mean_reward) / ext_std_reward
            else:
                dt["state"] = normalize_state(dt["state"], mean, std)
                dt["next_state"] = normalize_state(dt["next_state"], mean, std)
                dt["reward"] = (dt["reward"] - mean_reward) / std_reward

    return data_train, data_holdout, data_test, mean, std, mean_reward, std_reward
    
def print_array(data, name):
    print(name, end=": ")
    for i in range(data.shape[0]):
        print(data[i], end=",")
    print()

def qlearning_dataset_(
    dsname: str,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
    env = gym.make(dsname)
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



    return {
        "state": np.array(obs_),
        "action": np.array(action_),
        "next_state": np.array(next_obs_),
        "reward": np.array(reward_),
        "done": np.array(done_),
        "episode": np.array(episode_),
    }


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
    seq_idx: int = -1,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    '''
    output:
    states: (batch_size, sequence_num, state_dim)
    actions: (batch_size, sequence_num+future_num-1+out_state_num-1, action_dim)
    next_states: (batch_size, future_num, out_state_num * state_dim)
    rewards: (batch_size, future_num, out_state_num)
    '''
    if randomize:
        current_time = int(time.time())
        np.random.seed(current_time)
    N = dataset["state"].shape[0]
    samples_left = batch_size
    state_dim = dataset["state"].shape[1]
    action_dim = dataset["action"].shape[1]
    states = torch.empty((0, sequence_num+future_num-1, state_dim))
    actions = torch.empty((0, sequence_num+future_num+out_state_num-2, action_dim))
    rewards = torch.empty((0, sequence_num+future_num-1, 1))
    next_states = torch.empty((0, future_num, out_state_num*state_dim))
    nrewards = torch.empty((0, future_num, out_state_num))
    while samples_left > 0 and seq_idx < N-1:
        if seq_idx == -1:
            index = np.random.randint(0, N)
            start = dataset["start"][index]
            stop_ = dataset["stop"][index]
        else:
            for i in range(seq_idx, N):
                if dataset["episode"][i] == 0:
                    start = dataset["start"][i]
                    stop_ = dataset["stop"][i+1]
                    seq_idx = stop_
                    break
        #print("start", start, "stop_", stop_, "N", N)
        if stop_-start < sequence_num+future_num+out_state_num-1:
            continue
        step = min(samples_left, stop_-start-sequence_num-future_num-out_state_num+2)
        stop = start + step
        #print("start", start, "stop", stop, "stop_", stop_, "samples_left", samples_left)
        states_ = np.stack([dataset["state"][i:i+sequence_num+future_num-1] for i in range(start, stop)], axis=0)
        states = torch.cat((states, torch.tensor(states_, dtype=torch.float32)), dim=0)

        actions_ = np.stack([dataset["action"][i:i+sequence_num+future_num+out_state_num-2] for i in range(start, stop)], axis=0)
        actions = torch.cat((actions, torch.tensor(actions_, dtype=torch.float32)), dim=0)

        #rewards__ = np.expand_dims(dataset["reward"], axis=-1)
        #rewards_ = np.stack([rewards__[i:i+sequence_num+future_num-1] for i in range(start, stop)], axis=0)
        #rewards = torch.cat((rewards, torch.tensor(rewards_, dtype=torch.float32)), dim=0)
        #print("rewards__ shape", rewards__.shape, "rewards_ shape", rewards_.shape, "rewards shape", rewards.shape)
        next_states__ = np.array([np.concatenate(dataset["next_state"][i:i+out_state_num], axis=-1)
                                        for i in range(start+sequence_num-1, stop+sequence_num+future_num-1-1)])
        next_states_ = np.stack([next_states__[i:i+future_num] for i in range(0, step)], axis=0)        
        next_states = torch.cat((next_states, torch.tensor(next_states_, dtype=torch.float32)), dim=0)

        #print("rewards", dataset["reward"].shape)
        # (DS, 1)
        nrewards___ = np.expand_dims(dataset["reward"], axis=-1)
        # (future+step-1, out_state)
        nrewards__ = np.array([np.concatenate(nrewards___[i:i+out_state_num], axis=-1)
                                        for i in range(start+sequence_num-1, stop+sequence_num+future_num-1-1)])
        # (batch, future, out_state)
        nrewards_ = np.stack([nrewards__[i:i+future_num] for i in range(0, step)], axis=0)
        nrewards = torch.cat((nrewards, torch.tensor(nrewards_, dtype=torch.float32)), dim=0)

        samples_left -= step
    if seq_idx == -1:
        return states, actions, next_states, nrewards
    else:
        return states, actions, next_states, nrewards, seq_idx

def get_rsquare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true-y_mean)**2)
    ss_res = np.sum((y_true-y_pred)**2)
    return 1 - ss_res/ss_tot

def get_vae_sample(data, batch_size, device):
    N = data['state'].shape[0]
    indices = np.random.choice(N, batch_size)
    states = data['state'][indices]
    actions = data['action'][indices]
    return Tensor(np.concatenate((states, actions), axis=-1)).to(device)