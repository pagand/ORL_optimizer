import os

import math
import uuid
from copy import deepcopy

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

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
    dataset_name: str = "halfcheetah-medium-v2"
    state_dim: int = 17
    action_dim: int = 6
    min_action: float = -1.0
    max_action: float = 1.0
    chkpt_path: str = "/home/james/sfu/ORL_optimizer/OtherModels/chkpt/halfcheetah_medium_v2.pt"
    load_chkpt: bool = True    
    sequence_num: int = 5
    hidden_dim: int = 256
    dones_cutoff: float = 0.6
    eval_episodes: int = 10
    eval_every: int = 50
    eval_seed: int = 42
    num_epochs: int = 10000
    batch_size: int = 1024
    dynamics_lr: float = 1e-3
    rewards_lr: float = 1e-3
    dones_lr: float = 1e-3

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)
            #new_accumulators[key] = (value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}
        #return {k: np.array(v[0]) for k, v in self.accumulators.items()}

def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def sample_batch_online(
    env: gym.Env, batch_size: int, sequence_num: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    env.seed(np.random.randint(0, 1e18))
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

@pyrallis.wrap()
def main(config: Config):
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    wandb.mark_preempting()
    train_env = make_env(config.dataset_name, config.eval_seed)
    batch = sample_batch_online(train_env, config.batch_size, config.sequence_num)

    dynamics_nn = Dynamics(config.state_dim, config.action_dim, config.hidden_dim, config.sequence_num)
    reward_nn = Reward(config.state_dim, config.action_dim, config.hidden_dim, config.sequence_num)
    dones_nn = Dones(config.state_dim, config.action_dim, config.hidden_dim, config.sequence_num)

    if config.load_chkpt and os.path.exists(config.chkpt_path):
        checkpoint = torch.load(config.chkpt_path)
        dynamics_nn.load_state_dict(checkpoint["dynamics"])
        reward_nn.load_state_dict(checkpoint["reward"])
        dones_nn.load_state_dict(checkpoint["dones"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path)
    
    dynamics_optimizer = optim.Adam(dynamics_nn.parameters(), lr=config.dynamics_lr)
    reward_optimizer = optim.Adam(reward_nn.parameters(), lr=config.rewards_lr)
    dones_optimizer = optim.Adam(dones_nn.parameters(), lr=config.dones_lr)

    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)
    reward_nn.to(device)
    dones_nn.to(device)
    
    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        batch = sample_batch_online(train_env, config.batch_size, config.sequence_num)
        for k, v in batch.items():
            batch[k] = v.to(device)

        dynamics_nn.train()
        reward_nn.train()
        dones_nn.train()
        dynamics = dynamics_nn(torch.cat((batch["states"], batch["actions"]), dim=-1))
        reward = reward_nn(torch.cat((batch["states"], batch["actions"]), dim=-1))
        dones = dones_nn(torch.cat((batch["states"], batch["actions"]), dim=-1))
        dynamics_loss = criterion(dynamics, batch["next_states"][:,-1,:])
        reward_loss = criterion(reward, batch["rewards"][:,-1])
        dones_loss = criterion(dones, batch["dones"][:,-1])
        dynamics_optimizer.zero_grad()
        reward_optimizer.zero_grad()
        dones_optimizer.zero_grad()
        dynamics_loss.backward()
        reward_loss.backward()
        dones_loss.backward()
        dynamics_optimizer.step()
        reward_optimizer.step()
        dones_optimizer.step()
        t.set_description(f"DL:{dynamics_loss:.8f} RL: {reward_loss:.8f} DL:{dones_loss:.8f}")
        wandb.log({"dynamics_loss": dynamics_loss, "reward_loss": reward_loss, "dones_loss": dones_loss})
    
    wandb.finish()
    # save models to checkpoint
    torch.save({
        "dynamics": dynamics_nn.state_dict(),
        "reward": reward_nn.state_dict(),
        "dones": dones_nn.state_dict(),
        "config": asdict(config)
    }, config.chkpt_path)

    print("Checkpoint saved to", config.chkpt_path)

if __name__ == "__main__":
    main()