import os

import uuid

import numpy as np
import flax.linen as nn
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

from env_util import Config, make_env, sample_batch_online, get_env_info


@pyrallis.wrap()
def main(config: Config):
    config.name = "env_eval_online"
    config.refresh_name()
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
    eval_env = make_env(config.env_name, config.eval_seed)
    state_dim, action_dim = get_env_info(eval_env)

    dynamics_nn = Dynamics(state_dim, action_dim, config.hidden_dim, config.sequence_num)
    reward_nn = Reward(state_dim, action_dim, config.hidden_dim, config.sequence_num)
    dones_nn = Dones(state_dim, action_dim, config.hidden_dim, config.sequence_num)

    if config.load_chkpt and os.path.exists(config.chkpt_path):
        checkpoint = torch.load(config.chkpt_path)
        dynamics_nn.load_state_dict(checkpoint["dynamics"])
        reward_nn.load_state_dict(checkpoint["rewards"])
        dones_nn.load_state_dict(checkpoint["dones"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)
    reward_nn.to(device)
    dones_nn.to(device)

    criterion = nn.MSELoss()

    dynamics_losses = np.array([])
    rewards_losses = np.array([])
    dones_losses = np.array([])
    
    t = trange(config.eval_episodes, desc="Evaluating")
    for epoch in t:
        batch = sample_batch_online(eval_env, config.batch_size, config.sequence_num, config.eval_randomize)
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.inference_mode(): 
            dynamics = dynamics_nn(batch["states"], batch["actions"])
            reward = reward_nn(batch["states"], batch["actions"])
            dones = dones_nn(batch["states"], batch["actions"])
        dynamics_loss = criterion(dynamics, batch["next_states"][:,-1,:])
        reward_loss = criterion(reward, batch["rewards"][:,-1])
        dones_loss = criterion(dones, batch["dones"][:,-1])
        t.set_description(f"DL:{dynamics_loss:.6f} RL: {reward_loss:.6f} DL:{dones_loss:.6f}")
        dynamics_losses = np.append(dynamics_losses, dynamics_loss.cpu().numpy())
        rewards_losses = np.append(rewards_losses, reward_loss.cpu().numpy())
        dones_losses = np.append(dones_losses, dones_loss.cpu().numpy())
        wandb.log({"dynamics_loss": dynamics_loss, 
                "dynamics_loss_mean": np.mean(dynamics_losses),
                "dynamics_loss_std": np.std(dynamics_losses),                
                "reward_loss": reward_loss,
                "reward_loss_mean": np.mean(rewards_losses),
                "reward_loss_std": np.std(rewards_losses),
                "dones_loss": dones_loss,
                "dones_loss_mean": np.mean(dones_losses),
                "dones_loss_std": np.std(dones_losses),
                   })
    
    wandb.finish()

if __name__ == "__main__":
    main()