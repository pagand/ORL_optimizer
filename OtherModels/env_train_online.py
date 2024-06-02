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
    config.name = "env_train_online"
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
    train_env = make_env(config.env_name, config.eval_seed)
    state_dim, action_dim = get_env_info(train_env)

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
    
    dynamics_optimizer = optim.Adam(dynamics_nn.parameters(), lr=config.dynamics_lr)
    reward_optimizer = optim.Adam(reward_nn.parameters(), lr=config.rewards_lr)
    dones_optimizer = optim.Adam(dones_nn.parameters(), lr=config.dones_lr)

    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)
    reward_nn.to(device)
    dones_nn.to(device)
    
    dynamics_losses = np.array([])
    rewards_losses = np.array([])
    dones_losses = np.array([])

    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        batch = sample_batch_online(train_env, config.batch_size, config.sequence_num)
        for k, v in batch.items():
            batch[k] = v.to(device)

        dynamics_nn.train()
        reward_nn.train()
        dones_nn.train()
        dynamics = dynamics_nn(batch["states"], batch["actions"])
        reward = reward_nn(batch["states"], batch["actions"])
        dones = dones_nn(batch["states"], batch["actions"])
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
        t.set_description(f"DL:{dynamics_loss:.6f} RL: {reward_loss:.6f} DL:{dones_loss:.6f}")
        
        dynamics_losses = np.append(dynamics_losses, dynamics_loss.cpu().detach().numpy())
        rewards_losses = np.append(rewards_losses, reward_loss.cpu().detach().numpy())
        dones_losses = np.append(dones_losses, dones_loss.cpu().detach().numpy())
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
        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            # save models to checkpoint
            torch.save({
                "dynamics": dynamics_nn.state_dict(),
                "rewards": reward_nn.state_dict(),
                "dones": dones_nn.state_dict(),
                "config": asdict(config),
                "state_dim": state_dim,
                "action_dim": action_dim,
            }, config.chkpt_path)

    wandb.finish()
    print("Checkpoint saved to", config.chkpt_path)

if __name__ == "__main__":
    main()