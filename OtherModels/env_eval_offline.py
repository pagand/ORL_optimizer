import os

import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
from torch import Tensor
import torch
import wandb
from dataclasses import dataclass, asdict
import pyrallis
import uuid
from tqdm.auto import trange

from env_util_offline import Config, qlearning_dataset, get_env_info, sample_batch_offline
from env import Dynamics

@pyrallis.wrap()
def main(config: Config):
    config.name = "env_eval_offline"
    config.refresh_name()
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    np.random.seed(config.eval_seed)
    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    wandb.mark_preempting()
    env = gym.make(config.dataset_name)
    dataset = qlearning_dataset(env)
    state_dim, action_dim = get_env_info(env)

    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num)
    if os.path.exists(config.chkpt_path):
        checkpoint = torch.load(config.chkpt_path)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path)
    else:
        print("No checkpoint found at", config.chkpt_path)
        return

    criterion = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)

    dynamics_losses = np.array([])
 
    t = trange(config.eval_episodes, desc="Training")
    for epoch in t:
        states, actions, next_states, rewards = sample_batch_offline(
            dataset, config.batch_size, config.sequence_num, config.future_num, 
            config.out_state_num, is_eval=True, randomize=config.eval_randomize)
        rewards = rewards.to(device)
        states = states.to(device)
        actions = actions.to(device)
        #next_states = next_states.to(device)
        with torch.inference_mode():
            next_states_pred, rewards_pred = dynamics_nn(states, actions, is_eval=True)

        next_states_pred = next_states_pred.cpu().detach()
        #print("next_states_pred", next_states_pred[0,-1,:5])
        #print("next_states", next_states[0,-1,:5])
        #loss = criterion(next_states_pred[:,-10:,:state_dim], next_states[:,-10:,:state_dim])
        loss = criterion(next_states_pred[:,:,:state_dim], next_states[:,:,:state_dim])
        loss_ = loss.cpu().detach().numpy()
        dynamics_losses = np.append(dynamics_losses, loss_)
        t.set_description(f"DL:{loss_:.6f} DLM:{np.mean(dynamics_losses):.6f})")

        wandb.log({"dynamics_loss": loss.item(),
                     "dynamics_loss_mean": np.mean(dynamics_losses),
                     "dynamics_loss_std": np.std(dynamics_losses)
        })

    wandb.finish()

if __name__ == "__main__":
    main()
        



    