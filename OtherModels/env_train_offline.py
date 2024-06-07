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
    config.name = "env_train_offline"
    config.refresh_name()
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    np.random.seed(config.train_seed)
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
    if config.load_chkpt and os.path.exists(config.chkpt_path):
        checkpoint = torch.load(config.chkpt_path)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", config.chkpt_path)

    dynamics_optimizer = torch.optim.Adam(dynamics_nn.parameters(), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    criterion = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn.to(device)

    dynamics_losses = np.array([])
 
    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        states, actions, next_states = sample_batch_offline(
            dataset, config.batch_size, config.sequence_num, config.future_num, config.out_state_num, is_eval=False,
            randomize=config.train_randomize)
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        dynamics_nn.train()
        next_states_pred = dynamics_nn(states, actions)
        loss = criterion(next_states_pred, next_states)
        dynamics_optimizer.zero_grad()
        loss.backward()
        dynamics_optimizer.step()
        loss_ = loss.cpu().detach().numpy()
        dynamics_losses = np.append(dynamics_losses, loss_)
        t.set_description(f"DL:{loss_:.6f} DLM:{np.mean(dynamics_losses):.6f})")

        wandb.log({"dynamics_loss": loss.item(),
                     "dynamics_loss_mean": np.mean(dynamics_losses),
                     "dynamics_loss_std": np.std(dynamics_losses)
        })

        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            torch.save({
                "dynamics_nn": dynamics_nn.state_dict(),
                "config": asdict(config)
            }, config.chkpt_path)

    wandb.finish()
    print("Checkpoint saved to", config.chkpt_path)

if __name__ == "__main__":
    main()
        



    