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
from env_mod import Dynamics
from torch import nn

class MeanFourthPowerError(nn.Module):
    def __init__(self):
        super(MeanFourthPowerError, self).__init__()

    def forward(self, inputs, targets):
        # Compute the fourth power of the element-wise difference
        loss = torch.mean((inputs - targets) ** 4)
        return loss

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num, device=device)
    if config.is_ar:
        chkpt_path = config.chkpt_path_ar
    else:
        chkpt_path = config.chkpt_path_nar
    if config.load_chkpt and os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", chkpt_path)

    dynamics_optimizer = torch.optim.Adam(dynamics_nn.parameters(), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    criterion = torch.nn.MSELoss()
    #criterion = MeanFourthPowerError()


    dynamics_nn.to(device)

    dynamics_losses = np.array([])
    hold_out_losses = np.array([])
 
    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        states, actions, next_states, rewards = sample_batch_offline(
            dataset, config.batch_size, config.sequence_num, config.future_num, config.out_state_num, is_eval=False,
            is_holdout=False, randomize=config.train_randomize)
        rewards = rewards.to(device)
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        dynamics_nn.train()
        next_states_pred, rewards_pred = dynamics_nn(states, actions, is_eval=False, is_ar=config.is_ar)
        #print("rewards_pred", rewards_pred.shape, "rewards", rewards.shape)
        #print("next_states_pred", next_states_pred.shape, "next_states", next_states.shape)
        loss = criterion(next_states_pred, next_states) + criterion(rewards_pred, rewards)
        dynamics_optimizer.zero_grad()
        loss.backward()
        dynamics_optimizer.step()
        loss_ = loss.cpu().detach().numpy()
        dynamics_losses = np.append(dynamics_losses, loss_)
        if config.holdout_per>0 and len(hold_out_losses)>0:
            t.set_description(f"DL:{loss_:.6f} DLM:{np.mean(dynamics_losses):.6f} HLM:{np.mean(hold_out_losses[:]):.6f}")
        else:
            t.set_description(f"DL:{loss_:.6f} DLM:{np.mean(dynamics_losses):.6f})")

        wandb.log({"dynamics_loss": loss.item(),
                     "dynamics_loss_mean": np.mean(dynamics_losses),
                     "dynamics_loss_std": np.std(dynamics_losses)
        })

        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            torch.save({
                "dynamics_nn": dynamics_nn.state_dict(),
                "config": asdict(config)
            }, chkpt_path)

        #evaluate holdout data
        if config.holdout_per>0 and (epoch % config.holdout_per == 0 or epoch == config.num_epochs-1):
            for j in range(config.holdout_num):
                states, actions, next_states, rewards = sample_batch_offline(
                    dataset, config.holdout_num, config.sequence_num, config.future_num, config.out_state_num, is_eval=True,
                    is_holdout=True, randomize=config.holdout_randomize)
                rewards = rewards.to(device)
                states = states.to(device)
                actions = actions.to(device)
                next_states = next_states.to(device)
                dynamics_nn.eval()
                with torch.inference_mode():
                    next_states_pred, rewards_pred = dynamics_nn(states, actions, is_eval=True, is_ar=config.is_ar)
                loss = criterion(next_states_pred, next_states)
                loss = loss.cpu().detach().numpy()
                hold_out_losses = np.append(hold_out_losses, loss)                
                wandb.log({"holdout_loss": loss, 
                            "holdout_loss_mean": np.mean(hold_out_losses),
                            "holdout_loss_std": np.std(hold_out_losses)
                })

    wandb.finish()
    print("Checkpoint saved to", chkpt_path)

if __name__ == "__main__":
    main()
        



    