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

from env_util_offline import Config, qlearning_dataset, get_env_info, sample_batch_offline, qlearning_dataset2, str_to_floats
from env_mod import Dynamics, GRU_update
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
    torch.manual_seed(config.train_seed)
    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )

    wandb.mark_preempting()
    # comma separated list of datasets
    dsnames = config.dataset_name.split(",")
    env = gym.make(dsnames[0])
    data_train, data_holdout, *_ = qlearning_dataset2(dsnames, verbose=True)
    state_dim, action_dim = get_env_info(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_nn = Dynamics(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim,
                            sequence_num=config.sequence_num, out_state_num=config.out_state_num,
                            future_num=config.future_num, device=device, train_gamma=config.gamma)
    gru = GRU_update(state_dim+1, state_dim, (state_dim+action_dim)*config.sequence_num, state_dim+1, 1, config.future_num, 
                     train_gamma=config.gamma).to(device)
    
    if config.is_ar:
        chkpt_path = config.chkpt_path_ar
    else:
        chkpt_path = config.chkpt_path_nar
    if config.load_chkpt and os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=device)
        dynamics_nn.load_state_dict(checkpoint["dynamics_nn"])
        gru.load_state_dict(checkpoint["gru_nn"])
        config_dict = checkpoint["config"]
        print("Checkpoint loaded from", chkpt_path)

    dynamics_optimizer = torch.optim.Adam(dynamics_nn.parameters(), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    #gru_optimizer = torch.optim.Adam(gru.parameters(), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    gru_optimizer = torch.optim.Adam(list(dynamics_nn.parameters())+list(gru.parameters()), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)
    criterion = torch.nn.MSELoss()
    #criterion = MeanFourthPowerError()

    dynamics_nn.to(device)

    dynamics_losses = np.array([])
    hold_out_losses = np.array([])

    # generate a geometric sequence of gammas
    hold_out_losses = np.array([])
    hold_out_loss = np.array([])
    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        states, actions, next_states, next_rewards = sample_batch_offline(
            data_train, config.batch_size, config.sequence_num, config.future_num, config.out_state_num, 
            randomize=config.train_randomize)
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        next_rewards = next_rewards.to(device)
        dynamics_nn.train()
        
        dynamics_optimizer.zero_grad()
        next_states_pred, rewards_pred, loss = dynamics_nn(states, actions, next_state=next_states, 
                                                           next_reward=next_rewards, is_eval=False, is_ar=config.is_ar)
        if config.use_gru_update:
            gru.train()
            gru_optimizer.zero_grad()
            input = torch.cat((states, actions), dim=2)
            input = input[:, :config.sequence_num]
            #pred_features = torch.cat((next_states_pred.detach(), rewards_pred.detach()), dim=2)
            pred_features = torch.cat((next_states_pred, rewards_pred), dim=2)
            g_pred, loss2 = gru(pred_features, input, next_states, next_rewards)
            #print("g_states_pred", g_pred.shape)
            g_states_pred = g_pred[:, :, :state_dim]
            g_rewards_pred = g_pred[:, :, state_dim:]
            #print("rewards_pred", rewards_pred.shape, "rewards", rewards.shape)
            #print("next_states_pred", next_states_pred.shape, "next_states", next_states.shape)
            #loss = criterion(next_states_pred, next_states) + criterion(rewards_pred, rewards)
            loss2.backward()
            gru_optimizer.step()
        else:
            # only use the dynamics loss
            loss.backward()
            dynamics_optimizer.step()

        if config.use_gru_update:
            loss_ = loss2.cpu().detach().numpy()
        else:
            loss_ = loss.cpu().detach().numpy()
        dynamics_losses = np.append(dynamics_losses, loss_)
        if config.holdout_per>0 and len(hold_out_loss)>0:
            t.set_description(f"DL:{loss_:.2f} DLM:{np.mean(dynamics_losses):.2f} HLM:{np.mean(hold_out_loss[:]):.2f}")
        else:
            t.set_description(f"DL:{loss_:.2f} DLM:{np.mean(dynamics_losses):.2f})")

        if config.use_gru_update:
            wandb.log({"dynamics_loss": loss.item(),
                        "dynamics_loss_mean": np.mean(dynamics_losses),
                        "dynamics_loss_std": np.std(dynamics_losses),
                        "GRU_loss": loss2.item(),
            })
        else:
            wandb.log({"dynamics_loss": loss.item(),
                        "dynamics_loss_mean": np.mean(dynamics_losses),
                        "dynamics_loss_std": np.std(dynamics_losses),
            })

        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            torch.save({
                "dynamics_nn": dynamics_nn.state_dict(),
                "gru_nn": gru.state_dict(),
                "config": asdict(config)
            }, chkpt_path)

        #evaluate holdout data
        if config.holdout_per>0 and (epoch % config.holdout_per == 0 or epoch == config.num_epochs-1):
            hold_out_loss = np.array([])
            for j in range(config.holdout_num):
                states, actions, next_states, next_rewards = sample_batch_offline(
                    data_holdout, config.holdout_num, config.sequence_num, config.future_num, config.out_state_num,
                    randomize=config.holdout_randomize)
                states = states.to(device)
                actions = actions.to(device)
                next_states = next_states.to(device)
                next_rewards = next_rewards.to(device)
                dynamics_nn.eval()
                
                with torch.inference_mode():
                    next_states_pred, rewards_pred, loss = dynamics_nn(states, actions,  
                                                                       next_state=next_states, next_reward=next_rewards, 
                                                                       is_eval=True, is_ar=config.is_ar)
                loss = loss.cpu().detach().numpy()
                if config.use_gru_update:
                    gru.eval()
                    input = torch.cat((states, actions), dim=2)
                    input = input[:, :config.sequence_num]
                    pred_features = torch.cat((next_states_pred, rewards_pred), dim=2)
                    g_pred, loss2 = gru(pred_features, input, next_states, next_rewards)
                    g_states_pred = g_pred[:, :, :state_dim]
                    g_rewards_pred = g_pred[:, :, state_dim:]
                    loss2 = loss2.cpu().detach().numpy()
                #loss = criterion(next_states_pred, next_states)
               
                if config.use_gru_update:
                    hold_out_loss = np.append(hold_out_loss, loss2)
                    hold_out_losses = np.append(hold_out_losses, loss2)
                else:
                    hold_out_loss = np.append(hold_out_loss, loss)                
                    hold_out_losses = np.append(hold_out_losses, loss)
                wandb.log({"holdout_loss": np.mean(hold_out_loss), 
                            "holdout_loss_mean": np.mean(hold_out_losses),
                            "holdout_loss_std": np.std(hold_out_losses)
                })
    wandb.finish()
    print("Checkpoint saved to", chkpt_path)

if __name__ == "__main__":
    main()
        



    