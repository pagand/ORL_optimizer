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

from env_util import *
from env_model import SeqModel
from torch import nn

class MeanFourthPowerError(nn.Module):
    def __init__(self):
        super(MeanFourthPowerError, self).__init__()

    def forward(self, inputs, targets):
        # Compute the fourth power of the element-wise difference
        loss = torch.mean((inputs - targets) ** 4)
        return loss

@pyrallis.wrap(config_path="MBORL/config/hopper/env_hopper_medium_v2.yaml")
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
    data_train, data_holdout, *_ = qlearning_dataset3(dsnames, verbose=True)

    state_dim, action_dim = get_env_info(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seqModel = SeqModel(config.use_gru_update, state_dim, action_dim, config.hidden_dim,
                            sequence_num=config.sequence_num, future_num=config.future_num, 
                            device=device, train_gamma=config.gamma).to(device)
    if config.is_ar:
        chkpt_path = config.chkpt_path_ar
    else:
        chkpt_path = config.chkpt_path_nar
    if config.load_chkpt and os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location=device)
        config_dict = checkpoint["config"]
        seqModel.load_state_dict(checkpoint["seq_model"])
        print("Checkpoint loaded from", chkpt_path)
    seq_optimizer = torch.optim.Adam(seqModel.parameters(), lr=config.dynamics_lr, weight_decay=config.dynamics_weight_decay)

    t = trange(config.num_epochs, desc="Training")
    N = len(data_train["state"])
    N_holdout = len(data_holdout["state"])
    for epoch in t:
        update_cnt = 0
        losses1 = np.array([])
        losses2 = np.array([])
        
        while update_cnt < config.min_updates_per_epoch:
            # choose a random number between 0 and N - batch_size
            idx = np.random.randint(0, N - config.batch_size)
            t_len = len(data_train["state"][idx])
            if t_len < config.future_num + 1:
                continue
            lalign = epoch % 2 == 0
            past_states, past_actions, indices = sample_batch_init(data_train, config.batch_size, config.sequence_num, 
                                                                    tragectory_idx=idx, left_align=lalign, device=device)
            
            for u in range(t_len-config.future_num):
                hc = None
                next_states, next_actions, next_rewards= sample_batch_offline3(data_train,
                                                                config.batch_size, 
                                                                config.future_num, 
                                                                tragectory_idx=idx, 
                                                                indices=indices,
                                                                device=device)
                states = past_states[:, -config.sequence_num:].to(device)
                actions = past_actions[:, -1].to(device)
                seqModel.train()
                next_states_pred, rewards_pred, hc, loss1, loss2 = seqModel(states, actions, hc, next_state=next_states, 
                                                                        next_reward=next_rewards)

                seq_optimizer.zero_grad()
                loss = loss1 + 3 * loss2 if loss2 is not None else loss1
                #loss = loss2
                loss.backward()
                #loss1.backward()                
                seq_optimizer.step()

                losses1 = np.append(losses1, loss1.cpu().detach().numpy())
                if loss2 is not None:
                    losses2 = np.append(losses2, loss2.cpu().detach().numpy())
                next_states_pred = next_states_pred.detach()
                past_states = torch.cat((past_states, next_states_pred[:, 0:1]), dim=1)
                past_actions = torch.cat((past_actions, next_actions[:, 0:1]), dim=1)
                indices += 1
                #print("indices", indices)
            # end of for loop
            update_cnt += u

        wandb.log({"training_loss1_mean": np.mean(losses1),
                    "training_loss2_mean": np.mean(losses2),
        })

        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            torch.save({
                "seq_model": seqModel.state_dict(),
                "config": asdict(config)
            }, chkpt_path)

        #evaluate holdout data
        if config.holdout_per>0 and (epoch % config.holdout_per == 0 or epoch == config.num_epochs-1):
            seqModel.use_gru_update = config.use_gru_update
            states_stat_true = np.empty((1, 0, state_dim))
            rewards_stat_true = np.empty((1, 0))
            states_stat_ar = np.empty((1, 0, state_dim))
            rewards_stat_ar = np.empty((1, 0))
            tids = np.random.choice(N_holdout, config.holdout_num, replace=False)
            losses1 = np.array([])
            losses2 = np.array([])
            for id in trange(config.holdout_num, desc="AR", leave=False):
            #for tid in trange(N, desc="NAR", leave=False):
                #print("tid", tid, "N", N)
                tid = tids[id]
                tlen_ = len(data_holdout['state'][tid])
                hc = None
                if tlen_ < config.future_num + 1:
                    continue

                states_stat_true = np.concatenate((states_stat_true, 
                                np.array(data_holdout['state'][tid][1:tlen_-config.future_num+1]).reshape(1,-1,state_dim)), axis=1)
                rewards_stat_true = np.concatenate((rewards_stat_true, 
                                np.array(data_holdout['reward'][tid][1:tlen_-config.future_num+1]).reshape(1,-1)), axis=1)

                past_states, past_actions, indices = sample_batch_init(data_holdout, 1, config.sequence_num, 
                                                                    tragectory_idx=tid, left_align=True, device=device)
                for s in range(tlen_-config.future_num):
                    next_states, next_actions, next_rewards = sample_batch_offline3(
                            data_holdout, 1, config.future_num, 
                            tragectory_idx=tid, indices=indices, device=device)
                    states = past_states[:, -config.sequence_num:].to(device)
                    actions = past_actions[:, -1].to(device)

                    with torch.inference_mode():
                        seqModel.eval()
                        ss_pred_ar_, sr_pred_ar_, hc, loss1, loss2 = seqModel(states, actions, hc, next_states, next_rewards)
                    losses1 = np.append(losses1, loss1.cpu().detach().numpy())
                    if loss2 is not None:
                        losses2 = np.append(losses2, loss2.cpu().detach().numpy())
                    ss_pred_ar_ = ss_pred_ar_.detach()
                    past_states = torch.cat((past_states, ss_pred_ar_[:, 0:1]), dim=1)
                    past_actions = torch.cat((past_actions, next_actions[:, 0:1]), dim=1)
                    states_stat_ar = np.concatenate((states_stat_ar, ss_pred_ar_[:, 0:1].cpu().detach().numpy()), axis=1)
                    rewards_stat_ar = np.concatenate((rewards_stat_ar, sr_pred_ar_[:, 0:1].squeeze(-1).cpu().detach().numpy()), axis=1)
                    indices += 1
                #print("states_stat_true", states_stat_true.shape, "states_stat_ar", states_stat_ar.shape)
                #print("rewards_stat_true", rewards_stat_true.shape, "rewards_stat_ar", rewards_stat_ar.shape)
                #print("states_stat_true", states_stat_true.shape, "states_stat_ar", states_stat_ar.shape)
            s_r2_ar = get_rsquare(states_stat_ar, states_stat_true)
            r_r2_ar = get_rsquare(rewards_stat_ar, rewards_stat_true)
                
            wandb.log({"validation_states_r2": s_r2_ar,
                            "validation_rewards_r2": r_r2_ar,
                            "validation_loss1_mean": np.mean(losses1),
                            "validation_loss2_mean": np.mean(losses2),
                    })
            
        t.set_description(f"ML:{np.mean(losses2):.3f} SR:{s_r2_ar:.3f} RR:{r_r2_ar:.3f}")

    wandb.finish()
    print("Checkpoint saved to", chkpt_path)

if __name__ == "__main__":
    main()
        



    