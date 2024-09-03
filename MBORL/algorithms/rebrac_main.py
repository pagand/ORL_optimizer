from rebrac_model import DetActor, Critic, EnsembleCritic, TrainState
from rebrac_util import make_env, get_env_info, get_d4rl_dataset, sample_batch_d4rl, Config, Metrics

from rebrac_update import update_actor, update_critic 

from typing import Dict, Tuple
import wandb
import torch
from torch import Tensor
import pyrallis
from torch import nn
from dataclasses import dataclass, asdict
from copy import deepcopy

import numpy as np

import uuid
from tqdm.auto import trange
import gym

import sys
import os

from sklearn.metrics import r2_score
from rebrac_eval import evaluate, evaluate_simulator, augment_replay_buffer
from replay_buffer import ReplayBuffer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from simulators import env_model

def get_rsquare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true-y_mean)**2)
    ss_res = np.sum((y_true-y_pred)**2)
    return 1 - ss_res/ss_tot

@pyrallis.wrap(config_path="MBORL/config/hopper/rebrac_hopper_medium_v2.yaml")
def main(config: Config):

    #config.name = "rebrac_gym_main"
    config.refresh_name()
    dict_config = asdict(config)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get basic info about the environment
    env = make_env(config.dataset_name, config.train_seed)
    eval_env = make_env(config.dataset_name, config.eval_seed)
    state_dim, action_dim = get_env_info(env)

    # simulated environment
    myenv = env_model.MyEnv(config.chkpt_path, state_dim, action_dim, device, config.eval_step_limit, 
                          vae_chkpt_path=config.vae_chkpt_path, kappa=config.sim_kappa)

    dataset = get_d4rl_dataset(env)
    # augmented or pure replay buiffer use_augment_date to false for pure replay buffer
    rbuffer = ReplayBuffer(state_dim, action_dim, config.replay_buffer_size, dataset, device)
    actor = DetActor(state_dim=state_dim, action_dim=action_dim, 
                     hidden_dim=config.hidden_dim, layernorm=config.actor_ln, 
                     n_hiddens=config.actor_n_hiddens)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate,
                                   betas=(0.9, 0.999), eps=1e-8)
    actor_target_model = deepcopy(actor)

    critic = EnsembleCritic(state_dim=state_dim, action_dim=action_dim, 
                            hidden_dim=config.hidden_dim, num_critics=2,
                            layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens)
    critic_target_model = deepcopy(critic)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate,
                                    betas=(0.9, 0.999), eps=1e-8)
    '''
    if os.path.exists(config.save_chkpt_path):
        checkpoint = torch.load(config.save_chkpt_path, map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        actor_target_model.load_state_dict(checkpoint["actor_target"])
        critic.load_state_dict(checkpoint["critic"])
        critic_target_model.load_state_dict(checkpoint["critic_target"])
        print("Checkpoint loaded from", config.save_chkpt_path)
    '''
    actor_state = TrainState(actor, actor_target_model, actor_optim)
    critic_state = TrainState(critic, critic_target_model, critic_optim)
    actor_state.to(device)
    critic_state.to(device)

    metrics = Metrics.create(["critic_loss", "qmin", "actor_loss", "bc_mse_policy", "action_mse"])

    t = trange(config.num_epochs, desc="Training")
    sim_rewards = []
    gym_rewards = []
    init_obs = np.random.uniform(-1, 1, (config.eval_episodes, state_dim))            
    steps = np.ones(config.eval_episodes) * config.eval_step_limit    
    for epoch in t:
        for i in range(config.num_updates_on_epoch):
            #batch = sample_batch_d4rl(dataset, config.batch_size, randomize=False, device=device)
            batch = rbuffer.sample(config.batch_size, config.d4rl_ratio, config.myenv_ratio, 
                                   config.highreward_ratio, config.lowreward_ratio)
            update_critic(actor_state, critic_state, batch, metrics,
                            config.gamma, config.critic_bc_coef, config.tau, 
                            config.policy_noise, config.noise_clip)            
            if i % config.policy_freq == 0:
                update_actor(actor_state, critic_state, batch, metrics,
                             config.normalize_q, config.actor_bc_coef, config.tau)

        mean_metrics = metrics.compute()
        logs = {k: v for k, v in mean_metrics.items()}
        wandb.log(logs)
        metrics.reset()

        # save checkpoints
        if (config.save_chkpt_per>0) and (epoch>0) and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs - 1):
            torch.save({
                "actor": actor_state.get_model().state_dict(),
                "actor_target": actor_state.get_target_model().state_dict(),
                "critic": critic_state.get_model().state_dict(),
                "critic_target": critic_state.get_target_model().state_dict(),
                "config": dict_config}, 
                f"{config.save_chkpt_path}_{epoch}.pt")

        # evaluations
        sim_return_mean = 0.0
        if config.eval_every>0 and (epoch % config.eval_every == 0 or epoch == config.num_epochs - 1):
            if config.use_gym_env:
                eval_returns, init_obs, steps = evaluate(
                    eval_env,
                    actor_state.get_model(),
                    config.eval_episodes,
                    seed=config.eval_seed,
                    device=device,
                    step_limit=config.eval_step_limit
                )
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0

            if config.use_sim_env:
                sim_returns = evaluate_simulator(
                    myenv,
                    actor_state.get_model(),
                    config.eval_episodes,
                    init_obs,
                    config.eval_step_limit,
                    config.eval_total_steps,
                    config.elbo_cutoff,
                    device,
                )
                #sim_normalized_score = eval_env.get_normalized_score(sim_returns) * 100.0
                #sim_returns *= 100.0
                if np.mean(sim_returns) > 0:
                    sim_return_mean = np.mean(sim_returns)

            if config.use_gym_env:
                  #rsqr = 0.0
                #if len(sim_rewards) > 1:
                #    rsqr = r2_score(sim_rewards, gym_rewards)            
                wandb.log(
                    {
                        "epoch": epoch,
                        "eval/gym_return_mean": np.mean(eval_returns),
                        "eval/gym_return_std": np.std(eval_returns),
                        "eval/gym_normalized_score_mean": np.mean(normalized_score),
                        "eval/gym_normalized_score_std": np.std(normalized_score),
                        "eval/sim_return_mean": sim_return_mean,
                        #"eval/sim_gym_rsqr": rsqr,
                    }
                )              
                t.set_postfix(
                {
                    "RM": np.mean(eval_returns),
                    "SM": sim_return_mean,
                })
            else:
                wandb.log(
                    {
                        "epoch": epoch,
                        "eval/sim_return_mean": sim_return_mean,
                    }
                )
                t.set_postfix(
                {
                    "SM": sim_return_mean,
                })
        # augment data
        if config.use_augment_data and epoch % config.augment_per == 0 and epoch > 0:
            returns = augment_replay_buffer(
                rbuffer,
                myenv,
                actor_state.get_model(),
                init_obs,
                config.augment_step_limit,
                config.augment_total_steps,
                config.reward_penalize,
                config.sensitivity_threshold,
                config.elbo_cutoff,
                config.elbo_threshold,
                device,
            )        

if __name__ == "__main__":
    main()


