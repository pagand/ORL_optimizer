from rebrac_mod import DetActor, Critic, EnsembleCritic, TrainState

from rebrac_util import make_env, get_env_info, get_d4rl_dataset, sample_batch_d4rl, Config, Metrics

from rebrac_update import update_actor, update_critic

from typing import Dict, Tuple
import wandb
import torch
from torch import Tensor
import pyrallis
from torch import nn
from dataclasses import dataclass, asdict

import numpy as np

import uuid
from tqdm.auto import trange
import gym

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from simulators import env_mod

def evaluate(
    env: gym.Env,
    actor: nn.Module,
    num_episodes: int,
    seed: int,
    device: torch.device
):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    init_obs = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, done = env.reset(), False
        init_obs.append(obs)
        total_reward = 0.0
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = actor(obs).detach().cpu().numpy()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    init_obs = np.array(init_obs)
    return np.array(returns), init_obs

def evaluate_simulator(
    env: env_mod.MyEnv,
    actor: nn.Module,
    num_episodes: int,
    init_obs: np.ndarray,
    device: torch.device
):
    returns = []
    for i in trange(num_episodes, desc="Eval Simulator", leave=False):
        obs = init_obs[i:i+1, :]
        obs = Tensor(obs)
        env.reset(obs)
        done = False
        total_reward = 0.0
        while not done:
            obs = obs.to(device)
            with torch.no_grad():
                action = actor(obs).detach()
            obs, reward, done = env.step(action)
            total_reward += float(reward)
            #print("reward", reward, "total_reward", total_reward)
        returns.append(total_reward)
    return np.array(returns)

@pyrallis.wrap()
def main(config: Config):

    config.name = "rebrac_gym_main"
    config.refresh_name()
    dict_config = asdict(config)

    np.random.seed(config.train_seed)
    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_env(config.dataset_name, config.train_seed)
    eval_env = make_env(config.dataset_name, config.eval_seed)
    state_dim, action_dim = get_env_info(env)

    myenv = env_mod.MyEnv(config.chkpt_path, state_dim, action_dim, device, eval_env._max_episode_steps)

    dataset = get_d4rl_dataset(env)
    actor = DetActor(state_dim=state_dim, action_dim=action_dim, 
                     hidden_dim=config.hidden_dim, layernorm=config.actor_ln, 
                     n_hiddens=config.actor_n_hiddens)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate,
                                   betas=(0.9, 0.999), eps=1e-8)
    actor_state = TrainState(actor, actor_optim)
    critic = EnsembleCritic(state_dim=state_dim, action_dim=action_dim, 
                            hidden_dim=config.hidden_dim, num_critics=2,
                            layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate,
                                    betas=(0.9, 0.999), eps=1e-8)
    critic_state = TrainState(critic, critic_optim)
    actor_state.to(device)
    critic_state.to(device)

    metrics = Metrics.create(["critic_loss", "qmin", "actor_loss", "bc_mse_policy", "action_mse"])

    t = trange(config.num_epochs, desc="Training")
    for epoch in t:
        for i in range(config.num_updates_on_epoch):
            batch = sample_batch_d4rl(dataset, config.batch_size, randomize=False, device=device)
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
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns, init_obs = evaluate(
                eval_env,
                actor_state.get_model(),
                config.eval_episodes,
                seed=config.eval_seed,
                device=device
            )
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            sim_returns = evaluate_simulator(
                myenv,
                actor_state.get_model(),
                config.eval_episodes,
                init_obs,
                device
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                    "eval/normalized_score_mean": np.mean(normalized_score),
                    "eval/normalized_score_std": np.std(normalized_score),
                    "eval/sim_return_mean": np.mean(sim_returns),
                    "eval/sim_return_std": np.std(sim_returns),
                }
            )
        t.set_postfix(
            {
                "RM": np.mean(eval_returns),
                "SM": np.mean(sim_returns),
            }
        )
                
if __name__ == "__main__":
    main()


