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

import numpy as np

import uuid
from tqdm.auto import trange
import gym

import sys
import os

from sklearn.metrics import r2_score
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


# with GYM env
def evaluate(
    env: gym.Env,
    actor: nn.Module,
    num_episodes: int,
    seed: int,
    device: torch.device,
    step_limit: int = 1e8  
):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    init_obs = []
    steps = []
    for _ in trange(num_episodes, desc="Eval Gym", leave=False):
        obs, done = env.reset(), False
        init_obs.append(obs)
        total_reward = 0.0
        step = 0
        while (not done) and step < step_limit:
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = actor(obs).detach().cpu().numpy()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step+=1
            #print("reward", reward, "total_reward", total_reward, "step", step, "done", done)   
        returns.append(total_reward)
        steps.append(step+1)
    init_obs = np.array(init_obs)
    print("gym steps", steps)
    return np.array(returns), init_obs, steps

def hopper_is_done(state_):
    healthy_state_range=(-100.0, 100.0)
    healthy_z_range=(0.7, float("inf"))
    healthy_angle_range=(-0.2, 0.2)
    
    z, angle = state_[0:2]
    state = state_[1:]
    #print("z", z, "angle", angle, "state", state)

    min_state, max_state = healthy_state_range
    min_z, max_z = healthy_z_range
    min_angle, max_angle = healthy_angle_range

    healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
    healthy_z = min_z < z < max_z
    healthy_angle = min_angle < angle < max_angle

    is_healthy = all((healthy_state, healthy_z, healthy_angle))
    return not is_healthy

# with Simulated env
@torch.no_grad()
def evaluate_simulator(
    env: env_model.MyEnv,
    actor: nn.Module,
    num_episodes: int,
    init_obs: np.ndarray,
    step_limit: int,
    total_steps: int,
    elbo_cutoff: float,
    device: torch.device,
):
    rewards = []
    step = 0
    need_init = True
    steps = []
    for i in trange(total_steps, desc="Eval Simulator", leave=False):
        if need_init:
            total_reward = 0.0
            itrajectory = np.random.randint(0, init_obs.shape[0])
            obs = init_obs[itrajectory:itrajectory+1, :]
            # use a random init obs
            obs = np.random.randn(*obs.shape)
            obs = Tensor(obs).to(device)
            env.reset(obs)
            need_init = False

        action = actor(obs)
        obs, reward, done,  prob, elbo, discounted_reward, *_ = env.step(action, 
                            use_sensitivity=True)
            #print("step", step, "reward", reward)
            #done = is_done_walker2d(obs[0])
        total_reward += float(reward.cpu())
            #total_elbo += float(elbo)

        step+=1
        # let step > ??? to stabilize the elbo values
        if (step>15) and (done or step >= step_limit or elbo > elbo_cutoff):
            #print("step", step, "total_reward", total_reward, "done", done, "elbo", elbo)
            if step > 20:
                rewards.append(total_reward)
                steps.append(step)
            need_init = True
            step = 0                
            
    print("sim steps", steps)
    if len(rewards) == 0:
        rewards = [0.0]
    return np.array(rewards)


# add simulator data to replay buffer (sensitivity and reward penalization)
@torch.no_grad()
def augment_replay_buffer(
    rbuffer: ReplayBuffer,
    env: env_model.MyEnv,
    actor: nn.Module,
    init_obs: np.ndarray,
    step_limit: int,
    total_steps: int,
    reward_penalize: bool,
    sensitivity_threshold: float,
    elbo_cutoff: float,
    elbo_threshold: float,
    device: torch.device,
):
    returns = []
    need_init = True
    for i in trange(total_steps, desc="Augment Data", leave=False):
        if need_init:
            e_data = []
            total_reward = 0.0
            itrajectory = np.random.randint(0, init_obs.shape[0])
            obs = init_obs[itrajectory:itrajectory+1, :]
            # use a random init value
            obs = np.random.randn(*obs.shape)
            obs = Tensor(obs).to(device)
            env.reset(obs)
            step = 0
            need_init = False

        action = actor(obs)
        obs_, reward, done,  prob, elbo, discounted_reward, s_states, s_rewards = env.step(action, 
                            use_sensitivity=True)
        e_data.append([obs.cpu().numpy(), 
                               action.cpu().numpy(), 
                               obs_.cpu().numpy(), 
                               reward.item(), 
                               prob.item(),
                               elbo.item(), 
                               discounted_reward.item(), 
                               s_states.item(), 
                               s_rewards.item()])   
        total_reward += float(reward.cpu())
        step+=1
     
        # let step > ??? to stabilize the elbo values
        if (step>15) and (done or step >= step_limit or elbo > elbo_cutoff):
            #print("step", step, "total_reward", total_reward, "done", done, "elbo", elbo)
            if step > 20:
                returns.append(total_reward)
                # add to replay buffer
                reward_1 = 0.0
                reward_diffs = []
                for c in range(len(e_data)-1):
                    done = c == len(e_data) - 2
                    elbo = e_data[c][5]
                    if elbo > elbo_threshold:
                        continue
                    if reward_penalize:
                        reward_ = e_data[c][6]
                    else:
                        reward_ = e_data[c][3]
                    reward_diff = reward_ - reward_1
                    reward_diffs.append(reward_diff)
                    reward_1 = reward_                    
                    s_states = e_data[c][7]
                    if s_states > sensitivity_threshold:
                        continue
                    rbuffer.add_transition_np(
                        e_data[c][0],
                        e_data[c][1],
                        e_data[c][2],
                        e_data[c+1][1],
                        reward_,
                        done,
                        type="myenv",
                    )
                    if reward_diff > 0.2:
                        rbuffer.add_transition_np(
                            e_data[c][0],
                            e_data[c][1],
                            e_data[c][2],
                            e_data[c+1][1],
                            reward_,
                            done,
                            type="highreward",
                        )
                    if reward_diff < -0.1:
                        rbuffer.add_transition_np(
                            e_data[c][0],
                            e_data[c][1],
                            e_data[c][2],
                            e_data[c+1][1],
                            reward_,
                            done,
                            type="lowreward",
                        )
            need_init = True
    return np.array(returns)

def is_done_walker2d(state):
    height = state[1]  # Height of the walker
    angle = state[2]   # Angle of the torso

    # Define the thresholds
    min_height = 0.8
    max_height = 2.0
    max_angle = 1.0

    # Check if the walker has fallen
    if height < min_height or height > max_height or abs(angle) > max_angle:
        return True
    return False

@pyrallis.wrap()
def main(config: Config):

    #config.name = "rebrac_eval"
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
    eval_env = make_env(config.dataset_name, config.eval_seed)
    state_dim, action_dim = get_env_info(eval_env)

    myenv = env_model.MyEnv(config.chkpt_path, state_dim, action_dim, device, eval_env._max_episode_steps, 
                          vae_chkpt_path=config.vae_chkpt_path, kappa=config.sim_kappa)

    actor = DetActor(state_dim=state_dim, action_dim=action_dim, 
                     hidden_dim=config.hidden_dim, layernorm=config.actor_ln, 
                     n_hiddens=config.actor_n_hiddens).to(device)

    
    sim_rewards = []
    gym_rewards = []
    checkpoint = torch.load(f"{config.save_chkpt_path}_0.pt")
    eval_config = checkpoint["config"]
    t = trange(config.num_epochs, desc="Eval")
    for epoch in t:
        # load checkpoints
        if epoch % eval_config["save_chkpt_per"] == 0 or epoch == config.num_epochs - 1:
            # if file exist
            if not os.path.exists(f"{config.save_chkpt_path}_{epoch}.pt"):
                break
            checkpoint = torch.load(f"{config.save_chkpt_path}_{epoch}.pt")
            actor.load_state_dict(checkpoint["actor"])
        else:
            continue

        # evaluations
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            if config.use_gym_env:
                eval_returns, init_obs, steps = evaluate(
                    eval_env,
                    actor,
                    config.eval_episodes,
                    seed=config.eval_seed,
                    device=device,
                    step_limit=config.eval_step_limit
                )
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            else:
                init_obs = np.random.uniform(-1, 1, (config.eval_episodes, state_dim))            
                steps = np.random.randint(1, config.eval_step_limit, config.eval_episodes)
            sim_returns = evaluate_simulator(
                myenv,
                actor,
                config.eval_episodes,
                init_obs,
                steps,
                device,
                config.eval_step_limit
            )
            sim_normalized_score = eval_env.get_normalized_score(sim_returns) * 100.0
            if config.use_gym_env:
                sim_rewards.append(np.mean(sim_returns))
                gym_rewards.append(np.mean(eval_returns))      
                rsqr = 0.0
                #if len(sim_rewards) > 1:
                #    rsqr = r2_score(sim_rewards, gym_rewards)            
                wandb.log(
                    {
                        "epoch": epoch,
                        "eval/gym_return_mean": np.mean(eval_returns),
                        "eval/gym_return_std": np.std(eval_returns),
                        "eval/gym_normalized_score_mean": np.mean(normalized_score),
                        "eval/gym_normalized_score_std": np.std(normalized_score),
                        "eval/sim_return_mean": np.mean(sim_returns),
                        "eval/sim_return_std": np.std(sim_returns),
                        "eval/sim_normalized_score_mean": np.mean(sim_normalized_score),
                        "eval/sim_normalized_score_std": np.std(sim_normalized_score),
                        #"eval/sim_gym_rsqr": rsqr,
                    }
                )              
                t.set_postfix(
                {
                    "RM": np.mean(eval_returns),
                    "SM": np.mean(sim_returns),
                })
            else:
                wandb.log(
                    {
                        "epoch": epoch,
                        "eval/sim_return_mean": np.mean(sim_returns),
                        "eval/sim_return_std": np.std(sim_returns),
                        "eval/sim_normalized_score_mean": np.mean(sim_normalized_score),
                        "eval/sim_normalized_score_std": np.std(sim_normalized_score),
                    }
                )
                t.set_postfix(
                {
                    "SM": np.mean(sim_returns),
                })
  

if __name__ == "__main__":
    main()


