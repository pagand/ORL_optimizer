from rebrac_mod import DetActor, Critic, EnsembleCritic, TrainState
from rebrac_util import make_env, get_env_info, get_d4rl_dataset, sample_batch_d4rl, Config

from typing import Dict, Tuple
import torch
import pyrallis

def update_actor(
    actor: TrainState,
    critic: TrainState,
    batch: Dict,
    normalize_q: bool = True,
    beta: float = 0.01,
    tau: float = 0.005,
):
    actor_model = actor.get_model()
    actor_model.train()
    action = actor_model(batch["state"])
    bc_penalty = ((action - batch["action"]) ** 2).sum(-1)
    critic_model = critic.get_model()
    with torch.no_grad():
        q_values = critic_model(batch["state"], action)
    q_value_min = q_values.min(0).values
    lmbda = 1
    if normalize_q:
        lmbda = 1 / q_value_min.detach().mean()

    loss = (beta * bc_penalty - lmbda * q_values).mean()
    actor.get_optimizer().zero_grad()
    loss.backward()
    actor.get_optimizer().step()
    actor.soft_update(tau)
    critic.soft_update(tau)
    return loss.item()

def update_critic(
    actor: TrainState,
    critic: TrainState,
    batch: Dict,
    gamma: float = 0.99,
    beta: float = 0.01,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
):
    with torch.no_grad():
        next_action = actor.get_target_model()(batch["next_state"])
        noise = torch.randn_like(next_action) * policy_noise
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (next_action + noise).clamp(-1, 1)
        bc_penalty = ((next_action - batch["next_action"]) ** 2).sum(-1)
        next_q = critic.get_target_model()(batch["next_state"], next_action).min(0).values
        next_q = next_q - beta * bc_penalty
        target_q = batch["reward"] + (1 - batch["done"]) * gamma * next_q
        target_q = target_q.squeeze(0).detach()

    critic_model = critic.get_model()
    critic_model.train()
    q = critic_model(batch["state"], batch["action"])
    #print("q_min", q.min(0).values.shape)
    q_min = q.min(0).values.mean()
    #print("q_diff", ((q - target_q) ** 2).mean(1).shape)
    loss = ((q - target_q) ** 2).mean(1).sum(0)
    critic.get_optimizer().zero_grad()
    loss.backward()
    critic.get_optimizer().step()
    return loss, q_min

def test(config: Config):
    env_name = "halfcheetah-medium-v2"
    seed = 0
    env = make_env(env_name, seed)
    state_dim, action_dim = get_env_info(env)
    dataset = get_d4rl_dataset(env)
    print(dataset.keys())
    actor = DetActor(state_dim, action_dim, 128, True, 3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor_state = TrainState(actor, actor_optimizer)
    critic = EnsembleCritic(state_dim, action_dim, 128, 2, True, 3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    critic_state = TrainState(critic, critic_optimizer)
    batch = sample_batch_d4rl(dataset, 1024)
    for _ in range(100):
        loss = update_actor(actor_state, critic_state, batch)
        print("actor loss", loss)
        loss, q_min = update_critic(actor_state, critic_state, batch)
        print("critic loss", loss, "q_min", q_min)

@pyrallis.wrap()
def main(config: Config):
    test(config)

if __name__ == "__main__":
    main()