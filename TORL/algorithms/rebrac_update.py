from rebrac_mod import DetActor, Critic, EnsembleCritic, TrainState
from rebrac_util import make_env, get_env_info, get_d4rl_dataset, sample_batch_d4rl, Config, Metrics

from typing import Dict, Tuple
import torch
import pyrallis

def update_actor(
    actor: TrainState,
    critic: TrainState,
    batch: Dict,
    metrics: Metrics,
    normalize_q: bool = True,
    beta: float = 0.001,
    tau: float = 0.005,
):
    actor_model = actor.get_model()
    actor_model.train()
    action = actor_model(batch["state"])
    bc_penalty = ((action - batch["action"]) ** 2).sum(-1)
    bc_penalty = bc_penalty
    
    lmbda = 1

    critic_model = critic.get_model()
    q_values = critic_model(batch["state"], action)

    q_value_min = q_values.min(0).values
    if normalize_q:
        lmbda = 1 / (torch.abs(q_value_min).mean())
    lmbda = lmbda.detach()

    actor.get_optimizer().zero_grad()
    loss = (beta * bc_penalty - lmbda * q_values).mean()
    loss.backward()
    actor.get_optimizer().step()

    #print("beta", beta, "bc_penalty", bc_penalty, "lmbda", lmbda, "q_values", q_values)
    with torch.no_grad():
        actor.soft_update(tau)
        critic.soft_update(tau)

    loss = loss.detach().cpu()
    metrics.update({"actor_loss": loss,
                    "bc_mse_policy": bc_penalty.mean().detach().cpu(),
                    "action_mse": ((action - batch["action"]) ** 2).mean().detach().cpu(),
                    "actor_bc_penalty_mean": bc_penalty.mean()
                    })
    #print("actor loss", loss, "bc_mse_policy", bc_penalty.mean().detach().cpu(), "action_mse", ((action - batch["action"]) ** 2).mean().detach().cpu())
    return loss

def update_critic(
    actor: TrainState,
    critic: TrainState,
    batch: Dict,
    metrics: Metrics,
    gamma: float = 0.99,
    beta: float = 0.01,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
):
    with torch.no_grad():
        next_action = actor.get_target_model()(batch["next_state"])
        #print("next_action", next_action)
        '''
        noise = torch.randn_like(next_action) * policy_noise
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (next_action + noise).clamp(-1, 1)
        '''
        #print("next_action", next_action.shape, "batch[next_action]", batch["next_action"].shape)
        #bc_penalty = ((next_action - batch["next_action"]) ** 2).sum(-1)
        act_diff = next_action - batch["next_action"]
        #c_penalty = torch.abs(act_diff).sum(1).mean()
        bc_penalty = torch.square(act_diff).sum(-1)
        #print("bc_penalty", bc_penalty.shape)
        #print("critic target model", critic.get_target_model())
        next_q = critic.get_target_model()(batch["next_state"], next_action)
        #print("next_q", next_q)
        next_q, _ = next_q.min(0)
        #print("next_q", next_q)
        next_q = next_q - beta * bc_penalty
        #print("next_q - beta * bc_penalty", next_q)
        target_q = batch["reward"] + (1 - batch["done"]) * gamma * next_q
        target_q = target_q.squeeze(0).detach()
        #print("target_q", target_q)
        #print("target_q", target_q)
    
    critic_model = critic.get_model()
    critic_model.train()

    q = critic_model(batch["state"], batch["action"])
    q_min, _ = torch.min(q, dim=0)
    q_min = torch.mean(q_min)
        # print("q.shape", q.shape, "target_q.shape", target_q.shape)
    loss = ((q - target_q.unsqueeze(0)) ** 2).mean(1).sum(0)
        
    critic.get_optimizer().zero_grad()
    loss.backward()
    critic.get_optimizer().step()
    loss = loss.detach().cpu()
    qmin = q_min.detach().cpu()
    metrics.update({"critic_loss": loss, "qmin": qmin, "critic_bc_penalty_mean": bc_penalty.mean()})
    #print("critic loss", loss, "qmin", qmin)
    return loss, q_min

def test1(config: Config):
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

def test2(config: Config):
    torch.set_printoptions(precision=10)
    state_dim = 17
    action_dim = 6
    batch_size = 1
    batch = {
        "state": torch.ones((batch_size, state_dim)),
        "action": torch.ones((batch_size, action_dim)),
        "next_state": torch.ones((batch_size, state_dim)),
        "next_action": torch.ones((batch_size, action_dim)),
        "reward": torch.ones((batch_size, 1)),
        "done": torch.zeros((batch_size, 1))
    }
    actor = DetActor(state_dim, action_dim, 3, False, 3)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actor_state = TrainState(actor, actor_optimizer)
    critic = EnsembleCritic(state_dim, action_dim, 3, 2, True, 3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    critic_state = TrainState(critic, critic_optimizer)
    metrics = Metrics.create(["critic_loss", "qmin", "actor_loss", "bc_mse_policy", "action_mse"])
    for _ in range(1000):
        update_critic(actor_state, critic_state, batch, metrics)
        update_actor(actor_state, critic_state, batch, metrics)
    mean_metrics = metrics.compute()
    print(mean_metrics)

@pyrallis.wrap()
def main(config: Config):
    test2(config)

if __name__ == "__main__":
    main()