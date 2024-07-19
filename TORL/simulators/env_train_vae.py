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

from env_util_offline import Config, qlearning_dataset, get_env_info, sample_batch_offline, qlearning_dataset2, str_to_floats, get_vae_sample
from env_mod import Dynamics, GRU_update, VAE
from torch import nn


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
    vae = VAE(input_dim=state_dim+action_dim, latent_dim=32, hidden_dim=256).to(device)
    if config.load_chkpt and os.path.exists(config.vae_chkpt_path):
        checkpoint = torch.load(config.vae_chkpt_path)
        vae.load_state_dict(checkpoint["vae"])
        print("Checkpoint loaded from", config.vae_chkpt_path)

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    t = trange(config.num_epochs, desc="Training VAE")
    for epoch in t:
        samples = get_vae_sample(data_train, 1024, device)
        vae.train()
        elbo_loss, *_ = vae(samples)
        vae_optimizer.zero_grad()
        elbo_loss.backward()
        vae_optimizer.step()
        vae.eval()
        with torch.no_grad():
            elbo_values = vae.estimate(samples)
        threshold = torch.quantile(elbo_values, 0.95)
        t.set_description(f"EL: {elbo_loss.item():.2f}, PT: {threshold:.2f}")
        wandb.log({"elbo": elbo_loss.item()})
    
        if epoch>0 and config.save_chkpt_per>0 and (epoch % config.save_chkpt_per == 0 or epoch == config.num_epochs-1):
            torch.save({
                "vae": vae.state_dict()
            }, config.vae_chkpt_path)
    wandb.finish()
    print("Checkpoint saved to", config.vae_chkpt_path)

if __name__ == "__main__":
    main()