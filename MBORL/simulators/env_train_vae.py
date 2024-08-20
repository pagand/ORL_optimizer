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
from env_model import VAE
from torch import nn
import sys

@torch.no_grad()
def evaluate_vae(config: Config):
    np.random.seed(config.train_seed)
    torch.manual_seed(config.train_seed)
    # comma separated list of datasets
    dsnames = config.dataset_name.split(",")
    env = gym.make(dsnames[0])
    data_train, data_holdout, data_test,  *_ = qlearning_dataset2(dsnames, verbose=True)
    state_dim, action_dim = get_env_info(env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(input_dim=(state_dim+action_dim)*config.vae_sequence_num, latent_dim=config.vae_latent_dim, hidden_dim=config.vae_hidden_dim).to(device)
    if config.load_chkpt and os.path.exists(config.vae_chkpt_path):
        checkpoint = torch.load(config.vae_chkpt_path, map_location=device)
        vae.load_state_dict(checkpoint["vae"])
        print("Checkpoint loaded from", config.vae_chkpt_path)

    for _ in range(20):
        samples = get_vae_sample(data_test, config.vae_batch_size, config.vae_sequence_num, device)
        hold_num = 10
        for i in range(hold_num):
            noise = torch.randn_like(samples) * (8 * 0.5 ** i)
            noisy_samples = samples + noise
            elbo_values = vae.estimate(noisy_samples)
            print("i", i, "noise", (8 * 0.5 ** i), "elbo mean", elbo_values.mean().item(), "std", elbo_values.std().item())


@pyrallis.wrap()
def main(config: Config):
    evaluate_vae(config)
    return
    config.name = "env_train_vae"
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
    vae = VAE(input_dim=(state_dim+action_dim)*config.vae_sequence_num, latent_dim=config.vae_latent_dim, hidden_dim=config.vae_hidden_dim).to(device)
    if config.load_chkpt and os.path.exists(config.vae_chkpt_path):
        checkpoint = torch.load(config.vae_chkpt_path, map_location=device)
        vae.load_state_dict(checkpoint["vae"])
        print("Checkpoint loaded from", config.vae_chkpt_path)

    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    t = trange(config.vae_num_epochs, desc="Training VAE")
    for epoch in t:
        elbo_losses = []
        thresholds = []
        elbo_holdout_losses = []
        for _ in range(config.vae_num_updates_per_epoch):
            samples = get_vae_sample(data_train, config.vae_batch_size, config.vae_sequence_num, device)
            vae.train()
            elbo_loss, *_ = vae(samples)
            vae_optimizer.zero_grad()
            elbo_loss.backward()
            vae_optimizer.step()
            elbo_losses.append(elbo_loss.cpu().detach())
            vae.eval()
            with torch.no_grad():
                elbo_values = vae.estimate(samples)
            threshold = torch.quantile(elbo_values, 0.05)
            thresholds.append(threshold.cpu().detach())
            #print("elbo_values", elbo_values)
            #filtered_samples = samples[elbo_values<threshold]
            #print("Filtered samples", filtered_samples.shape)
            hold_out_samples = get_vae_sample(data_holdout, config.vae_batch_size, config.vae_sequence_num, device)
            with torch.no_grad():
                elbo_loss2, *_ = vae(hold_out_samples)
                elbo_holdout_losses.append(elbo_loss2.cpu().detach())
            '''
            hold_num = 5
            for i in range(hold_num):
                noise = torch.randn_like(hold_out_samples) * (0.5 ** i)
                noisy_samples = hold_out_samples + noise
                with torch.no_grad():
                    elbo_values2 = vae.estimate(noisy_samples)
                    print("i", i, "elbo mean", elbo_values2.mean().item(), "std", elbo_values2.std().item())
            '''
        elbo_loss_mean = np.mean(elbo_losses)
        threshold_mean = np.mean(thresholds)
        elbo_holdout_loss_mean = np.mean(elbo_holdout_losses)
        t.set_description(f"EL: {elbo_loss_mean:.2f}, PT: {threshold_mean:.2f}, HL: {elbo_holdout_loss_mean:.2f}")
        wandb.log({"elbo": elbo_loss_mean, "pt": threshold_mean, "elbo_holdout": elbo_holdout_loss_mean})
    
        if epoch>0 and config.vae_save_chkpt_per>0 and (epoch % config.vae_save_chkpt_per == 0 or epoch == config.vae_num_epochs-1):
            torch.save({
                "vae": vae.state_dict(),
                "threshold": threshold_mean,
                "config": dict_config
            }, config.vae_chkpt_path)
    wandb.finish()
    print("Checkpoint saved to", config.vae_chkpt_path)

if __name__ == "__main__":
    main()