## Usage

cd ORL_optimizer 

then run

python TORL/simulators/env_plot.py --config TORL/config/hopper/env_hopper_medium_v2.yaml

## Files

- env_model.py

contains models and MyEnv

- env_util.py

common utility classes for config parsing and getting sample batch

- env_train_offline.py

training for environment using offline d4rl data, use env_... . yaml file both AR and NAR

- env_train_vae.py

train VAE from the model in env_mod and save checkpt in config folder

- env_plot.py

plotting and returns R2 data, predicted reward and few state AR vs NAR
