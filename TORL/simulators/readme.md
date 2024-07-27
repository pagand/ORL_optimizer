## Usage

cd sfu/ORL_optimizer # change this to your own ORL_optimizer folder

then run

python3 TORL/simulators/env_plot.py --config TORL/config/halfcheetah/env_halfcheetah_medium_v2.yaml


## Files

env_mod.py

contains models and MyEnv

env_util.py

common utility classes

env_util_offline.py

utility classes for offline d4rl

env_train_offline.py

training for environment using offline d4rl data

env_eval_offline.py

evaluating the environment 

env_plot.py

plotting and returns R2 data