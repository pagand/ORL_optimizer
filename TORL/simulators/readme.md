usage:

first change 

chkpt_path_nar: /home/james/sfu/ORL_optimizer/TORL/config/halfcheetah_medium_v2_nar.pt
chkpt_path_ar: /home/james/sfu/ORL_optimizer/TORL/config/halfcheetah_medium_v2_ar.pt

in ./config/env_halfcheetah_medium_v2.yaml 

to your local path that contains these .pt files

then run

python ./simulators/env_plot.py --config ./config/env_halfcheetah_medium_v2.yaml


------------------------
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