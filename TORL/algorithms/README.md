# Usage
cd sfu/ORL_optimizer  #replace with your own ORL_optimizer folder

To evaluate with the gym environment: 

python ./TORL/algorithms/rebrac_gym_main.py --config ./TORL/config/rebrac_halfcheetah_medium_v2.yaml

# FILES
rebrac_gym_main.py

training and evaluation

rebrac_mod.py

Pytorch modules

rebrac_update.py

update actor and critic routines 

rebrac_util.py

config and other utilities