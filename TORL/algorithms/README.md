## Usage

cd sfu/ORL_optimizer  #replace with your own ORL_optimizer folder

To evaluate with myenv simulator and the gym environment with Rebrac:

python3 TORL/algorithms/rebrac_main.py --config TORL/config/hopper/rebrac_hopper_medium_v2.yaml

## FILES

- rebrac_main.py

The main file to train Rebrac

- rebrac_model.py

the model architecture of the components

- rebrac_update.py

update actor and critic moduels based on RebRac

- rebrac_util.py

config files and get data loader and D4RL

- rebrac_eval.py

evaluate gym - evalute simulated env - get transition from simulated env for data augmentation
