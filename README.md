# Offline RL optimizer

### Disclaimer

This code is is based on  the paper DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning accepted at AAAI'2022.

### Usage

The code of combustion simulator is in `Simulator/simrnn_model.py`, the code of model-based offline RL framework, MORE, is in `RL/primal_dual_ddpg.py`.



For rebrac:

`python3 CORL/algorithms/offline/rebrac.py --config_path="CORL/configs/offline/rebrac/halfcheetah/medium_v2.yaml"`



Decision support system:
1. fully automation:
	Show GPS coordinates and speeds for the captains to follow
2. human-in-the-loop:
	a. show only the current value of fuel consumption
	b. show the current value as well as the predicted values of fuel consumption in a time frame for the captains to make judgements

Estimated time of arrival calculation: