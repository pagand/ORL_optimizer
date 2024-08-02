# Offline RL optimizer

### Disclaimer

This code is is based on  the paper DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning accepted at AAAI'2022.

### Usage

The code of combustion simulator is in `Simulator/simrnn_model.py`, the code of model-based offline RL framework, MORE, is in `RL/primal_dual_ddpg.py`.



For rebrac:

`python3 CORL/algorithms/offline/rebrac.py --config_path="CORL/configs/offline/rebrac/halfcheetah/medium_v2.yaml"`
