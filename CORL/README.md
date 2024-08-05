# CORL (Clean Offline Reinforcement Learning)

## Getting started

You can use the same env varible as the main project.

```bash
# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```

## Algorithms Implemented

| Algorithm                                                                                                                         | Variants Implemented                                                                                   | Wandb Report                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Offline and Offline-to-Online**                                                                                           |                                                                                                        |                                                                                                                                                                                                |
| ✅[Conservative Q-Learning for Offline Reinforcement Learning `<br>`(CQL)](https://arxiv.org/abs/2006.04779)                       | [`offline/cql.py`](algorithms/offline/cql.py) <br /> [`finetune/cql.py`](algorithms/finetune/cql.py)     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-CQL--VmlldzoyNzA2MTk5) <br /><br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-CQL--Vmlldzo0NTQ3NTMz)   |
| ✅[Accelerating Online Reinforcement Learning with Offline Datasets `<br>`(AWAC)](https://arxiv.org/abs/2006.09359)                | [`offline/awac.py`](algorithms/offline/awac.py) <br /> [`finetune/awac.py`](algorithms/finetune/awac.py) | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-AWAC--VmlldzoyNzA2MjE3) <br /><br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-AWAC--VmlldzozODAyNzQz) |
| ✅[Offline Reinforcement Learning with Implicit Q-Learning `<br>`(IQL)](https://arxiv.org/abs/2110.06169)                          | [`offline/iql.py`](algorithms/offline/iql.py)  <br /> [`finetune/iql.py`](algorithms/finetune/iql.py)    | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-IQL--VmlldzoyNzA2MTkx) <br /><br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-IQL--VmlldzozNzE1MTEy)   |
| **Offline-to-Online only**                                                                                                  |                                                                                                        |                                                                                                                                                                                                |
| ✅[Supported Policy Optimization for Offline Reinforcement Learning `<br>`(SPOT)](https://arxiv.org/abs/2202.06239)                | [`finetune/spot.py`](algorithms/finetune/spot.py)                                                       | [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-SPOT--VmlldzozODk5MTgx)                                                                                             |
| ✅[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning `<br>`(Cal-QL)](https://arxiv.org/abs/2303.05479)   | [`finetune/cal_ql.py`](algorithms/finetune/cal_ql.py)                                                   | [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-Cal-QL--Vmlldzo0NTQ3NDk5)                                                                                           |
| **Offline only**                                                                                                            |                                                                                                        |                                                                                                                                                                                                |
| ✅ Behavioral Cloning`<br>`(BC)                                                                                                 | [`offline/any_percent_bc.py`](algorithms/offline/any_percent_bc.py)                                     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC--VmlldzoyNzA2MjE1)                                                                                                                   |
| ✅ Behavioral Cloning-10%`<br>`(BC-10%)                                                                                         | [`offline/any_percent_bc.py`](algorithms/offline/any_percent_bc.py)                                     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC-10---VmlldzoyNzEwMjcx)                                                                                                               |
| ✅[A Minimalist Approach to Offline Reinforcement Learning `<br>`(TD3+BC)](https://arxiv.org/abs/2106.06860)                       | [`offline/td3_bc.py`](algorithms/offline/td3_bc.py)                                                     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-TD3-BC--VmlldzoyNzA2MjA0)                                                                                                               |
| ✅[Decision Transformer: Reinforcement Learning via Sequence Modeling `<br>`(DT)](https://arxiv.org/abs/2106.01345)                | [`offline/dt.py`](algorithms/offline/dt.py)                                                             | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-Decision-Transformer--VmlldzoyNzA2MTk3)                                                                                                 |
| ✅[Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble `<br>`(SAC-N)](https://arxiv.org/abs/2110.01548)   | [`offline/sac_n.py`](algorithms/offline/sac_n.py)                                                       | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-SAC-N--VmlldzoyNzA1NTY1)                                                                                                                |
| ✅[Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble `<br>`(EDAC)](https://arxiv.org/abs/2110.01548)    | [`offline/edac.py`](algorithms/offline/edac.py)                                                         | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-EDAC--VmlldzoyNzA5ODUw)                                                                                                                 |
| ✅[Revisiting the Minimalist Approach to Offline Reinforcement Learning `<br>`(ReBRAC)](https://arxiv.org/abs/2305.09836)          | [`offline/rebrac.py`](algorithms/offline/rebrac.py)                                                     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-ReBRAC--Vmlldzo0ODkzOTQ2)                                                                                                               |
| ✅[Q-Ensemble for Offline RL: Don&#39;t Scale the Ensemble, Scale the Batch Size `<br>`(LB-SAC)](https://arxiv.org/abs/2211.11092) | [`offline/lb_sac.py`](algorithms/offline/lb_sac.py)                                                     | [`Offline Gym-MuJoCo`](https://wandb.ai/tlab/CORL/reports/LB-SAC-D4RL-Results--VmlldzozNjIxMDY1)                                                                                                |

## Citing CORL

If you use CORL in your work, please use the following bibtex

```bibtex
@inproceedings{
tarasov2022corl,
  title={{CORL}: Research-oriented Deep Offline Reinforcement Learning Library},
  author={Denis Tarasov and Alexander Nikulin and Dmitry Akimov and Vladislav Kurenkov and Sergey Kolesnikov},
  booktitle={3rd Offline RL Workshop: Offline RL as a ''Launchpad''},
  year={2022},
  url={https://openreview.net/forum?id=SyAS49bBcv}
}
```
