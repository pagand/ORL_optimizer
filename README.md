# Model-based Offline RL optimizer

### Contents

1. [Setup](#setup)
2. [Usage](#usage)
3. [Disclaimer](#disclaimer)

## Setup

 Clone the repository and build a conda environment from requirement.txt:

```Shell
git clone https://github.com/pagand/ORL_optimizer
cd ORL_OPTIMIZER
conda create -n orl python=3.10
conda activate orl
pip install --upgrade pip
pip install -r requirements.txt
```

Now install the following extra packages:

```Shell
pip install 'cython<3'
pip install scipy==1.12
```

Install the correct version of pytorch given your CUDA from [start locally](https://pytorch.org/get-started/locally/) or [previous versions](https://pytorch.org/get-started/previous-versions/).  For CUDA 11.8

```Shell
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118
```

Install Jax (only required for the CORL):

```Shell
pip install "jax[cuda11_cudnn86]"==0.4.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

The code of combustion simulator is in `Simulator/simrnn_model.py`, the code of model-based offline RL framework, MORE, is in `RL/primal_dual_ddpg.py`.

* For Model-based offline RL:

```
cd MBORL
```

* For different offline RL models

```
cd CORL
```

* For vessel training model and simulator

```
cd VesselModel
```

* For MORE paper implementation

```
cd MORE
```


## Disclaimer

This code is is  heavily based on

- paper DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning  [https://github.com/ryanxhr/DeepThermal]()
- CORL (Clean Offline Reinforcement Learning)  github page by Thinkof [https://github.com/tinkoff-ai/CORL]()
- Model optimize vessel [https://github.com/pagand/model_optimze_vessel]()
