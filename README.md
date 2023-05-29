# Jax OffBench

This repository contains the official JAX implementation of paper

[Improving and Benchmarking Offline Reinforcement Learning Algorithms](), [Bingyi Kang*](), [Xiao Ma*](), Yirui Wang, [Yang Yue](), [Shuicheng Yan](), ArXiv Preprint. 2023. (*equal contribution)

We provide a holistic benchmark for offline reinforcement learning across both the [D4RL](https://github.com/Farama-Foundation/D4RL) and the [RL Unplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged) datasets. In addition, by ablating 20 low-level implementation choices, we introduce CQL+ and CRR+, which improve [CQL](https://arxiv.org/abs/2006.04779) and [CRR](https://arxiv.org/abs/2006.15134) by only choosing better implementations, and achieve the state-of-the-art results.

## Main Results

### Ablating the implementation choices
![image](https://github.com/sail-sg/offbench/assets/17567744/2f51faa4-cce7-4bc0-8f45-877084d70adb)


### Results on D4RL dataset
<img width="1144" alt="image" src="https://user-images.githubusercontent.com/17567744/224491030-ae50f00d-fc3c-4cef-aac3-94789f5057c1.png">

## Setup the Environment

To setup the environment, we recommend to use docker. Simply run
```bash
./docker_run.sh
```

If you wish to setup the environment manually, you might need to first install some additional packages

To run the experiment, you might might need to install some additional packages
```bash
sudo apt-get install libglew-dev patchelf
```

Next,
```bash
pip install -e .
```

Apart from this, you'll have to setup your MuJoCo environment and key as well. Please follow [D4RL](https://github.com/Farama-Foundation/D4RL) repo and setup the environment accordingly.

## Run Experiments

An algorithm is associated with a config file defined in `configs/`. For example, `configs/cql_cfgs.py` uniquely defines CQL with its corresponding parameters.

You can run experiments using the following command:
```bash
python -m experiments.main --env 'walker2d-medium-v2' --logging.output_dir './experiment_output' --algo_cfg=./configs/<algo>_cfgs.py
```

For example, to reproduce CRR, please run:
```bash
python -m experiments.main --logging.output_dir=./experiment_output --algo_cfg=./configs/crr_cfgs.py
```

To specify additional parameters with command line, simply add them behind. For example:
```bash
python -m experiments.main --logging.output_dir=./experiment_output --algo_cfg=./configs/crr_cfgs.py --algo_cfg.training.norm_reward=True --algo_cfg.agent.policy_lr=3e-4
```

To reproduce IQL (non-JNT) which, following OnestepRL, performs policy improvement with a fixed learned value network:
```bash
python -m experiments.main --logging.output_dir=./experiment_output --algo_cfg=./configs/iql_onestep_cfgs.py --algo_cfg.training.qf_n_epochs=1000 --algo_cfg.training.pi_n_epochs=1000 --algo_cfg.training.n_epochs=2000
```

To reproduce 10%BC:
```bash
python -m experiments.main --logging.output_dir=./experiment_output --algo_cfg=./configs/bc_cfgs.py --topn=10
```

## Weights and Biases Online Visualization Integration 
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable and add `--logging.online` when launching the script.
Alternatively, you could simply run `wandb login`.

## TODOs
- [ ] Release the MuZero implementation.

## Credits
This project borrows from the [Jax CQL implementation](https://github.com/young-geng/JaxCQL). We also credit to the [official IQL implementation](https://github.com/ikostrikov/implicit_q_learning), the [official TD3+BC implementation](https://github.com/sfujim/TD3_BC), and [Acme](https://github.com/deepmind/acme) for their CRR implementation.
