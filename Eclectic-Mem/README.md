# Eclectic Memory: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Control 

Eclectic-Mem: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Memory-Based Expectation Maximization

Eclectic-Mem: An Eclectic Mix Of Contrastive Learning, Episodic Control, And Transformer-Style Self-Attention In Continuous Control

Eclectic Memory: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Memory-Based Control 

Eclectic Memory: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Control AGI Tasks

Eclectic Memory: Efficient Contrastive-Learner-Episodic-Controller-Transformer In Continuous Control Tasks

Eclectic Memory: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Control RL Tasks

Eclectic Memory: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Control Locomotive Generalization

Eclectic: Efficient Contrastive Learner Episodic Controller Transformer In Continuous Control

Eclectic: Efficient Contrastive Learner, Episodic Controller, And Transformer In Continuous Control

Eclectic: Efficient Contrastive Learner, Episodic Controller, And Transformer In Continuous Action Spaces

Eclectic-Mem: Efficient Contrastive Learner, Episodic Controller, And Transformer All In One Consolidated Lifelong Memory Architecture

Eclectic-Mem: Efficient Contrastive Learner, Episodic Controller, Transformer, And Continuous Controller All In One Consolidated Lifelong Memory Architecture

Eclectic-Mem: Efficient Contrastive Learner, Episodic Controller, And Transformer In Continuous-Control-Continual-General-Few-Shot-Meta-RL MDPs

Eclectic-Mem: Efficient Contrastive Learning For Episodic Control By Time Irregular Concepts To Memory

This repository is an extension of [CURL](https://mishalaskin.github.io/curl/) for the DeepMind control experiments. Atari experiments were done in a separate codebase available [here](https://github.com/aravindsrinivas/curl_rainbow). Implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats. 

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train an Eclectic agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | CU_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | CU_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | CU_LOSS: 0.0000
```

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh. 

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. For environment troubleshooting issues, see the DeepMind control documentation.
