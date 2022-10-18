# Official Implementation of TarGF: Learning Target Gradient Field for Object Rearrangement 

[[Website](https://sites.google.com/view/targf/)] [[Arxiv](https://arxiv.org/abs/2209.00853)]

We study object rearrangement without explicit goal specification. The agent is given examples from a target distribution and aims at rearranging objects to increase the likelihood of the distribution. Our key idea is to learn a **target gradient field** that indicates the fastest direction to increase the likelihood from examples via score-matching. We further incoporates the target gradient field with reinforcement learning or model-based planner to tackle this task in model-free and model-based setting respectively. Our method significantly outperforms the state-of-the-art methods in the quality of the terminal state, the efficiency of the control process, and scalability.

The environments used in this work are demonstrated as follows:

|<img src="Assets/demos/circling_demo.gif" align="middle" width="200"/>  | <img src="Assets/demos/clustering_demo.gif" align="middle" width="200"/>  | <img src="Assets/demos/hybrid_demo.gif" align="middle" width="200"/>    | <img src="Assets/demos/room_demo.gif" align="middle" width="200"/> |
| *Circling* | *Clustering* | *Circling + Clustering* | *Room Rearrangement* |

## Installation

### Requirements

```
Ubuntu >= 18.04

Anaconda3
```

### Install Global Dependencies

```
git clone https://github.com/AaronAnima/TarGF

cd TarGF

conda create -n targf python=3.9

conda activate targf

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

conda install pyg -c pyg

pip install opencv-python tensorboard pytorch_fid ipdb imageio 
```

### Install *Ball Rearrangement* Dependencies

`pip install gym pybullet`

### Install *Room Rearrangement* Dependencies

Please follow the README in [this page](https://github.com/AaronAnima/TarGF/tree/main/Envs).

If you do not need to run this experiment, you can skip this procedure. 


## Training 

We assign an `$EXP_NAME` for each experiement, which can be modified in each `xxxx.sh` file.

The training log (Tensorboard), checkpoints and evaluation results will be (automatically) saved in `../logs/$EXP_NAME`.

To visualise the training log (Tensorboard), you can modify `tb.sh` script with your customed `$EXP_NAME` accordingly.

In this repo, we provide the original training/evaluations scripts of *Ours(SAC)* and *Ours(ORCA)* used in our experiments on paper. 
All the bash commands are stored in comments of scripts, you can uncomment or even specify your own configs to run the experiments.



- Training the *Target Score Network*: 
`bash sde_ball.sh`. The ode-sampler results are visualised in `../logs/$EXP_NAME/test_batch/`

- Training *Ours(SAC)*: 
`bash sac_ball.sh` 

For all scripts mentioned above, you can modify the scripts to specify your customed `$EXP_NAME` or choose different hyperparameters.

### For Room Rearrangement

- Training the *Target Score Network*: 
`bash sde_room.sh`. The ode-sampler results (starting from unseen scene condition) are visualised in `../logs/$EXP_NAME/test_batch/`

- Training *Ours(SAC)*: 
`bash sac_ball.sh`. The visualisation results in tensorboard are sampled starting from unseen scene condition.

## Evaluation

In `ball_sac.sh` and `room_sac.sh`, the evaluation scripts are stored below the training scripts. We seperate them using titles `---Training---` and `---Evaluation---`. 
To evaluate *Ours(ORCA)*, 

### For Ball Rearrangement
- Evaluating *Ours(SAC)*: 
Uncomment commands under the `Evaluation` of `sac_ball.sh`. Then assign `$EXP_NAME` to `--exp_name` accordingly. You can visualise the episodes by changing `--eval_mode fullmetric` to `--eval_mode analysis`. Results are saved in `../logs/$EXP_NAME/analysis`.


- Evaluating *Ours(ORCA)*: 
Assign `$EXP_NAME` to `--exp_name` accordingly and run `bash orca_ball.sh`. You can visualise the episodes by changing `--mode eval` to `--mode debug`. Results are saved in `../logs/$EXP_NAME`.

### For Room Rearrangement

- Evaluating *Ours(SAC)*: 
Uncomment commands under the `Evaluation` of `sac_room.sh`. Then assign `$EXP_NAME` to `--exp_name` accordingly. You can visualise the episodes by changing `--save_video False` to `--save_video True` (It will take tens of minutes to collect trajectories before start saving videos). Results are saved in `../logs/$EXP_NAME/eval_$EXP_NAME`.








