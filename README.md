# TarGF
Official Implementation of Learning Gradient Fields for Object Rearrangement

## Installation
For `Ubuntu >= 18.04` and `Anaconda3`, you can successfully run ball rearrangement tasks following the instructions below:

```
conda create -n targf python=3.9

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

conda install pyg -c pyg

pip install opencv-python tensorboard pytorch_fid gym pybullet ipdb imageio 
```

To further run our room rearrangement tasks, you need to further install `igibson`, download the preprocessed version of `3DFRONT` and meta-data used in our experiments following the instructions below:

First download the `3DFRONT` into `$YOUR_DOWNLOAD_PATH` (Preferably outside of `./TarGF`), then:

```
cd Targf
pip install igibson==1.0.3
python replace.py # modify some files in igibson, and construct a softlink to data folder

```

Then download the `3DFRONT` dataset preprocessed by us:
** 2022/8/31 update: Due to the license issue, we temporarily canceled the sharing link below. If you need this dataset urgently, please email the authors. **

```
cd data
wget https://www.dropbox.com/s/f1kha4922t8xfqr/threedfront_dataset.zip
unzip threedfront_dataset.zip
rm -rf threedfront_dataset.zip
```

Besides, you need to download the meta-data of our cleaned data:

```
wget https://www.dropbox.com/s/x6b2vuv8di8fyj8/RoomMetas.zip # download metadata
unzip RoomMetas.zip
rm -rf RoomMetas.zip
mkdir ../../ExpertDatasets # the metadata should be placed in there
cp RoomMetas/* ../../ExpertDatasets/ -r
```

## Training 
We assign an `$EXP_NAME` for each experiement, which can be modified in each `xxxx.sh` file.

The training log (Tensorboard), checkpoints and evaluation results will be (automatically) saved in `../logs/$EXP_NAME`.

To visualise the training log (Tensorboard), you can modify `tb.sh` script with your customed `$EXP_NAME` accordingly.

In this repo, we provide the original training/evaluations scripts of *Ours(SAC)* and *Ours(ORCA)* used in our experiments on paper. 
All the bash commands are stored in comments of scripts, you can uncomment or even specify your own configs to run the experiments.

<!-- For more baselines' scripts and implementation, we defer to XXXXX. -->

### For Ball Rearrangement
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








