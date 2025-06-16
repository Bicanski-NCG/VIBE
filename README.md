# 🧠 Algonauts Decoding

> End-to-end pipeline for voxel-wise decoding of fMRI time-series from
> multimodal stimulus features (audio · video · text).

## Directory layout

```
Algonauts-Decoding
├── algonauts
│   ├── cli
│   │   ├── fit.py                     <- Entrypoint for full train + retrain
│   │   ├── make_submission.py         <- Entrypoint for inference on season 7
│   │   ├── retrain.py                 <- Entrypoint for retrainin on full dataset
│   │   └── train.py                   <- Entrypoint for train-validation loop
│   ├── data
│   │   ├── data.py                    <- Dataset
│   │   └── loader.py                  <- Data loaders for train-val / full dataset
│   ├── features                       <- Feature extractors
│   ├── models
│   │   ├── fmri.py                    <- Main FMRIModel
│   │   ├── rope.py                    <- RoPE models
│   │   └── utils.py                   <- Model utils
│   ├── training
│   │   ├── loop.py                    <- Train/val and retrain loops
│   │   ├── losses.py                  <- Loss functions 
│   │   ├── metrics.py                 <- Data loaders for train-val / full dataset
│   │   └── optim.py                   <- Scheduler / optimizer 
│   └── utils                          <- Util functions
├── configs
│   ├── features.yaml                  <- Feature specification for model
│   ├── params.yaml                    <- Model and training parameters
│   └── sweep.yaml                     <- Parameter specs for WandB sweeps
├── data
│   ├── features                       <- Extracted features (~100 GB)
│   │   ├── Audio
│   │   ├── Emotional
│   │   ├── Omni
│   │   ├── Text
│   │   └── Visual
│   ├── outputs                        <- Output files
│   │   ├── checkpoints
│   │   ├── job_logs
│   │   ├── submissions
│   │   └── wandb
│   └── raw                       
│       ├── fmri
│       └── stimuli
├── notebooks
├── scripts
│   ├── setup_end.sh.example           <- Sample environment setup
│   ├── fit.sh                         <- SLURM scripts for launching the entrypoints
│   ├── retrain.sh
│   ├── submission.sh
│   ├── sweep.sh
│   └── train.sh
└── tests
```

## Install 

To intstall the python env, follow these steps:

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Load conda module: `module load anaconda/3/2023.03`
3. Sync the venv: `uv sync`
4. Install algonauts `uv pip install -e .`
4. Set up WandB: `wandb login`

## Setup cluster

To get started with running the model on the cluster, first make a copy of the `setup_env.sh.example` file and edit the paths to data directories in that file:
```
#!/bin/bash

module purge
module load anaconda/3/2023.03

# The following command replaces `conda init` for the current session
# without touching the .bashrc file:
eval "$(conda shell.bash hook)"

export FEATURES_DIR="/path/to/features"
export DATA_DIR="/path/to/fmri"
export OUTPUTS_DIR="/path/to/outputs"
```
You can also add whatever other setup you want for all the runs. Now, you can use the packaged slurm scripts to launch jobs. By default the scripts will look for features in directories specified in these environment variables. If is possible to override these with parameters to the scripts as detailed below.

## Training model

To start training-validation loop, run `algonauts-train`, or start a batch job with `scripts/train.sh`.

```
usage: algonauts-train [-h] [--features FEATURES] [--features_dir FEATURES_DIR] [--data_dir DATA_DIR] [--params PARAMS] [--seed SEED] [--name NAME] [--device DEVICE] [--wandb_project WANDB_PROJECT] [--diagnostics]

Training entrypoint

options:
  -h, --help            show this help message and exit
  --features FEATURES   Path to features YAML file (default: configs/features.yaml)
  --params PARAMS       Path to training parameters YAML file (default: configs/params.yaml)
  --features_dir FEATURES_DIR
                        Directory for features, overrides FEATURES_DIR environment variable
  --data_dir DATA_DIR   Directory for fMRI data, overrides DATA_DIR environment variable
  --checkpoint_dir CHECKPOINT_DIR
                        Directory containing checkpoints, overrides CHECKPOINT_DIR environment variable
  --seed SEED           Random seed for reproducibility
  --name NAME           Run name for W&B
  --device DEVICE       Device to use for training (default: cuda)
  --wandb_project WANDB_PROJECT
                        W&B project name
  --no_diagnostics      Skip diagnostics after training
```

### Override local paths

To specify alternative data locations, either set the appropriate environment variables, or specify the correct path using the flags above.

### Parameters & feature specification

Parameters and features are loaded from the `params.yaml` and `features.yaml` files respectively. Pass the locations with the `--params` and `--features` options. By default, the script will look in `configs` directory.

## Retrain model

To start full retrain loop, run `algonauts-retrain`, or start a batch job with `scripts/retrain.sh`.

```
usage: algonauts-retrain [-h] [--checkpoint CHECKPOINT] [--checkpoint_dir CHECKPOINT_DIR] [--wandb_project WANDB_PROJECT] [--device DEVICE] [--diagnostics]

Retrain a model on the full dataset after initial training

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Model checkpoint (same as wandb run ID)
  --checkpoint_dir CHECKPOINT_DIR
                        Directory containing checkpoints, overrides CHECKPOINT_DIR environment variable
  --wandb_project WANDB_PROJECT
                        W&B project name
  --device DEVICE       Device to use for training (default: cuda)
  --no_diagnostics      Skip diagnostics after retraining
```

## Fit model

The `algonauts-fit` command launches training and retraining sequentially for the same model.

```
usage: algonauts-fit [-h] [--features FEATURES] [--features_dir FEATURES_DIR] [--data_dir DATA_DIR] [--checkpoint_dir CHECKPOINT_DIR] [--params PARAMS] [--seed SEED] [--name NAME] [--device DEVICE] [--wandb_project WANDB_PROJECT] [--diagnostics]

Fit a model to the dataset

options:
  -h, --help            show this help message and exit
  --features FEATURES   Path to features YAML file (default: configs/features.yaml)
  --params PARAMS       Path to training parameters YAML file (default: configs/params.yaml)
  --features_dir FEATURES_DIR
                        Directory for features, overrides FEATURES_DIR environment variable
  --data_dir DATA_DIR   Directory for fMRI data, overrides DATA_DIR environment variable
  --checkpoint_dir CHECKPOINT_DIR
                        Directory containing checkpoints, overrides CHECKPOINT_DIR environment variable
  --seed SEED           Random seed for reproducibility
  --name NAME           Run name for W&B
  --device DEVICE       Device to use for training (default: cuda)
  --wandb_project WANDB_PROJECT
                        W&B project name
  --no_diagnostics      Skip diagnostics after training and retraining
```

## Running a parameter sweep

Setup the sweep configuration in a yaml file. Set the parameter choices to sweep over. A sweep template might look like this:
```
program: train.py
name: sweep_name
method: bayes

metric:
  goal: minimize
  name: val/neg_corr.min

parameters:
  param1:
    max: 10
    min: 1
    distribution: int_uniform
  param2:
    max: 0.1
    min: 0
    distribution: uniform
  param3:
    values: [4, 8, 16, 32]

command:
  - ${env}
  - ${interpreter}
  - ${program}
```

Launch the sweep in wandb `wandb sweep /path/to/sweep.yaml --project project_name --name sweep_name`. In the ouput, wandb will log the sweep id. We will use this to launch the sweep later.

The wandb init will now override parameters from the yaml file as needed for each of the sweep agents.

You can view the sweep specification on WandB. To start the sweep, run the batch job script and specify the sweep id:
```
sbatch scripts/sweep.sh your_sweep_id
```

This will launch 16 nodes for 12 hours to explore the parameter space. To stop the sweep, arrest it on the WandB website or run `wandb sweep --stop your_sweep_id`. The nodes will wind down and exit gracefully.

## Make submission

To make a submisison, run the `algonauts-submit` or the corresponding batch script.

```
usage: algonauts-submit [-h] --checkpoint CHECKPOINT [--name NAME]

Make submission for fMRI predictions

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Checkpoint to load
  --name NAME           Name of output file
```

Pass the `--checkpoint CHECKPOINT` option to the script to specify which model to make predictions from. The checkpoint is the WandB ID of the run. The script will load the fully trained model, and use that to make predictions on season 7 of friends.

## TODO:
- [ ] feature extraction entrypoint