# Algonauts-Decoding

## Install 

To intstall the python env, follow these steps:

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Load conda module: `module load anaconda/3/2023.03`
3. Sync the venv: `uv sync`
4. Set up WandB: `wandb login`

## Training model

To train the model, run the training script `train.py`.

```
usage: train.py [-h] [--features FEATURES] [--features_dir FEATURES_DIR] [--data_dir DATA_DIR] [--params PARAMS] [--seed SEED] [--name NAME] [--device DEVICE]
                [--wandb_project WANDB_PROJECT]

Training entrypoint

options:
  -h, --help            show this help message and exit
  --features FEATURES   Path to features YAML file
  --features_dir FEATURES_DIR
                        Directory for features
  --data_dir DATA_DIR   Directory for fMRI data
  --params PARAMS       Path to training parameters YAML file
  --seed SEED           Random seed for reproducibility
  --name NAME           Run name for W&B
  --device DEVICE       Device to use for training (default: cuda)
  --wandb_project WANDB_PROJECT
                        W&B project name
```

### Local paths

To set the paths to feature or data directories, either add environmental variables to your slurm file, or pass options to script. If no options are passed, features are assumed to be in a local subdirectory `Features`, and data in `fmri`.
```
export FEATURES_DIR="/path/to/features"
export DATA_DIR="/path/to/fmri_data"
```
or 
```
uv run python train.py --features_dir /path/to/features --data_dir /path/to/fmri_data
```

### Parameters & feature specification

Parameters and features are loaded from the `params.yaml` and `features.yaml` files respectively. Pass the locations with the `--params` and `--features` options. By default, the script will look in `config` directory.

### Running on the cluster

Below is a sample SLUM script for running on raven
```
#!/bin/bash -l
# Standard output and error:
#SBATCH -o job_logs/%x-%j.out
#SBATCH -e job_logs/%x-%j.err
# Initial working directory:
#SBATCH -D /path/to/project         # Add path to your git repo
# Job name
#SBATCH -J algonauts-train
#
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
#SBATCH --mail-type=NONE
#SBATCH --mail-user=your.email@cbs.mpg.de       # Add your email
#SBATCH --time=01:30:00

module purge
module load anaconda/3/2023.03

export FEATURES_DIR="/path/to/features"
export DATA_DIR="/path/to/fmri"

uv run python train.py "$@"
```

Launch the training with `sbatch train.sh`, add options as desired.

The training script will initialize the environment, set the paths to features and data, and pass arguments to the script. You can monitor the progress on wandb.

## Running a parameter sweep

Setup the sweep configuration in a yaml file. Set the parameter choices to sweep over.
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

Launch the sweep in wandb `wandb sweep /path/to/sweep.yaml --project project name`. In the ouput, wandb will log the sweep id. We will use this to la

The wandb init will now override parameters from the yaml file as needed for each of the sweep agents.

Launch a SLURM job array for the sweep workers.
```
#!/bin/bash -l
# Standard output and error:
#SBATCH -o job_logs/%x-%j.out
#SBATCH -e job_logs/%x-%j.err
# Initial working directory:
#SBATCH -D /path/to/project
# Job name
#SBATCH -J algonauts-sweep
#SBATCH --array=0-12
#SBATCH --ntasks=1
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
#SBATCH --mail-type=NONE
#SBATCH --mail-user=your.email@cbs.mpg.de
#SBATCH --time=08:00:00

module purge
module load anaconda/3/2023.03

export FEATURES_DIR="/u/danielcs/algonauts/Algonauts-Decoding/features"
export DATA_DIR="/u/danielcs/algonauts/Algonauts-Decoding/data/fmri"

SWEEP_ID="sweep_id"

uv run wandb agent $SWEEP_ID
```

Your sweep will now start and be available on WandB.

## Make submission

To make a submisison, run the `make_submission.py` script.

```
usage: make_submission.py [-h] [--checkpoint CHECKPOINT] [--name NAME]

Make submission for fMRI predictions

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Checkpoint to load
  --name NAME           Name of output file
```

Pass the `--checkpoint CHECKPOINT` option to the script to specify which model to make predictions from. The checkpoint is the WandB ID of the run.