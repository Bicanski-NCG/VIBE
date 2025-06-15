#!/bin/bash -l
# Standard output and error:
#SBATCH -o data/outputs/job_logs/%x-%j.out
#SBATCH -e data/outputs/job_logs/%x-%j.err
# Initial working directory:
#SBATCH -D /u/danielcs/algonauts/Algonauts-Decoding
# Job name
#SBATCH -J algonauts-fit
#
#SBATCH --ntasks=1
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=NONE
#SBATCH --mail-user=schad@cbs.mpg.de
#SBATCH --time=01:30:00

. scripts/setup_env.sh

export FEATURES_DIR="/u/danielcs/algonauts/Algonauts-Decoding/data/features"
export DATA_DIR="/u/danielcs/algonauts/Algonauts-Decoding/data/raw/fmri"
export CHECKPOINTS_DIR="/u/danielcs/algonauts/Algonauts-Decoding/data/outputs/checkpoints"

uv run algonauts-fit "$@"