#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J vda
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dixit@cbs.mpg.de
#SBATCH --time=12:00:00

srun uv run python extract_features.py "$@"