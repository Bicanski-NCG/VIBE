#!/bin/bash -l
# Standard output and error:
#SBATCH -o data/outputs/job_logs/%x-%j.out
#SBATCH -e data/outputs/job_logs/%x-%j.err
# Initial working directory:
# Job name
#SBATCH -J algonauts-shapley
#
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
#SBATCH --time=24:00:00


uv run algonauts-shapley "$@"