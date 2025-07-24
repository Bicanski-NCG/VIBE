#!/bin/bash -l
# Standard output and error:
#SBATCH -o runs/job_logs/%x-%j.out
#SBATCH -e runs/job_logs/%x-%j.err
# Job name
#SBATCH -J vibe-sweep
#SBATCH --array=0-19
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
# 
#SBATCH --time=12:00:00

. scripts/env.sh

uv run wandb agent "$@"