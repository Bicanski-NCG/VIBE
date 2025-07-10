#!/bin/bash -l
# Standard output and error:
#SBATCH -o data/outputs/job_logs/%x-%j.out
#SBATCH -e data/outputs/job_logs/%x-%j.err
# Initial working directory:
#SBATCH -D /u/danielcs/algonauts/Algonauts-Decoding
# Job name
#SBATCH -J algonauts-notebook
#
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
#SBATCH --time=00:30:00

. scripts/env.sh

jupyter notebook --no-browser --ip="*" --port=9779 --NotebookApp.token=letmepass --notebook-dir="./tests"
