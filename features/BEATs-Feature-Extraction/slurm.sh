#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J qwen_omni
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dixit@cbs.mpg.de
#SBATCH --time=00:45:00

module purge
module load anaconda/3/2023.03
module unload cuda
module load cuda/12.6
conda deactivate
srun uv run python extract_audio_BEATs.py "$@"