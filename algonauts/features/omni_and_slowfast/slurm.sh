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
#SBATCH --mem=80000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dixit@cbs.mpg.de
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03
module unload cuda
module load cuda/12.6
conda deactivate
source .venv/bin/activate
module load ffmpeg/4.4
module load gcc/13
srun uv run python extract_omni_features.py "$@"