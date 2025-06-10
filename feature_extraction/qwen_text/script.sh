#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dixit@cbs.mpg.de
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03
module unload cuda
module load cuda/12.6

source .venv/bin/activate

# Read the arguments for modality and stimulus
input_folder=$1
output_folder=$2


srun uv run python extract_text_features.py "$@"