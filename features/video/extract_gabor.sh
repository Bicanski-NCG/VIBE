#!/bin/bash -l
# Standard output and error:
#SBATCH -o runs/job_logs/%x-%j.out
#SBATCH -e runs/job_logs/%x-%j.err
# Initial working directory:
#SBATCH -D /u/danielcs/algonauts/Algonauts-Decoding
# Job name
#SBATCH -J feat-gabor
#
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=125000M
#
#SBATCH --time=01:30:00

module load ffmpeg

uv run python -u algonauts/features/video/gabor.py --input_folder data/raw/stimuli/movies --output_folder data/feat_out/gabor_filters_new --tr 1.49 --decode_height 128