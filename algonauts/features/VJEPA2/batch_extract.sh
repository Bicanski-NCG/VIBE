#!/bin/bash -l
# Standard output and error:
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Initial working directory:
#SBATCH -D /your/working/directory
# Job name
#SBATCH -J algonauts-extract
#
#SBATCH --ntasks=1
#
# --- use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-core=2
#SBATCH --mem=200000M

#SBATCH --time=24:00:00
#SBATCH --array=0-20   # <--- This must correspond with --num_chunks in fucntion calling

module purge
module load anaconda/3/2023.03
module load cuda/12.1
module load ffmpeg/4.4

# Pass SLURM_ARRAY_TASK_ID to Python as an argument
python3 get_vjepa_ac_features.py \
  --input_folder /path/to/input/stimuli \
  --output_folder /path/to/output/features \
  --chunk_length 6 \
  --seconds_before_chunk 3 \
  --chunk_id $SLURM_ARRAY_TASK_ID \
  --num_chunks 21
