#!/bin/bash
# This script is used to fit all models in the Algonauts Decoding project.
# It assumes that the necessary configurations and data are already set up.

. scripts/env.sh

echo "Starting fitting process for all models..."
sbatch --array=0-19 scripts/fit.sh --no_diagnostics --name model1 --features configs/features_model1.yaml --params configs/params_model1.yaml --output_dir $OUTPUTS_DIR
sbatch --array=0-19 scripts/fit.sh --no_diagnostics --name model1_fr --features configs/features_model1_fr.yaml --params configs/params_model1_fr.yaml --output_dir $OUTPUTS_DIR
sbatch --array=0-19 scripts/fit.sh --no_diagnostics --name model_visual --features configs/features_visual.yaml --params configs/params_visual.yaml --output_dir $OUTPUTS_DIR
sbatch --array=0-19 scripts/fit.sh --no_diagnostics --name model_default --features configs/features_default.yaml --params configs/params_default.yaml --output_dir $OUTPUTS_DIR
echo "Fitting process initiated. Check the job scheduler for progress."