#!/bin/bash
# This script is used to fit all models in the Algonauts Decoding project.
# It assumes that the necessary configurations and data are already set up.

. scripts/env.sh

echo "Starting submit process for all models..."
sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model1/checkpoints --name model1 --output_dir $OUTPUTS_DIR/model1 --test_names s07
sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model1/checkpoints --name model1 --output_dir $OUTPUTS_DIR/model1

sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model1_fr/checkpoints --name model1_fr --output_dir $OUTPUTS_DIR/model1_fr --test_names passepartout

sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model_visual/checkpoints --name model_visual --output_dir $OUTPUTS_DIR/model_visual --test_names s07
sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model_visual/checkpoints --name model_visual --output_dir $OUTPUTS_DIR/model_visual

sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model_default/checkpoints --name model_default --output_dir $OUTPUTS_DIR/model_default --test_names s07
sbatch scripts/submit.sh --checkpoint_dir $OUTPUTS_DIR/model_default/checkpoints --name model_default --output_dir $OUTPUTS_DIR/model_default
echo "Submit process initiated. Check the job scheduler for progress."
