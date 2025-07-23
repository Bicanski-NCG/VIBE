#!/bin/bash
# This script is used to fit all models in the Algonauts Decoding project.
# It assumes that the necessary configurations and data are already set up.

. scripts/env.sh

echo "Starting merge process for all models..."
uv run algonauts-merge --config configs/merge_plan_s07.yaml --output_dir $OUTPUTS_DIR --data_dir $DATA_DIR
uv run algonauts-merge --config configs/merge_plan_ood.yaml --output_dir $OUTPUTS_DIR --data_dir $DATA_DIR
echo "Merge process completed."