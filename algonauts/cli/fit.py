import wandb
import argparse
import os
from algonauts.utils import logger
from algonauts.utils import ensure_paths_exist
from algonauts.cli.train import main as train_main
from algonauts.cli.retrain import main as retrain_main


def main():
    parser = argparse.ArgumentParser(description="Fit a model to the dataset")
    parser.add_argument("--features", default=None, type=str,
                        help="Path to features YAML file")
    parser.add_argument("--params", default=None, type=str,
                        help="Path to training parameters YAML file")
    parser.add_argument("--features_dir", default=None, type=str,
                        help="Directory with extracted features "
                             "(default $FEATURES_DIR or data/features)")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Directory with raw fMRI data "
                             "(default $DATA_DIR or data/raw/fmri)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root directory for outputs & checkpoints "
                             "(default $OUTPUT_DIR or data/outputs)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--name", default=None, type=str,
                        help="Run name for W&B")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for training (default: cuda)")
    parser.add_argument("--wandb_project", default="fmri-model", type=str,
                        help="W&B project name")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Plot diagnostics after training", default=True)
    args = parser.parse_known_args()[0]

    features_dir = args.features_dir or os.getenv("FEATURES_DIR", "data/features")
    data_dir = args.data_dir or os.getenv("DATA_DIR", "data/raw/fmri")
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR", "data/outputs")

    ensure_paths_exist(
        (features_dir, "features_dir"),
        (data_dir,     "data_dir"),
        (output_dir,   "output_dir"),
    )

    # Run the training command
    with logger.step("🚀 Starting training..."):
        run_id, n_epochs = train_main(args)

    # Run the retraining command
    with logger.step("🚀 Starting retraining..."):
        retrain_main(args, run_id=run_id, n_epochs=n_epochs)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("💥 Run crashed")
        logger.error(str(e))
        if wandb.run:
            wandb.finish(exit_code=1)