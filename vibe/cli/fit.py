import wandb
import argparse
import os
from vibe.utils import logger
from vibe.utils import ensure_paths_exist
from vibe.cli.train import main as train_main
from vibe.cli.retrain import main as retrain_main


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
                             "(default $OUTPUT_DIR or runs)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--name", default=None, type=str,
                        help="Run name for W&B")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for training (default: cuda)")
    parser.add_argument("--wandb_project", default=None, type=str,
                        help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, type=str,
                        help="W&B entity (team) name")
    parser.add_argument("--no_diagnostics", action="store_true",
                        help="Skip diagnostics after training")
    parser.add_argument("--profile", action="store_true",
                    help="Enable PyTorch profiling and export trace to output_dir/checkpoints/<run_id>/profiler_trace.json")
    args = parser.parse_known_args()[0]

    # Run the training command
    with logger.step("🚀 Starting training..."):
        run_id, n_epochs = train_main(args)

    # Run the retraining command
    with logger.step("🚀 Starting retraining..."):
        args.no_diagnostics = True
        args.output_dir = os.path.join(args.output_dir or os.getenv("OUTPUT_DIR", "runs"), args.name or "default")
        retrain_main(args, run_id=run_id, n_epochs=n_epochs)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("💥 Run crashed")
        logger.error(str(e))
        if wandb.run:
            wandb.finish(exit_code=1)