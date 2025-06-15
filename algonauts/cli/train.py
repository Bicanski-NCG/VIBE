import argparse
import random
import wandb
from pathlib import Path
import os
import torch

from algonauts.data import get_train_val_loaders
from algonauts.models import save_initial_state, build_model
from algonauts.training import train_val_loop, create_optimizer_and_scheduler
from algonauts.utils import Config, set_seed
from algonauts.utils.viz import plot_diagnostics
from algonauts import logger

torch.backends.cudnn.enabled = True  # Enable cuDNN for better performance
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes
torch.set_float32_matmul_precision("high") # Use high precision for matrix multiplication


def main(args=None):

    # -------------------- CLI ARGUMENTS & SEED --------------------
    with logger.step("🔧 Parsing CLI arguments and setting seed..."):
        if not args:
            parser = argparse.ArgumentParser(description="Training entrypoint")
            parser.add_argument("--features", default=None, type=str,
                                help="Path to features YAML file")
            parser.add_argument("--features_dir", default=None, type=str,
                                help="Directory for features")
            parser.add_argument("--data_dir", default=None, type=str,
                                help="Directory for fMRI data")
            parser.add_argument("--params", default=None, type=str,
                                help="Path to training parameters YAML file")
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

        # Set seed
        if args.seed is None:
            chosen_seed = random.SystemRandom().randint(0, 2**32 - 1)
            logger.info(f"No --seed provided; using generated seed={chosen_seed}")
        else:
            chosen_seed = args.seed
            logger.info(f"Using user-provided seed={chosen_seed}")
        set_seed(chosen_seed)

    # -------------------- PATH SANITY CHECKS --------------------
    # Get features and params paths
    features_path = args.features or os.getenv("FEATURES_PATH", "configs/features.yaml")
    params_path = args.params or os.getenv("PARAMS_PATH", "configs/params.yaml")

    # Same for dirs 
    features_dir = args.features_dir or os.getenv("FEATURES_DIR", "data/features")
    data_dir = args.data_dir or os.getenv("DATA_DIR", "data/raw/fmri")

    features_path = Path(features_path)
    params_path = Path(params_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features YAML file not found: {features_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters YAML file not found: {params_path}")
    if not Path(features_dir).exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info("✅ Paths validated.")

    # -------------------- CONFIG & W&B SETUP --------------------
    with logger.step("📄 Loading config and initializing W&B..."):
        # Load config from YAML files
        config = Config.from_yaml(features_path, params_path, chosen_seed, args.name,
                                features_dir, data_dir, args.device)

        wandb.init(project="fmri-model", config=vars(config), 
                   name=config.run_name, dir="data/outputs/wandb")

        # If running a W&B sweep, wandb.config contains overridden hyperparameters.
        # Merge them back into our local config dict so later code picks them up.
        config = Config(**wandb.config)

    # -------------------- MODEL --------------------
    with logger.step("🛠️ Building model..."):
        # Build model outside the train loop
        model = build_model(config)
        wandb.log({"model/num_params": sum(p.numel() for p in model.parameters())}, commit=False)

        # Define W&B metrics
        wandb.define_metric("epoch")
        wandb.define_metric("retrain_epoch")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("val/neg_corr", step_metric="epoch", summary="min")

        # Prepare checkpoint directory
        run_id = wandb.run.id
        logger.info(f"Run ID: {run_id}")
        # Create a directory for checkpoints
        ckpt_dir = Path("data/outputs/checkpoints") / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save the model’s initial state
        save_initial_state(model, ckpt_dir / "initial_model.pt", ckpt_dir / "initial_random_state.pt")

        # Watch the model
        wandb.watch(model, "val/neg_corr", log_graph=True, log_freq=200)

        # Optimizer & scheduler created outside the loop
        with logger.step("⚙️ Creating optimizer & scheduler..."):
            optimizer, scheduler = create_optimizer_and_scheduler(model, config)

        # Persist the config YAML to checkpoints directory
        config_path = ckpt_dir / "config.yaml"
        config.save(config_path)

    # -------------------- DATA --------------------
    with logger.step("📥 Building DataLoaders..."):
        train_loader, valid_loader = get_train_val_loaders(config)

    # -------------------- TRAIN --------------------
    with logger.step("🚀 Starting training loop..."):
        best_val_epoch = train_val_loop(model, optimizer, scheduler,train_loader, 
                                        valid_loader, ckpt_dir, config)
        
        # Save the number of epochs trained
        (ckpt_dir / "n_epochs.txt").write_text(str(best_val_epoch))
    
    # -------------------- DIAGNOSTICS --------------------
    if args.diagnostics:
        with logger.step("📊 Generating validation diagnostics..."):
            out_dir = ckpt_dir / "val_diagnostics"
            model.load_state_dict(torch.load(ckpt_dir / "best_model.pt"))
            plot_diagnostics(model, valid_loader, config, out_dir)

    # -------------------- FINISH --------------------
    logger.info("🏁 Training complete. Saving best model and W&B run summary.")
    wandb.run.summary["best_val_epoch"] = best_val_epoch
    wandb.finish()
    return run_id, best_val_epoch


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("💥 Run crashed")
        logger.error(str(e))
        if wandb.run:
            wandb.alert(title="Run crashed", text=str(e))
        raise
