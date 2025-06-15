import argparse
import torch
import wandb
from pathlib import Path
import os

from algonauts.data import get_full_loader
from algonauts.models import load_model_from_ckpt, load_initial_state
from algonauts.training import create_optimizer_and_scheduler, full_loop
from algonauts.utils import Config, logger
from algonauts.utils.viz import plot_diagnostics


def main(args=None, run_id=None, n_epochs=100):
    # -------------------- CLI ARGUMENTS --------------------
    if not args:
        parser = argparse.ArgumentParser(
            description="Retrain a model on the full dataset after initial training"
        )
        parser.add_argument("--checkpoint", type=str, default=os.getenv("ALGONAUTS_RUN_ID", None),
                            help="Model checkpoint (same as wandb run ID)")
        parser.add_argument("--checkpoint_dir", type=str, default=None,
                            help="Directory containing checkpoints")
        parser.add_argument("--wandb_project", type=str, default="fmri-model",
                            help="W&B project name")
        parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use for training (default: cuda)")    
        parser.add_argument("--diagnostics", action="store_true", default=False,
                            help="Plot diagnostics after retraining")
        args = parser.parse_known_args()[0]

        checkpoint = args.checkpoint
    else:
        checkpoint = run_id or args.checkpoint or os.getenv("ALGONAUTS_RUN_ID", None)
    checkpoint_dir = args.checkpoint_dir or os.getenv("CHECKPOINTS_DIR", "data/outputs/checkpoints")
    ckpt_dir = Path(checkpoint_dir) / checkpoint

    if not checkpoint:
        raise ValueError("Please provide a checkpoint to load.")
    
    # Continue wandb run from the checkpoint
    wandb.init(id=checkpoint, resume="must", project=args.wandb_project,
               dir="data/outputs/wandb")
    
    with logger.step("📦 Loading checkpoint and config …"):
        try:
            # Load the model and config from the checkpoint
            model, config = load_model_from_ckpt(
                model_ckpt_path=ckpt_dir / "initial_model.pt",
                params_path=ckpt_dir / "config.yaml",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model from checkpoint {checkpoint}: {e}")
        else:
            logger.info(f"Using checkpoint: {checkpoint}")

    # Set the device for the model
    device = torch.device(args.device)
    config.device = device
    with logger.step("🖥️ Moving model to device …"):
        model.to(device)

    # Create optimizer and scheduler for full retrain
    with logger.step("⚙️ Creating optimizer & scheduler …"):
        optimizer_full, scheduler_full = create_optimizer_and_scheduler(model, config)

    # Load initial model and random state
    with logger.step("🔄 Restoring initial state …"):
        load_initial_state(model, ckpt_dir / "initial_model.pt", ckpt_dir / "initial_random_state.pt")

    # Construct full data loader
    with logger.step("📥 Building full DataLoader …"):
        full_loader = get_full_loader(config)

    # Retrain the model on the full dataset
    with logger.step("🚀 Starting full retrain …"):
        full_loop(model, optimizer_full, scheduler_full, full_loader, ckpt_dir, config, n_epochs)

    # Plot diagnostics after retraining
    if args.diagnostics:
        with logger.step("📊 Generating diagnostics …"):
            out_dir = ckpt_dir / "full_diagnostics"
            model.load_state_dict(torch.load(ckpt_dir / "final_model.pt"))
            plot_diagnostics(model, full_loader, config, out_dir)

    # Finish wandb run
    with logger.step("🏁 Finishing W&B run"):
        wandb.run.summary["final_model_path"] = str(ckpt_dir / "final_model.pt")
        wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("💥 Run crashed")
        logger.error(str(e))
        if wandb.run:
            wandb.alert(title="Run crashed", text=str(e))
        raise