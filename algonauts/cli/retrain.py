import argparse
import torch
import wandb
from pathlib import Path
import os

from algonauts.data import get_full_loader
from algonauts.data.loader import get_train_val_loaders
from algonauts.models import load_model_from_ckpt, load_initial_state
from algonauts.training import create_optimizer_and_scheduler, full_loop
from algonauts.utils import logger, ensure_paths_exist
from algonauts.utils.viz import plot_diagnostics


def main(args=None, run_id=None, n_iters=None):
    # -------------------- CLI ARGUMENTS --------------------
    if not args:
        parser = argparse.ArgumentParser(
            description="Retrain a model on the full dataset after initial training"
        )
        parser.add_argument("--checkpoint", type=str, default=os.getenv("ALGONAUTS_RUN_ID", None),
                            help="Model checkpoint (same as wandb run ID)")
        parser.add_argument("--output_dir", default=None, type=str,
                            help="Root directory for outputs & checkpoints "
                                 "(default $OUTPUT_DIR or data/outputs)")
        parser.add_argument("--wandb_project", default=None, type=str,
                            help="W&B project name")
        parser.add_argument("--wandb_entity", default=None, type=str,
                                help="W&B entity (team) name")
        parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use for training (default: cuda)")    
        parser.add_argument("--no_diagnostics", action="store_true",
                            help="Skip diagnostics after training")
        args = parser.parse_known_args()[0]

        checkpoint = args.checkpoint
    else:
        checkpoint = run_id or args.checkpoint or os.getenv("ALGONAUTS_RUN_ID", None)

    if not checkpoint:
        raise ValueError("Please provide a checkpoint to load.")


    output_dir = Path(args.output_dir or os.getenv("OUTPUT_DIR", "data/outputs"))
    ckpt_dir = output_dir / 'checkpoints' / checkpoint
    model_path = ckpt_dir / "initial_model.pt"
    params_path = ckpt_dir / "config.yaml"
    ensure_paths_exist(
        (ckpt_dir,    "checkpoint_dir"),
        (output_dir,  "output_dir"),
        (model_path,  "initial_model.pt"),
        (params_path, "config.yaml"),
    )

    # Continue wandb run from the checkpoint
    project_name = args.wandb_project or os.getenv("WANDB_PROJECT", "fmri-model")
    entity_name = args.wandb_entity or os.getenv("WANDB_ENTITY", None)
    wandb.init(id=checkpoint, resume="must", project=project_name, entity=entity_name,
               dir=output_dir / "wandb")
    
    with logger.step("📦 Loading checkpoint and config …"):
        try:
            # Load the model and config from the checkpoint
            model, config = load_model_from_ckpt(
                model_ckpt_path=model_path,
                params_path=params_path,
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

    # Load initial model and random state
    with logger.step("🔄 Restoring initial state …"):
        load_initial_state(model, ckpt_dir / "initial_model.pt", ckpt_dir / "initial_random_state.pt")

    # Construct full data loader
    with logger.step("📥 Building full DataLoader …"):
        full_loader = get_full_loader(config)

    # Create optimizer and scheduler for full retrain
    with logger.step("⚙️ Creating optimizer & scheduler …"):
        num_train_batches = len(get_train_val_loaders(config)[0])
        optimizer_full, scheduler_full = create_optimizer_and_scheduler(model, config, num_train_batches)

    # Retrain the model on the full dataset
    with logger.step("🚀 Starting full retrain …"):
        # iter file has format <best_val_iter>
        with open(ckpt_dir / "n_iters.txt", "r") as f:
            best_val_iter = int(f.readline().strip())
        full_loop(model, optimizer_full, scheduler_full, full_loader, ckpt_dir, config, best_val_iter)

    # Plot diagnostics after retraining
    if not args.no_diagnostics:
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