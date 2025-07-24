import argparse
import random
import wandb
from pathlib import Path
import os
import torch
from torch.profiler import profile, ProfilerActivity

from vibe.dataset import get_train_val_loaders
from vibe.models import save_initial_state, build_model
from vibe.training import train_val_loop, create_optimizer_and_scheduler
from vibe.utils import logger, Config, set_seed, ensure_paths_exist, run_feature_analyses
from vibe.utils.viz import plot_diagnostics

torch.backends.cudnn.enabled = True  # Enable cuDNN for better performance
torch.backends.cudnn.benchmark = True  # Enable for faster training on fixed input sizes
torch.set_float32_matmul_precision("high") # Use high precision for matrix multiplication


def main(args=None):

    if not args:
        parser = argparse.ArgumentParser(description="Training entrypoint")
        parser.add_argument("--features", default=None, type=str,
                            help="Path to feature-set YAML")
        parser.add_argument("--params", default=None, type=str,
                            help="Path to training-parameter YAML")
        parser.add_argument("--features_dir", default=None, type=str,
                            help="Directory with extracted features "
                                 "(if unset uses $FEATURES_DIR)")
        parser.add_argument("--data_dir", default=None, type=str,
                            help="Directory with raw fMRI data "
                                 "(if unset uses $DATA_DIR)")
        parser.add_argument("--output_dir", default=None, type=str,
                            help="Root directory for outputs & checkpoints "
                                 "(if unset uses $OUTPUT_DIR)")
        parser.add_argument("--seed", default=None, type=int,
                            help="Random seed for reproducibility")
        parser.add_argument("--name", default=None, type=str,
                            help="Run name. If unset, uses 'default'. Runs are stores in <output_dir>/<name>")
        parser.add_argument("--device", default="cuda", type=str,
                            help="Device to use for training (default: cuda)")
        parser.add_argument("--wandb_project", default=None, type=str,
                            help="W&B project name")
        parser.add_argument("--wandb_entity", default=None, type=str,
                            help="W&B entity (team) name")
        parser.add_argument("--no_diagnostics", action="store_true",
                            help="Skip diagnostics after training")
        parser.add_argument("--profile", action="store_true",
                            help="Enable PyTorch profiling and export trace to <output_dir>/checkpoints/<run_id>/profiler_trace.json")
        args = parser.parse_known_args()[0]

    with logger.step("🔧 Parsing CLI arguments and setting seed..."):
        if args.seed is None:
            chosen_seed = random.SystemRandom().randint(0, 2**32 - 1)
            logger.info(f"No --seed provided; using generated seed={chosen_seed}")
        else:
            chosen_seed = args.seed
            logger.info(f"Using user-provided seed={chosen_seed}")
        set_seed(chosen_seed)

    # -------------------- PATH SANITY CHECKS --------------------
    features_path = args.features or os.getenv("FEATURES_PATH")
    params_path = args.params or os.getenv("PARAMS_PATH")

    features_dir = args.features_dir or os.getenv("FEATURES_DIR")
    data_dir = args.data_dir or os.getenv("DATA_DIR")
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR")

    if args.name:
        output_dir = os.path.join(output_dir, args.name)
    else:
        output_dir = os.path.join(output_dir, "default")

    features_path = Path(features_path)
    params_path = Path(params_path)
    features_dir = Path(features_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    ensure_paths_exist(
        (features_path, "features YAML"),
        (params_path,   "params YAML"),
        (features_dir,  "features_dir"),
        (data_dir,      "data_dir"),
    )

    try:
        ensure_paths_exist(
            (output_dir,   "output_dir"),
        )
    except FileNotFoundError:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("✅ Paths validated.")

    # -------------------- CONFIG & W&B SETUP --------------------
    with logger.step("📄 Loading config and initializing W&B..."):
        config = Config.from_yaml(features_path, params_path, chosen_seed, args.name,
                                  features_dir, data_dir, args.device)
        
        project_name = args.wandb_project or os.getenv("WANDB_PROJECT", "fmri-model")
        entity_name = args.wandb_entity or os.getenv("WANDB_ENTITY", None)

        wandb.init(entity=entity_name, project=project_name, config=vars(config), 
                   name=config.run_name, dir=output_dir / "wandb")

        wandb.save(str(features_path), base_path=features_path.parent, policy="now")
        wandb.save(str(params_path), base_path=params_path.parent, policy="now")

        # If running a W&B sweep, wandb.config contains overridden hyperparameters.
        # Merge them back into our local config dict so later code picks them up.
        config = Config(**wandb.config)

    # -------------------- MODEL --------------------
    with logger.step("🛠️ Building model..."):
        model = build_model(config)
        wandb.log({"model/num_params": sum(p.numel() for p in model.parameters())}, commit=False)

        wandb.define_metric("epoch")
        wandb.define_metric("retrain_epoch")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("val/neg_corr", step_metric="epoch", summary="min")

        run_id = wandb.run.id
        logger.info(f"Run ID: {run_id}")

        ckpt_dir = output_dir / "checkpoints" / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_initial_state(model, ckpt_dir / "initial_model.pt", ckpt_dir / "initial_random_state.pt")

        wandb.watch(model, "val/neg_corr", log_graph=True, log_freq=200)

        with logger.step("⚙️ Creating optimizer & scheduler..."):
            optimizer, scheduler = create_optimizer_and_scheduler(model, config)

        config_path = ckpt_dir / "config.yaml"
        config.save(config_path)

    # -------------------- DATA --------------------
    with logger.step("📥 Building DataLoaders..."):
        train_loader, valid_loader = get_train_val_loaders(config)

    # -------------------- TRAIN --------------------
    with logger.step("🚀 Starting training loop..."):
        if args.profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                best_val_epoch = train_val_loop(model, optimizer, scheduler, train_loader,
                                                valid_loader, ckpt_dir, config)
            prof.export_chrome_trace(str(ckpt_dir / "profiler_trace.json"))
            logger.info(f"📝 Profiling trace saved to {ckpt_dir / 'profiler_trace.json'}")
        else:
            best_val_epoch, max_roi_epoch = train_val_loop(model, optimizer, scheduler, train_loader,
                                                           valid_loader, ckpt_dir, config)
        # Save the number of epochs trained
        with open(ckpt_dir / "n_epochs.txt", "w") as f:
            f.write(f"{best_val_epoch}\n")
    
    # -------------------- DIAGNOSTICS --------------------
    if not args.no_diagnostics:
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pt"))
        with logger.step("📊 Generating validation diagnostics..."):
            out_dir = ckpt_dir / "val_diagnostics"
            plot_diagnostics(model, valid_loader, config, out_dir)
        with logger.step("🔹 Running feature analyses on validation set..."):
            run_feature_analyses(model, valid_loader, config.device)

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
