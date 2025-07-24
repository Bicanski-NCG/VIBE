import wandb
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from vibe.dataset import FMRI_Dataset, split_dataset_by_name, collate_fn, make_group_weights
from vibe.utils import logger


def get_train_val_loaders(config):
    """Return train and validation DataLoaders."""

    # Prepend the features directory to each feature path
    features_dir = Path(config.features_dir)
    config.features = {n: str(features_dir / p) for n, p in config.features.items()}

    ds = FMRI_Dataset(config.data_dir,
                      feature_paths=config.features,
                      input_dims=config.input_dims,
                      modalities=config.modalities,
                      noise_std=config.train_noise_std,
                      normalization_stats=None,
                      oversample_factor=config.oversample_factor,
                      modality_dropout_mode = config.modality_dropout_mode,
                      modality_dropout_prob = config.modality_dropout_prob,
                      normalize_features = config.use_normalization,
                      loss_masks_path=config.loss_masks_path,
                      
                      )

    if config.filter_name is not None:
        def filter_fn(sample):
            return all([filter_key not in sample["dataset_name"] for filter_key in config.filter_name])

        ds = ds.filter_samples(filter_fn)
    
    logger.info(f"Dataset size: {len(ds)} samples")

    train_ds, valid_ds = split_dataset_by_name(
        ds,
        val_name=config.val_name,
        val_run=config.val_run,
        train_noise_std=config.train_noise_std,
        normalize_validation_bold=config.normalize_validation_bold,
    )

    if config.stratification_variable:
        train_weights = make_group_weights(train_ds, filter_on=config.stratification_variable)
        logger.info(f"Using stratification variable: {config.stratification_variable}")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        logger.info("🎯 Using weighted sampler for class balance.")
        shuffle = False
    else:
        train_weights = torch.ones(len(train_ds), dtype=torch.float32)
        sampler = None
        shuffle = True

    logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )

    wandb.log({
        "dataset/train_samples": len(train_ds),
        "dataset/val_samples": len(valid_ds),
        "dataset/full_samples": len(ds),
    }, commit=False)
    return train_loader, valid_loader


def get_full_loader(config):
    """
    Return a DataLoader over the *entire* dataset (train + val merged).
    """
    norm_stats = torch.load("normalization_stats.pt")
    logger.info("📏 Loaded voxel‑wise normalization stats.")

    features_dir = Path(config.features_dir)
    config.features = {n: str(features_dir / p) for n, p in config.features.items()}

    ds_full = FMRI_Dataset(
        config.data_dir,
        feature_paths=config.features,
        input_dims=config.input_dims,
        modalities=config.modalities,
        noise_std=0.0,
        normalization_stats=norm_stats,
        modality_dropout_mode = config.modality_dropout_mode,
        modality_dropout_prob = config.modality_dropout_prob,
        normalize_features = config.use_normalization,
        loss_masks_path=config.loss_masks_path
    )

    if config.filter_name is not None:
        def filter_fn(sample):
            return all([filter_key not in sample["dataset_name"] for filter_key in config.filter_name])

        ds_full = ds_full.filter_samples(filter_fn)
    
    logger.info(f"Dataset size: {len(ds_full)} samples")

    full_loader = DataLoader(
        ds_full,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
    )

    wandb.log({"dataset/full_samples": len(ds_full)}, commit=False)
    return full_loader