import torch
import wandb

def create_optimizer_and_scheduler(model, config, num_train_batches):
    """Build AdamW optimiser and a warm‑up + cosine scheduler."""
    if getattr(model, "use_hrf_conv", False) and getattr(model, "learn_hrf", False):
        hrf_params = [model.hrf_conv.weight]
        other_params = [p for n, p in model.named_parameters() if n != "hrf_conv.weight"]
        param_groups = [
            {"params": hrf_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": config.weight_decay},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": config.weight_decay}]

    optimizer = torch.optim.AdamW(param_groups, lr=config.lr)
    # Log initial optimiser settings
    wandb.log({
        "train/initial_lr": config.lr,
        "train/weight_decay": config.weight_decay,
        "train/warmup_steps_ratio": config.warmup_steps_ratio,
    })
    total_steps = num_train_batches * config.epochs
    warmup_steps = int(config.warmup_steps_ratio * total_steps)        # 5 % warm-up
    main_steps = total_steps - warmup_steps

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_steps
    )
    
    if config.warmup_steps_ratio > 0:
        warm_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.warmup_start_lr_factor, 
            end_factor=1.0, total_iters=warmup_steps
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = main_scheduler

    return optimizer, scheduler
