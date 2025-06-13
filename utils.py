from collections import defaultdict
from pathlib import Path
import random
import wandb
import numpy as np
import torch
import yaml

from config import Config
from model import FMRIModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model/total_params": total_params})


def save_initial_state(model, path="initial_model.pt", random_path="initial_random_state.pt"):
    torch.save(model.state_dict(), path)
    random_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(random_state, random_path)


def load_initial_state(model, path="initial_model.pt", random_path="initial_random_state.pt"):
    model.load_state_dict(torch.load(path))
    random_state = torch.load(random_path, weights_only=False)
    random.setstate(random_state["random"])
    np.random.set_state(random_state["numpy"])
    torch.set_rng_state(random_state["torch"])
    if torch.cuda.is_available() and random_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(random_state["cuda"])


def collect_predictions(loader, model, device):
    """
    Run model over `loader` (no grad) and return:
        fmri_true  : list of (T, V) arrays per subject in order
        fmri_pred  : list of (T, V) arrays per subject in same order
        subj_order : list of subject IDs as strings ("01", "02", ...)
        atlas_paths: list of atlas paths (looked up from dataset samples)
    Assumes each batch contains a single subject only (Algonauts starter).
    """
    model.eval()
    subj_to_true = defaultdict(list)
    subj_to_pred = defaultdict(list)
    subj_to_atlas = {}
    sid_map = {v: k for k, v in loader.dataset.subject_name_id_dict.items()}
    with torch.no_grad():
        for batch in loader:
            subj_ids = batch["subject_ids"]      # tensor shape (B,)
            run_ids  = batch["run_ids"]          # tensor shape (B,)
            fmri     = batch["fmri"].to(device)
            attn     = batch["attention_masks"].to(device)
            run_ids  = batch["run_ids"]
            feats    = {k: batch[k].to(device) for k in loader.dataset.modalities}

            pred = model(feats, subj_ids, run_ids, attn)

            for i, sid in enumerate(subj_ids):
                sid = sid_map[sid]
                subj_to_true[sid].append(fmri[i].cpu().numpy())
                subj_to_pred[sid].append(pred[i].cpu().numpy())
                if sid not in subj_to_atlas:
                    atlas_path = loader.dataset.samples[0]["subject_atlas"].format(subject=sid)
                    subj_to_atlas[sid] = atlas_path

    fmri_true, fmri_pred, subj_order, atlas_paths = [], [], [], []
    for sid in sorted(subj_to_true.keys()):
        fmri_true.append(np.concatenate(subj_to_true[sid], axis=0))
        fmri_pred.append(np.concatenate(subj_to_pred[sid], axis=0))
        subj_order.append(sid)
        atlas_paths.append(subj_to_atlas[sid])
    return fmri_true, fmri_pred, subj_order, atlas_paths


def load_model_from_ckpt(model_ckpt_path, params_path, device="cuda"):
    """
    Rebuild an FMRIModel from a saved state-dict and the YAML parameters file.

    Returns
    -------
    model  : torch.nn.Module – the reconstructed model with weights loaded
    config : Config          – the Config object instantiated from YAML
    """
    model_ckpt_path = Path(model_ckpt_path)
    params_path     = Path(params_path)

    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    # read hyper-parameters
    with params_path.open("r") as fp:
        cfg_dict = yaml.safe_load(fp)

    # rebuild Config and model
    config = Config(**cfg_dict)
    model  = build_model(config)
    state  = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config


def build_model(config):
    """Instantiate the FMRIModel and move to device."""
    model = FMRIModel(
        config.input_dims,
        config.output_dim,
        # fusion-stage hyper-params
        fusion_hidden_dim=config.fusion_hidden_dim,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        fusion_dropout=config.fusion_dropout,
        subject_dropout_prob=config.subject_dropout_prob,
        use_fusion_transformer=config.use_fusion_transformer,
        proj_layers=config.proj_layers,
        fuse_mode=config.fuse_mode,
        subject_count=config.subject_count,
        # temporal predictor hyper-params
        pred_layers=config.pred_layers,
        pred_heads=config.pred_heads,
        pred_dropout=config.pred_dropout,
        rope_pct=config.rope_pct,
        # HRF-related
        use_hrf_conv=config.use_hrf_conv,
        learn_hrf=config.learn_hrf,
        hrf_size=config.hrf_size,
        tr_sec=config.tr_sec,
        # training
        mask_prob=config.mask_prob,
    )
    model.to(config.device)
    return model