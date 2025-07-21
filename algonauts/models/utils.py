import torch
import random
import numpy as np
from pathlib import Path
import yaml

from algonauts.utils import Config
from algonauts.models import FMRIModel


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


def load_model_from_ckpt(model_ckpt_path, params_path,device = None):
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

    # Modalities list contains the order of modalities in the model.
    modality_order = config.modalities
    config.input_dims = {modality: config.input_dims[modality] for modality in modality_order if modality in config.input_dims}
    if device:
        config.device = device
    model  = build_model(config)
    state  = torch.load(model_ckpt_path)
    model.load_state_dict(state)
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
        use_gmu=config.use_gmu,
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
        # padding
        num_pre_tokens=config.num_pre_tokens,
        n_prepend_zeros=config.n_prepend_zeros,
    )
    model.to(config.device)
    return model