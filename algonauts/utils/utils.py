from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from functools import lru_cache
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.stats import pearsonr
import os
from tqdm import tqdm

def voxelwise_pearsonr(y_true: np.ndarray,
                       y_pred: np.ndarray) -> np.ndarray:
    """
    Voxel-wise Pearson correlation.

    Parameters
    ----------
    y_true, y_pred : (T, V) float arrays

    Returns
    -------
    (V,)  ndarray of Pearson *r*.
    """
    return np.array(
        [pearsonr(y_true[:, v], y_pred[:, v])[0]
         for v in range(y_true.shape[1])],
        dtype=np.float32
    )


@lru_cache(maxsize=None) # We only ever use one atlas size
def get_atlas(n_rois: int = 1000, yeo_networks: int = 7):
    """
    Fetch the Schaefer 2018 atlas with the specified number of ROIs.
    Caches the result to avoid repeated downloads.
    """
    return fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks, verbose=0)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

                # Only keep "valid" timepoints
                valid_mask = attn[i].bool()
                fmri_i = fmri[i][valid_mask].cpu().numpy()
                pred_i = pred[i][valid_mask].cpu().numpy()
                if fmri_i.shape[0] == 0 or pred_i.shape[0] == 0:
                    continue

                subj_to_true[sid].append(fmri_i)
                subj_to_pred[sid].append(pred_i)
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


def evaluate_corr(model,
                  loader,
                  *,
                  device: str | torch.device = "cpu",
                  preprocess=None) -> np.ndarray:
    """
    Voxel-wise Pearson r on *loader* using the same masking logic as
    `collect_predictions`: only time-points whose attention_mask == 1
    are included in the correlation.

    Parameters
    ----------
    preprocess : callable(batch) → None
        Optional in-place batch mutation used by ablation / permutation
        diagnostics *before* tensors are moved to device.

    Returns
    -------
    (V,) ndarray of Pearson r   (mean across all subjects / clips).
    """
    model.eval()
    pred_chunks, true_chunks = [], []

    with torch.no_grad():
        for batch in loader:
            if preprocess is not None:
                preprocess(batch)

            # --- move tensors ---
            attn = batch["attention_masks"].to(device, non_blocking=True)   # (B,T)
            fmri = batch["fmri"].to(device, non_blocking=True)              # (B,T,V)
            feats = {k: batch[k].to(device, non_blocking=True)
                     for k in loader.dataset.modalities}
            subj = batch["subject_ids"]
            run  = batch["run_ids"]

            # --- forward ---
            pred = model(feats, subj, run, attn)                            # (B,T,V)

            # --- apply collect_predictions masking ---
            for i in range(pred.size(0)):
                valid = attn[i].bool()                                      # (T,)
                if valid.any():
                    pred_chunks.append(pred[i][valid].cpu().numpy())
                    true_chunks.append(fmri[i][valid].cpu().numpy())

    y_pred = np.concatenate(pred_chunks, axis=0)   # (T_total, V)
    y_true = np.concatenate(true_chunks, axis=0)
    return voxelwise_pearsonr(y_true, y_pred)


def ensure_paths_exist(*pairs):
    """
    Assert that every provided path exists, otherwise raise FileNotFoundError.

    Parameters
    ----------
    *pairs : tuple[str | Path, str]
        Each entry is (path, human_readable_name).  The second element is only
        used to make the error message clearer.

    Example
    -------
    ensure_paths_exist(
        ("/data/features", "features_dir"),
        (Path("configs/params.yaml"), "params YAML"),
    )
    """
    for p, pretty in pairs:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"{pretty} not found: {p}")
        


