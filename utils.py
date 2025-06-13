from collections import defaultdict
import random
import wandb
import numpy as np
import torch

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