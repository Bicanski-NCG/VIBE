import argparse
import os
from pathlib import Path
import glob
import random
import numpy as np
import torch
import zipfile

from tqdm import tqdm
import wandb
from algonauts.data.loader import get_train_val_loaders
from algonauts.models import load_model_from_ckpt
from algonauts.models.ensemble import EnsembleAverager, ROIAdaptiveEnsemble
from algonauts.utils import ensure_paths_exist, collect_predictions
from algonauts.utils.utils import collect_predictions_per_sample, voxelwise_pearsonr
from algonauts.utils.viz import load_and_label_atlas, plot_glass_brain, roi_table

model, config = load_model_from_ckpt(
                model_ckpt_path="few/checkpoints/3tyx97qv/final_model.pt",
                params_path="few/checkpoints/3tyx97qv/config.yaml",
            )
print("Loaded model")
wandb.init(project="try", config=vars(config), )

model.to("cuda")
config.val_name = "figures"
config.filter_name = []
config.batch_size = 1
train_loader, valid_loader = get_train_val_loaders(config)
print(len(train_loader), len(valid_loader))
print("Loaded data")
out_dir="many/Figures"

fmri_true, fmri_pred, meta = collect_predictions_per_sample(
    valid_loader, model, device="cuda"
)

print(f"n samples: {len(fmri_true)}")
print("Collected predictions")

for true, pred, m in tqdm(zip(fmri_true, fmri_pred, meta), total=len(fmri_true)):
    sid        = m["subject"]
    atlas_path = m["atlas_path"]
    dataset_name = m["dataset_name"]

    r = voxelwise_pearsonr(true, pred)
    masker = load_and_label_atlas(atlas_path,
                                  yeo_networks=config.yeo_networks,
                                  anatomical=False)

    plot_glass_brain(r, sid, masker, out_dir=str(out_dir), dataset_name=dataset_name)
    # plot_corr_histogram(r, sid, out_dir=str(out_dir))
    df_roi = roi_table(r, sid, masker, out_dir=str(out_dir))


    
