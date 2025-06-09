import gzip, pickle, os
from pathlib import Path
import numpy as np
from nilearn import plotting, datasets
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

def load_and_label_atlas(path):
    atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=1000)
    atlas_data.labels = np.insert(atlas_data.labels, 0, "7Networks_NA_Background_0")
    schaefer_labels = np.array([b.decode('utf-8').split('_')[2] for b in atlas_data.labels])
    atlas_masker = NiftiLabelsMasker(labels_img=path, labels=list(schaefer_labels))
    atlas_masker.fit()
    return atlas_masker

def voxelwise_pearsonr(fmri_val, fmri_val_pred):
    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    return encoding_accuracy

def plot_glass_brain(r, subj_id, masker, out_dir="plots", filename=None):

    mean_r = np.round(np.mean(r), 3)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###

    r_nii = masker.inverse_transform(r)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, {subj_id}, mean accuracy: " + str(mean_r)
    display = plotting.plot_glass_brain(
        r_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    if filename is not None:
        outfile = out_dir / f"{filename}_{subj_id}.png"
    else:
        outfile = out_dir / f"glass_brain_{subj_id}.png"
    display.savefig(outfile, dpi=150)
    display.close()
    return outfile

def plot_corr_hist(r, subj_id, out_dir="plots"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r, bins=60, edgecolor="k", alpha=0.9, color='blue')
    ax.set_xlabel("Pearson $r$")
    ax.set_ylabel("Voxel count")
    ax.set_title(subj_id)
    fig.tight_layout()
    outfile = out_dir / f"corr_hist_{subj_id}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile

def roi_table(r, subj_id, masker, out_dir="plots"):
    """
    Save a CSV of mean voxel-wise r per Schaefer ROI and return the DataFrame.
    """

    labels_array = np.array(masker.labels[1:])           # skip background
    roi_masks = {name: np.where(labels_array == name)[0]
                 for name in np.unique(labels_array)}

    rows = [{"label": name, "mean_r": np.mean(r[idxs])}
            for name, idxs in roi_masks.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_r", ascending=False)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_file = out_dir / f"roi_table_{subj_id}.csv"
        df.to_csv(csv_file, index=True)

    return df

def plot_time_corr(r_t, subj_id, out_dir="plots"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    window = 100
    smoothed = np.convolve(r_t, np.ones(window)/window, mode="same")
    k = max(1, int(0.1 * len(r_t)))
    worst_idx = np.argsort(smoothed)[:k]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_t, lw=0.1, linestyle="-", color="b", alpha=0.3)
    ax.plot(smoothed, color="b", lw=0.3)
    ax.set_ylabel("Pearson's r")
    ax.set_xlabel("Timepoint (TR)")
    ax.set_title(f"Smoothed correlation over time {subj_id}")
    ax.vlines(np.where(np.isnan(r_t))[0], ymin=np.nanmin(r_t), ymax=np.nanmax(r_t), color='r', lw=0.1)
    ax.scatter(worst_idx, smoothed[worst_idx], color="red", label="worst 5 %")

    outfile = out_dir / f"time_corr_{subj_id}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_glass_bads(fmri_pred, fmri_true, subj_id, masker, out_dir="plots", pct_bads=.1):
    r_t = np.array([pearsonr(fmri_pred[t], fmri_true[t])[0] for t in range(fmri_true.shape[0])])
    
    window = 100
    smoothed = np.convolve(r_t, np.ones(window)/window, mode="same")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = max(1, int(pct_bads * len(r_t)))
    worst_idx = np.argsort(smoothed)[:k]
    
    r = voxelwise_pearsonr(fmri_pred[worst_idx], fmri_true[worst_idx])
    
    plot_glass_brain(r, subj_id, masker, out_dir=out_dir, filename="glass_bads")
