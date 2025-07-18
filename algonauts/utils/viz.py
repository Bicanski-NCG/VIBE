"""
Visualization utilities for Algonauts-Decoding.

All “plot_…” functions save their output into *out_dir* and return the
`Path` to the generated file so the caller can pass it to W-and-B or any
other logger.
"""

from __future__ import annotations

# ————————————————————————
# Standard library
# ————————————————————————
import gzip
import os
import pickle
from pathlib import Path
from typing import Iterable, Literal, Sequence

# ————————————————————————
# Third-party
# ————————————————————————
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import welch
from scipy.stats import pearsonr
import pandas as pd
import wandb
import torch
import glob

from algonauts.utils.utils import voxelwise_pearsonr

# ----------------------------------------------------------------------
# W&B helper
# ----------------------------------------------------------------------
def _wandb_log_image(img_path: Path, key_prefix: str = "viz"):
    """
    Log *img_path* to Weights & Biases under key ``f"{key_prefix}/{name}"``.
    Does nothing if no active wandb run.
    """
    if wandb.run is not None:
        wandb.log({f"{key_prefix}/{img_path.name}": wandb.Image(str(img_path))},
                  commit=False)

# ————————————————————————
# Local imports
# ————————————————————————
from algonauts.utils.utils import get_atlas
from algonauts.utils import logger, collect_predictions

# ————————————————————————
# Module-level constants
# ————————————————————————
_DEFAULT_CMAP = "hot_r"
_RESIDUAL_CMAP = "cold_hot"


# ==========================================================================
# Helper utilities
# ==========================================================================

def ensure_dir(path: os.PathLike | str) -> Path:
    """Create *path* if necessary and return it as `Path` instance."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==========================================================================
# Atlas helpers
# ==========================================================================

def load_and_label_atlas(atlas_path: os.PathLike | str,
                         *,
                         yeo_networks: int = 7,
                         anatomical: bool = False,
                         n_rois: int = 1000) -> NiftiLabelsMasker:
    """
    Load Schaefer-2018 labels and attach them to an unlabeled atlas.

    Parameters
    ----------
    atlas_path
        Path to the atlas NIfTI whose integer IDs follow the Schaefer order.
    n_rois
        Number of parcels (100, 200, 400, … 1000).  Must match *atlas_path*.

    Returns
    -------
    NiftiLabelsMasker
        Ready-fit masker with `.labels` attribute.
    """
    schaefer = get_atlas(n_rois=n_rois, yeo_networks=yeo_networks)
    all_labels = np.insert(schaefer.labels, 0,
                           "7Networks_NA_Background_0").astype(str)
    if anatomical:
        # Extract the anatomical labels from the end of the label string
        network_labels = np.array([l.split('_')[3] if not l.split('_')[3].isdigit() else l.split('_')[2] for l in all_labels])
    else:
        # Extact the functional labels from the label string
        network_labels = np.array([l.split('_')[2] for l in all_labels])

    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        labels=network_labels.tolist(),
        verbose=0
    )
    masker.fit()
    return masker

# ==========================================================================
# Scalar diagnostics
# ==========================================================================

def roi_table(r: np.ndarray,
              subj_id: str,
              masker: NiftiLabelsMasker,
              out_dir: os.PathLike | str | None = "plots") -> pd.DataFrame:
    """
    Compute mean *r* per ROI and optionally save as CSV.

    Parameters
    ----------
    r
        Voxel-wise Pearson *r* (V,).
    subj_id
        Subject label or `"group"`.
    masker
        Atlas masker whose labels match the voxel order.
    out_dir
        Directory for the CSV.  If *None*, nothing is written.

    Returns
    -------
    DataFrame
        Columns: `label`, `mean_r` (sorted descending).
    """
    labels = np.array(masker.labels[1:])  # skip background (idx 0)
    roi_masks = {lab: np.where(labels == lab)[0]
                 for lab in np.unique(labels)}
    df = (pd.DataFrame(
        [{"label": lab, "mean_r": float(r[idx].mean())}
         for lab, idx in roi_masks.items()])
          .sort_values("mean_r", ascending=False))

    if out_dir is not None:
        csv_path = ensure_dir(out_dir) / f"roi_table_{subj_id}.csv"
        df.to_csv(csv_path, index=False)

    return df


# ==========================================================================
# Plot helpers (all return Path to image)
# ==========================================================================

def plot_glass_brain(r: np.ndarray,
                     subj_id: str,
                     masker: NiftiLabelsMasker,
                     *,
                     out_dir: os.PathLike | str = "plots",
                     filename: str | None = None,
                     cmap: str = _DEFAULT_CMAP) -> Path:
    """Glass-brain of voxel-wise *r*."""
    out_dir = ensure_dir(out_dir)
    nii = masker.inverse_transform(r)
    title = f"Encoding accuracy – {subj_id}  (mean={r.mean():.3f})"

    disp = plotting.plot_glass_brain(
        nii, display_mode="lyrz", cmap=cmap,
        symmetric_cbar=False, plot_abs=False,
        colorbar=True, title=title
    )
    cbar = disp._cbar
    cbar.set_label("Pearson $r$", rotation=90, labelpad=12, fontsize=12)
    name = f"{filename or 'glass_brain'}_{subj_id}.png"
    path = out_dir / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    disp.close()
    _wandb_log_image(path)
    return path

def plot_voxel_contributions(
    contrib: np.ndarray,
    masker,
    *,
    out_dir: os.PathLike | str = "plots",
    filename: str | None = None,
    cmap=None,
    neg_color: str = "#2166ac",   # blue (used when data contain negatives)
    pos_color: str = "#b2182b",   # red
    zero_color: str = "white",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar_label: str = "Contribution",
) -> Path:
    out_dir = ensure_dir(out_dir)
    contrib = np.asarray(contrib)

    if vmax is None:
        vmax = np.nanmax(contrib)
    if vmin is None:
        vmin = np.nanmin(contrib)

    # Build cmap if not provided
    if cmap is None:
        if vmin < 0 < vmax:  # signed data
            cmap = sns.blend_palette((neg_color, zero_color, pos_color), as_cmap=True)
            sym_cbar = True
        else:  # nonnegative data
            # sequential white→pos_color
            cmap = sns.blend_palette((zero_color, pos_color), as_cmap=True)
            sym_cbar = False
    else:
        # trust user-supplied cmap; infer whether to symmetrize
        sym_cbar = (vmin < 0 < vmax)

    nii = masker.inverse_transform(contrib)

    if title is None:
        title = "Voxel contributions (white=0)"

    disp = plotting.plot_glass_brain(
        nii,
        display_mode="lyrz",
        cmap=cmap,
        symmetric_cbar=sym_cbar,
        plot_abs=False,
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
        title=title,
    )
    cbar = disp._cbar
    cbar.set_label(colorbar_label, rotation=90, labelpad=12, fontsize=12)

    name = f"{filename or 'voxel_contrib'}.pdf"
    path = Path(out_dir) / name
    plt.savefig(path, dpi=600, bbox_inches="tight")
    disp.close()
    _wandb_log_image(path)
    return path


def plot_corr_histogram(r: np.ndarray,
                        subj_id: str,
                        *,
                        out_dir: os.PathLike | str = "plots",
                        bins: int = 60) -> Path:
    """Histogram of voxel-wise *r*."""
    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r, bins=bins, edgecolor="k", alpha=.9, color="royalblue")
    ax.set(xlabel="Pearson $r$", ylabel="Voxel count", title=subj_id)
    path = out_dir / f"corr_hist_{subj_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _wandb_log_image(path)
    plt.close(fig)
    return path


def plot_time_correlation(r_t: np.ndarray,
                          subj_id: str,
                          *,
                          out_dir: os.PathLike | str = "plots",
                          smooth_window: int = 100,
                          highlight_pct: float = .10) -> Path:
    """
    Time-point pattern correlation with smoothing and “worst-k” highlight.
    """
    out_dir = ensure_dir(out_dir)
    smoothed = np.convolve(
        r_t, np.ones(smooth_window) / smooth_window, mode="same"
    )

    k = max(1, int(highlight_pct * len(r_t)))
    worst_idx = np.argsort(smoothed)[:k]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r_t, lw=.2, color="steelblue", alpha=.3)
    ax.plot(smoothed, lw=.5, color="steelblue")
    ax.scatter(worst_idx, smoothed[worst_idx], color="crimson", s=10,
               label=f"worst {highlight_pct:.0%}")
    ax.set(xlabel="TR", ylabel="Pearson $r$",
           title=f"Pattern correlation over time – {subj_id}")
    ax.legend(frameon=False, loc="lower left")

    path = out_dir / f"time_corr_{subj_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _wandb_log_image(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# “Bad TR” glass-brain
# ----------------------------------------------------------------------
def plot_glass_bads(y_pred: np.ndarray,
                    y_true: np.ndarray,
                    subj_id: str,
                    masker: NiftiLabelsMasker,
                    *,
                    out_dir: os.PathLike | str = "plots",
                    pct_bads: float = .10) -> Path:
    """
    Plot glass-brain of the worst *pct_bads* TRs (lowest smoothed corr)."""
    r_t = np.array([pearsonr(y_pred[t], y_true[t])[0]
                    for t in range(y_true.shape[0])])
    smooth = np.convolve(r_t, np.ones(100) / 100, mode="same")
    k = max(1, int(pct_bads * len(r_t)))
    worst = np.argsort(smooth)[:k]

    r_bad = voxelwise_pearsonr(y_true[worst], y_pred[worst])
    return plot_glass_brain(r_bad, subj_id, masker,
                            out_dir=out_dir, filename="glass_bads")


# ----------------------------------------------------------------------
# Residual diagnostics
# ----------------------------------------------------------------------
def plot_residual_glass(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        subj_id: str,
                        masker: NiftiLabelsMasker,
                        *,
                        out_dir: os.PathLike | str = "plots",
                        metric: Literal["mean", "median"] = "mean") -> Path:
    """Glass-brain of mean/median residual (pred − true)."""
    err = (y_pred - y_true)
    voxel_err = err.mean(0) if metric == "mean" else np.median(err, 0)
    vmax = np.percentile(np.abs(voxel_err), 99) or 1e-6

    out_dir = ensure_dir(out_dir)
    nii = masker.inverse_transform(voxel_err.astype(np.float32))
    disp = plotting.plot_glass_brain(
        nii, display_mode="lyrz", cmap=_RESIDUAL_CMAP, colorbar=True,
        symmetric_cbar=True, vmin=-vmax, vmax=vmax, plot_abs=False,
        title=f"Residual {metric} – {subj_id}"
    )
    cbar = disp._cbar
    cbar.set_label("ΔBOLD (pred − true)", rotation=90, labelpad=12, fontsize=12)
    path = out_dir / f"glass_residual_{subj_id}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    disp.close()
    _wandb_log_image(path)
    return path


def plot_pred_vs_true_scatter(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              subj_id: str,
                              *,
                              out_dir: os.PathLike | str = "plots",
                              max_points: int = 50_000) -> Path:
    """Hex-bin scatter of predicted vs. true amplitudes."""
    out_dir = ensure_dir(out_dir)
    x, y = y_true.ravel(), y_pred.ravel()
    if x.size > max_points:
        idx = np.random.choice(x.size, max_points, replace=False)
        x, y = x[idx], y[idx]

    fig, ax = plt.subplots(figsize=(4, 4))
    hb = ax.hexbin(x, y, gridsize=120, mincnt=1)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], "k--", lw=.8)
    ax.set(xlabel="True BOLD (z)", ylabel="Predicted",
           title=f"Pred vs. True – {subj_id}")
    fig.colorbar(hb, ax=ax, label="count")
    path = out_dir / f"scatter_pred_true_{subj_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    _wandb_log_image(path)
    plt.close(fig)
    return path


def plot_residual_psd(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      subj_id: str,
                      *,
                      out_dir: os.PathLike | str = "plots",
                      sample_vox: int = 1_000,
                      fs: float = 1.0) -> Path:
    """Median power spectral density of residuals across *sample_vox* voxels."""
    err = y_pred - y_true
    T, V = err.shape
    vox = np.random.choice(V, min(sample_vox, V), replace=False)
    freqs, pxx = welch(err[:, vox], fs=fs, axis=0, nperseg=min(256, T))
    p_med = np.median(pxx, 1)

    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.loglog(freqs[1:], p_med[1:])
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD",
           title=f"Median PSD residuals – {subj_id}")
    path = out_dir / f"psd_residual_{subj_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _wandb_log_image(path)
    plt.close(fig)
    return path

def plot_diagnostics(model, loader, config, out_dir):
    """Generate subject‑level and group‑level visual diagnostics and log them to W&B."""
    fmri_true, fmri_pred, subj_ids, atlas_paths = collect_predictions(loader, model, config.device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist validation predictions for later analysis
    pred_path = out_dir / "val_predictions.pkl.gz"
    with gzip.open(pred_path, "wb") as f:
        pickle.dump(
            {
                "subjects": subj_ids,
                "atlas_paths": atlas_paths,
                "fmri_true": fmri_true,   # lists of (T, V) ndarrays
                "fmri_pred": fmri_pred,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ----- Per‑subject diagnostics -----
    for true, pred, sid, atlas_path in zip(fmri_true, fmri_pred, subj_ids, atlas_paths):
        logger.info(f"📊 Diagnostics for subject {sid} …")

        r = voxelwise_pearsonr(true, pred)

        masker = load_and_label_atlas(atlas_path, yeo_networks=config.yeo_networks, anatomical=False)

        plot_glass_brain(r, sid, masker, out_dir=str(out_dir))

        # plot_corr_histogram(r, sid, out_dir=str(out_dir))

        df_roi = roi_table(r, sid, masker, out_dir=str(out_dir))
        table_roi = wandb.Table(dataframe=df_roi.astype({"mean_r": float}))
        bar_chart = wandb.plot.bar(
            table_roi,
            "label",      # x‑axis
            "mean_r",     # y‑axis
            title=f"ROI mean Pearson r – {sid}",
        )
        wandb.log({f"viz/roi_bar_{sid}": bar_chart}, commit=False)

        r_t = np.array([pearsonr(true[t], pred[t])[0] for t in range(true.shape[0])])
        plot_time_correlation(r_t, sid, out_dir=str(out_dir))

        # plot_glass_bads(true, pred, sid, masker, out_dir=str(out_dir), pct_bads=config.pct_bads)

        # plot_residual_glass(true, pred, sid, masker, out_dir=str(out_dir))

        # plot_pred_vs_true_scatter(true, pred, sid, out_dir=str(out_dir))

        # plot_residual_psd(true, pred, sid, out_dir=str(out_dir), fs=1/1.49)

    # ----- Group diagnostics -----
    logger.info("📊 Group diagnostics …")
    group_mean_r = np.mean([voxelwise_pearsonr(true, pred) for true, pred in zip(fmri_true, fmri_pred)], axis=0)
    group_masker = load_and_label_atlas(atlas_paths[0], yeo_networks=config.yeo_networks, anatomical=False)  # use first atlas for group

    plot_glass_brain(group_mean_r, "group", group_masker, out_dir=str(out_dir))

    # plot_corr_histogram(group_mean_r, "group", out_dir=str(out_dir))

    df_group_roi = roi_table(group_mean_r, "group", group_masker, out_dir=str(out_dir))
    table_roi = wandb.Table(dataframe=df_group_roi.astype({"mean_r": float}))
    bar_chart = wandb.plot.bar(
        table_roi,
        "label",       # x‑axis
        "mean_r",      # y‑axis
        title="Group ROI mean Pearson r",
    )
    wandb.log({"viz/roi_bar_group": bar_chart}, commit=False)

    r_t_list = [
        np.array([pearsonr(pred[t], true[t])[0]
                for t in range(true.shape[0])])
        for true, pred in zip(fmri_true, fmri_pred)
    ]
    max_T   = max(arr.size for arr in r_t_list)
    r_t_mat = np.full((len(r_t_list), max_T), np.nan)
    for i, arr in enumerate(r_t_list):
        r_t_mat[i, :arr.size] = arr

    group_r_t = np.nanmean(r_t_mat, axis=0)   # (max_T,)
    plot_time_correlation(group_r_t, "group", out_dir=str(out_dir))

    # group_res_true = np.concatenate([t for t in fmri_true], 0)
    # group_res_pred = np.concatenate([p for p in fmri_pred], 0)
    # plot_residual_glass(group_res_true, group_res_pred, "group", group_masker, out_dir=str(out_dir))

    # plot_glass_bads(
    #     group_res_true,
    #     group_res_pred,
    #     "group",
    #     group_masker,
    #     out_dir=str(out_dir),
    #     pct_bads=config.pct_bads
    # )

    # plot_pred_vs_true_scatter(
    #     group_res_true, group_res_pred, "group", out_dir=str(out_dir), max_points=config.max_scatter_points
    # )

    # plot_residual_psd(group_res_true, group_res_pred, "group", out_dir=str(out_dir), fs=1/1.49)

    # (W&B logging of images is now done directly in each plot_… function.)
