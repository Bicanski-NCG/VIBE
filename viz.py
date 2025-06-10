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
import numpy as np
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy.signal import welch
from scipy.stats import pearsonr
import pandas as pd

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
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    all_labels = np.insert(schaefer.labels, 0,
                           "7Networks_NA_Background_0").astype(str)
    network_labels = [lab.split("_")[2] for lab in all_labels]

    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        labels=network_labels,
        verbose=0
    )
    masker.fit()
    return masker


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
    plt.close(fig)
    return path