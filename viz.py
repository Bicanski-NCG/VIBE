import os, pathlib, itertools
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy.stats import pearsonr
from nilearn.maskers import NiftiLabelsMasker


# ----------------------------------------------------------------------
# 1. Glass-brain plots  (per-subject + group mean)
# ----------------------------------------------------------------------
def plot_glass_brain_set(
    fmri_true, fmri_pred,                 # list of (T,V) arrays, one per subject
    subj_ids,                             # ["01","02",...]
    atlas_paths,                          # list of .nii.gz label atlases per subj
    out_dir="plots", prefix="val",
    cmap="hot_r",
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    group_r = []

    for true, pred, sid, atlas_file in zip(fmri_true, fmri_pred, subj_ids, atlas_paths):
        r = np.array([pearsonr(true[:, v], pred[:, v])[0]
                      for v in range(true.shape[1])], dtype=np.float32)
        group_r.append(r)

        masker = NiftiLabelsMasker(labels_img=atlas_file).fit()
        statmap = masker.inverse_transform(r)

        title = f"sub-{sid}  |  mean r = {r.mean():.3f}"
        display = plotting.plot_glass_brain(
            statmap, display_mode="lyrz", cmap=cmap, colorbar=True,
            plot_abs=False, symmetric_cbar=False, title=title, vmax=None
        )
        display._cbar.set_label("Pearson $r$")
        fname = os.path.join(out_dir, f"{prefix}_glass_sub-{sid}.png")
        display.savefig(fname, dpi=150)
        display.close()

    # group mean
    group_r = np.stack(group_r).mean(axis=0)
    masker = NiftiLabelsMasker(labels_img=atlas_paths[0]).fit()
    statmap = masker.inverse_transform(group_r)

    display = plotting.plot_glass_brain(
        statmap, display_mode="lyrz", cmap=cmap, colorbar=True,
        plot_abs=False, symmetric_cbar=False,
        title=f"group-mean  |  mean r = {group_r.mean():.3f}",
        vmin=0.0
    )
    display._cbar.set_label("Pearson $r$")
    fname = os.path.join(out_dir, f"{prefix}_glass_group.png")
    display.savefig(fname, dpi=150)
    display.close()


# ----------------------------------------------------------------------
# 2. Histogram(s) of voxelwise correlations
# ----------------------------------------------------------------------
def plot_corr_hist_set(
    r_per_subj,             # list of 1-D arrays (V,)
    subj_ids,
    out_dir="plots", prefix="val",
    nbins=60,
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # per-subject
    for r, sid in zip(r_per_subj, subj_ids):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(r, bins=nbins, edgecolor="k", alpha=0.75)
        ax.set_xlabel("Pearson $r$")
        ax.set_ylabel("Voxel count")
        ax.set_title(f"sub-{sid}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_hist_sub-{sid}.png"), dpi=150)
        plt.close(fig)

    # group
    r_all = np.concatenate(r_per_subj)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(r_all, bins=nbins, edgecolor="k", alpha=0.9, color="slateblue")
    ax.set_xlabel("Pearson $r$")
    ax.set_ylabel("Voxel count")
    ax.set_title("group mean")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_hist_group.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Time × voxel imshow with correlation colour-bar
# ----------------------------------------------------------------------
def plot_time_voxel_imshow(
    true, pred,            # (T, V) ndarray for ONE subject
    subj_id, stim_tag,
    out_dir="plots", prefix="val",
    vmax=None,
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    T, V = true.shape
    r = np.array([pearsonr(true[:, v], pred[:, v])[0] for v in range(V)])

    # Optionally reorder voxels by r
    order = np.argsort(-r)          # descending correlation
    true_ord = true[:, order]
    pred_ord = pred[:, order]
    r_ord = r[order]

    vmax = vmax or np.nanpercentile(np.abs(np.concatenate([true_ord, pred_ord])), 99)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 5, 0.3], wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(true_ord.T, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax0.set_title("Ground truth")
    ax0.set_ylabel("voxels  (sorted by $r$)")
    ax0.set_xlabel("time")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(pred_ord.T, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax1.set_title("Prediction")
    ax1.set_yticks([]); ax1.set_xlabel("time")

    # side strip: voxel-wise r
    ax2 = fig.add_subplot(gs[2])
    sc = ax2.scatter(np.zeros_like(r_ord), np.arange(V), c=r_ord, cmap="viridis",
                     vmin=0, vmax=1, s=4)
    ax2.set_ylim(-0.5, V-0.5)
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(sc, ax=ax2, fraction=0.8, label="Pearson $r$")

    fig.suptitle(f"sub-{subj_id} | {stim_tag}")
    fname = os.path.join(out_dir, f"{prefix}_imshow_sub-{subj_id}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname