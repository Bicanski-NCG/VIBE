#!/usr/bin/env python3
"""
merging.py  —  three‑level override merger
==========================================

YAML schema (exactly what the user asked for)
---------------------------------------------
atlas: /atlas/schaefer1000_7net.nii.gz   # required, NIfTI parcellation
output_path: /out/path                   # where to write the merged .npy
merge:
  base: "data/outputs/ood_model_1_fixed/submissions/ensemble.npy"    # required
  movies:                                                             # optional
    passepartout: "data/..._fr/submissions/ensemble.npy"
  networks:                                                           # optional
    Vis:     "data/..._vis/submissions/ensemble.npy"
    Default: "data/..._dmn/submissions/ensemble.npy"

*Step order is fixed*:
1.   start with **base**
2.   replace *entire clips* specified in `movies`
3.   graft only the voxels of each ROI specified in `networks`

No subject‑level cherry‑picks; whatever subjects are inside the files get used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict
import zipfile

import numpy as np
import yaml
from algonauts.utils.viz import load_and_label_atlas


# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #
def load_submission(path: Path, cache: Dict[str, dict]) -> dict:
    """Load a .npy submission once and memoize."""
    p = str(path.expanduser().resolve())
    if p not in cache:
        p_path = Path(p)
        if not p_path.exists():
            sys.exit(f"Submission file not found: {p_path}")
        cache[p] = np.load(p_path, allow_pickle=True).item()
    return cache[p]


def get_roi_masks(atlas_path: Path) -> Dict[str, np.ndarray]:
    """Return boolean masks for the Yeo‑7 networks."""
    masker = load_and_label_atlas(str(atlas_path), yeo_networks=7)
    labels = np.asarray(masker.labels[1:])  # drop background
    return {roi: labels == roi for roi in labels}


# ------------------------------------------------------------------ #
# main logic                                                         #
# ------------------------------------------------------------------ #
def merge_predictions(cfg_path: Path) -> None:
    # ---- read YAML ------------------------------------------------ #
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    # minimal schema validation
    if "atlas" not in cfg or "merge" not in cfg or "base" not in cfg["merge"]:
        sys.exit("YAML must contain keys: atlas, merge.base")

    atlas_path = Path(cfg["atlas"]).expanduser().resolve()
    if not atlas_path.exists():
        sys.exit(f"Atlas file not found: {atlas_path}")

    output_path = Path(cfg.get("output_path", cfg_path.with_suffix(".zip"))).expanduser().resolve()

    masks = get_roi_masks(atlas_path)

    cache: Dict[str, dict] = {}

    # ---- 1. load base -------------------------------------------- #
    base_path = Path(cfg["merge"]["base"])
    merged = load_submission(base_path, cache)  # dict {subject: {clip: array}}

    # ---- 2. movie‑level overrides -------------------------------- #
    for movie, path in (cfg["merge"].get("movies") or {}).items():
        sub_dict = load_submission(Path(path), cache)
        for subj, clips in sub_dict.items():
            for clip_name, arr in clips.items():
                if movie in clip_name:
                    if subj not in merged:
                        merged[subj] = {}
                    merged[subj][clip_name] = arr.copy()

    # ---- 3. network‑level overrides ------------------------------ #
    for net, path in (cfg["merge"].get("networks") or {}).items():
        if net not in masks:
            sys.exit(f"Network label '{net}' not in atlas.")
        mask = masks[net]
        sub_dict = load_submission(Path(path), cache)
        for subj, clips in sub_dict.items():
            for clip_name, arr in clips.items():
                if subj not in merged or clip_name not in merged[subj]:
                    # clip absent so far → take full array first
                    merged.setdefault(subj, {})[clip_name] = arr.copy()
                merged[subj][clip_name][..., mask] = arr[..., mask]

    # ---- save ----------------------------------------------------- #
    npy_path = output_path.with_suffix(".npy")
    zip_path = output_path.with_suffix(".zip")
    np.save(npy_path, merged, allow_pickle=True)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(npy_path, arcname=npy_path.name)
    print(f"Saved merged predictions ➜  {output_path}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge predictions with base → movie → ROI overrides")
    parser.add_argument("-c", "--config", required=True, help="YAML file as described above")
    args = parser.parse_args()

    cfg_file = Path(args.config).expanduser().resolve()
    if not cfg_file.exists():
        sys.exit(f"Config file not found: {cfg_file}")

    merge_predictions(cfg_file)