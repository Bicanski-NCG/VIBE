#!/usr/bin/env python3
"""
Usage
-----
```bash
python align_feat.py \
    --root featX \
    --val-pattern "**/life/*.pt" \
    --out-dir aligned_featX
```

Options
-------
--root         Root directory that contains the original ``.pt`` files.
--val-pattern  Globbing pattern **relative to --root** that *selects* validation
               files.  Everything else is considered training.
--out-dir      Destination root.  The script recreates sub‑directories as
               needed.
--eps          Ridge term added to covariances for numerical stability.
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path
from typing import List, Tuple

import torch

################################################################################
# Utility functions
################################################################################

def list_pt_files(root: Path) -> List[Path]:
    """Recursively list all ``*.pt`` files under *root*."""
    return sorted(root.rglob("*.pt"))

def split_train_val(all_files: List[Path], root: Path, val_pattern: str) -> Tuple[List[Path], List[Path]]:
    """Partition *all_files* into (train, val) by fnmatch on **relative** path."""
    val, train = [], []
    for p in all_files:
        rel = p.relative_to(root).as_posix()  # forward‑slash style for fnmatch
        (val if fnmatch.fnmatch(rel, val_pattern) else train).append(p)
    if not val:
        raise RuntimeError("Validation split is empty – check --val-pattern")
    if not train:
        raise RuntimeError("Training split is empty – adjust --val-pattern")
    return train, val

def load_concat(files: List[Path]) -> torch.Tensor:
    """Load and concatenate 2‑D tensors row‑wise."""
    tensors = []
    for f in files:
        x = torch.load(f, map_location="cpu").float()
        if x.dim() != 2:
            raise ValueError(f"{f}: expected 2‑D tensor, got {x.shape}")
        tensors.append(x)
    return torch.cat(tensors, dim=0)

def mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    cov = xc.t().matmul(xc) / (x.size(0) - 1)
    return mu.squeeze(0), cov

def coral_matrix(c_s: torch.Tensor, c_t: torch.Tensor, eps: float) -> torch.Tensor:
    d = c_s.size(0)
    eye = torch.eye(d, device=c_s.device)
    c_s = c_s + eps * eye
    c_t = c_t + eps * eye
    es, vs = torch.linalg.eigh(c_s)
    et, vt = torch.linalg.eigh(c_t)
    inv_sqrt_s = vs @ torch.diag(torch.rsqrt(es.clamp(min=eps))) @ vs.t()
    sqrt_t = vt @ torch.diag(torch.sqrt(et)) @ vt.t()
    return inv_sqrt_s @ sqrt_t  # (D, D)

def align_tensor(x: torch.Tensor, A: torch.Tensor, mu_s: torch.Tensor, mu_t: torch.Tensor) -> torch.Tensor:
    return (x - mu_t) @ A.t() + mu_s

################################################################################
# Main
################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CORAL to selected validation .pt tensors while keeping folder hierarchy.")
    parser.add_argument("--root", required=True, type=Path, help="Root directory holding original .pt files")
    parser.add_argument("--val-pattern", required=True, help="Glob pattern (relative to --root) that defines validation files")
    parser.add_argument("--out-dir", required=True, type=Path, help="Destination root directory")
    parser.add_argument("--eps", type=float, default=1e-5, help="Numerical stabiliser added to covariances")
    args = parser.parse_args()

    all_files = list_pt_files(args.root)
    train_files, val_files = split_train_val(all_files, args.root, args.val_pattern)

    print(f"Training files:   {len(train_files)}", flush=True)
    print(f"Validation files: {len(val_files)}")

    # Compute CORAL statistics
    print("Computing statistics …")
    X_s = load_concat(train_files)
    mu_s, cov_s = mean_cov(X_s)
    del X_s  # free RAM
    X_t = load_concat(val_files)
    mu_t, cov_t = mean_cov(X_t)
    del X_t

    print("Deriving CORAL transform …")
    A = coral_matrix(cov_s, cov_t, eps=args.eps)

    # Prepare output tree
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save_tensor(src_path: Path, tensor: torch.Tensor):
        dst = args.out_dir / src_path.relative_to(args.root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, dst)

    # 1) Process validation (target) files with CORAL
    print("Aligning validation tensors …")
    for p in val_files:
        x = torch.load(p, map_location="cpu").float()
        x_aligned = align_tensor(x, A, mu_s, mu_t)
        save_tensor(p, x_aligned)

    # 2) Copy / save training files unchanged
    print("Copying training tensors …")
    for p in train_files:
        dst = args.out_dir / p.relative_to(args.root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Use shutil to avoid extra serialise‑deserialise round‑trip.
        shutil.copy2(p, dst)

    print("All done – aligned data written to", args.out_dir)

if __name__ == "__main__":
    main()
