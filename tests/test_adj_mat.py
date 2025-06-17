"""
Compare legacy `adjacency_matrices_old` with the refactored `adjaceny_matrices`
to ensure numerical equivalence (within float tolerance).

Usage:
    pytest -q tests/test_adj_mat.py
"""

import importlib
import numpy as np
import torch
import pytest

def _banner(msg):
    print(f"🧪  {msg}")

# Import both modules
old = importlib.import_module("ref.ref_adjacency_matrices")
new = importlib.import_module("algonauts.utils.adjacency_matrices")

N  = 1000        # ROI count small enough for fast CI
SIGMA = 0.25
TOL = 1e-6      # numerical tolerance


# -------------------------------------------------------------------
# Helper: generic "close enough" assertion for ndarray / tensor
# -------------------------------------------------------------------
def _assert_same(a, b, tol=TOL):
    if isinstance(a, torch.Tensor):
        a, b = a.cpu().numpy(), b.cpu().numpy()
    assert np.allclose(a, b, atol=tol), "Outputs differ beyond tolerance"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
def test_spatial_adjacency():
    _banner("Testing spatial adjacency matrix")
    _assert_same(
        old.get_spatial_adjacency_matrix(sigma=SIGMA, n_rois=N),
        new.get_spatial_adjacency_matrix(sigma=SIGMA, n_rois=N),
    )


def test_network_adjacency():
    _banner("Testing network adjacency matrix")
    _assert_same(
        old.get_network_adjacency_matrix(N),
        new.get_network_adjacency_matrix(N),
    )


@pytest.mark.parametrize("lap_fn", ["spatial", "network"])
def test_laplacians(lap_fn):
    _banner(f"Testing {lap_fn} Laplacian")
    if lap_fn == "spatial":
        A_old = old.get_spatial_adjacency_matrix(sigma=SIGMA, n_rois=N)
        A_new = new.get_spatial_adjacency_matrix(sigma=SIGMA, n_rois=N)
    else:
        A_old = old.get_network_adjacency_matrix(N)
        A_new = new.get_network_adjacency_matrix(N)

    L_old = old.calculate_laplacian(torch.tensor(A_old, dtype=torch.float32))
    L_new = new.calculate_laplacian(torch.tensor(A_new, dtype=torch.float32))
    _assert_same(L_old, L_new)


def test_temporal_laplacian():
    _banner("Testing temporal Laplacian")
    _assert_same(
        old.temporal_Laplacian(1000, 6.0),
        new.temporal_laplacian(1000, 6.0),
    )

@pytest.mark.parametrize("use_knn_spatial_adjacency", [True, False])
def test_get_laplacians(use_knn_spatial_adjacency):
    _banner("Testing combined get_laplacians()")
    Ls_old = old.get_laplacians(SIGMA, use_knn_spatial_adjacency)
    Ls_new = new.get_laplacians(SIGMA, use_knn_spatial_adjacency)
    for Lo, Ln in zip(Ls_old, Ls_new):
        _assert_same(Lo, Ln)