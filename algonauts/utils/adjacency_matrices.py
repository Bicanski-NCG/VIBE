import nibabel as nb
import numpy as np
from scipy.ndimage import center_of_mass # For finding centroids
from scipy.spatial.distance import cdist # For pairwise distances
import torch

from algonauts.utils.utils import get_atlas

def get_network_masks(network_names, n_rois: int = 1000):
    """
    Return a dict {network_name: boolean_mask (len=n_rois)}.
    """
    atlas = get_atlas(n_rois)
    networks = np.array([lbl.decode("utf-8").split("_")[-2] for lbl in atlas["labels"]])
    return {name: networks == name for name in network_names}

def get_spatial_adjacency_matrix(sigma: float = 0.2, n_rois: int = 1000, thresh: float = 1e-2) -> np.ndarray:
    """
    Spatial affinity matrix  W  (shape n_rois × n_rois)
    ---------------------------------------------------
    W_ij = exp(-d_ij² / σ)  where  d_ij  is the centre-to-centre distance
    of ROI *i* & *j* in MNI space, normalised to [0, 1].

    Missing ROIs (not present in the volume) yield rows/cols of zeros.
    """

    atlas = get_atlas(n_rois)
    img   = nb.load(atlas.maps)
    data  = img.get_fdata()
    aff   = img.affine

    # ----- vectorised centres of mass (≈ 1 ms vs 2 min loop) -----------------
    labels = np.arange(1, n_rois + 1)
    # SciPy returns (z, y, x); shape (n_rois, 3)
    com = np.array(
        center_of_mass(np.ones_like(data), labels=data, index=labels),
        dtype=np.float32,
    )

    # Handle labels that are absent → centre_of_mass returns NaNs
    missing = np.isnan(com[:, 0])
    if missing.any():
        miss_idx = ", ".join(map(str, labels[missing]))
        print(f"⚠️  {missing.sum()} ROIs missing in volume: {miss_idx}")

    # (x, y, z) voxel → MNI
    mni = nb.affines.apply_affine(aff, com[:, ::-1])   # swap to (x, y, z) order
    mni[missing] = np.nan                              # keep NaNs for later

    # ----- distance & kernel --------------------------------------------------
    D = cdist(mni, mni, metric="euclidean")            # NaNs propagate
    D_max = np.nanmax(D)                               # ignore NaNs
    D /= D_max

    W = np.exp(-(D ** 2) / sigma)
    W[np.isnan(W)] = 0.0                               # rows with missing ROIs
    W[W < thresh]  = 0.0
    np.fill_diagonal(W, 0.0)
    return W

def get_network_adjacency_matrix(n_rois: int = 1000) -> np.ndarray:
    """
    Binary ROI-adjacency matrix (shape n_rois × n_rois) where
        A_ij = 1  ⇔  ROI *i* and *j* belong to the same Yeo network.
    """
    atlas = get_atlas(n_rois)

    # network label is always the penultimate token: 'Yeo17_RH_Default_37'
    nets = np.array([lbl.decode("utf-8").split("_")[2] for lbl in atlas.labels],
                    dtype="U16")   # small fixed-width dtype

    # vectorised equality → boolean matrix
    same_net = (nets[:, None] == nets[None, :]).astype(float)

    # zero diagonal
    np.fill_diagonal(same_net, 0.0)
    return same_net

def calculate_laplacian(A: torch.Tensor) -> torch.Tensor:
    """Unnormalised graph Laplacian L = I - D⁻¹A with safe divide."""
    deg = A.sum(1, keepdim=True).clamp(min=1e-6)
    return torch.eye(A.size(0), device=A.device) - A / deg

def temporal_laplacian(n: int = 1000, sigma: float = 8.0, thresh: float = 0.1):
    """Temporal Laplacian with exponential decay along diagonal."""
    idx = np.arange(n)
    W = np.exp(-np.abs(idx[:, None] - idx[None, :]) / sigma)
    W[W < thresh] = 0.0
    A = torch.tensor(W, dtype=torch.float32)
    return calculate_laplacian(A)

temporal_Laplacian = temporal_laplacian  # backward compatibility

def get_laplacians(sigma: float = 0.2):
    spatial = torch.tensor(get_spatial_adjacency_matrix(sigma), dtype=torch.float32)
    network = torch.tensor(get_network_adjacency_matrix(), dtype=torch.float32)
    return calculate_laplacian(spatial), calculate_laplacian(network)