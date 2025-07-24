import nibabel as nb
import numpy as np
from scipy.ndimage import center_of_mass # For finding centroids
from scipy.spatial.distance import cdist # For pairwise distances
import torch

from algonauts.utils import get_atlas


def get_network_masks(network_names, n_rois: int = 1000):
    """
    Return a dict {network_name: boolean_mask (len=n_rois)}.
    """
    atlas = get_atlas(n_rois)
    networks = np.array([lbl.decode("utf-8").split("_")[-2] for lbl in atlas["labels"]])
    return {name: networks == name for name in network_names}


def extract_mni_centroids(n_rois: int = 1000) -> np.ndarray:
    """
    Extract MNI coordinates of the centres of mass of ROIs in the atlas.
    Returns an array of shape (n_rois, 3) with MNI coordinates.
    """
    atlas = get_atlas(n_rois)
    img   = nb.load(atlas.maps)
    data  = img.get_fdata()
    aff   = img.affine

    labels = np.arange(1, n_rois + 1)

    com = np.array(
        center_of_mass(np.ones_like(data), labels=data, index=labels),
        dtype=np.float32,
    )

    missing = np.isnan(com[:, 0])
    if missing.any():
        miss_idx = ", ".join(map(str, labels[missing]))
        print(f"⚠️  {missing.sum()} ROIs missing in volume: {miss_idx}")

    mni = nb.affines.apply_affine(aff, com[:, ::-1])
    mni[missing] = np.nan
    return mni.astype(np.float32)


def get_spatial_adjacency_matrix(sigma: float = 0.2, n_rois: int = 1000, thresh: float = 1e-2) -> np.ndarray:
    """
    Spatial affinity matrix  W  (shape n_rois × n_rois)
    ---------------------------------------------------
    W_ij = exp(-d_ij² / σ)  where  d_ij  is the centre-to-centre distance
    of ROI *i* & *j* in MNI space, normalised to [0, 1].

    Missing ROIs (not present in the volume) yield rows/cols of zeros.
    """

    mni = extract_mni_centroids(n_rois)

    D = cdist(mni, mni, metric="euclidean")
    D_max = np.nanmax(D)
    D /= D_max

    W = np.exp(-(D ** 2) / sigma)
    W[np.isnan(W)] = 0.0
    W[W < thresh]  = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def get_network_adjacency_matrix(n_rois: int = 1000) -> np.ndarray:
    """
    Binary ROI-adjacency matrix (shape n_rois × n_rois) where
        A_ij = 1  ⇔  ROI *i* and *j* belong to the same Yeo network.
    """
    atlas = get_atlas(n_rois)

    nets = np.array([lbl.decode("utf-8").split("_")[2] for lbl in atlas.labels],
                    dtype="U16")   # small fixed-width dtype

    same_net = (nets[:, None] == nets[None, :]).astype(float)
    np.fill_diagonal(same_net, 0.0)
    return same_net


def spatial_adjacency_matrix_knn_homogenized(n_neighbors = 8, n_rois = 1000, sigma = 'local_max'):
    '''
    Compute a spatial adjacency matrix with fixed degree based on k-nearest neighbors and homogenization of distances
    '''
    
    centroids = extract_mni_centroids(n_rois)
    distance_matrix = cdist(centroids,centroids)
    knn_inds,knn_distances = naive_knn(distance_matrix,n_neighbors)
    
    W = np.zeros_like(distance_matrix)
    
    if sigma =='local_max':
        sigma = knn_distances.max(axis=-1,keepdims=True)
    elif sigma =='global_max':
        sigma = knn_distances.max()
        
    knn_distances/=sigma
    W[np.arange(knn_inds.shape[0])[:,None],knn_inds] = np.exp(-knn_distances**2)
    A = np.maximum(W,W.T)
    return A


def naive_knn(distance_matrix,k,exclude_self = True):
    """
    Naive k-nearest neighbors search on a distance matrix.
    """
    arginds = np.argsort(distance_matrix)
    if exclude_self:
        knn_inds = arginds[:,1:k+1]
    else:
        knn_inds = arginds[:,:k]
        
    knn_distances = distance_matrix[np.arange(distance_matrix.shape[0])[:,None],knn_inds]
    return knn_inds,knn_distances


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


def get_laplacians(sigma: float = 0.2, use_knn_spatial_adjacency: bool = False):
    if use_knn_spatial_adjacency:
        spatial = torch.tensor(spatial_adjacency_matrix_knn_homogenized(), dtype=torch.float32)
    else:
        spatial = torch.tensor(get_spatial_adjacency_matrix(sigma), dtype=torch.float32)
    network = torch.tensor(get_network_adjacency_matrix(), dtype=torch.float32)
    return calculate_laplacian(spatial), calculate_laplacian(network)