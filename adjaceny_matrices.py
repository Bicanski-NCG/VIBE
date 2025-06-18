import nilearn.datasets
import nibabel as nb
import numpy as np
from scipy.ndimage import center_of_mass # For finding centroids
from scipy.spatial.distance import cdist # For pairwise distances
import torch

def get_network_masks(network_names,n_rois = 1000):
    atlas_data = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)

    nets = atlas_data['labels']
    networks = np.array([net.decode('utf-8').split('_')[2] for net in nets])

    masks = {net_name: networks==net_name for net_name in network_names}

    return masks
    
def extract_mni_centroids(n_rois = 1000):
# Assume you've already fetched and loaded the atlas data as shown before:
    atlas_data = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas_img = nb.load(atlas_data.maps)
    atlas_data_array = atlas_img.get_fdata() # This is the NumPy array of voxel labels
   

    print("\nConstructing Volumetric Center-to-Center Distance Matrix...")

    # Get the affine transformation matrix from voxel coordinates to MNI coordinates
    affine = atlas_img.affine

   


    centroids_voxel_ordered = []
    valid_labels = []
    for label in range(1, n_rois + 1):
        roi_mask = (atlas_data_array == label)
        if np.any(roi_mask):
            # center_of_mass returns (z, y, x) for standard numpy array indexing
            com_z, com_y, com_x = center_of_mass(roi_mask)
            # Affine expects (x, y, z, 1) input vector
            centroids_voxel_ordered.append([com_x, com_y, com_z, 1])
            valid_labels.append(label)
        else:
            print(f"Warning: ROI label {label} not found in image data. Skipping.")


    centroids_voxel_ordered = np.array(centroids_voxel_ordered).T # Transpose for matrix multiplication

    # Apply affine transform to get MNI coordinates
    centroids_mni = affine @ centroids_voxel_ordered
    centroids_mni = centroids_mni[:3, :].T # Take the first 3 rows (x, y, z) and transpose back

    # Check if all ROIs were found, otherwise the matrix size will be smaller
    if len(valid_labels) < n_rois:
        print(f"Note: Calculated centroids for only {len(valid_labels)} out of {n_rois} ROIs.")
        # You might need to map these back to the original 1000 indices if you want a 1000x1000 matrix
        # with NaNs for missing ROIs. For simplicity here, we'll build the distance matrix
        # only for the valid ROIs found.
        
    return centroids_mni




def get_spatial_adjacency_matrix(sigma=0.2,threshold = 1e-2, n_rois = 1000):


    centroids_mni = extract_mni_centroids(n_rois)

    # Calculate the pairwise Euclidean distances between all MNI centroids
    # cdist calculates distance between all pairs of rows in two matrices
    adj_spatial_distance = cdist(centroids_mni, centroids_mni, metric='euclidean')

    print("Volumetric Center-to-Center Distance Matrix constructed.")
    # adj_spatial_distance is now a (number_of_valid_rois) x (number_of_valid_rois) matrix
    # containing the Euclidean distances.


    # normalize distances
    D = adj_spatial_distance/adj_spatial_distance.max()

    # scale distances exponentially
    W = np.exp((-D**2)/sigma)

    W[W<threshold]=0.0

    #diagonal should be zero

    W -= np.diag(np.diag(W))

    return W

def knn_distance_constraint_network_adjacency(n_neighbors = 8,n_rois=1000,sigma = 'local_max'):


    centroids = extract_mni_centroids(n_rois)

    distance_matrix = cdist(centroids,centroids)

    network_adj = get_network_adjacency_matrix(n_rois)

    distance_matrix[~(network_adj.astype(bool))] = np.inf



    knn_inds,knn_distances = naive_knn(distance_matrix,n_neighbors)
    
    W = np.zeros_like(distance_matrix)
    
    #we may either normalize locally by largest distance, globally by largest distance, or by custom sigma
    if sigma =='local_max':
        sigma = knn_distances.max(axis=-1,keepdims=True)
    elif sigma =='global_max':
        sigma = knn_distances.max()
        
    knn_distances/=sigma

    W[np.arange(knn_inds.shape[0])[:,None],knn_inds] = np.exp(-knn_distances**2)


    #W = W-np.diag(np.diag(W))

    #symmetrize
    A = np.maximum(W,W.T)
    
    
    return A

def spatial_adjacency_matrix_knn_homogenized(n_neighbors = 8,n_rois = 1000,sigma = 'local_max'):
    '''
    Compute a spatial adjacency matrix with fixed degree based on k-nearest neighbors and homogenization of distances
    '''
    
    centroids = extract_mni_centroids(n_rois)
    
    distance_matrix = cdist(centroids,centroids)
    
    knn_inds,knn_distances = naive_knn(distance_matrix,n_neighbors)
    
    W = np.zeros_like(distance_matrix)
    
    #we may either normalize locally by largest distance, globally by largest distance, or by custom sigma
    if sigma =='local_max':
        sigma = knn_distances.max(axis=-1,keepdims=True)
    elif sigma =='global_max':
        sigma = knn_distances.max()
        
    knn_distances/=sigma

    W[np.arange(knn_inds.shape[0])[:,None],knn_inds] = np.exp(-knn_distances**2)


    #W = W-np.diag(np.diag(W))

    #symmetrize
    A = np.maximum(W,W.T)
    
    
    return A

    

    
def naive_knn(distance_matrix,k,exclude_self = True):
    
    
    arginds = np.argsort(distance_matrix)
    
    if exclude_self:
        knn_inds = arginds[:,1:k+1]
    else:
        knn_inds = arginds[:,:k]
        
    knn_distances = distance_matrix[np.arange(distance_matrix.shape[0])[:,None],knn_inds]
    
    
    return knn_inds,knn_distances


def get_network_adjacency_matrix(n_rois = 1000):
    atlas_data = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)

    print("\nConstructing Network Adjacency Matrix from labels (corrected)...")

    # Get the list of ROI labels
    roi_labels = atlas_data.labels

    # Extract the network name from each label string
    # We need to decode the bytes object to a string first
    roi_networks = []
    for label_bytes in roi_labels: # Skip 'Background' (which might also be bytes)
        # Decode the bytes label to a string using UTF-8 encoding
        label_str = label_bytes.decode('utf-8')

        parts = label_str.split('_')
        if len(parts) > 2:
            # Assuming the format 'YeoNetwork_Hemisphere_NetworkName_Index'
            network_name = parts[2]
            roi_networks.append(network_name)
        else:
            # Handle unexpected format - this shouldn't happen for Schaefer cortical labels
            print(f"Warning: Label '{label_str}' has unexpected format.")
            roi_networks.append("Unknown")

    # Check if we got 1000 network names
    if len(roi_networks) != n_rois:
        print(f"Error: Expected {n_rois} network names, but found {len(roi_networks)}. Check label parsing.")
        # You might need to inspect the roi_labels content if this fails

    # Initialize the network adjacency matrix with zeros
    adj_network = np.zeros((n_rois, n_rois), dtype=bool)

    # Iterate through all unique pairs of ROIs (0-indexed: 0 to 999)
    for i in range(n_rois):
        for j in range(n_rois):
            # Exclude the diagonal
            if i == j:
                continue

            # Check if ROI i and ROI j belong to the same network
            if roi_networks[i] == roi_networks[j]:
                adj_network[i, j] = True
                # Adjacency is symmetric

    print("Network Adjacency Matrix constructed from labels.")
    # adj_network is now a 1000x1000 boolean matrix where True means in the same network
    adj_network_int = adj_network.astype(int)

    return adj_network_int.astype(float)




def get_laplacians(sigma=0.2,use_knn_spatial_adjacency = False):


    if use_knn_spatial_adjacency:
        spatial_adjacency = spatial_adjacency_matrix_knn_homogenized()
    else:
        spatial_adjacency = get_spatial_adjacency_matrix(sigma)

    network_adjacency = get_network_adjacency_matrix()


    spatial = torch.tensor(spatial_adjacency).float()

    network = torch.tensor(network_adjacency).float()


    spatial_laplacian = calculate_laplacian(spatial)

    network_laplacian = calculate_laplacian(network)

    return spatial_laplacian,network_laplacian

def calculate_laplacian(A):
    return torch.eye(A.shape[0]) - A/A.sum(axis=1,keepdim=True)


def temporal_Laplacian(n=1000,sigma=8.0):

    W = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            W[i,j] = np.exp((-abs(i-j))/sigma)

    W[W<0.1]=0.0

    A = torch.tensor(W).float()

    Laplacian = calculate_laplacian(A)

    return Laplacian


     