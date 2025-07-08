import os
import re
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
from tqdm import tqdm 

class FMRI_Dataset(Dataset):
    def __init__(
        self,
        root_folder_fmri,
        feature_paths,
        input_dims,
        modalities,
        noise_std=0.0,
        normalization_stats=None,
        oversample_factor=1,
        samples=None,
        normalize_bold=False,
        modality_dropout_mode = 'zeros',
        modality_dropout_prob = 0.1,
        normalize_features = False,
        loss_masks_path=None # None or paths
    ):
        super().__init__()

        self.root_folder = root_folder_fmri
        self.feature_paths = feature_paths  # Dict of {modality: path}

        

        self.input_dims = input_dims
        self.modalities = modalities

        self.modality_dropout_prob = modality_dropout_prob
        self.modality_dropout_mode = modality_dropout_mode

        self.noise_std = noise_std
        self.normalize_features = normalize_features
        if (normalization_stats is None) and ((self. modality_dropout_mode =='gaussian') or (self.normalize_features)):
            warnings.warn(f"No normalization stats provided but required for the options you chose. ' \
            'Will try to fetch them from current working directory {os.getcwd()}")
            fname = 'normalization_stats.pt'
            
            normalization_stats_valid = False

            if os.path.isfile('normalization_stats.pt'):
                normalization_stats_valid = True

                print('Found file at', os.getcwd(),fname)
                normalization_stats = torch.load(fname)
                for modality,path in feature_paths.items():
                    if not normalization_stats.get(modality):
                        warnings.warn("Did not find all features of this dataset in the normalization_stats.pt. Will recompute them to be safe.")
                        normalization_stats_valid = False
                        break



            if not normalization_stats_valid:
                warnings.warn('No valid normalization stats provided. Will compute normalization stats and save in current directory. This might take a while.')
                normalization_stats = compute_statistics(feature_paths)
                torch.save(normalization_stats,fname)







        self.normalization_stats = normalization_stats

        self.oversample_factor = oversample_factor
        self.normalize_bold = normalize_bold  # enable/disable run‑wise z‑score

        self._h5_cache = {}  # Cache for h5 files to avoid reopening

        # Precompute feature file index
        self._feature_index = {modality: {} for modality in feature_paths.keys()}
        for modality, root_path in self.feature_paths.items():
            for file_path in glob.glob(os.path.join(root_path, "**", "*.*"), recursive=True):
                basename = os.path.splitext(os.path.basename(file_path))[0]
                self._feature_index[modality][basename] = file_path

        self.subject_name_id_dict = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}

        if samples is not None:
            self.samples = samples
        else:
            self.fmri_files = sorted(
                glob.glob(os.path.join(root_folder_fmri, "sub-0?", "func", "*.h5")))
            self.samples = []
            for fmri_file in self.fmri_files:
                subject_id = os.path.basename(
                    os.path.dirname(os.path.dirname(fmri_file))
                )
                subject_atlas = glob.glob(
                    os.path.join(os.path.dirname(os.path.dirname(fmri_file)), "atlas","*.nii.gz")
                )
                if not subject_atlas:
                    raise FileNotFoundError(
                        f"No atlas found for subject {subject_id} in {fmri_file}"
                    )
                else:
                    subject_atlas = subject_atlas[0]
                
                with h5py.File(fmri_file, "r") as h5file:
                    for dataset_name in h5file.keys():
                        num_samples = h5file[dataset_name].shape[0]
                        name_re = re.compile(r"^(?:ses-\d+_task-)(?P<file_name>(?P<name>s\d+|[a-zA-Z]+)[a-zA-Z0-9]+)(?:_run-(?P<run>\d*)$)?")
                        match = name_re.match(dataset_name)                            
                        file_name = match.group("file_name")
                        name = match.group("name")
                        run = match.group("run") if match.group("run") else 1
                        has_multiple_runs = match.group("run") is not None
                        sample = {
                            "subject_id": subject_id,
                            "fmri_file": fmri_file,
                            "subject_atlas": subject_atlas,
                            "dataset_name": dataset_name,
                            "num_samples": num_samples,
                            "is_movie": "movie" in fmri_file.lower(),
                            "file_name": file_name,
                            "name": name,
                            "run": int(run) - 1,
                            "has_multiple_runs": has_multiple_runs
                        }
                        self.samples.append(sample)

        if loss_masks_path:
            self.loss_masks = torch.load(loss_masks_path)
            #should be a dictionary with (subject_id,dataset_name)
        else:
            self.loss_masks = None
    def __len__(self):
        return len(self.samples)

    def filter_samples(self, filter_fn):
        """
        Filters the samples based on a provided filter function.
        The filter function should take a sample dict and return True if the sample should be included.
        """
        self.samples = [s for s in self.samples if filter_fn(s)]
        return self

    def find_feature_file(self, modailty, file_name):
        try:
            return self._feature_index[modailty][file_name]
        except KeyError:
            # feature may have prefixes like "friends_" or "movies10_"
            for key in self._feature_index[modailty].keys():
                if file_name in key:
                    return self._feature_index[modailty][key]
            else: # If no match found, raise an error
                raise FileNotFoundError(
                    f"Feature file for {file_name} not found in {modailty} features."
                )

    def normalize(self, data, mean, std):
    
        return (data - mean) / std
    
    def _get_h5(self, path):
        handle = self._h5_cache.get(path, None)
        if handle is None:
            handle = h5py.File(path, "r", libver='latest', swmr=True)
            if handle is None:
                raise FileNotFoundError(f"Could not open HDF5 file: {path}")
            self._h5_cache[path] = handle
        return handle
    
    def _get_feature(self, path):
        if path.endswith(".npy"):
            try:
                arr = np.load(path, mmap_mode='r')
                return torch.tensor(arr, dtype=torch.float32).squeeze()
            except EOFError:
                raise ValueError(f"Corrupted feature file: {path}")
        elif path.endswith(".pt"):
            try:
                return torch.load(path, map_location="cpu").squeeze().float()
            except EOFError:
                raise ValueError(f"Corrupted feature file: {path}")
        else:
            raise ValueError(f"Unknown feature file extension: {path}")
    
    def get_loss_mask(self,dataset_name,subject_id,fmri_file):
        if self.loss_masks:

            loss_mask = self.loss_masks.get((subject_id,dataset_name),None)
        else:
            loss_mask =None

        if loss_mask is None:
            loss_mask = torch.ones_like(fmri_file)

        return loss_mask

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject_id = self.subject_name_id_dict[sample_info["subject_id"]]
        run_id = int(sample_info["has_multiple_runs"]) + sample_info["run"]
        fmri_file = sample_info["fmri_file"]
        dataset_name = sample_info["dataset_name"]

        file_name_features = (
            f"{dataset_name.split('-')[-1]}"
        )

        fmri_response = self._get_h5(fmri_file)[dataset_name][:]

        fmri_response_tensor = torch.tensor(fmri_response, dtype=torch.float32)
        # ---- Run‑wise (per‑key) z‑scoring ---------------------------------
        if self.normalize_bold:
            mu    = fmri_response_tensor.mean(dim=0, keepdim=True)
            sigma = fmri_response_tensor.std(dim=0, keepdim=True) + 1e-6
            fmri_response_tensor = (fmri_response_tensor - mu) / sigma
        # -------------------------------------------------------------------

        features = {}
        min_samples = fmri_response_tensor.shape[0]

        for modality, root_path in self.feature_paths.items():
            path = self.find_feature_file(modality, file_name_features)
            data = self._get_feature(path)

            if data.isnan().any():
                data = torch.nan_to_num(data, nan=0.0)

            if self.modality_dropout_prob>0: 
                if self.modality_dropout_mode =='zeros':
                    if float(torch.rand(1)) < self.modality_dropout_prob:
                        # Replace with zeros; keeps tensor shape & device
                        data = torch.zeros_like(data)


                elif self.modality_dropout_mode == 'gaussian':
                    if float(torch.rand(1)) < self.modality_dropout_prob:
                        
                        X = torch.randn_like(data)
                        mean = self.normalization_stats[modality]['mean']
                        std = self.normalization_stats[modality]['std']
                        data = mean + std*X






            if (self.normalize_features and self.normalization_stats 
                and self.normalization_stats.get(modality)
                  and self.normalization_stats.get(modality).get('mean') ):
                
                data = self.normalize(
                    data,
                    self.normalization_stats[modality]["mean"],
                    self.normalization_stats[modality]["std"]
                )

            if self.noise_std > 0:
                data += torch.randn_like(data) * self.noise_std

            min_samples = min(min_samples, data.shape[0])
            features[modality] = data

        # Truncate all to same length
        for key in features:
            features[key] = features[key][:min_samples]
        fmri_response_tensor = fmri_response_tensor[:min_samples]

        loss_mask = self.get_loss_mask(dataset_name,subject_id,fmri_response_tensor)

        return subject_id, run_id, features, fmri_response_tensor,loss_mask


def compute_mean_std(dataset):
    sums = {key: [] for key in dataset.feature_paths}
    for _, features, _ in DataLoader(dataset, batch_size=1):
        for modality, feat in features.items():
            sums[modality].append(feat.squeeze(0))
    stats = {}
    for modality, data in sums.items():
        data_all = torch.cat(data, dim=0)
        stats[f"{modality}_mean"] = data_all.mean(dim=0)
        stats[f"{modality}_std"] = data_all.std(dim=0) + 1e-8
    return stats


def split_dataset_by_name(dataset, val_name="06", val_run="all",
                          train_noise_std=0.00, normalize_validation_bold=False):
    """
    Splits the dataset into training and validation sets based on the `val_name` and `val_run` parameters.
    If `val_name` is found in the sample name, it is considered a validation sample.
    If `val_run` is specified, only validation samples from that run will be included.
    If `val_run` is "all", no filtering is applied.

    Parameters
    ----------
    dataset : FMRI_Dataset
        The dataset to split.
    val_name : str
        The substring to match for validation samples. Default is "06".
    train_noise_std : float
        Standard deviation of noise to add to training samples. Default is 0.00.
    normalize_validation_bold : bool
        Whether to normalize BOLD responses in the validation set. Default is False.
    filter_run : str
        If specified, will filter validation samples by this run name.
        If "all", no filtering is applied. Default is "all".
    Returns
    -------
    train_ds : FMRI_Dataset
        The training dataset.
    val_ds : FMRI_Dataset
        The validation dataset.
    """
    train_samples, val_samples = [], []

    for sample in dataset.samples:
        if val_name.lower() in sample["name"].lower():
                if val_run == "all" or str(sample["run"]) == str(val_run):
                    val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_ds = FMRI_Dataset(
        root_folder_fmri=dataset.root_folder,
        feature_paths=dataset.feature_paths,
        input_dims=dataset.input_dims,
        modalities=dataset.modalities,
        normalization_stats=dataset.normalization_stats,
        noise_std=train_noise_std,
        samples=train_samples,
        normalize_bold=dataset.normalize_bold,
        modality_dropout_mode = dataset.modality_dropout_mode,
        modality_dropout_prob =  dataset.modality_dropout_prob,
        normalize_features = dataset.normalize_features
        
    )

    val_ds = FMRI_Dataset(
        root_folder_fmri=dataset.root_folder,
        feature_paths=dataset.feature_paths,
        input_dims=dataset.input_dims,
        modalities=dataset.modalities,
        noise_std=0.0,
        samples=val_samples,
        normalize_bold=normalize_validation_bold,
        modality_dropout_mode = dataset.modality_dropout_mode,
        modality_dropout_prob =  0.0, #no dropout in validation set
        normalize_features = dataset.normalize_features
    )

    return train_ds, val_ds


def collate_fn(batch):
    subject_ids, run_ids, features_list, fmri_responses,loss_mask = zip(*batch)

    all_modalities = features_list[0].keys()
    padded_features = {
        modality: pad_sequence(
            [f[modality] for f in features_list], batch_first=True, padding_value=0
        )
        for modality in all_modalities
    }

    fmri_padded = pad_sequence(fmri_responses, batch_first=True, padding_value=0)

    loss_mask_padded = pad_sequence(loss_mask, batch_first=True, padding_value=0)

    # Example attention mask (you can adapt this per modality if needed)
    seq_lengths = [next(iter(f.values())).shape[0] for f in features_list]
    max_len = max(seq_lengths)
    idx = torch.arange(max_len)
    attention_masks = (idx.unsqueeze(0) < torch.tensor(seq_lengths).unsqueeze(1)).bool()

    return {
        "subject_ids": subject_ids,
        "run_ids": run_ids,
        **padded_features,
        "fmri": fmri_padded,
        "attention_masks": attention_masks,
        "loss_mask":loss_mask_padded
    }


def make_group_weights(dataset, filter_on: str):
    """
    Return a weight‐vector w ∈ ℝᴺ that equalizes the total number of TRs
    drawn from each distinct value of `filter_on`.

    For example:
      • filter_on="is_movie":
          • groups are {False} and {True}, each group’s weights sum to 1.
      • filter_on="dataset_name":
          • one group per unique dataset_name, each group’s weights sum to 1.

    The rule is simple: if sample i has
        lengths[i] = dataset.samples[i]["num_samples"], 
        value[i] = dataset.samples[i][filter_on],

    then, within each group G = {i | value[i] = v}, we set
        w[i] = lengths[i] / sum_{j ∈ G} lengths[j].

    If for some group G the total sum is zero (i.e. all num_samples==0),
    we simply assign equal nonzero weights (1 / |G|) for that group.

    Parameters
    ----------
    dataset : FMRI_Dataset
        Any instance whose `dataset.samples` is a list of dicts, each having
        at least the key `"num_samples"` and the key `filter_on`.

    filter_on : str
        The sample‐dict key to group by. Common choices:
          • "is_movie"
          • "dataset_name"
          • (or any other key present in sample_dict)

    Returns
    -------
    weights : torch.FloatTensor, shape (N,)
        A vector of per‐sample weights. For each distinct value v of `filter_on`,
        the subarray `weights[G]` (where G = { i | sample[i][filter_on] == v }) 
        will sum to 1.

    Example
    -------
    # Per‐domain (movie vs friends)
    w_movie = make_group_weights(train_ds, filter_on="is_movie")

    # Per‐stimulus weights (one group per dataset_name)
    w_stim = make_group_weights(train_ds, filter_on="dataset_name")
    """
    # 1) Extract lengths (number of TRs) and the grouping‐key values
    lengths = torch.tensor(
        [sample["num_samples"] for sample in dataset.samples],
        dtype=torch.float32,
    )

    # Ensure filter_on exists in each sample:
    try:
        values = [sample[filter_on] for sample in dataset.samples]
    except KeyError:
        raise KeyError(f"Key '{filter_on}' not found in sample dict keys.")

    # 2) Build a mapping from each distinct key‐value → list of indices
    value_to_indices = {}
    for idx, v in enumerate(values):
        value_to_indices.setdefault(v, []).append(idx)

    # 3) Allocate output weight tensor
    N = len(dataset.samples)
    weights = torch.zeros(N, dtype=torch.float32)

    # 4) For each group, normalize within that group
    for v, idx_list in value_to_indices.items():
        group_lengths = lengths[idx_list]
        subtotal = float(group_lengths.sum().item())

        if subtotal == 0.0:
            # Avoid division by zero: assign uniform weights
            uniform_w = 1.0 / len(idx_list)
            for i in idx_list:
                weights[i] = uniform_w
        else:
            # w[i] = lengths[i] / total_length_of_this_group
            weights[idx_list] = group_lengths / subtotal

    return weights




def compute_statistics(feature_paths, cov_univariate=True,include_ood = False):

    normalization_stats = {}

    for feature, path in tqdm(feature_paths.items()):
        #path = os.path.join(config.features_dir,path)
        #print(path)
        M = 0
        mean = None
        M2 = None  # For Welford's algorithm to compute variance/covariance

        for root, _, files in os.walk(path):
            for file in tqdm(files):
                full_path = os.path.join(root, file)
                if include_ood ==False:
                    if 'ood' in full_path:
                        continue

                if full_path.endswith('.npy'):
                    data = torch.from_numpy(np.load(full_path)).float()
                elif full_path.endswith('.pt'):
                    data = torch.load(full_path, map_location='cpu').float()
                else:
                    continue # Skip non-data files

                N = data.shape[-2]

                if M == 0:
                    mean = torch.mean(data, dim=-2)
                    if cov_univariate:
                        M2 = torch.sum((data - mean.unsqueeze(-2))**2, dim=-2)
                    else:
                        centered_data = data - mean.unsqueeze(-2)
                        M2 = torch.matmul(centered_data.transpose(-1, -2), centered_data)
                else:
                    old_mean = mean.clone()
                    mean, M2_new_contribution = increment_mean_and_M2(data, mean, M, cov_univariate)
                    M2 += M2_new_contribution

                M += N

        if M > 0:
            if cov_univariate:
                std = torch.sqrt(M2 / (M-1))
            else:
                # For covariance, M2 is already the sum of outer products of centered data
                # We need to divide by M-1 for sample covariance, or M for population covariance.
                # Assuming sample covariance for 'state-of-the-art' for statistical purposes.
                # If M=1, covariance is undefined, so handle that case.
                if M > 1:
                    std = M2 / (M - 1)
                else:
                    std = torch.zeros_like(M2) # Or raise an error, depending on desired behavior
        
            normalization_stats[feature] = {"mean": mean, "std": std}

    return normalization_stats

def increment_mean_and_M2(data, mean, M, cov_univariate):
    N = data.shape[-2]
    
    # Current mean of the new batch
    batch_mean = torch.mean(data, dim=-2)

    # Calculate new combined mean
    new_mean = (M * mean + N * batch_mean) / (M + N)

    if cov_univariate:
        # Update M2 (sum of squared differences) using a stable formula
        delta = batch_mean - mean
        M2_new_contribution = torch.sum((data - batch_mean.unsqueeze(-2))**2, dim=-2)
        M2_new_contribution += M * N * delta**2 / (M + N)
    else:
        # Update M2 (sum of outer products of centered data) for covariance
        # M2_B (sum of outer products for the new batch relative to its own mean)
        centered_data_batch = data - batch_mean.unsqueeze(-2)
        M2_new_batch = torch.matmul(centered_data_batch.transpose(-1, -2), centered_data_batch)

        # Correction term for combining M2 values
        # delta between old mean and new batch mean
        delta_mean = batch_mean - mean
        
        M2_new_contribution = M2_new_batch + (M * N / (M + N)) * torch.outer(delta_mean, delta_mean)

    return new_mean, M2_new_contribution
