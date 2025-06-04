import os
import re
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


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
    ):
        super().__init__()
        self.root_folder = root_folder_fmri

        self.feature_paths = feature_paths  # Dict of {modality: path}
        self.input_dims = input_dims
        self.modalities = modalities

        self.noise_std = noise_std
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
                glob.glob(os.path.join(root_folder_fmri, "sub-0?", "func", "*.h5"))            )
            self.samples = []
            for fmri_file in self.fmri_files:
                subject_id = os.path.basename(
                    os.path.dirname(os.path.dirname(fmri_file))
                )
                with h5py.File(fmri_file, "r") as h5file:
                    for dataset_name in h5file.keys():
                        num_samples = h5file[dataset_name].shape[0]
                        name_re = re.compile(r"^(?:ses-\d+_task-)(?P<file_name>(?P<name>s\d+|[a-zA-Z]+)[a-zA-Z0-9]+)(?:_run-\d*$)?")
                        match = name_re.match(dataset_name)                            
                        file_name = match.group("file_name")
                        name = match.group("name")
                        sample = {
                            "subject_id": subject_id,
                            "fmri_file": fmri_file,
                            "dataset_name": dataset_name,
                            "num_samples": num_samples,
                            "is_movie": "movie" in fmri_file.lower(),
                            "file_name": file_name,
                            "name": name,
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

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
            arr = np.load(path, mmap_mode='r')
            return torch.tensor(arr, dtype=torch.float32).squeeze()
        elif path.endswith(".pt"):
            return torch.load(path, map_location="cpu").squeeze().float()
        else:
            raise ValueError(f"Unknown feature file extension: {path}")

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject_id = self.subject_name_id_dict[sample_info["subject_id"]]
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

            if (
                self.normalization_stats
                and modality + "_mean" in self.normalization_stats
            ):
                data = self.normalize(
                    data,
                    self.normalization_stats[f"{modality}_mean"],
                    self.normalization_stats[f"{modality}_std"],
                )

            if self.noise_std > 0:
                data += torch.randn_like(data) * self.noise_std

            min_samples = min(min_samples, data.shape[0])
            features[modality] = data

        # Truncate all to same length
        for key in features:
            features[key] = features[key][:min_samples]
        fmri_response_tensor = fmri_response_tensor[:min_samples]

        return subject_id, features, fmri_response_tensor


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


def split_dataset_by_name(dataset, val_name="06", train_noise_std=0.00,
                          normalize_validation_bold=False):
    train_samples, val_samples = [], []

    for sample in dataset.samples:
        if val_name.lower() in sample["name"].lower():
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
        
    )

    val_ds = FMRI_Dataset(
        root_folder_fmri=dataset.root_folder,
        feature_paths=dataset.feature_paths,
        input_dims=dataset.input_dims,
        modalities=dataset.modalities,
        noise_std=0.0,
        samples=val_samples,
        normalize_bold=normalize_validation_bold,
    )

    return train_ds, val_ds


def collate_fn(batch):
    subject_ids, features_list, fmri_responses = zip(*batch)

    all_modalities = features_list[0].keys()
    padded_features = {
        modality: pad_sequence(
            [f[modality] for f in features_list], batch_first=True, padding_value=0
        )
        for modality in all_modalities
    }

    fmri_padded = pad_sequence(fmri_responses, batch_first=True, padding_value=0)

    # Example attention mask (you can adapt this per modality if needed)
    seq_lengths = [next(iter(f.values())).shape[0] for f in features_list]
    max_len = max(seq_lengths)
    idx = torch.arange(max_len)
    attention_masks = (idx.unsqueeze(0) < torch.tensor(seq_lengths).unsqueeze(1)).bool()

    return {
        "subject_ids": subject_ids,
        **padded_features,
        "fmri": fmri_padded,
        "attention_masks": attention_masks,
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