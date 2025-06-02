import os
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
        noise_std=0.0,
        normalization_stats=None,
        oversample_factor=1,
        samples=None,
        normalize_bold=True,
    ):
        super().__init__()
        self.root_folder = root_folder_fmri

        self.feature_paths = feature_paths  # Dict of {modality: path}

        self.noise_std = noise_std
        self.normalization_stats = normalization_stats
        self.oversample_factor = oversample_factor
        self.normalize_bold = normalize_bold  # enable/disable run‑wise z‑score

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
                        sample = {
                            "subject_id": subject_id,
                            "fmri_file": fmri_file,
                            "dataset_name": dataset_name,
                            "num_samples": num_samples,
                            "is_movie": "movie" in fmri_file.lower(),
                        }
                        self.samples.append(sample)

                        if is_movie_sample(fmri_file):
                            for _ in range(self.oversample_factor - 1):
                                self.samples.append(sample.copy())

    def __len__(self):
        return len(self.samples)

    def find_feature_file(self, feature_root, file_name):
        matches = glob.glob(
            os.path.join(feature_root, "**", f"*{file_name}.*"), recursive=True
        )
        if not matches:
            raise FileNotFoundError(
                f"Feature file '{file_name}' not found in '{feature_root}'"
            )
        return matches[0]

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject_id = self.subject_name_id_dict[sample_info["subject_id"]]
        fmri_file = sample_info["fmri_file"]
        dataset_name = sample_info["dataset_name"]

        file_name_features = (
            f"{dataset_name.split('-')[-1]}"
        )

        with h5py.File(fmri_file, "r") as h5file:
            fmri_response = h5file[dataset_name][:]

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
            path = self.find_feature_file(root_path, file_name_features)
            if path.endswith(".npy"):
                data = torch.tensor(np.load(path), dtype=torch.float32).squeeze()
            elif path.endswith(".pt"):
                data = torch.load(path, map_location="cpu").squeeze().float()
            else:
                raise ValueError(
                    f"Unknown feature file extension: {path}"
                )
            
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


def split_dataset_by_name(dataset, val_name="friends_06", train_noise_std=0.00,
                          normalize_validation_bold=False):
    train_samples, val_samples = [], []

    for sample in dataset.samples:
        if val_name.lower() in sample["dataset_name"].lower():
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_ds = FMRI_Dataset(
        dataset.root_folder,
        dataset.feature_paths,
        noise_std=train_noise_std,
        samples=train_samples,
        normalize_bold=dataset.normalize_bold,
        
    )

    val_ds = FMRI_Dataset(
        dataset.root_folder,
        dataset.feature_paths,
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
    attention_masks = torch.zeros(
        (len(seq_lengths), max(seq_lengths)), dtype=torch.bool
    )
    for i, length in enumerate(seq_lengths):
        attention_masks[i, :length] = 1

    return {
        "subject_ids": subject_ids,
        **padded_features,
        "fmri": fmri_padded,
        "attention_masks": attention_masks,
    }


def is_movie_sample(dataset_name):
    # Customize this check based on how movie files are named
    return "movie" in dataset_name.lower()


def make_balanced_weights(dataset):
    """
    Return a weight tensor wᵢ that equalises the *expected number of TRs*
    drawn from each domain per epoch.
    """

    lengths = torch.tensor(
        [sample["num_samples"] for sample in dataset.samples], dtype=torch.float32
    )

    is_movie = torch.tensor(
        [sample["is_movie"] for sample in dataset.samples], dtype=torch.bool
    )

    L_movies = lengths[is_movie].sum()
    L_friends = lengths[~is_movie].sum()

    weights = torch.where(
        is_movie,
        lengths / L_movies,
        lengths / L_friends
    )
    return weights