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
    ):
        super().__init__()
        self.root_folder = root_folder_fmri
        self.fmri_files = sorted(
            glob.glob(os.path.join(root_folder_fmri, "sub-0?", "func", "*.h5"))
        )

        self.feature_paths = feature_paths  # Dict of {modality: path}

        self.noise_std = noise_std
        self.normalization_stats = normalization_stats
        self.oversample_factor = oversample_factor

        self.subject_name_id_dict = {"sub-01": 0, "sub-02": 1, "sub-03": 2, "sub-05": 3}

        self.samples = []
        for fmri_file in self.fmri_files:
            subject_id = os.path.basename(os.path.dirname(os.path.dirname(fmri_file)))
            with h5py.File(fmri_file, "r") as h5file:
                for dataset_name in h5file.keys():
                    num_samples = h5file[dataset_name].shape[0]
                    sample = {
                        "subject_id": subject_id,
                        "fmri_file": fmri_file,
                        "dataset_name": dataset_name,
                        "num_samples": num_samples,
                    }
                    self.samples.append(sample)

                    if is_movie_sample(fmri_file):
                        for _ in range(self.oversample_factor - 1):  # 1 already added
                            self.samples.append(sample.copy())

    def __len__(self):
        return len(self.samples)

    def find_feature_file(self, feature_root, file_name):
        matches = glob.glob(
            os.path.join(feature_root, "**", f"*{file_name}"), recursive=True
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

        file_name_features = f"{dataset_name.split('-')[-1]}.npy"

        with h5py.File(fmri_file, "r") as h5file:
            fmri_response = h5file[dataset_name][:]

        fmri_response_tensor = torch.tensor(fmri_response, dtype=torch.float32)

        features = {}
        min_samples = fmri_response_tensor.shape[0]

        for modality, root_path in self.feature_paths.items():
            path = self.find_feature_file(root_path, file_name_features)
            data = torch.tensor(np.load(path), dtype=torch.float32).squeeze()

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


def split_dataset_by_season(dataset, val_season="6", train_noise_std=0.01):
    train_samples, val_samples = [], []

    for sample in dataset.samples:
        if f"s0{val_season}" in sample["dataset_name"].split("-")[-1].lower():
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_ds = FMRI_Dataset(
        dataset.root_folder,
        dataset.feature_paths,
        noise_std=train_noise_std,
    )
    train_ds.samples = train_samples

    val_ds = FMRI_Dataset(
        dataset.root_folder,
        dataset.feature_paths,
        noise_std=0.0,
    )
    val_ds.samples = val_samples

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
