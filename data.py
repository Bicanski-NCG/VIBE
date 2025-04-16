import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class FMRI_Dataset(Dataset):
    def __init__(self, root_folder_fmri, audio_feature_path, video_feature_path, text_feature_path, noise_std=0.0):
        super().__init__()
        self.root_folder = root_folder_fmri
        self.fmri_files = sorted(glob.glob(os.path.join(root_folder_fmri, "sub-0?", "func", "*.h5")))

        self.audio_feature_path = audio_feature_path
        self.video_feature_path = video_feature_path
        self.text_feature_path = text_feature_path

        self.noise_std = noise_std  # <-- Add this

        self.subject_name_id_dict = {
            "sub-01": 0,
            "sub-02": 1,
            "sub-03": 2,
            "sub-05": 3
        }

        self.samples = []
        for fmri_file in self.fmri_files:
            subject_id = os.path.basename(os.path.dirname(os.path.dirname(fmri_file)))
            with h5py.File(fmri_file, 'r') as h5file:
                for dataset_name in h5file.keys():
                    data_shape = h5file[dataset_name].shape
                    num_samples = data_shape[0]
                    self.samples.append({
                        'subject_id': subject_id,
                        'fmri_file': fmri_file,
                        'dataset_name': dataset_name,
                        'num_samples': num_samples
                    })

    def __len__(self):
        return len(self.samples)

    def find_feature_file(self, feature_root, file_name):
        matches = glob.glob(os.path.join(feature_root, '**', f"*{file_name}"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Feature file '{file_name}' not found in '{feature_root}'")
        return matches[0]

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject_id = self.subject_name_id_dict[sample_info['subject_id']]
        fmri_file = sample_info['fmri_file']
        dataset_name = sample_info['dataset_name']

        file_name_features = f"{dataset_name.split('-')[-1]}.npy"

        with h5py.File(fmri_file, 'r') as h5file:
            fmri_response = h5file[dataset_name][:]

        fmri_response_tensor = torch.tensor(fmri_response, dtype=torch.float32)

        audio_path = self.find_feature_file(self.audio_feature_path, file_name_features)
        video_path = self.find_feature_file(self.video_feature_path, file_name_features)
        text_path = self.find_feature_file(self.text_feature_path, file_name_features)

        audio_features = torch.tensor(np.load(audio_path), dtype=torch.float32)
        video_features = torch.tensor(np.load(video_path), dtype=torch.float32)
        text_features = torch.tensor(np.load(text_path), dtype=torch.float32)

        min_samples = min(fmri_response_tensor.shape[0], audio_features.shape[0], video_features.shape[0], text_features.shape[0])

        # Add Gaussian noise if requested
        if self.noise_std > 0:
            audio_features = audio_features[:min_samples] + torch.randn_like(audio_features[:min_samples]) * self.noise_std
            video_features = video_features[:min_samples] + torch.randn_like(video_features[:min_samples]) * self.noise_std
            text_features = text_features[:min_samples] + torch.randn_like(text_features[:min_samples]) * self.noise_std
        else:
            audio_features = audio_features[:min_samples]
            video_features = video_features[:min_samples]
            text_features = text_features[:min_samples]

        return subject_id, audio_features, video_features.squeeze(), text_features, fmri_response_tensor[:min_samples]


def split_dataset_by_season(dataset, val_season="6", train_noise_std=0.01):
    train_samples = []
    val_samples = []

    for sample in dataset.samples:
        dataset_name = sample['dataset_name'].lower()
        if f"s0{val_season}" in dataset_name.split("-")[-1]:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_dataset = FMRI_Dataset(
        dataset.root_folder,
        dataset.audio_feature_path,
        dataset.video_feature_path,
        dataset.text_feature_path,
        noise_std=train_noise_std  # <-- Noise only for train
    )
    train_dataset.samples = train_samples

    val_dataset = FMRI_Dataset(
        dataset.root_folder,
        dataset.audio_feature_path,
        dataset.video_feature_path,
        dataset.text_feature_path,
        noise_std=0.0  # <-- No noise for val
    )
    val_dataset.samples = val_samples

    return train_dataset, val_dataset
