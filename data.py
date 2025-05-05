import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class FMRI_Dataset(Dataset):
    def __init__(self, root_folder_fmri, audio_feature_path, video_feature_path, text_feature_path,
                 noise_std=0.0, normalization_stats=None, oversample_factor=1):
        super().__init__()
        self.root_folder = root_folder_fmri
        self.fmri_files = sorted(glob.glob(os.path.join(root_folder_fmri, "sub-0?", "func", "*.h5")))

        self.audio_feature_path = audio_feature_path
        self.video_feature_path = video_feature_path
        self.text_feature_path = text_feature_path

        self.noise_std = noise_std
        self.normalization_stats = normalization_stats
        self.oversample_factor = oversample_factor

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
                    num_samples = h5file[dataset_name].shape[0]
                    sample = {
                        'subject_id': subject_id,
                        'fmri_file': fmri_file,
                        'dataset_name': dataset_name,
                        'num_samples': num_samples
                    }
                    self.samples.append(sample)

                    if is_movie_sample(fmri_file):
                        for _ in range(self.oversample_factor - 1):  # 1 already added
                            self.samples.append(sample.copy())

    def __len__(self):
        return len(self.samples)

    def find_feature_file(self, feature_root, file_name):
        matches = glob.glob(os.path.join(feature_root, '**', f"*{file_name}"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Feature file '{file_name}' not found in '{feature_root}'")
        return matches[0]

    def normalize(self, data, mean, std):
        return (data - mean) / std

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

        min_samples = min(fmri_response_tensor.shape[0], audio_features.shape[0],
                          video_features.shape[0], text_features.shape[0])

        audio_features = audio_features[:min_samples]
        video_features = video_features[:min_samples]
        text_features = text_features[:min_samples]

        if self.normalization_stats is not None:
            audio_features = self.normalize(audio_features, self.normalization_stats['audio_mean'], self.normalization_stats['audio_std'])
            video_features = self.normalize(video_features, self.normalization_stats['video_mean'], self.normalization_stats['video_std'])
            text_features = self.normalize(text_features, self.normalization_stats['text_mean'], self.normalization_stats['text_std'])

        if self.noise_std > 0:
            audio_features += torch.randn_like(audio_features) * self.noise_std
            video_features += torch.randn_like(video_features) * self.noise_std
            text_features += torch.randn_like(text_features) * self.noise_std

        return subject_id, audio_features, video_features.squeeze(), text_features, fmri_response_tensor[:min_samples]


def compute_mean_std(dataset):
    audio_all, video_all, text_all = [], [], []

    for _, audio, video, text, _ in DataLoader(dataset, batch_size=1):
        audio_all.append(audio.squeeze(0))
        video_all.append(video.squeeze(0))
        text_all.append(text.squeeze(0))

    audio_all = torch.cat(audio_all, dim=0)
    video_all = torch.cat(video_all, dim=0)
    text_all = torch.cat(text_all, dim=0)

    return {
        'audio_mean': audio_all.mean(dim=0),
        'audio_std': audio_all.std(dim=0) + 1e-8,
        'video_mean': video_all.mean(dim=0),
        'video_std': video_all.std(dim=0) + 1e-8,
        'text_mean': text_all.mean(dim=0),
        'text_std': text_all.std(dim=0) + 1e-8,
    }


def split_dataset_by_season(dataset, val_season="6", train_noise_std=0.01):
    train_samples, val_samples = [], []

    for sample in dataset.samples:
        if f"s0{val_season}" in sample['dataset_name'].split("-")[-1].lower():
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    train_ds = FMRI_Dataset(dataset.root_folder, dataset.audio_feature_path, dataset.video_feature_path,
                            dataset.text_feature_path, noise_std=train_noise_std)
    train_ds.samples = train_samples

    val_ds = FMRI_Dataset(dataset.root_folder, dataset.audio_feature_path, dataset.video_feature_path,
                          dataset.text_feature_path, noise_std=0.0)
    val_ds.samples = val_samples

    return train_ds, val_ds


def collate_fn(batch):
    subject_ids, audio_feats, video_feats, text_feats, fmri_responses = zip(*batch)

    audio_padded = pad_sequence(audio_feats, batch_first=True, padding_value=0)
    video_padded = pad_sequence(video_feats, batch_first=True, padding_value=0)
    text_padded = pad_sequence(text_feats, batch_first=True, padding_value=0)
    fmri_padded = pad_sequence(fmri_responses, batch_first=True, padding_value=0)

    attention_masks = torch.zeros(audio_padded.shape[:2], dtype=torch.bool)
    for i, length in enumerate([af.shape[0] for af in audio_feats]):
        attention_masks[i, :length] = 1

    return {
        'subject_ids': subject_ids,
        'audio': audio_padded,
        'video': video_padded,
        'text': text_padded,
        'fmri': fmri_padded,
        'attention_masks': attention_masks
    }

def is_movie_sample(dataset_name):
    # Customize this check based on how movie files are named
    return "movie" in dataset_name.lower() 