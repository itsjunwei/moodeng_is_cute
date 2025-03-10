import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
from torch.hub import download_url_to_file
import numpy as np
import torch.nn.functional as F

dataset_dir = r"./data/TAU-urban-acoustic-scenes-2022-mobile-development"
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile dataset' location in variable " \
                                "'dataset_dir'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/record/6337421"

dataset_config = {
    "dataset_name": "tau24",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": os.path.join(dataset_dir, "TAU-urban-acoustic-scenes-2024-mobile-evaluation"),
    "eval_fold_csv": os.path.join(dataset_dir, "TAU-urban-acoustic-scenes-2024-mobile-evaluation",
                                  "evaluation_setup", "fold1_test.csv")
}


class BasicDCASE24Dataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads data from files
    """

    def __init__(self, meta_csv):
        """
        @param meta_csv: meta csv file for the dataset
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(dataset_dir, self.files[index]))
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)


class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for different splits
        return: waveform, file, label, device, city
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.available_indices)


class RollDataset(TorchDataset):
    """A dataset implementing time rolling of waveforms.
    """

    def __init__(self, dataset: TorchDataset, shift_range: int, axis=1):
        """
        @param dataset: dataset to load data from
        @param shift_range: maximum shift range
        return: waveform, file, label, device, city
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), file, label, device, city

    def __len__(self):
        return len(self.dataset)


class AugmentDataset(TorchDataset):
    """
    A dataset that randomly applies one of the following augmentations to a waveform:
      1. Roll the waveform (time shift)
      2. Add white Gaussian noise
    Default parameters are chosen for acoustic scene classification:
      - shift_range: 4410 samples (~0.1 sec at 44.1 kHz)
      - noise_std: 0.005 (small perturbation)
    """
    def __init__(self, dataset: TorchDataset, 
                 shift_range: int = 4410, 
                 noise_std: float = 0.005,
                 sample_rate: int = 44100, 
                 axis: int = 1,
                 speed_low: float = 0.9,
                 speed_high: float = 1.1):
        """
        @param dataset: The original dataset.
        @param shift_range: Maximum absolute shift (in samples) for rolling the waveform.
        @param noise_std: Standard deviation for white Gaussian noise.
        @param sample_rate: The sample rate of the audio.
        @param axis: The axis along which to roll the waveform (typically 1 for (channels, samples)).
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.noise_std = noise_std
        self.sample_rate = sample_rate
        self.axis = axis
        self.speed_low = speed_low
        self.speed_high = speed_high

    def speed_perturb(self, x: torch.Tensor, speed_factor: float, final_length: int) -> torch.Tensor:
        """
        Applies speed perturbation by resampling the waveform.
        After resampling, the waveform is cropped or padded to final_length.
        
        @param x: Input waveform tensor of shape (channels, samples)
        @param speed_factor: The speed factor to apply (e.g. 0.9 for slowing down, 1.1 for speeding up)
        @param final_length: Desired output length (number of samples)
        @return: Speed-perturbed waveform of shape (channels, final_length)
        """
        channels, orig_length = x.shape
        # Calculate the new length after applying the speed factor.
        new_length = int(orig_length / speed_factor)
        
        # Resample the waveform using linear interpolation.
        # Input needs to be 3D: (batch=1, channels, time)
        x_unsq = x.unsqueeze(0)  # shape: (1, channels, orig_length)
        x_resampled = F.interpolate(x_unsq, size=new_length, mode='linear', align_corners=False)
        x_resampled = x_resampled.squeeze(0)  # shape: (channels, new_length)
        
        # If the resampled signal is shorter than final_length, pad at the end.
        if new_length < final_length:
            pad_amount = final_length - new_length
            x_out = F.pad(x_resampled, (0, pad_amount))
        else:
            # If longer, crop to final_length.
            x_out = x_resampled[:, :final_length]
            
        return x_out

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        # Randomly choose one augmentation
        aug_choice = np.random.choice(['roll', 'noise', 'speed'])
        
        if aug_choice == 'roll':
            sf = int(np.random.randint(-self.shift_range, self.shift_range + 1))
            x_aug = x.roll(sf, dims=self.axis)
        elif aug_choice == 'noise':
            noise = torch.randn_like(x) * self.noise_std
            x_aug = x + noise
        elif aug_choice == 'speed':
            # Choose a random speed factor within the specified range.
            speed_factor = np.random.uniform(self.speed_low, self.speed_high)
            x_aug = self.speed_perturb(x, speed_factor, final_length=self.sample_rate)
        else:
            # If none chosen (should not happen), return original.
            x_aug = x
        return x_aug, file, label, device, city


    def __len__(self):
        return len(self.dataset)


def get_training_set(split=100, roll=False):
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = get_base_training_set(dataset_config['meta_csv'], subset_split_file)
    if roll:
        # ds = RollDataset(ds, shift_range=roll)
        ds = AugmentDataset(ds)
    return ds


def get_base_training_set(meta_csv, train_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    train_subset_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv),
                                train_subset_indices)
    return ds


def get_test_set():
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = get_base_test_set(dataset_config['meta_csv'], test_split_csv)
    return ds


def get_base_test_set(meta_csv, test_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv), test_indices)
    return ds


class BasicDCASE24EvalDataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, eval_dir):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.eval_dir = eval_dir

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(self.eval_dir, self.files[index]))
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)


def get_eval_set():
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = get_base_eval_set(dataset_config['eval_fold_csv'], dataset_config['eval_dir'])
    return ds


def get_base_eval_set(meta_csv, eval_dir):
    ds = BasicDCASE24EvalDataset(meta_csv, eval_dir)
    return ds
