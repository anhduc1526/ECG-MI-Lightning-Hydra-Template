"""ECGDataModule — exact copy of data loading logic from ecg/ECGDataLoader.py.

Includes all transform classes (BaselineWander, GaussianNoise, PowerlineNoise,
ChannelResize, BaselineShift, ComposeTransforms) and the LightningDataModule.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split

from src.data.components.ecg_dataset import ECGDataset


# ---------------------------------------------------------------------------
#  Transform / noise classes — verbatim from ecg/ECGDataLoader.py
# ---------------------------------------------------------------------------


def Tnoise_powerline(fs=100, N=1000, C=1, fn=50., K=3, channels=1):
    """powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    """
    t = torch.arange(0, N / fs, 1. / fs)

    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2 * math.pi)
    for k in range(1, K + 1):
        ak = random.uniform(0, 1)
        signal += C * ak * torch.cos(2 * math.pi * k * fn * t + phi1)
    signal = C * signal[:, None]
    if channels > 1:
        channel_gains = torch.empty(channels).uniform_(-1, 1)
        signal = signal * channel_gains[None]
    return signal


def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01, channels=1,
                           independent_channels=False):
    """baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/"""
    if fdelta is None:
        fdelta = fs / N

    K = int((fc / fdelta) + 0.5)
    t = torch.arange(0, N / fs, 1. / fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2 * math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2 * math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C * res


class ChannelResize:
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range))
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T
        wave = resize_factors * wave
        return wave


class GaussianNoise:
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave


class BaselineShift:
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave


class BaselineWander:
    def __init__(self, prob=1.0, freq=1000, C=1.0):
        self.freq = freq
        self.prob = prob
        self.C = C

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave, C=self.C)
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave


class PowerlineNoise:
    def __init__(self, prob=1.0, freq=1000, C=1.0):
        self.freq = freq
        self.prob = prob
        self.C = C

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(fs=self.freq, N=len_wave, C=self.C, channels=channels).T
            wave = wave + noise
        return wave


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave):
        for t in self.transforms:
            wave = t(wave)
        return wave


# ---------------------------------------------------------------------------
#  LightningDataModule — mirrors ecg/ECGDataLoader.py exactly
# ---------------------------------------------------------------------------


class ECGDataModule(LightningDataModule):
    """DataModule for ECG heartbeat classification.

    Exactly mirrors ecg/ECGDataLoader.py, including:
    - 5-fold cross-validation support via fold_train / fold_test
    - Heartbeat-level random split (split_with_patient=False, matching original)
    - Same transforms: BaselineWander, GaussianNoise, PowerlineNoise, ChannelResize, BaselineShift
    """

    def __init__(
        self,
        csv_file: str = "ptb_fold.csv",
        data_dir: str = "/kaggle/input/datasets/tphdng/ecg-dataset",
        fold_train: List[int] = [1, 2, 3, 4],
        fold_test: List[int] = [0],
        batch_size: int = 256,
        num_workers: int = 2,
        split_ratio: float = 0.8,
        sample_before: int = 198,
        sample_after: int = 400,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Any] = None
        self.data_val: Optional[Any] = None
        self.data_test: Optional[Any] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Mirrors ecg/ECGDataLoader.py setup() exactly.
        """
        # Only setup once
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        dataset = ECGDataset(
            csv_file=self.hparams.csv_file,
            data_dir=self.hparams.data_dir,
            fold_list=self.hparams.fold_train,
            sample_before=self.hparams.sample_before,
            sample_after=self.hparams.sample_after,
            transform=ComposeTransforms([
                BaselineWander(prob=0.5, C=0.0001),
                GaussianNoise(prob=0.5, scale=0.0001),
                PowerlineNoise(prob=0.5, C=0.0001),
                ChannelResize(magnitude_range=(0.5, 2.0)),
                BaselineShift(prob=0.5, scale=0.01),
            ])
        )

        # Chọn cách chia theo bệnh nhân nếu không thì sẽ theo nhịp tim
        split_with_patient = False

        if split_with_patient:
            # Chia theo bệnh nhân (patient-wise split)
            dataset_patient_id_list = list(dataset.info["patient_number"].unique())
            random.shuffle(dataset_patient_id_list)

            split_point = int(len(dataset_patient_id_list) * self.hparams.split_ratio)
            train_patient_id_list = dataset_patient_id_list[:split_point]

            train_indices = dataset.info[dataset.info["patient_number"].isin(train_patient_id_list)].index.tolist()
            val_indices = dataset.info[~dataset.info["patient_number"].isin(train_patient_id_list)].index.tolist()

            self.data_train = Subset(dataset, train_indices)
            self.data_val = Subset(dataset, val_indices)
        else:
            # Chia theo nhịp tim (heartbeat-level split)
            total_size = len(dataset)
            train_size = int(total_size * self.hparams.split_ratio)
            val_size = total_size - train_size
            self.data_train, self.data_val = random_split(
                dataset, [train_size, val_size]
            )

        self.data_test = ECGDataset(
            csv_file=self.hparams.csv_file,
            data_dir=self.hparams.data_dir,
            fold_list=self.hparams.fold_test,
            sample_before=self.hparams.sample_before,
            sample_after=self.hparams.sample_after
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = ECGDataModule()
