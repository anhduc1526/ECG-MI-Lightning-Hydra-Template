"""ECGDataset — exact copy from ecg/ECGDataset.py.

Returns (heartbeat, label, patient_number, rpeak_index, pt_path) — 5 items.
"""

import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

# NOTE: Do NOT set random.seed() here at module level!
# In the template, this module is imported AFTER L.seed_everything(30),
# so a module-level random.seed(42) would override the global seed.
# Seeding is handled centrally by L.seed_everything() in train.py.


class ECGDataset(Dataset):
    def __init__(
        self,
        csv_file,
        data_dir,
        fold_list,
        sample_before=0,
        sample_after=0,
        transform=None
    ):
        self.info = pd.read_csv(os.path.join(data_dir, csv_file))
        self.fold_list = fold_list
        self.sample_before = sample_before
        self.sample_after = sample_after
        self.data_dir = data_dir
        self.label_dict = {"MI": 0, "Healthy": 1}
        self.transform = transform
        self._signal_cache = {}

        if fold_list is not None:
            self.info = self.info[self.info["fold"].isin(fold_list)].reset_index(drop=True)
        self.info = self.info[self.info["label"].isin(["MI", "Healthy"])].reset_index(drop=True)

        valid_mask = (self.info["r_peak_index"] - self.sample_before > 0) & \
                     (self.info["length"] - self.info["r_peak_index"] + 1 > self.sample_after)
        self.info = self.info[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.info)

    def _load_signal(self, path):
        if path in self._signal_cache:
            return self._signal_cache[path]
        self._signal_cache[path] = torch.load(path, weights_only=True)

        return self._signal_cache[path]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rpeak_index = self.info.iloc[idx, 2]
        pt_path = self.info.iloc[idx, 1]
        signal = self._load_signal(os.path.join(self.data_dir, pt_path)).t()
        start = rpeak_index - self.sample_before
        end = rpeak_index + self.sample_after + 1
        heartbeat = signal[:, start:end]

        # Chuẩn hóa shape
        target_len = self.sample_before + self.sample_after + 1
        cur_len = heartbeat.shape[1]
        if cur_len < target_len:
            # Padding nếu thiếu
            pad_size = target_len - cur_len
            heartbeat = torch.nn.functional.pad(heartbeat, (0, pad_size))
        elif cur_len > target_len:
            # Cắt nếu thừa
            heartbeat = heartbeat[:, :target_len]

        label = self.label_dict[self.info.iloc[idx, 3]]
        patient_number = self.info.iloc[idx, 0]
        if self.transform:
            heartbeat = self.transform(heartbeat)

        return heartbeat, label, patient_number, rpeak_index, pt_path
