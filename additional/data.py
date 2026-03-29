"""Dataset and collate helpers for frozen sequence feature experiments."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapSequenceDataset(Dataset):
    def __init__(self, feats_mm, lens_mm, labels, seq_len: int):
        self.feats = feats_mm
        self.lens = lens_mm
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.feats[idx], dtype=np.float32))
        length = int(self.lens[idx])
        length = max(1, min(length, self.seq_len))
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, length, y


class MemmapSequenceTestDataset(Dataset):
    def __init__(self, feats_mm, lens_mm, seq_len: int):
        self.feats = feats_mm
        self.lens = lens_mm
        self.seq_len = seq_len

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.feats[idx], dtype=np.float32))
        length = int(self.lens[idx])
        length = max(1, min(length, self.seq_len))
        return x, length


def collate_train(batch):
    xs, lens, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    lengths = torch.tensor(lens, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, lengths, y


def collate_infer(batch):
    xs, lens = zip(*batch)
    x = torch.stack(xs, dim=0)
    lengths = torch.tensor(lens, dtype=torch.long)
    return x, lengths
