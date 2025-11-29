# server/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def horizontal_flip_keypoints(seq):
    # seq: (T, D) where D = (33+21+21)*3
    # For skeleton symmetry flipping we must swap left<->right indices.
    # This function does a simple x = 1-x flip; for correct results you should swap left/right landmark indices.
    seq_flipped = seq.copy()
    # flip x coordinate assuming normalized x in [0,1]
    seq_flipped[:, 0::3] = 1.0 - seq_flipped[:, 0::3]
    return seq_flipped

class KeypointDataset(Dataset):
    """
    Expects manifest CSV with columns: file,label
    Each file is an .npz saved with keypoints: array shape (T,input_dim) and optional 'label'
    During training sequences are resampled/padded to target_seq_len.
    """
    def __init__(self, manifest_csv, label_encoder, seq_len=64, augment=False):
        import pandas as pd
        self.df = pd.read_csv(manifest_csv)
        self.seq_len = seq_len
        self.augment = augment
        self.label_encoder = label_encoder  # sklearn LabelEncoder or dict mapping label->idx

    def __len__(self):
        return len(self.df)

    def _resample(self, seq):
        # simple linear resample by nearest indices
        L = seq.shape[0]
        if L == self.seq_len:
            return seq
        if L < 2:
            # pad with repeated frames
            pad = np.repeat(seq, self.seq_len, axis=0)[:self.seq_len]
            return pad
        idx = np.linspace(0, L - 1, self.seq_len).astype(np.int32)
        return seq[idx]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(row['file'])['keypoints'].astype(np.float32)  # (T, D)
        arr = self._resample(arr)
        if self.augment:
            # random horizontal flip
            if random.random() < 0.5:
                arr = horizontal_flip_keypoints(arr)
            # small noise
            arr += np.random.normal(0, 1e-3, arr.shape).astype(np.float32)
        label = row['label']
        # label encoder might be dict or sklearn LabelEncoder-like object
        if isinstance(self.label_encoder, dict):
            lbl = int(self.label_encoder[label])
        else:
            lbl = int(self.label_encoder.transform([label])[0])
        return torch.from_numpy(arr), torch.tensor(lbl, dtype=torch.long)
