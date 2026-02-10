"""Dataset for GUR (Generalized User Representations) autoencoder training.

Unlike ColesDataset which splits sequences into contrastive pairs,
GURDataset provides raw sequences directly — the autoencoder is
self-supervised via reconstruction loss, no class labels needed.
"""

import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict


class GURDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.gur.GURModule.

    Each item is a raw feature dict (event sequence for one user).
    No splitting or augmentation — the autoencoder learns from full sequences.

    Args:
        data: List of feature dicts, each with keys like
            'event_time', 'mcc_code', 'amount', etc.
        min_seq_len: Minimum sequence length to include (default: 1).
    """

    def __init__(self, data, min_seq_len=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        if min_seq_len > 1:
            self.data = [d for d in self.data if self.get_seq_len(d) >= min_seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for feature_arrays in self.data:
            yield feature_arrays

    @staticmethod
    def collate_fn(batch):
        """Collate feature dicts into PaddedBatch + dummy labels.

        Returns (PaddedBatch, dummy_labels) to match ABSModule.training_step
        which expects batch = (x, y). The dummy labels are unused by GURModule.
        """
        padded = collate_feature_dict(batch)
        dummy_labels = torch.zeros(len(batch), dtype=torch.float)
        return padded, dummy_labels


class GURIterableDataset(GURDataset, torch.utils.data.IterableDataset):
    """Iterable variant of GURDataset for streaming data."""
    pass
