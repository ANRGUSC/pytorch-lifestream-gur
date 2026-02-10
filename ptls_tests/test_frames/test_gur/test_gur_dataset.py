"""Tests for GURDataset."""

import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.gur.gur_dataset import GURDataset


def make_sample_data(n=10):
    """Generate simple feature dicts."""
    return [{
        'event_time': torch.arange(seq_len, dtype=torch.float),
        'mcc_code': torch.randint(0, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
    } for seq_len in torch.randint(5, 50, (n,))]


def test_gur_dataset_len():
    data = make_sample_data(20)
    ds = GURDataset(data)
    assert len(ds) == 20


def test_gur_dataset_getitem():
    data = make_sample_data(5)
    ds = GURDataset(data)
    item = ds[0]
    assert isinstance(item, dict)
    assert 'event_time' in item
    assert 'mcc_code' in item


def test_gur_dataset_collate_fn():
    data = make_sample_data(8)
    ds = GURDataset(data)
    batch = [ds[i] for i in range(4)]
    padded, labels = GURDataset.collate_fn(batch)

    assert isinstance(padded, PaddedBatch)
    assert isinstance(padded.payload, dict)
    assert padded.seq_lens.shape == (4,)
    assert labels.shape == (4,)
    assert labels.dtype == torch.float


def test_gur_dataset_min_seq_len():
    data = [
        {'event_time': torch.tensor([1.0]), 'mcc_code': torch.tensor([0])},
        {'event_time': torch.tensor([1.0, 2.0, 3.0]), 'mcc_code': torch.tensor([0, 1, 2])},
        {'event_time': torch.arange(10, dtype=torch.float), 'mcc_code': torch.zeros(10, dtype=torch.long)},
    ]
    ds = GURDataset(data, min_seq_len=3)
    assert len(ds) == 2  # only sequences with len >= 3


def test_gur_dataset_empty_sequence_filtered():
    data = [
        {'event_time': torch.tensor([]), 'mcc_code': torch.tensor([], dtype=torch.long)},
        {'event_time': torch.tensor([1.0, 2.0]), 'mcc_code': torch.tensor([0, 1])},
    ]
    ds = GURDataset(data, min_seq_len=1)
    # Empty sequence (len 0) should be filtered even with min_seq_len=1
    assert len(ds) == 1


def test_gur_dataset_iteration():
    data = make_sample_data(5)
    ds = GURDataset(data)
    items = list(ds)
    assert len(items) == 5
