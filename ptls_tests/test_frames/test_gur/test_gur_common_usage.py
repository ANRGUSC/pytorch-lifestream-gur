"""Integration test: full GUR train + inference pipeline.

Mirrors test_common_usage.py::test_datamodule_way but for GUR.
"""

from functools import partial

import pytest
import pytorch_lightning as pl
import torch

from ptls.data_load.datasets.dataloaders import inference_data_loader
from ptls.frames import PtlsDataModule
from ptls.frames.gur import GURModule, GURDataset
from ptls.frames.supervised import SequenceToTarget
from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder


def get_dataset(n=500):
    return [{
        'mcc_code': torch.randint(1, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
        'event_time': torch.arange(seq_len, dtype=torch.float),
    } for seq_len in torch.randint(20, 100, (n,))]


@pytest.fixture(scope='module')
def seq_encoder():
    return AutoEncoderSeqEncoder(
        trx_encoder=TrxEncoder(
            embeddings={'mcc_code': {'in': 10, 'out': 4}},
            numeric_values={'amount': 'identity'},
        ),
        latent_dim=16,
        encoder_hidden_dims=[32],
        time_windows=[7, 30, 180],
        time_unit=1.0,
        dropout=0.0,
    )


@pytest.fixture(scope='module')
def data():
    dataset = get_dataset(500)
    split = int(len(dataset) * 0.9)
    return dataset[:split], dataset[split:], dataset


def test_gur_train_and_inference(data, seq_encoder):
    """Full pipeline: train GUR, extract embeddings, verify shapes."""
    train_data, valid_data, full_dataset = data

    # Train
    train_ds = GURDataset(data=train_data)
    valid_ds = GURDataset(data=valid_data)

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=64,
        train_num_workers=0,
        valid_data=valid_ds,
        valid_batch_size=64,
        valid_num_workers=0,
    )

    module = GURModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=0.001),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
    )

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='cpu',
        logger=None,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=dm)

    # Inference — use SequenceToTarget (same pattern as CoLES)
    inference_dl = inference_data_loader(full_dataset, num_workers=0, batch_size=256)
    inference_model = SequenceToTarget(seq_encoder)
    inference_trainer = pl.Trainer(accelerator='cpu', logger=None)
    embeddings = torch.vstack(inference_trainer.predict(inference_model, inference_dl))

    assert embeddings.shape == (500, 16), f"Expected (500, 16), got {embeddings.shape}"
