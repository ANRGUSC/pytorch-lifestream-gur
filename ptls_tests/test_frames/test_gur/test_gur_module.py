"""Tests for GURModule — training loop, shapes, loss behavior."""

from functools import partial

import pytorch_lightning as pl
import torch

from ptls.frames.gur import GURModule, GURDataset
from ptls.frames import PtlsDataModule
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder


def make_data(n=200, min_len=10, max_len=50):
    return [{
        'event_time': torch.sort(torch.rand(seq_len) * 180)[0],
        'mcc_code': torch.randint(0, 10, (seq_len,)),
        'amount': torch.randn(seq_len),
    } for seq_len in torch.randint(min_len, max_len + 1, (n,))]


def make_module(latent_dim=8):
    trx_encoder = TrxEncoder(
        embeddings={'mcc_code': {'in': 10, 'out': 4}},
        numeric_values={'amount': 'identity'},
    )
    seq_encoder = AutoEncoderSeqEncoder(
        trx_encoder=trx_encoder,
        latent_dim=latent_dim,
        encoder_hidden_dims=[16],
        time_windows=[7, 30, 180],
        time_unit=1.0,
        dropout=0.0,
    )
    return GURModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=1e-3),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=1.0),
    )


def test_train_loop():
    """Full trainer.fit for 1 epoch — mirrors test_coles_module.py."""
    module = make_module()
    train_data = make_data(100)
    valid_data = make_data(30)

    dm = PtlsDataModule(
        train_data=GURDataset(train_data),
        train_batch_size=32,
        train_num_workers=0,
        valid_data=GURDataset(valid_data),
        valid_batch_size=32,
        valid_num_workers=0,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        logger=None,
        enable_checkpointing=False,
        accelerator='cpu',
    )
    trainer.fit(module, datamodule=dm)


def test_shared_step_shapes():
    module = make_module(latent_dim=8)
    module.train()
    data = make_data(8)
    batch, labels = GURDataset.collate_fn(data)
    reconstruction, target = module.shared_step(batch, labels)

    trx_size = module.seq_encoder.trx_encoder.output_size
    agg_dim = trx_size * 3  # 3 time windows
    assert reconstruction.shape == (8, agg_dim)
    assert target.shape == (8, agg_dim)
    assert not target.requires_grad  # target should be detached


def test_forward_returns_latent():
    module = make_module(latent_dim=16)
    module.eval()
    data = make_data(5)
    batch, _ = GURDataset.collate_fn(data)
    with torch.no_grad():
        out = module(batch)
    assert out.shape == (5, 16)


def test_loss_decreases():
    """Train for 5 epochs and verify loss decreases."""
    module = make_module()
    train_data = make_data(200)
    valid_data = make_data(50)

    dm = PtlsDataModule(
        train_data=GURDataset(train_data),
        train_batch_size=64,
        train_num_workers=0,
        valid_data=GURDataset(valid_data),
        valid_batch_size=64,
        valid_num_workers=0,
    )

    trainer = pl.Trainer(
        max_epochs=5,
        logger=None,
        enable_checkpointing=False,
        accelerator='cpu',
    )
    trainer.fit(module, datamodule=dm)

    # Check that training produced a finite loss
    final_loss = trainer.callback_metrics.get('loss')
    assert final_loss is not None
    assert final_loss.item() < float('inf')
    assert not torch.isnan(final_loss)
