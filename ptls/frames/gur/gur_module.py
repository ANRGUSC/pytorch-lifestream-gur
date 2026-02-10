"""GUR training module — Generalized User Representations via autoencoder.

Implements the self-supervised training framework from the Spotify paper
(arXiv 2403.00584). The model learns compact user embeddings by training
an autoencoder on multi-timescale aggregated event features.

Interface is identical to CoLESModule so the two can be swapped seamlessly.
"""

import torch
import torch.nn as nn
import torchmetrics
from functools import partial

from ptls.frames.abs_module import ABSModule
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder


class ReconstructionMSE(torchmetrics.MeanMetric):
    """Validation metric: mean reconstruction MSE across batches, weighted by batch size."""

    def update(self, preds, target):
        mse = nn.functional.mse_loss(preds, target)
        super().update(mse, weight=preds.shape[0])


class GURModule(ABSModule):
    """Generalized User Representations training module.

    Trains an AutoEncoderSeqEncoder by minimizing reconstruction loss
    on multi-timescale aggregated event embeddings.

    Usage is identical to CoLESModule:
        >>> from ptls.frames.gur import GURModule
        >>> module = GURModule(seq_encoder=my_ae_encoder, ...)
        >>> trainer.fit(module, datamodule=dm)
        >>> embeddings = module(batch)  # (B, latent_dim)

    Args:
        seq_encoder: AutoEncoderSeqEncoder instance.
        loss: Loss function (default: nn.MSELoss).
        validation_metric: Metric for validation (default: ReconstructionMSE).
        optimizer_partial: Partial optimizer constructor (params missing).
        lr_scheduler_partial: Partial scheduler constructor (optimizer missing).
    """

    def __init__(
        self,
        seq_encoder=None,
        loss=None,
        validation_metric=None,
        optimizer_partial: partial = None,
        lr_scheduler_partial: partial = None,
    ):
        if not isinstance(seq_encoder, AutoEncoderSeqEncoder):
            raise TypeError(
                f"GURModule requires an AutoEncoderSeqEncoder, got {type(seq_encoder).__name__}. "
                "Use AutoEncoderSeqEncoder from ptls.nn.seq_encoder.autoencoder_encoder."
            )

        if loss is None:
            loss = nn.MSELoss()

        if validation_metric is None:
            validation_metric = ReconstructionMSE()

        super().__init__(
            validation_metric,
            seq_encoder,
            loss,
            optimizer_partial,
            lr_scheduler_partial,
        )

    @property
    def metric_name(self):
        return 'reconstruction_mse'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        """Forward pass + reconstruction for loss computation.

        Args:
            x: PaddedBatch of event sequences.
            y: Dummy labels (unused — reconstruction is self-supervised).

        Returns:
            (reconstruction, target) for loss computation.
        """
        latent = self(x)  # AutoEncoderSeqEncoder.forward -> (B, latent_dim)
        reconstruction = self._seq_encoder.reconstruct(latent)  # (B, agg_dim)
        target = self._seq_encoder.get_cached_aggregated()      # (B, agg_dim), detached + cleared
        if target is None:
            raise RuntimeError(
                "Aggregated features cache is empty. This should not happen — "
                "forward() should populate the cache before shared_step reads it."
            )
        return reconstruction, target
