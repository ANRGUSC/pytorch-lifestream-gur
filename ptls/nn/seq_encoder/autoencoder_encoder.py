"""Autoencoder-based sequence encoder for Generalized User Representations.

Implements the Spotify GUR approach (arXiv 2403.00584) as a
pytorch-lifestream SeqEncoderContainer-compatible module.

The architecture:
    Event Sequences -> TrxEncoder -> MultiTimescaleAggregator -> Encoder MLP -> Latent Embedding
                                                                  Decoder MLP <- (training only)
"""

import warnings

import torch
import torch.nn as nn

from ptls.data_load.padded_batch import PaddedBatch


class MultiTimescaleAggregator(nn.Module):
    """Aggregates event embeddings over multiple time windows via masked mean pooling.

    For each time window, computes the mean of event embeddings whose timestamps
    fall within that window (measured backward from the most recent event).

    Args:
        time_windows: Time window sizes in days. Default: [7, 30, 180] per the paper.
        time_unit: Scale factor converting event_time units to days.
            'auto' (default) detects epoch-seconds vs day offsets.
            Use 86400.0 for epoch-seconds, 1.0 for day units.
    """

    def __init__(self, time_windows=None, time_unit='auto'):
        super().__init__()
        self.time_windows = time_windows or [7, 30, 180]
        self.time_unit = time_unit

    @property
    def num_windows(self):
        return len(self.time_windows)

    def _detect_time_unit(self, event_time):
        max_val = event_time.max().item()
        if max_val > 1e6:
            return 86400.0
        return 1.0

    def forward(self, embeddings, event_time, seq_lens):
        """
        Args:
            embeddings: (B, T, H) per-event embeddings from TrxEncoder.
            event_time: (B, T) timestamps, or None for simple mean pooling.
            seq_lens: (B,) actual sequence lengths.

        Returns:
            (B, num_windows * H) concatenated per-window mean embeddings.
        """
        B, T, H = embeddings.shape
        device = embeddings.device

        # Padding mask: True for valid positions
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_mask = positions < seq_lens.unsqueeze(1)  # (B, T)

        if event_time is None:
            # Fallback: mean pool all valid events, repeat for each window
            warnings.warn(
                "event_time not found in input data. "
                "MultiTimescaleAggregator is falling back to simple mean pooling "
                "(all time windows will be identical). Pass event_time in your "
                "feature dicts for proper multi-timescale aggregation.",
                stacklevel=3,
            )
            mask_f = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
            counts = mask_f.sum(dim=1).clamp(min=1)    # (B, 1)
            mean_emb = (embeddings * mask_f).sum(dim=1) / counts  # (B, H)
            return mean_emb.repeat(1, self.num_windows)

        # Resolve time unit
        if self.time_unit == 'auto':
            time_unit = self._detect_time_unit(event_time)
        else:
            time_unit = float(self.time_unit)

        # Time delta from most recent event per sequence (in days)
        masked_time = event_time.float().clone()
        masked_time[~valid_mask] = float('-inf')
        max_time = masked_time.max(dim=1, keepdim=True).values  # (B, 1)
        time_delta = (max_time - event_time.float()) / time_unit  # (B, T) in days

        window_embeddings = []
        for window_days in self.time_windows:
            window_mask = valid_mask & (time_delta <= window_days)        # (B, T)
            mask_f = window_mask.unsqueeze(-1).float()                   # (B, T, 1)
            counts = mask_f.sum(dim=1).clamp(min=1)                      # (B, 1)
            window_avg = (embeddings * mask_f).sum(dim=1) / counts       # (B, H)
            window_embeddings.append(window_avg)

        return torch.cat(window_embeddings, dim=-1)  # (B, num_windows * H)


def _build_selu_mlp(in_dim, hidden_dims, out_dim, dropout):
    """Build an MLP with SELU activations and AlphaDropout."""
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(d, h), nn.SELU(), nn.AlphaDropout(p=dropout)])
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class AutoEncoderSeqEncoder(nn.Module):
    """Sequence encoder using multi-timescale aggregation + autoencoder.

    Follows the SeqEncoderContainer interface so it can be used as a
    drop-in replacement in ABSModule-based training frameworks.

    The full gradient path is: loss -> decoder -> encoder -> aggregated -> TrxEncoder.
    This trains TrxEncoder end-to-end to produce features that reconstruct well.
    The reconstruction target is detached to prevent competing gradients.

    Args:
        trx_encoder: TrxEncoder instance for embedding raw event features.
        input_size: Override for trx_encoder.output_size if needed.
        latent_dim: Latent representation size (paper default: 120).
        encoder_hidden_dims: Encoder MLP hidden layer sizes.
            Default: [agg_dim, agg_dim // 2].
        decoder_hidden_dims: Decoder MLP hidden layer sizes.
            Default: mirrors encoder_hidden_dims reversed.
        time_windows: Aggregation windows in days (default: [7, 30, 180]).
        time_unit: Time unit for event_time ('auto', 86400.0, 1.0).
        dropout: AlphaDropout rate (default: 0.1).
        col_time: Event time column name (default: 'event_time').
    """

    def __init__(
        self,
        trx_encoder=None,
        input_size=None,
        latent_dim=120,
        encoder_hidden_dims=None,
        decoder_hidden_dims=None,
        time_windows=None,
        time_unit='auto',
        dropout=0.1,
        col_time='event_time',
    ):
        super().__init__()

        self.trx_encoder = trx_encoder
        self.col_time = col_time
        self._is_reduce_sequence = True
        self._latent_dim = latent_dim

        trx_size = input_size if input_size is not None else trx_encoder.output_size
        self.aggregator = MultiTimescaleAggregator(
            time_windows=time_windows,
            time_unit=time_unit,
        )
        agg_dim = trx_size * self.aggregator.num_windows
        self._agg_dim = agg_dim

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [agg_dim, agg_dim // 2]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))

        self.encoder = _build_selu_mlp(agg_dim, encoder_hidden_dims, latent_dim, dropout)
        self.decoder = _build_selu_mlp(latent_dim, decoder_hidden_dims, agg_dim, dropout)

        self._cached_aggregated = None
        self._init_weights()

    def _init_weights(self):
        """Lecun-normal init for SELU compatibility."""
        for module in [self.encoder, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # --- SeqEncoderContainer interface ---

    @property
    def is_reduce_sequence(self):
        return self._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self._is_reduce_sequence = value

    @property
    def embedding_size(self):
        return self._latent_dim

    @property
    def category_max_size(self):
        return self.trx_encoder.category_max_size

    @property
    def category_names(self):
        return set(self.trx_encoder.embeddings.keys())

    # --- Public API for GURModule ---

    def reconstruct(self, latent):
        """Decode latent embeddings back to aggregated feature space.

        Args:
            latent: (B, latent_dim) tensor from encoder.

        Returns:
            (B, agg_dim) reconstructed aggregated features.
        """
        return self.decoder(latent)

    def get_cached_aggregated(self):
        """Return and clear the cached aggregated features from last forward().

        The cache is populated during forward() and contains the detached
        aggregated features used as the reconstruction target.

        Returns:
            (B, agg_dim) detached tensor, or None if forward() hasn't been called.
        """
        val = self._cached_aggregated
        self._cached_aggregated = None
        return val

    def forward(self, x, names=None, seq_len=None):
        """
        Args:
            x: PaddedBatch with dict payload containing event features.
            names: Unused, for interface compatibility.
            seq_len: Unused, for interface compatibility.

        Returns:
            Tensor (B, latent_dim) user embeddings.
        """
        # Extract event_time before TrxEncoder transforms the payload
        event_time = None
        if isinstance(x.payload, dict) and self.col_time in x.payload:
            event_time = x.payload[self.col_time].float()

        # Embed events via TrxEncoder
        x = self.trx_encoder(x)  # PaddedBatch(B, T, H_trx)

        # Aggregate over time windows
        aggregated = self.aggregator(
            x.payload, event_time, x.seq_lens,
        )  # (B, num_windows * H_trx)

        # Cache detached target for reconstruction loss in shared_step
        self._cached_aggregated = aggregated.detach()

        # Encode to latent space
        latent = self.encoder(aggregated)  # (B, latent_dim)
        return latent
