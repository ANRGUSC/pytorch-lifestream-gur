"""Tests for AutoEncoderSeqEncoder and MultiTimescaleAggregator."""

import torch
import pytest

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import (
    AutoEncoderSeqEncoder,
    MultiTimescaleAggregator,
)


# --- Fixtures ---

def make_trx_encoder():
    return TrxEncoder(
        embeddings={'mcc_code': {'in': 10, 'out': 4}},
        numeric_values={'amount': 'identity'},
    )


def make_padded_batch(batch_size=4, max_seq_len=20):
    """Create a PaddedBatch with event_time, mcc_code, amount."""
    seq_lens = torch.randint(5, max_seq_len + 1, (batch_size,))
    event_time = torch.zeros(batch_size, max_seq_len)
    mcc_code = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    amount = torch.zeros(batch_size, max_seq_len)

    for i in range(batch_size):
        L = seq_lens[i].item()
        event_time[i, :L] = torch.sort(torch.rand(L) * 180)[0]  # days 0-180
        mcc_code[i, :L] = torch.randint(0, 10, (L,))
        amount[i, :L] = torch.randn(L)

    payload = {
        'event_time': event_time,
        'mcc_code': mcc_code,
        'amount': amount,
    }
    return PaddedBatch(payload, seq_lens)


# --- MultiTimescaleAggregator Tests ---

def test_multi_timescale_aggregator_basic():
    agg = MultiTimescaleAggregator(time_windows=[7, 30, 180], time_unit=1.0)
    B, T, H = 4, 20, 8
    embeddings = torch.randn(B, T, H)
    event_time = torch.zeros(B, T)
    seq_lens = torch.full((B,), T, dtype=torch.long)
    for i in range(B):
        event_time[i] = torch.sort(torch.rand(T) * 180)[0]

    out = agg(embeddings, event_time, seq_lens)
    assert out.shape == (B, 3 * H), f"Expected ({B}, {3 * H}), got {out.shape}"


def test_multi_timescale_aggregator_no_event_time():
    agg = MultiTimescaleAggregator(time_windows=[7, 30], time_unit=1.0)
    B, T, H = 3, 10, 5
    embeddings = torch.randn(B, T, H)
    seq_lens = torch.full((B,), T, dtype=torch.long)

    with pytest.warns(UserWarning, match="event_time not found"):
        out = agg(embeddings, None, seq_lens)

    assert out.shape == (B, 2 * H)
    # All windows should be identical (same mean pooling)
    torch.testing.assert_close(out[:, :H], out[:, H:])


def test_multi_timescale_aggregator_single_event():
    agg = MultiTimescaleAggregator(time_windows=[7, 30, 180], time_unit=1.0)
    B, T, H = 2, 5, 4
    embeddings = torch.randn(B, T, H)
    seq_lens = torch.ones(B, dtype=torch.long)  # only 1 event per sequence
    event_time = torch.zeros(B, T)
    event_time[:, 0] = 100.0  # single event at day 100

    out = agg(embeddings, event_time, seq_lens)
    assert out.shape == (B, 3 * H)
    # All windows should contain the same single event
    torch.testing.assert_close(out[:, :H], out[:, H:2*H])
    torch.testing.assert_close(out[:, :H], out[:, 2*H:])


def test_multi_timescale_aggregator_empty_window():
    """Events only in last 2 days — the 7-day window sees them but
    a hypothetical 0-day window would not (tested via very small window)."""
    agg = MultiTimescaleAggregator(time_windows=[0.5, 7, 180], time_unit=1.0)
    B, H = 2, 4
    T = 5
    embeddings = torch.randn(B, T, H)
    seq_lens = torch.full((B,), T, dtype=torch.long)
    # All events at day 100 (time_delta = 0 for all)
    event_time = torch.full((B, T), 100.0)

    out = agg(embeddings, event_time, seq_lens)
    assert out.shape == (B, 3 * H)
    # All events have time_delta=0, so all windows include all events
    # (even 0.5 day window includes events at delta=0)
    torch.testing.assert_close(out[:, :H], out[:, H:2*H])


def test_multi_timescale_aggregator_respects_padding():
    """Padding positions should not contribute to aggregation."""
    agg = MultiTimescaleAggregator(time_windows=[180], time_unit=1.0)
    B, T, H = 1, 10, 3
    embeddings = torch.zeros(B, T, H)
    embeddings[0, 0, :] = 1.0  # only first event is non-zero
    embeddings[0, 5:, :] = 999.0  # padding positions with large values
    seq_lens = torch.tensor([5])  # only first 5 positions are valid
    event_time = torch.arange(T).unsqueeze(0).float()  # 0..9

    out = agg(embeddings, event_time, seq_lens)
    # Should average over positions 0-4 only, ignoring 999s in 5-9
    expected_mean = embeddings[0, :5, :].mean(dim=0)
    torch.testing.assert_close(out[0, :H], expected_mean)


# --- AutoEncoderSeqEncoder Tests ---

def test_autoencoder_seq_encoder_forward():
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(
        trx_encoder=trx_enc,
        latent_dim=16,
        encoder_hidden_dims=[32],
        time_windows=[7, 30, 180],
        time_unit=1.0,
        dropout=0.0,
    )
    x = make_padded_batch(batch_size=4)
    ae.train()
    out = ae(x)
    assert out.shape == (4, 16), f"Expected (4, 16), got {out.shape}"


def test_autoencoder_seq_encoder_cached_aggregated():
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(
        trx_encoder=trx_enc, latent_dim=8, encoder_hidden_dims=[16],
        time_windows=[30], time_unit=1.0, dropout=0.0,
    )
    x = make_padded_batch(batch_size=3)
    ae.train()
    ae(x)

    cached = ae.get_cached_aggregated()
    assert cached is not None
    trx_size = trx_enc.output_size
    assert cached.shape == (3, 1 * trx_size)  # 1 window
    assert not cached.requires_grad  # should be detached

    # Cache should be cleared after get
    assert ae.get_cached_aggregated() is None


def test_autoencoder_seq_encoder_reconstruct():
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(
        trx_encoder=trx_enc, latent_dim=8, encoder_hidden_dims=[16],
        time_windows=[7, 30], time_unit=1.0, dropout=0.0,
    )
    latent = torch.randn(4, 8)
    recon = ae.reconstruct(latent)
    expected_agg_dim = trx_enc.output_size * 2
    assert recon.shape == (4, expected_agg_dim)


def test_autoencoder_seq_encoder_embedding_size():
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(trx_encoder=trx_enc, latent_dim=64)
    assert ae.embedding_size == 64


def test_autoencoder_seq_encoder_gradient_flow():
    """Verify gradients flow through encoder to TrxEncoder."""
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(
        trx_encoder=trx_enc, latent_dim=8, encoder_hidden_dims=[16],
        time_windows=[30], time_unit=1.0, dropout=0.0,
    )
    ae.train()
    x = make_padded_batch(batch_size=2)
    latent = ae(x)
    target = ae.get_cached_aggregated()
    reconstruction = ae.reconstruct(latent)
    loss = torch.nn.functional.mse_loss(reconstruction, target)
    loss.backward()

    # TrxEncoder embedding should have gradients
    for name, param in trx_enc.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for TrxEncoder param: {name}"
            break


def test_autoencoder_seq_encoder_category_names():
    trx_enc = make_trx_encoder()
    ae = AutoEncoderSeqEncoder(trx_encoder=trx_enc, latent_dim=8)
    names = ae.category_names
    assert 'mcc_code' in names
