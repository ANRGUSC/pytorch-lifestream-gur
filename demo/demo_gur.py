"""Demo: GUR (Generalized User Representations) — Autoencoder training with pytorch-lifestream.

This script demonstrates the full pipeline:
1. Generate synthetic sequential event data
2. Configure AutoEncoderSeqEncoder with TrxEncoder + MultiTimescaleAggregator
3. Train GURModule with PyTorch Lightning
4. Extract user embeddings
5. Show interface compatibility with CoLES
"""

import numpy as np
import torch
from functools import partial

import pytorch_lightning as pl

from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder
from ptls.frames.gur import GURModule, GURDataset
from ptls.frames import PtlsDataModule


# --- Step 1: Generate synthetic sequential event data ---

def generate_synthetic_data(n_users=500, min_events=10, max_events=200, seed=42):
    """Create synthetic event sequences mimicking transaction data.

    Each user has a sequence of events with:
    - event_time: timestamps in epoch-seconds (sorted)
    - mcc_code: merchant category code (categorical, 0-99)
    - amount: transaction amount (numerical)
    """
    rng = np.random.RandomState(seed)
    data = []

    for _ in range(n_users):
        seq_len = rng.randint(min_events, max_events + 1)
        # Random timestamps over ~6 months, sorted
        base_time = 1700000000  # ~Nov 2023
        times = np.sort(rng.uniform(0, 180 * 86400, size=seq_len)) + base_time

        data.append({
            'event_time': torch.FloatTensor(times),
            'mcc_code': torch.LongTensor(rng.randint(0, 100, size=seq_len)),
            'amount': torch.FloatTensor(rng.exponential(50, size=seq_len)),
        })

    return data


def main():
    pl.seed_everything(42)

    print("=" * 60)
    print("GUR Demo: Generalized User Representations")
    print("=" * 60)

    # --- Step 1: Data ---
    print("\n[1/5] Generating synthetic data...")
    all_data = generate_synthetic_data(n_users=500)
    train_data = all_data[:400]
    valid_data = all_data[400:]
    print(f"  Train: {len(train_data)} users, Valid: {len(valid_data)} users")

    # --- Step 2: Build model ---
    print("\n[2/5] Building AutoEncoderSeqEncoder...")

    trx_encoder = TrxEncoder(
        embeddings={'mcc_code': {'in': 100, 'out': 24}},
        numeric_values={'amount': 'identity'},
    )
    trx_size = trx_encoder.output_size
    print(f"  TrxEncoder output_size: {trx_size}")

    seq_encoder = AutoEncoderSeqEncoder(
        trx_encoder=trx_encoder,
        latent_dim=64,               # smaller than paper's 120 for this demo
        encoder_hidden_dims=[128],   # single hidden layer for simplicity
        time_windows=[7, 30, 180],   # 1 week, 1 month, 6 months
        time_unit=86400.0,           # event_time is epoch seconds
        dropout=0.1,
    )

    agg_dim = trx_size * 3  # 3 time windows
    print(f"  Aggregated dim: {agg_dim}")
    print(f"  Latent dim: {seq_encoder.embedding_size}")
    print(f"  Encoder: {agg_dim} -> 128 -> {seq_encoder.embedding_size}")
    print(f"  Decoder: {seq_encoder.embedding_size} -> 128 -> {agg_dim}")

    # --- Step 3: Train ---
    print("\n[3/5] Training GURModule...")

    module = GURModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=1e-3),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9),
    )

    train_ds = GURDataset(train_data, min_seq_len=5)
    valid_ds = GURDataset(valid_data, min_seq_len=5)

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=64,
        train_num_workers=0,
        valid_data=valid_ds,
        valid_batch_size=64,
        valid_num_workers=0,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        enable_checkpointing=False,
        enable_model_summary=True,
        accelerator='auto',
    )
    trainer.fit(module, datamodule=dm)

    # --- Step 4: Extract embeddings ---
    print("\n[4/5] Extracting user embeddings...")

    module.eval()
    with torch.no_grad():
        # Get a batch from validation set
        batch_data = [valid_data[i] for i in range(min(16, len(valid_data)))]
        padded, _ = GURDataset.collate_fn(batch_data)
        embeddings = module(padded)

    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Expected: (16, {seq_encoder.embedding_size})")
    assert embeddings.shape == (min(16, len(valid_data)), seq_encoder.embedding_size)
    print("  Shape verified!")

    # --- Step 5: Interface compatibility ---
    print("\n[5/5] Verifying CoLES-compatible interface...")

    # The seq_encoder can be accessed just like CoLES
    print(f"  module.seq_encoder is accessible: {module.seq_encoder is not None}")
    print(f"  embedding_size: {module.seq_encoder.embedding_size}")
    print(f"  is_reduce_sequence: {module.seq_encoder.is_reduce_sequence}")

    # Embeddings can be used for downstream tasks (classification, retrieval, etc.)
    # Just like CoLES embeddings would be used.
    print(f"  Embedding L2 norms (sample): {torch.linalg.norm(embeddings[:5], dim=-1).tolist()}")

    print("\n" + "=" * 60)
    print("Demo complete! GUR module is fully functional and CoLES-compatible.")
    print("=" * 60)


if __name__ == '__main__':
    main()
