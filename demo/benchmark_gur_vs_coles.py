"""Benchmark: GUR vs CoLES on Age Group Prediction.

Compares self-supervised embedding quality on a 4-class age-bin classification task.
Uses the bundled test CSV files (ptls_tests/age-transactions.csv, ptls_tests/age-bin.csv).

Pipeline:
    1. Load & preprocess transaction data into feature dicts
    2. Train CoLES (RNN-based contrastive) and GUR (autoencoder-based) with matched configs
    3. Extract embeddings via InferenceModule
    4. Classify with GradientBoostingClassifier
    5. Report accuracy for both methods
"""

import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.seq_encoder.autoencoder_encoder import AutoEncoderSeqEncoder
from ptls.nn.seq_encoder.containers import RnnSeqEncoder
from ptls.frames.gur import GURModule, GURDataset
from ptls.frames.coles import CoLESModule, ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.datasets.dataloaders import inference_data_loader


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

EMBEDDING_DIM = 64
MAX_EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3
NUM_WORKERS = 0
SEED = 42

DATA_DIR = PROJECT_ROOT / "ptls_tests"
TRX_CSV = DATA_DIR / "age-transactions.csv"
TARGET_CSV = DATA_DIR / "age-bin.csv"


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load transaction & target CSVs, convert to list-of-dicts (feature dict format)."""
    df_trx = pd.read_csv(TRX_CSV)
    df_target = pd.read_csv(TARGET_CSV)

    # small_group needs +1 because TrxEncoder embedding index 0 is reserved for padding
    max_small_group = df_trx["small_group"].max() + 1

    # Group transactions by client_id, creating feature dicts
    records = []
    for client_id, group in df_trx.groupby("client_id"):
        group = group.sort_values("trans_date")
        records.append({
            "client_id": client_id,
            "event_time": torch.FloatTensor(group["trans_date"].values),
            "small_group": torch.LongTensor(group["small_group"].values),
            "amount_rur": torch.FloatTensor(group["amount_rur"].values),
        })

    # Attach labels
    target_map = dict(zip(df_target["client_id"], df_target["bins"]))
    for rec in records:
        rec["target"] = target_map.get(rec["client_id"], -1)

    # Filter out records without target
    records = [r for r in records if r["target"] >= 0]

    print(f"  Loaded {len(records)} clients, max_small_group={max_small_group}")
    print(f"  Label distribution: {pd.Series([r['target'] for r in records]).value_counts().sort_index().to_dict()}")
    return records, max_small_group


def split_data(records, test_size=0.3):
    """Split records into train/test, preserving client-level splits."""
    targets = [r["target"] for r in records]
    train_idx, test_idx = train_test_split(
        range(len(records)), test_size=test_size, random_state=SEED, stratify=targets
    )
    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    return train_records, test_records


# ═══════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════

def build_trx_encoder(max_small_group):
    """Shared TrxEncoder config for both models."""
    return TrxEncoder(
        embeddings={"small_group": {"in": max_small_group + 1, "out": 16}},
        numeric_values={"amount_rur": "identity"},
    )


def build_coles(max_small_group):
    """Build CoLES module with RNN encoder."""
    trx_encoder = build_trx_encoder(max_small_group)
    input_size = trx_encoder.output_size

    seq_encoder = RnnSeqEncoder(
        trx_encoder=trx_encoder,
        hidden_size=EMBEDDING_DIM,
        type="gru",
        is_reduce_sequence=True,
    )

    module = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=LR),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9),
    )
    return module


def build_gur(max_small_group):
    """Build GUR module with autoencoder encoder."""
    trx_encoder = build_trx_encoder(max_small_group)

    seq_encoder = AutoEncoderSeqEncoder(
        trx_encoder=trx_encoder,
        latent_dim=EMBEDDING_DIM,
        encoder_hidden_dims=[128],
        time_windows=[7, 30, 180],
        time_unit=1.0,  # trans_date in the CSV is in day-offset units
        dropout=0.1,
    )

    module = GURModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=LR),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9),
    )
    return module


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_coles(module, train_records, valid_records):
    """Train CoLES with contrastive learning on subsequence splits."""
    train_ds = ColesDataset(
        data=MemoryMapDataset(train_records),
        splitter=SampleSlices(split_count=3, cnt_min=3, cnt_max=20),
    )
    valid_ds = ColesDataset(
        data=MemoryMapDataset(valid_records),
        splitter=SampleSlices(split_count=3, cnt_min=3, cnt_max=20),
    )

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=BATCH_SIZE,
        train_num_workers=NUM_WORKERS,
        valid_data=valid_ds,
        valid_batch_size=BATCH_SIZE,
        valid_num_workers=NUM_WORKERS,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="auto",
    )

    t0 = time.time()
    trainer.fit(module, datamodule=dm)
    train_time = time.time() - t0
    return train_time


def train_gur(module, train_records, valid_records):
    """Train GUR with autoencoder reconstruction loss."""
    train_ds = GURDataset(train_records, min_seq_len=1)
    valid_ds = GURDataset(valid_records, min_seq_len=1)

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=BATCH_SIZE,
        train_num_workers=NUM_WORKERS,
        valid_data=valid_ds,
        valid_batch_size=BATCH_SIZE,
        valid_num_workers=NUM_WORKERS,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="auto",
    )

    t0 = time.time()
    trainer.fit(module, datamodule=dm)
    train_time = time.time() - t0
    return train_time


# ═══════════════════════════════════════════════════════════════════════
# Embedding extraction & classification
# ═══════════════════════════════════════════════════════════════════════

def extract_embeddings(module, records):
    """Extract embeddings by running seq_encoder on batched data."""
    dl = inference_data_loader(records, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    module.eval()
    all_embs = []
    with torch.no_grad():
        for batch in dl:
            emb = module.seq_encoder(batch)
            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def evaluate_downstream(module, train_records, test_records, model_name):
    """Extract embeddings and classify with GBM."""
    print(f"\n  [{model_name}] Extracting embeddings...")
    X_train = extract_embeddings(module, train_records)
    X_test = extract_embeddings(module, test_records)

    y_train = np.array([r["target"] for r in train_records])
    y_test = np.array([r["target"] for r in test_records])

    print(f"  [{model_name}] Training GBM classifier (X_train: {X_train.shape}, X_test: {X_test.shape})...")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return train_acc, test_acc


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    pl.seed_everything(SEED)

    print("=" * 70)
    print("Benchmark: GUR vs CoLES on Age Group Prediction (4-class)")
    print("=" * 70)

    # --- Data ---
    print("\n[1/5] Loading data...")
    records, max_small_group = load_data()
    train_records, test_records = split_data(records)

    # Further split train into train/valid for self-supervised training
    train_ssl, valid_ssl = split_data(train_records, test_size=0.2)
    print(f"  SSL train: {len(train_ssl)}, SSL valid: {len(valid_ssl)}, Test: {len(test_records)}")

    # --- Train CoLES ---
    print(f"\n[2/5] Training CoLES ({MAX_EPOCHS} epochs)...")
    coles_module = build_coles(max_small_group)
    coles_time = train_coles(coles_module, train_ssl, valid_ssl)
    print(f"  CoLES training time: {coles_time:.1f}s")

    # --- Train GUR ---
    print(f"\n[3/5] Training GUR ({MAX_EPOCHS} epochs)...")
    gur_module = build_gur(max_small_group)
    gur_time = train_gur(gur_module, train_ssl, valid_ssl)
    print(f"  GUR training time: {gur_time:.1f}s")

    # --- Evaluate ---
    print("\n[4/5] Downstream evaluation (GBM on embeddings)...")
    coles_train_acc, coles_test_acc = evaluate_downstream(
        coles_module, train_records, test_records, "CoLES"
    )
    gur_train_acc, gur_test_acc = evaluate_downstream(
        gur_module, train_records, test_records, "GUR"
    )

    # --- Report ---
    print("\n[5/5] Results")
    print("=" * 70)
    print(f"  {'Method':<12} {'Embed Dim':<12} {'Train Time':<14} {'Train Acc':<12} {'Test Acc':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
    print(f"  {'CoLES':<12} {EMBEDDING_DIM:<12} {coles_time:<14.1f} {coles_train_acc:<12.4f} {coles_test_acc:<12.4f}")
    print(f"  {'GUR':<12} {EMBEDDING_DIM:<12} {gur_time:<14.1f} {gur_train_acc:<12.4f} {gur_test_acc:<12.4f}")
    print("=" * 70)

    print("\nNotes:")
    print("  - Dataset: 100 clients, ~10 transactions each (small test set)")
    print("  - CoLES: RNN (GRU) encoder with contrastive loss")
    print("  - GUR: Autoencoder encoder with multi-timescale aggregation + MSE loss")
    print("  - Downstream: GradientBoostingClassifier on extracted embeddings")
    print("  - For production benchmarks, use the full Age Group Prediction dataset")
    print("    from HuggingFace: dllllb/age-group-prediction")


if __name__ == "__main__":
    main()
