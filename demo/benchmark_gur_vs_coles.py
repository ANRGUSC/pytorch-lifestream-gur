"""Benchmark: GUR vs CoLES on Age Group Prediction.

Compares self-supervised embedding quality on a 4-class age-bin classification task.

Modes:
    default:  Uses bundled test CSVs (~100 clients, for quick smoke tests)
    --full:   Downloads full HuggingFace dataset (~30K clients, ~26.5M transactions)

Pipeline:
    1. Load & preprocess transaction data into feature dicts
    2. Train CoLES (RNN-based contrastive) and GUR (autoencoder-based) with matched configs
    3. Extract embeddings via InferenceModule
    4. Classify with GradientBoostingClassifier
    5. Report accuracy for both methods
"""

import argparse
import sys
import time
import urllib.request
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

SEED = 42
LR = 1e-3
EMBEDDING_DIM = 64

SMALL_CONFIG = {
    "max_epochs": 15,
    "batch_size": 32,
    "num_workers": 0,
    "coles_split_count": 3,
    "coles_cnt_min": 3,
    "coles_cnt_max": 20,
    "gur_min_seq_len": 1,
    "coles_min_seq_len": 0,
}

FULL_CONFIG = {
    "max_epochs": 5,
    "batch_size": 64,
    "num_workers": 0,
    "coles_split_count": 5,
    "coles_cnt_min": 25,
    "coles_cnt_max": 200,
    "gur_min_seq_len": 25,
    "coles_min_seq_len": 25,
}

# Data paths
DATA_DIR = PROJECT_ROOT / "ptls_tests"
TRX_CSV = DATA_DIR / "age-transactions.csv"
TARGET_CSV = DATA_DIR / "age-bin.csv"

FULL_DATA_DIR = PROJECT_ROOT / "demo" / "data" / "age-group-prediction"
HF_TRX_URL = "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz"
HF_TARGET_URL = "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv"


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def download_full_dataset():
    """Download full Age Group Prediction dataset from HuggingFace (cached)."""
    FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    trx_path = FULL_DATA_DIR / "transactions_train.csv.gz"
    target_path = FULL_DATA_DIR / "train_target.csv"

    for url, path in [(HF_TRX_URL, trx_path), (HF_TARGET_URL, target_path)]:
        if path.exists():
            print(f"  Cached: {path.name}")
        else:
            print(f"  Downloading {path.name}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved: {path}")

    return trx_path, target_path


def load_data_full():
    """Load full dataset from HuggingFace with frequency encoding for small_group."""
    trx_path, target_path = download_full_dataset()

    print("  Reading transactions CSV...")
    df_trx = pd.read_csv(trx_path, compression="gzip")
    df_target = pd.read_csv(target_path)

    print(f"  {len(df_trx):,} transactions, {df_trx['client_id'].nunique():,} clients")

    # Frequency-encode small_group: rank by frequency, 1-indexed (0 = padding)
    print("  Frequency-encoding small_group...")
    freq = df_trx["small_group"].value_counts()
    freq_map = {cat: rank + 1 for rank, (cat, _) in enumerate(freq.items())}
    df_trx["small_group"] = df_trx["small_group"].map(freq_map)
    max_small_group = len(freq_map)

    # Group by client_id, sort by trans_date, convert to feature dicts
    print("  Grouping by client_id...")
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
    records = [r for r in records if r["target"] >= 0]

    print(f"  {len(records)} clients with labels, max_small_group={max_small_group}")
    print(f"  Label distribution: {pd.Series([r['target'] for r in records]).value_counts().sort_index().to_dict()}")
    return records, max_small_group


def load_data():
    """Load bundled test CSV data, convert to list-of-dicts (feature dict format)."""
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

def train_coles(module, train_records, valid_records, cfg):
    """Train CoLES with contrastive learning on subsequence splits."""
    splitter = SampleSlices(
        split_count=cfg["coles_split_count"],
        cnt_min=cfg["coles_cnt_min"],
        cnt_max=cfg["coles_cnt_max"],
    )

    i_filters = []
    if cfg["coles_min_seq_len"] > 0:
        from ptls.data_load.iterable_processing import SeqLenFilter
        i_filters.append(SeqLenFilter(min_seq_len=cfg["coles_min_seq_len"]))

    train_ds = ColesDataset(
        data=MemoryMapDataset(train_records, i_filters=i_filters),
        splitter=splitter,
    )
    valid_ds = ColesDataset(
        data=MemoryMapDataset(valid_records, i_filters=i_filters),
        splitter=splitter,
    )

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=cfg["batch_size"],
        train_num_workers=cfg["num_workers"],
        valid_data=valid_ds,
        valid_batch_size=cfg["batch_size"],
        valid_num_workers=cfg["num_workers"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="auto",
    )

    t0 = time.time()
    trainer.fit(module, datamodule=dm)
    train_time = time.time() - t0
    return train_time


def train_gur(module, train_records, valid_records, cfg):
    """Train GUR with autoencoder reconstruction loss."""
    train_ds = GURDataset(train_records, min_seq_len=cfg["gur_min_seq_len"])
    valid_ds = GURDataset(valid_records, min_seq_len=cfg["gur_min_seq_len"])

    dm = PtlsDataModule(
        train_data=train_ds,
        train_batch_size=cfg["batch_size"],
        train_num_workers=cfg["num_workers"],
        valid_data=valid_ds,
        valid_batch_size=cfg["batch_size"],
        valid_num_workers=cfg["num_workers"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
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

def extract_embeddings(module, records, cfg):
    """Extract embeddings by running seq_encoder on batched data."""
    dl = inference_data_loader(records, num_workers=cfg["num_workers"], batch_size=cfg["batch_size"])

    module.eval()
    all_embs = []
    with torch.no_grad():
        for batch in dl:
            emb = module.seq_encoder(batch)
            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def evaluate_downstream(module, train_records, test_records, model_name, cfg):
    """Extract embeddings and classify with GBM."""
    print(f"\n  [{model_name}] Extracting embeddings...")
    X_train = extract_embeddings(module, train_records, cfg)
    X_test = extract_embeddings(module, test_records, cfg)

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
    parser = argparse.ArgumentParser(description="Benchmark: GUR vs CoLES on Age Group Prediction")
    parser.add_argument("--full", action="store_true",
                        help="Use full HuggingFace dataset (~30K clients) instead of bundled test data")
    args = parser.parse_args()

    cfg = FULL_CONFIG if args.full else SMALL_CONFIG
    mode_label = "FULL (~30K clients)" if args.full else "SMALL (~100 clients)"

    pl.seed_everything(SEED)

    print("=" * 70)
    print("Benchmark: GUR vs CoLES on Age Group Prediction (4-class)")
    print(f"Mode: {mode_label}")
    print("=" * 70)

    # --- Data ---
    print("\n[1/5] Loading data...")
    if args.full:
        records, max_small_group = load_data_full()
    else:
        records, max_small_group = load_data()
    train_records, test_records = split_data(records)

    # Further split train into train/valid for self-supervised training
    train_ssl, valid_ssl = split_data(train_records, test_size=0.2)
    print(f"  SSL train: {len(train_ssl)}, SSL valid: {len(valid_ssl)}, Test: {len(test_records)}")

    # --- Train CoLES ---
    print(f"\n[2/5] Training CoLES ({cfg['max_epochs']} epochs, batch_size={cfg['batch_size']})...")
    coles_module = build_coles(max_small_group)
    coles_time = train_coles(coles_module, train_ssl, valid_ssl, cfg)
    print(f"  CoLES training time: {coles_time:.1f}s")

    # --- Train GUR ---
    print(f"\n[3/5] Training GUR ({cfg['max_epochs']} epochs, batch_size={cfg['batch_size']})...")
    gur_module = build_gur(max_small_group)
    gur_time = train_gur(gur_module, train_ssl, valid_ssl, cfg)
    print(f"  GUR training time: {gur_time:.1f}s")

    # --- Evaluate ---
    print("\n[4/5] Downstream evaluation (GBM on embeddings)...")
    coles_train_acc, coles_test_acc = evaluate_downstream(
        coles_module, train_records, test_records, "CoLES", cfg
    )
    gur_train_acc, gur_test_acc = evaluate_downstream(
        gur_module, train_records, test_records, "GUR", cfg
    )

    # --- Report ---
    print("\n[5/5] Results")
    print("=" * 70)
    print(f"  {'Method':<12} {'Embed Dim':<12} {'Train Time':<14} {'Train Acc':<12} {'Test Acc':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
    print(f"  {'CoLES':<12} {EMBEDDING_DIM:<12} {coles_time:<14.1f} {coles_train_acc:<12.4f} {coles_test_acc:<12.4f}")
    print(f"  {'GUR':<12} {EMBEDDING_DIM:<12} {gur_time:<14.1f} {gur_train_acc:<12.4f} {gur_test_acc:<12.4f}")
    print("=" * 70)

    if args.full:
        print("\nNotes:")
        print("  - Dataset: ~30K clients from HuggingFace dllllb/age-group-prediction")
        print("  - CoLES: RNN (GRU) encoder with contrastive loss")
        print("  - GUR: Autoencoder encoder with multi-timescale aggregation + MSE loss")
        print("  - Downstream: GradientBoostingClassifier on extracted embeddings")
    else:
        print("\nNotes:")
        print("  - Dataset: ~100 clients (small test set)")
        print("  - CoLES: RNN (GRU) encoder with contrastive loss")
        print("  - GUR: Autoencoder encoder with multi-timescale aggregation + MSE loss")
        print("  - Downstream: GradientBoostingClassifier on extracted embeddings")
        print("  - For full benchmark, run with --full flag")


if __name__ == "__main__":
    main()
